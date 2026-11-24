"""Caption-only branch budgets, HTTP compatibility and context reservation."""
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from ov_streaming import OVClient, OVSessionError
from ov_streaming.serve import SessionRegistry, build_app
from ov_streaming.session import OVSession
from ov_streaming.types import Turn

from .fakes import BranchPlan, FakeBackend
from .test_renderers import IMAGE, make_ov21
from .test_session import make_session, messages, texts


class CaptionBudgetTest(unittest.IsolatedAsyncioTestCase):
    async def test_only_caption_budget_changes_and_finished_memory_commits(self):
        for budget in (None, 128, 2048):
            with self.subTest(budget=budget):
                caption = BranchPlan('caption body')
                answer = BranchPlan('answer body')
                caption.release.set()
                answer.release.set()
                backend, session = make_session(
                    ['caption', 'response', 'silence'], [caption, answer],
                    renderer=make_ov21(), enable=['caption'],
                    max_caption_tokens=budget,
                )
                try:
                    for index, plan in enumerate((caption, answer)):
                        await session.send_messages(messages=messages(index, image=IMAGE))
                        await session.drain()
                        await asyncio.wait_for(plan.completed.wait(), 1)
                    await session.send_messages(messages=messages(2, image=IMAGE))
                    await session.close()
                    self.assertEqual(backend.branch_max_tokens, [256 if budget is None else budget, 256])
                    self.assertIn('caption body', texts(backend)[-1])
                    events = [event async for event in session.events()]
                    created = [event.to_dict() for event in events if event.type == 'response.created']
                    if budget is None:
                        self.assertTrue(all('max_tokens' not in event for event in created))
                    else:
                        self.assertEqual([event['max_tokens'] for event in created], [budget, 256])
                finally:
                    await session.abort()

    async def test_validation_is_consistent(self):
        for value in (0, -1, True, 1.5, '2048'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    OVClient('http://test').session(max_caption_tokens=value)
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    make_session(renderer=make_ov21(), max_caption_tokens=value)
                registry = SessionRegistry(FakeBackend([]), make_ov21(), 256)
                transport = httpx.ASGITransport(app=build_app(registry=registry))
                async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                    response = await client.post('/v1/ov/sessions', json={'max_caption_tokens': value})
                    self.assertEqual(response.status_code, 422)
                    self.assertFalse(registry.sessions)

    async def test_sdk_and_http_acknowledge_opt_in_and_omit_default(self):
        for budget in (None, 2048):
            with self.subTest(budget=budget):
                registry = SessionRegistry(FakeBackend([]), make_ov21(), 256)
                sent, received = [], []

                async def request_hook(request):
                    sent.append(json.loads(request.content) if request.content else {})

                async def response_hook(response):
                    await response.aread()
                    received.append(response.json())

                http = httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=build_app(registry=registry)),
                    base_url='http://test',
                    event_hooks={'request': [request_hook], 'response': [response_hook]},
                )
                session = OVClient('http://test').session(max_caption_tokens=budget)
                with patch('ov_streaming.client.httpx.AsyncClient', return_value=http):
                    await session.__aenter__()
                self.assertEqual(registry.sessions[session.session_id].max_caption_tokens, budget)
                self.assertEqual(sent[0].get('max_caption_tokens'), budget)
                self.assertEqual(received[0].get('max_caption_tokens'), budget)
                if budget is None:
                    self.assertNotIn('max_caption_tokens', sent[0])
                    self.assertNotIn('max_caption_tokens', received[0])
                await session.abort()
                self.assertFalse(registry.sessions)

    async def test_old_server_ignoring_budget_is_rejected_and_session_closed(self):
        requests = []

        def handle(request):
            requests.append((request.method, request.url.path))
            return httpx.Response(201 if request.method == 'POST' else 200, json={'session_id': 'old'})

        http = httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url='http://test')
        session = OVClient('http://test').session(max_caption_tokens=2048)
        with patch('ov_streaming.client.httpx.AsyncClient', return_value=http):
            with self.assertRaisesRegex(OVSessionError, 'did not confirm'):
                await session.__aenter__()
        self.assertEqual(requests, [('POST', '/v1/ov/sessions'), ('DELETE', '/v1/ov/sessions/old')])
        self.assertTrue(http.is_closed)

    def test_context_is_reserved_only_if_caption_is_enabled(self):
        for budget, enable, compacts in [(None, ['caption'], False), (2048, [], False), (2048, ['caption'], True)]:
            with self.subTest(budget=budget, enable=enable):
                backend = FakeBackend([])
                backend.max_model_len = 4000
                backend.count_tokens = lambda text: 20
                session = OVSession(backend, make_ov21(), 'reserve', enable=enable, max_caption_tokens=budget)
                session._source_tokens = 2500
                session._turns = [Turn(0, tokens=1000), Turn(1, tokens=1000)]
                kept = session._select_kept_turns(Turn(2, tokens=1000))
                self.assertEqual(kept is not None, compacts)
                if compacts:
                    # Replay + control + 2048-token caption fits the context.
                    self.assertEqual(len(kept), 1)
        with self.assertRaisesRegex(ValueError, 'smaller than max_model_len'):
            OVSession(backend, make_ov21(), 'invalid', max_caption_tokens=4000)
