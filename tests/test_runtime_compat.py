"""Compatibility of the deployed evaluation interfaces with canonical features."""

import asyncio
import json
import threading
import unittest
from unittest.mock import patch

import httpx

from ov_streaming import OVClient
from ov_streaming.renderers import load_renderer
from ov_streaming.renderers.ov21 import OV21WindowRenderer
from ov_streaming.serve import SessionRegistry, build_app
from ov_streaming.session import OVSession
from ov_streaming.types import Memory, Turn

from .fakes import FakeBackend
from .test_renderers import IMAGE, make_ov21
from .test_session import messages, request


def window_renderer():
    renderer = make_ov21()
    renderer.__class__ = OV21WindowRenderer
    return renderer


class ClientOptionsTest(unittest.IsolatedAsyncioTestCase):
    async def test_enable_omission_empty_and_caption_with_public_limits(self):
        for enable in (None, [], ["caption"]):
            requests = []

            def handler(req):
                requests.append(req)
                return httpx.Response(201, json={"session_id": "s"})

            http = httpx.AsyncClient(
                base_url="http://test", transport=httpx.MockTransport(handler)
            )
            limits = httpx.Limits(max_keepalive_connections=0)
            client = OVClient("http://test", limits=limits)
            session = client.session(enable=enable, verification="road_inspection")
            with patch(
                "ov_streaming.client.httpx.AsyncClient", return_value=http
            ) as factory:
                await session.__aenter__()
            self.assertIs(factory.call_args.kwargs["limits"], limits)
            body = json.loads(requests[0].content)
            self.assertEqual(body["verification"], "road_inspection")
            if enable is None:
                self.assertNotIn("enable", body)
            else:
                self.assertEqual(body["enable"], enable)
            await session.abort()

        http = httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        )
        session = OVClient("http://test").session()
        with patch(
            "ov_streaming.client.httpx.AsyncClient", return_value=http
        ) as factory:
            await session.__aenter__()
        self.assertNotIn("limits", factory.call_args.kwargs)
        await session.abort()


class RendererCompatibilityTest(unittest.TestCase):
    def test_legacy_default_and_explicit_caption_modes(self):
        renderer = make_ov21()
        self.assertEqual(
            renderer.session_controls(None), (tuple(renderer.control_ids), "")
        )
        self.assertEqual(renderer.session_controls([]), (("silence", "response"), ""))
        self.assertEqual(
            renderer.session_controls(["caption", "caption"]),
            (("silence", "response", "caption"), "<caption_begin>\n"),
        )
        renderer.task_prefix = "task prefix\n"
        first = renderer.render_prompt(
            [Turn(0)], prefix=renderer.session_controls(["caption"])[1]
        )
        self.assertIn("user\ntask prefix\n<caption_begin>\n", first)
        with self.assertRaises(ValueError):
            renderer.session_controls(["unknown"])

    def test_window_rendering_matches_deployed_first_and_compacted_text(self):
        renderer = window_renderer()
        self.assertIs(load_renderer("ov21-window"), OV21WindowRenderer)
        question = Memory("question", 0, "Tell me when it moves.")
        first = renderer.render_prompt(
            [Turn(0, questions=(question,))], prefix="<caption_begin>\n"
        )
        head = (
            "<|im_start|>system\nYou are a real-time streaming video assistant.<|im_end|>\n"
            "<|im_start|>user\n<caption_begin>\n"
            '<question_memory>{"timestep": "00:00:00", "content": "Tell me when it moves."}</question_memory>\n'
        )
        self.assertEqual(
            first, head + "<0.000 seconds>\n<|im_end|>\n<|im_start|>assistant\n"
        )
        compacted = renderer.render_prompt(
            [Turn(2)], [question], prefix="<caption_begin>\n"
        )
        self.assertEqual(
            compacted, head + "<2.000 seconds>\n<|im_end|>\n<|im_start|>assistant\n"
        )
        self.assertEqual(
            int(renderer.render_frame(IMAGE, 2).data["patch_positions"][0, 0]), 60
        )
        self.assertEqual(
            int(make_ov21().render_frame(IMAGE, 2).data["patch_positions"][0, 0]), 2000
        )


class LifecycleAndBudgetTest(unittest.IsolatedAsyncioTestCase):
    async def test_feature_options_and_token_compaction_reserve_slots(self):
        renderer = make_ov21()
        backend = FakeBackend([])
        backend.max_model_len = 4096
        registry = SessionRegistry(backend, renderer, 4)
        app = build_app(registry=registry)
        result = await request(
            app, "POST", "/v1/ov/sessions", json={"enable": [], "max_responses": 1}
        )
        self.assertEqual(result.status_code, 201)
        session = registry.get(result.json()["session_id"])
        self.assertNotIn("caption", session._controls.values())
        self.assertEqual(result.json()["enable"], [])
        with self.assertRaisesRegex(RuntimeError, "no free request slots"):
            await registry.create(max_responses=1)
        registry.last_seen[session.session_id] = 0
        registry.watchers[session.session_id] = 1
        self.assertEqual(await registry.reap(1), [])
        registry.watchers[session.session_id] = 0
        self.assertEqual(await registry.reap(1), [session.session_id])
        self.assertFalse(registry.sessions)

    async def test_threaded_input_cannot_enqueue_after_close(self):
        renderer = make_ov21()
        session = OVSession(FakeBackend([]), renderer, "closing")
        entered, release = threading.Event(), threading.Event()
        original = session._read_input

        def slow_read(value):
            entered.set()
            release.wait(2)
            return original(value)

        with patch.object(session, "_read_input", side_effect=slow_read):
            sending = asyncio.create_task(
                session.send_messages(messages=messages(0, text="question"))
            )
            await asyncio.to_thread(entered.wait, 1)
            await session.close()
            release.set()
            with self.assertRaisesRegex(RuntimeError, "session is closing"):
                await sending
        self.assertIsNone(session._input_task)

    async def test_token_budget_rolls_window_and_keeps_question(self):
        renderer = window_renderer()
        backend = FakeBackend(
            [renderer.control_ids["silence"]] * 12,
            max_model_len=1700,
            image_tokens=20,
        )
        backend.count_tokens = len
        original_open = backend.open_source

        def open_source(*args, **kwargs):
            source = original_open(*args, **kwargs)
            step = source.step

            async def counted(prompt, frames, **options):
                result = await step(prompt, frames, **options)
                self.assertLessEqual(source.num_tokens + 257, backend.max_model_len)
                return result

            source.step = counted
            return source

        backend.open_source = open_source
        session = OVSession(backend, renderer, "budget", enable=[])
        for index in range(12):
            await session.send_messages(
                messages=messages(
                    index, image=IMAGE, text="Watch every move." if index == 0 else ""
                )
            )
            await session.drain()
        await session.close()
        self.assertGreater(len(backend.sources), 1)
        self.assertIn("Watch every move.", backend.steps[-1][1])
        self.assertTrue(all(len(text) + 257 <= 1700 for _, text, _ in backend.steps))
        self.assertEqual(session._image_tokens, 20)
        events = [event async for event in session.events()]
        self.assertTrue(any(event.type == "context.compacted" for event in events))
        self.assertTrue(
            all(
                "timing" in event.metadata
                for event in events
                if event.type == "input.done"
            )
        )
