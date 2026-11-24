from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from ov_streaming import Event, OVClient, OVResponseError, OVSessionError


class EventStream(httpx.AsyncByteStream):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def __aiter__(self):
        while (item := await self.queue.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item

    def send(self, *events):
        for event in events:
            self.queue.put_nowait(f"data: {json.dumps(event.to_dict())}\n\n".encode())


class ClientSessionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stream = EventStream()
        self.requests = []
        self.seen = asyncio.Queue()
        self.http = httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(self.handle)
        )
        self.session = OVClient("http://test").session()
        with patch("ov_streaming.client.httpx.AsyncClient", return_value=self.http):
            await self.session.__aenter__()
        self.addAsyncCleanup(self.session.abort)

    def handle(self, request):
        self.requests.append(request)
        if request.url.path == "/v1/ov/sessions":
            return httpx.Response(201, json={"session_id": "s"})
        if request.url.path.endswith("/events"):
            return httpx.Response(200, stream=self.stream)
        if request.url.path.endswith("/messages"):
            return httpx.Response(202, json=json.loads(request.content))
        if request.url.path.endswith("/commit"):
            waiting = request.url.path == "/v1/ov/sessions/s/responses/r/commit"
            return httpx.Response(200 if waiting else 404, json={})
        self.stream.send(Event("session.closed", "s", status="completed"))
        self.stream.queue.put_nowait(None)
        return httpx.Response(200, json={})

    def start_reader(self):
        async def consume():
            async for event in self.session.events():
                self.seen.put_nowait(event)

        reader = asyncio.create_task(consume())

        async def stop():
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

        self.addAsyncCleanup(stop)
        return reader

    async def next_event(self):
        return await asyncio.wait_for(self.seen.get(), 1)

    async def test_timestamp_is_a_content_field(self):
        input_id = await self.session.send_messages(
            input_id="frame:21",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": b"\xff\xd8\xffjpeg"},
                            "timestamp_sec": 21,
                        }
                    ],
                }
            ],
        )
        body = json.loads(self.requests[-1].content)
        self.assertEqual(input_id, "frame:21")
        self.assertNotIn("timestamp_sec", body)
        part = body["messages"][0]["content"][0]
        self.assertEqual(part["timestamp_sec"], 21)
        self.assertTrue(part["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    async def test_client_commits_what_the_user_heard(self):
        http = httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(self.handle)
        )
        session = OVClient("http://test").session(answer_commit="client")
        with patch("ov_streaming.client.httpx.AsyncClient", return_value=http):
            await session.__aenter__()
        self.addAsyncCleanup(session.abort)

        self.assertTrue(await session.commit_answer("r", "someone fell"))
        commit = self.requests[-1]
        self.assertFalse(await session.commit_answer("missing", "someone fell"))
        created = json.loads(self.requests[-3].content)
        self.assertEqual(created["answer_commit"], "client")
        self.assertEqual(commit.url.path, "/v1/ov/sessions/s/responses/r/commit")
        self.assertEqual(json.loads(commit.content), {"text": "someone fell"})

    async def test_verification_options_are_opt_in(self):
        default_body = json.loads(self.requests[0].content)
        self.assertNotIn("verification", default_body)
        self.assertNotIn("verification_timeout", default_body)
        http = httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(self.handle)
        )
        session = OVClient("http://test").session(
            verification="road_inspection", verification_timeout=2.5
        )
        with patch("ov_streaming.client.httpx.AsyncClient", return_value=http):
            await session.__aenter__()
        self.addAsyncCleanup(session.abort)
        created = json.loads(self.requests[-1].content)
        self.assertEqual(created["verification"], "road_inspection")
        self.assertEqual(created["verification_timeout"], 2.5)

    async def test_first_question_pinning_is_opt_in(self):
        self.assertNotIn("pin_first_question", json.loads(self.requests[0].content))
        http = httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(self.handle)
        )
        session = OVClient("http://test").session(pin_first_question=True)
        with patch("ov_streaming.client.httpx.AsyncClient", return_value=http):
            await session.__aenter__()
        self.addAsyncCleanup(session.abort)
        self.assertIs(json.loads(self.requests[-1].content)["pin_first_question"], True)

    async def test_late_waiter_without_prior_registration(self):
        self.start_reader()
        done = Event("response.done", "s", response_id="r", status="completed")
        self.stream.send(done)
        await self.next_event()
        self.assertEqual(await self.session.wait_response("r"), done)

    async def test_waiters_for_different_responses_and_cancellation(self):
        self.start_reader()
        cancelled = self.session.wait_response("r1")
        first = self.session.wait_response("r1")
        second = self.session.wait_response("r2")
        self.assertIsInstance(first, asyncio.Future)
        await asyncio.sleep(0)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        done = Event("response.done", "s", response_id="r2", status="completed")
        self.stream.send(done)
        self.assertEqual(await asyncio.wait_for(second, 1), done)
        self.assertFalse(first.done())
        self.stream.send(
            Event("response.done", "s", response_id="r1", status="completed")
        )
        await asyncio.wait_for(first, 1)

    async def test_failed_and_cancelled_responses_preserve_input_identity(self):
        self.start_reader()
        for status in ("failed", "cancelled"):
            with self.subTest(status=status):
                event = Event(
                    "response.done",
                    "s",
                    input_id="frame:7",
                    response_id=status,
                    status=status,
                    reason="capacity",
                )
                self.stream.send(event)
                with self.assertRaises(OVResponseError) as caught:
                    await asyncio.wait_for(self.session.wait_response(status), 1)
                self.assertEqual(caught.exception.event, event)
                self.assertEqual(str(caught.exception), "capacity")

    async def test_failed_result_can_arrive_before_any_waiter(self):
        self.start_reader()
        self.stream.send(Event("response.done", "s", response_id="r", status="failed"))
        await self.next_event()
        with self.assertRaises(OVResponseError):
            await self.session.wait_response("r")

    async def test_text_output_does_not_complete_response(self):
        self.start_reader()
        waiter = self.session.wait_response("r")
        self.stream.send(
            Event("response.output_text.done", "s", response_id="r", text="yes")
        )
        await self.next_event()
        self.assertFalse(waiter.done())
        self.stream.send(
            Event("response.done", "s", response_id="r", status="completed")
        )
        await asyncio.wait_for(waiter, 1)

    async def test_disconnect_fails_current_and_future_waiters(self):
        reader = self.start_reader()
        waiter = self.session.wait_response("r")
        self.stream.queue.put_nowait(None)
        with self.assertRaises(OVSessionError):
            await asyncio.wait_for(reader, 1)
        for pending in (waiter, self.session.wait_response("later")):
            with self.assertRaises(OVSessionError):
                await asyncio.wait_for(pending, 1)

    async def test_transport_error_keeps_its_cause(self):
        reader = self.start_reader()
        error = httpx.ReadError("connection lost")
        self.stream.queue.put_nowait(error)
        with self.assertRaises(OVSessionError) as caught:
            await asyncio.wait_for(reader, 1)
        self.assertIs(caught.exception.__cause__, error)

    async def test_server_error_reaches_reader_and_waiter(self):
        reader = self.start_reader()
        waiter = self.session.wait_response("r")
        self.stream.send(Event("error", "s", error="engine failed"))
        for pending in (reader, waiter):
            with self.assertRaisesRegex(OVSessionError, "engine failed"):
                await asyncio.wait_for(pending, 1)

    async def test_reader_cancellation_releases_waiters(self):
        reader = self.start_reader()
        self.stream.send(Event("response.created", "s", response_id="r"))
        await self.next_event()
        waiter = self.session.wait_response("r")
        reader.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await reader
        with self.assertRaises(OVSessionError):
            await asyncio.wait_for(waiter, 1)

    async def test_iterator_close_releases_waiters(self):
        events = self.session.events()
        self.stream.send(Event("response.created", "s", response_id="r"))
        await anext(events)
        waiter = self.session.wait_response("r")
        await events.aclose()
        with self.assertRaises(OVSessionError):
            await asyncio.wait_for(waiter, 1)

    async def test_server_close_is_normal_and_preserves_completed_results(self):
        reader = self.start_reader()
        done = Event("response.done", "s", response_id="r", status="completed")
        self.stream.send(done, Event("session.closed", "s", status="completed"))
        await asyncio.wait_for(reader, 1)
        self.assertEqual(await self.session.wait_response("r"), done)
        with self.assertRaises(OVSessionError):
            await self.session.wait_response("missing")

    async def test_close_drains_events_and_is_idempotent(self):
        reader = self.start_reader()
        self.stream.send(Event("response.created", "s", response_id="r"))
        await self.next_event()
        waiter = self.session.wait_response("r")
        self.stream.send(
            Event("response.done", "s", response_id="r", status="completed")
        )
        await asyncio.wait_for(self.session.close(), 1)
        await reader
        self.assertEqual((await waiter).status, "completed")
        self.assertEqual(self.requests[-1].method, "POST")
        self.assertTrue(self.requests[-1].url.path.endswith("/close"))
        count = len(self.requests)
        await self.session.close()
        self.assertEqual(len(self.requests), count)

    async def test_abort_without_reader_releases_waiters(self):
        waiter = self.session.wait_response("r")
        await self.session.abort()
        self.assertEqual(self.requests[-1].method, "DELETE")
        self.assertTrue(self.http.is_closed)
        with self.assertRaises(OVSessionError):
            await asyncio.wait_for(waiter, 1)

    async def test_invalid_lifecycle_event_fails_session(self):
        reader = self.start_reader()
        self.stream.send(Event("response.done", "s", status="completed"))
        with self.assertRaisesRegex(OVSessionError, "response_id"):
            await asyncio.wait_for(reader, 1)

    async def test_second_reader_does_not_stop_first_reader(self):
        reader = self.start_reader()
        self.stream.send(Event("response.created", "s", response_id="r"))
        await self.next_event()
        with self.assertRaisesRegex(RuntimeError, "one event consumer"):
            await anext(self.session.events())
        done = Event("response.done", "s", response_id="r", status="completed")
        self.stream.send(done)
        self.assertEqual(await self.session.wait_response("r"), done)
        self.assertFalse(reader.done())

    async def test_close_failure_still_closes_http_and_releases_waiters(self):
        waiter = self.session.wait_response("r")
        with patch.object(
            self.http, "request", side_effect=httpx.ConnectError("offline")
        ):
            with self.assertRaises(httpx.ConnectError):
                await self.session.close()
        self.assertTrue(self.http.is_closed)
        with self.assertRaises(OVSessionError):
            await asyncio.wait_for(waiter, 1)
