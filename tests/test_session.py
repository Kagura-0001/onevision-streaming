from __future__ import annotations

import asyncio
import base64
import io
import re
import unittest
from typing import Any
from unittest.mock import patch

import httpx
from PIL import Image

from ov_streaming.serve import SessionRegistry, build_app, parse_args
from ov_streaming.session import OVSession

from .fakes import BranchPlan, FakeBackend
from .test_renderers import FIXTURES, IMAGE, make_ov21, make_univia


def messages(timestamp_sec, *, text="", image=None, **source):
    content = []
    if image is not None:
        content.append(
            {
                "type": "image_pil",
                "image_pil": image,
                "timestamp_sec": timestamp_sec,
                **source,
            }
        )
    if text:
        content.append({"type": "text", "text": text, "timestamp_sec": timestamp_sec})
    return [{"role": "user", "content": content}]


def make_session(
    controls=(),
    branches=(),
    *,
    renderer=None,
    stop_source_after=None,
    max_model_len=None,
    image_tokens=0,
    **options,
):
    renderer = renderer or make_univia()
    backend = FakeBackend(
        [renderer.control_ids[name] for name in controls],
        branches,
        stop_source_after=stop_source_after,
        max_model_len=max_model_len,
        image_tokens=image_tokens,
    )
    return backend, OVSession(backend, renderer, "test", **options)


def texts(backend):
    """What the source holds at each step."""
    return [text for _, text, _ in backend.steps]


class OVSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_answers_are_written_back_in_completion_order(self) -> None:
        first = BranchPlan("first answer")
        second = BranchPlan("second answer")
        backend, session = make_session(
            ["response", "response", "silence"], [first, second]
        )

        await session.send_messages(input_id="q1", messages=messages(1, text="first?"))
        await session.send_messages(input_id="q2", messages=messages(2, text="second?"))
        await session.drain()
        second.release.set()
        await asyncio.wait_for(second.completed.wait(), 1)
        first.release.set()
        await asyncio.wait_for(first.completed.wait(), 1)
        await session.send_messages(input_id="frame", messages=messages(3, image=IMAGE))
        await session.drain()
        await session.close()

        appended = texts(backend)[-1].rpartition("<|im_start|>user\n")[2]
        self.assertLess(appended.index("second answer"), appended.index("first answer"))
        events = [event async for event in session.events()]
        responses = [
            event for event in events if event.type == "response.output_text.done"
        ]
        self.assertEqual(
            [(event.text, event.input_id, event.control) for event in responses],
            [("second answer", "q2", "response"), ("first answer", "q1", "response")],
        )

    async def test_branch_reaches_the_engine_before_the_source_resumes(self) -> None:
        submit = asyncio.Event()
        plan = BranchPlan("an answer", submit=submit)
        backend, session = make_session(["response", "silence"], [plan])
        self.addAsyncCleanup(session.abort)
        self.addCleanup(submit.set)
        await session.send_messages(messages=messages(0, text="what?", image=IMAGE))
        await session.send_messages(messages=messages(1, image=IMAGE))
        await asyncio.wait_for(plan.add_started.wait(), 1)
        await asyncio.sleep(0.01)
        self.assertEqual(len(backend.steps), 1)

        submit.set()
        await asyncio.wait_for(session.drain(), 1)
        response_id = session.renderer.control_ids["response"]
        self.assertEqual(backend.branch_suffix_token_ids, [[response_id]])
        self.assertEqual(backend.branch_source_ids, [backend.sources[0].request_id])
        # The next append starts with the control the source sampled.
        first, second = texts(backend)
        self.assertTrue(
            second.startswith(first + "</response><|im_end|>\n<|im_start|>user\n")
        )

    async def test_ov21_session_reproduces_the_frozen_stream(self) -> None:
        # Recorded from the OV2.1 adapter that renderers replaced.
        expected = re.split(
            r"^@@ .*\n",
            (FIXTURES / "ov21_session.txt").read_text(),
            flags=re.MULTILINE,
        )[1:]
        answer = BranchPlan("The door is still closed.")
        caption = BranchPlan("A man in a red apron walks in.")
        # The recording predates the switch and captions without opening the
        # session with <caption_begin>, which no session does now. Replay it
        # with captioning unconditional; it still guards the text layout.
        renderer = make_ov21()
        renderer.optional_controls = {}
        renderer.base_controls = tuple(renderer.control_ids)
        backend, session = make_session(
            ["silence", "response", "caption"] + ["silence"] * 4,
            [answer, caption],
            renderer=renderer,
            max_visual_frames=4,
            compact_to_visual_frames=1,
        )
        for item in (
            messages(0.0, image=IMAGE),
            messages(1.0, text="Tell me when the door opens.", image=IMAGE),
            messages(2.0, image=IMAGE, frame_index=60, fps=30.0),
        ):
            await session.send_messages(messages=item)
        await session.drain()
        caption.release.set()
        await asyncio.wait_for(caption.completed.wait(), 1)
        answer.release.set()
        await asyncio.wait_for(answer.completed.wait(), 1)
        for item in (
            messages(3.0, image=IMAGE),
            messages(3.5, text="What is he holding?"),
            messages(4.0, image=IMAGE),
            messages(5.0, image=IMAGE),
        ):
            await session.send_messages(messages=item)
        await session.drain()
        await session.close()

        self.assertEqual(texts(backend), expected)
        self.assertEqual(len(backend.sources), 2)

    async def test_close_waits_for_response_without_aborting_it(self) -> None:
        plan = BranchPlan("done")
        backend, session = make_session(["response"], [plan])
        await session.send_messages(messages=messages(0, text="query"))
        await session.drain()

        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        plan.release.set()
        await asyncio.wait_for(closing, 1)
        self.assertNotIn(backend.branch_ids[0], backend.aborted)
        self.assertTrue(backend.sources[0].closed)

    async def test_close_finishes_when_a_response_is_cancelled(self) -> None:
        backend, session = make_session(["response"], [BranchPlan("answer")])
        self.addAsyncCleanup(session.abort)
        await session.send_messages(messages=messages(0, text="query"))
        await session.drain()

        closing = asyncio.create_task(session.close())
        await asyncio.wait_for(asyncio.shield(session._input_task), 1)
        await session.cancel_response(backend.branch_ids[0])
        await asyncio.wait_for(closing, 1)

        events = [event async for event in session.events()]
        self.assertEqual(
            (events[-1].type, events[-1].status), ("session.closed", "completed")
        )

    async def test_cancel_on_response_created_keeps_session_usable(self) -> None:
        backend, session = make_session(["response", "silence"], [BranchPlan("answer")])
        self.addAsyncCleanup(session.abort)
        events = []

        async def consume():
            async for event in session.events():
                events.append(event)
                if event.type == "response.created":
                    await session.cancel_response(event.response_id)

        reader = asyncio.create_task(consume())
        self.addCleanup(reader.cancel)
        await session.send_messages(messages=messages(0, text="query"))
        await asyncio.wait_for(session.drain(), 1)
        await session.send_messages(messages=messages(1, image=IMAGE))
        await asyncio.wait_for(session.close(), 1)
        await asyncio.wait_for(reader, 1)

        done = [event for event in events if event.type == "response.done"]
        self.assertEqual(
            [(event.status, event.reason, event.control) for event in done],
            [("cancelled", "user", "response")],
        )
        self.assertEqual(backend.aborted, backend.branch_ids)
        inputs = [event.control for event in events if event.type == "input.done"]
        self.assertEqual(inputs, ["response", "silence"])
        self.assertEqual(events[-1].status, "completed")

    async def test_abort_leaves_no_responses_from_queued_inputs(self) -> None:
        for max_responses in (1, 3):
            with self.subTest(max_responses=max_responses):
                abort_started, release_abort = asyncio.Event(), asyncio.Event()

                class SlowAbortBackend(FakeBackend):
                    async def abort(self, request_id):
                        abort_started.set()
                        await release_abort.wait()
                        await super().abort(request_id)

                first, second = BranchPlan("first"), BranchPlan("second")
                renderer = make_univia()
                backend = SlowAbortBackend(
                    [renderer.control_ids["response"]] * 2, [first, second]
                )
                session = OVSession(
                    backend, renderer, "test", max_responses=max_responses
                )
                self.addAsyncCleanup(session.abort)
                self.addCleanup(release_abort.set)
                await session.send_messages(messages=messages(0, text="first"))
                await session.drain()
                await session.send_messages(messages=messages(1, text="second"))
                if max_responses == 1:
                    await asyncio.wait_for(abort_started.wait(), 1)
                aborting = asyncio.create_task(session.abort())
                await asyncio.wait_for(abort_started.wait(), 1)
                second_started = asyncio.create_task(second.add_started.wait())
                try:
                    await asyncio.wait(
                        {second_started, session._input_task},
                        timeout=1,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    second_started.cancel()
                    await asyncio.gather(second_started, return_exceptions=True)
                    release_abort.set()
                await asyncio.wait_for(aborting, 1)

                self.assertEqual(set(backend.aborted), set(backend.branch_ids))
                self.assertFalse(session._responses)
                self.assertEqual(
                    backend.aborted_sources, [backend.sources[0].request_id]
                )

    async def test_cancelling_drain_leaves_no_waiter_and_keeps_session_usable(
        self,
    ) -> None:
        submit = asyncio.Event()
        plan = BranchPlan("done", submit=submit)
        backend, session = make_session(["response"], [plan])
        self.addAsyncCleanup(session.abort)
        self.addCleanup(submit.set)
        await session.send_messages(messages=messages(0, text="query"))
        await asyncio.wait_for(plan.add_started.wait(), 1)

        tasks = asyncio.all_tasks()
        draining = asyncio.create_task(session.drain())
        await asyncio.sleep(0)
        draining.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await draining
        self.assertEqual(asyncio.all_tasks(), tasks)

        submit.set()
        await asyncio.wait_for(session.drain(), 1)
        plan.release.set()
        await asyncio.wait_for(session.close(), 1)
        events = [event async for event in session.events()]
        done = [event for event in events if event.type == "response.done"]
        self.assertEqual([event.status for event in done], ["completed"])

    async def test_fourth_response_cancels_oldest(self) -> None:
        plans = [BranchPlan(f"answer {index}") for index in range(4)]
        backend, session = make_session(["response"] * 4, plans, max_responses=3)
        for index in range(4):
            await session.send_messages(
                input_id=f"input-{index}",
                messages=messages(index, image=IMAGE),
            )
        await asyncio.wait_for(session.drain(), 1)

        self.assertEqual(backend.aborted, [backend.branch_ids[0]])
        await session.abort()
        events = [event async for event in session.events()]
        done = [event for event in events if event.type == "response.done"]
        self.assertEqual(
            (done[0].response_id, done[0].reason), (backend.branch_ids[0], "capacity")
        )

    async def test_abort_waits_for_submission_then_aborts_by_request_id(self) -> None:
        submit = asyncio.Event()
        plan = BranchPlan("late", submit=submit)
        backend, session = make_session(["response"], [plan])
        await session.send_messages(messages=messages(0, image=IMAGE))
        await asyncio.wait_for(plan.add_started.wait(), 1)

        aborting = asyncio.create_task(session.abort())
        await asyncio.sleep(0)
        self.assertFalse(aborting.done())
        submit.set()
        await asyncio.wait_for(aborting, 1)

        self.assertEqual(backend.aborted.count(backend.branch_ids[0]), 1)

    async def test_compaction_replays_kept_turns_on_a_new_source(self) -> None:
        backend, session = make_session(
            ["silence"] * 5, max_visual_frames=4, compact_to_visual_frames=3
        )
        for index in range(5):
            await session.send_messages(
                input_id=f"frame:{index}", messages=messages(index, image=IMAGE)
            )
        await session.drain()
        await session.close()

        old, new = backend.sources
        request_id, replay, frames = backend.steps[-1]
        self.assertEqual(request_id, new.request_id)
        self.assertEqual(len(frames), 3)
        self.assertTrue(replay.startswith("<|im_start|>user\n<2.0 seconds>"))
        self.assertIn("<4.0 seconds>", replay)
        self.assertEqual(backend.aborted_sources, [old.request_id])
        self.assertTrue(new.closed)
        self.assertEqual(old.cache_salt, new.cache_salt)

        events = [event async for event in session.events()]
        compacted = [event for event in events if event.type == "context.compacted"]
        self.assertEqual(len(compacted), 1)
        self.assertEqual(compacted[0].metadata["retained_turns"], 3)
        self.assertEqual(compacted[0].metadata["retained_visual_frames"], 3)

    async def test_source_is_compacted_before_it_outgrows_the_context(self) -> None:
        for image_tokens in (0, 150):
            with self.subTest(image_tokens=image_tokens):
                backend, session = make_session(
                    ["silence"] * 60, max_model_len=6000, image_tokens=image_tokens
                )
                for index in range(60):
                    await session.send_messages(
                        messages=messages(index, text=f"q{index}", image=IMAGE)
                    )
                await session.drain()
                await session.close()

                # Every source leaves room for a branch and its answer.
                room = 1 + session.renderer.max_response_tokens
                for source in backend.sources:
                    self.assertLessEqual(source.num_tokens + room, 6000)
                self.assertGreater(len(backend.sources), 1)
                events = [event async for event in session.events()]
                compacted = [e for e in events if e.type == "context.compacted"]
                self.assertEqual(len(compacted), len(backend.sources) - 1)
                # The first replay keeps the newest turns and archives the rest.
                replay = next(
                    text
                    for request_id, text, _ in backend.steps
                    if request_id == backend.sources[1].request_id
                )
                self.assertIn('"content": "q0"', replay)
                self.assertNotIn("<0.0 seconds>", replay)
                first = [
                    s for s in backend.steps if s[0] == backend.sources[0].request_id
                ]
                self.assertIn(f"<{len(first)}.0 seconds>", replay)

    async def test_concurrent_cancellations_finish_response_once(self) -> None:
        backend, session = make_session(["response"], [BranchPlan("answer")])
        self.addAsyncCleanup(session.abort)
        await session.send_messages(messages=messages(0, text="query"))
        await session.drain()
        response_id = backend.branch_ids[0]

        await asyncio.wait_for(
            asyncio.gather(
                session.cancel_response(response_id),
                session.cancel_response(response_id),
            ),
            1,
        )
        await session.close()
        events = [event async for event in session.events()]
        done = [event for event in events if event.type == "response.done"]
        self.assertEqual(
            [(event.status, event.reason) for event in done], [("cancelled", "user")]
        )
        self.assertEqual(backend.aborted, [response_id])

    async def test_input_failure_cancels_live_responses_and_finishes_events(
        self,
    ) -> None:
        backend, session = make_session(["response"], [BranchPlan("answer")])
        self.addAsyncCleanup(session.abort)
        await session.send_messages(messages=messages(0, text="query"))
        await session.drain()
        # The backend has no control for this second input.
        await session.send_messages(messages=messages(1, image=IMAGE))
        with self.assertRaisesRegex(RuntimeError, "session failed"):
            await asyncio.wait_for(session.drain(), 1)

        async def collect():
            return [event async for event in session.events()]

        events = await asyncio.wait_for(collect(), 1)
        self.assertEqual(
            (events[-1].type, events[-1].status), ("session.closed", "failed")
        )
        done = [event for event in events if event.type == "response.done"]
        self.assertEqual(
            [(event.status, event.reason) for event in done],
            [("cancelled", "session_error")],
        )
        self.assertEqual(backend.aborted, backend.branch_ids)
        self.assertEqual(backend.aborted_sources, [backend.sources[0].request_id])

    async def test_response_survives_compaction_and_is_written_back(self) -> None:
        response = BranchPlan("late answer")
        backend, session = make_session(
            ["response", "silence", "silence", "silence"],
            [response],
            max_visual_frames=2,
            compact_to_visual_frames=1,
        )

        await session.send_messages(messages=messages(0, image=IMAGE))
        await session.drain()
        for index in (1, 2):
            await session.send_messages(messages=messages(index, image=IMAGE))
        await session.drain()
        self.assertIn(backend.branch_source_ids[0], backend.aborted_sources)
        self.assertNotIn(backend.branch_ids[0], backend.aborted)

        response.release.set()
        await asyncio.wait_for(response.completed.wait(), 1)
        await session.send_messages(messages=messages(3, image=IMAGE))
        await session.drain()
        await session.close()

        request_id, text, _ = backend.steps[-1]
        self.assertEqual(request_id, backend.sources[-1].request_id)
        self.assertIn(
            '<answer_memory>{"timestep": "00:00:00", "content": "late answer"}', text
        )

    async def test_compaction_keeps_system_questions_and_memories(self) -> None:
        answer = BranchPlan("a man walks in")
        answer.release.set()
        backend, session = make_session(
            ["response"] + ["silence"] * 4,
            [answer],
            renderer=make_ov21(),
            max_visual_frames=2,
            compact_to_visual_frames=1,
        )

        await session.send_messages(
            messages=[
                {"role": "system", "content": "Watch the door."},
                *messages(0, text="Tell me if someone falls", image=IMAGE),
            ]
        )
        await session.drain()
        await asyncio.wait_for(answer.completed.wait(), 1)
        for index in range(1, 5):
            await session.send_messages(messages=messages(index, image=IMAGE))
        await session.drain()
        await session.close()

        self.assertEqual(len(backend.sources), 3)
        for source in backend.sources[1:]:
            for text in (
                "<|im_start|>system\nWatch the door.<|im_end|>",
                "Tell me if someone falls",
                "a man walks in",
            ):
                self.assertEqual(source.text.count(text), 1)

    async def test_raw_task_prefix_survives_repeated_compaction(self) -> None:
        renderer = make_ov21()
        renderer.task_prefix = "<caption_begin>\n"
        caption = BranchPlan("a man walks in")
        caption.release.set()
        backend, session = make_session(
            ["caption"] + ["silence"] * 5,
            [caption],
            renderer=renderer,
            max_visual_frames=2,
            compact_to_visual_frames=1,
        )
        self.addAsyncCleanup(session.abort)
        await session.send_messages(
            messages=messages(0, text="Watch the door", image=IMAGE)
        )
        await session.drain()
        await asyncio.wait_for(caption.completed.wait(), 1)
        for index in range(1, 5):
            await session.send_messages(messages=messages(index, image=IMAGE))
        await session.drain()
        await session.send_messages(messages=messages(5, text="Who arrived?"))
        await session.close()

        self.assertEqual(len(backend.sources), 3)
        for text in texts(backend):
            self.assertEqual(text.count("<caption_begin>"), 1)
            self.assertIn("<|im_start|>user\n<caption_begin>\n", text)
        for source in backend.sources[1:]:
            text = source.text
            positions = [
                text.index(item)
                for item in ("<caption_begin>", "Watch the door", "a man walks in")
            ]
            self.assertEqual(positions, sorted(positions))
        self.assertIn("Who arrived?", texts(backend)[-1])

    async def test_archive_keeps_the_order_dropped_turns_showed(self) -> None:
        # Turn 1 carries the answer to turn 0 and asks a new question.
        for renderer, order in (
            (make_univia(), ("Is the door open?", "the door is open", "Who is there?")),
            (make_ov21(), ("Is the door open?", "Who is there?", "the door is open")),
        ):
            with self.subTest(type(renderer).__name__):
                answer = BranchPlan("the door is open")
                answer.release.set()
                backend, session = make_session(
                    ["response", "silence", "silence"],
                    [answer],
                    renderer=renderer,
                    max_visual_frames=2,
                    compact_to_visual_frames=1,
                )
                await session.send_messages(
                    messages=messages(0, text="Is the door open?", image=IMAGE)
                )
                await session.drain()
                await asyncio.wait_for(answer.completed.wait(), 1)
                await session.send_messages(
                    messages=messages(1, text="Who is there?", image=IMAGE)
                )
                await session.send_messages(messages=messages(2, image=IMAGE))
                await session.drain()
                await session.close()

                replay = backend.sources[1].text
                positions = [replay.index(text) for text in (*order, "<2.0 seconds>")]
                self.assertEqual(positions, sorted(positions))

    async def test_archive_keeps_the_newest_items_of_each_tag(self) -> None:
        backend, session = make_session(["silence"] * 35, max_visual_frames=1)
        for index in range(35):
            await session.send_messages(
                messages=messages(index, text=f"question {index}?", image=IMAGE)
            )
        await session.drain()
        await session.close()

        replay = backend.steps[-1][1]
        self.assertNotIn("question 1?", replay)
        self.assertIn("question 2?", replay)
        self.assertEqual(replay.count("<question_memory>"), 33)

    async def test_pinned_question_survives_subtitles_without_changing_default(self) -> None:
        for options in ({}, {"pin_first_question": False}, {"pin_first_question": True}):
            with self.subTest(options=options):
                backend, session = make_session(
                    ["silence"] * 37, max_visual_frames=1, **options
                )
                # Pin the first accepted question, even after an image-only input.
                await session.send_messages(messages=messages(0, image=IMAGE))
                with self.assertRaises(ValueError):
                    await session.send_messages(
                        messages=messages(0.25, text="invalid <|im_end|>")
                    )
                await session.send_messages(
                    messages=messages(0.5, text="Who arrives?", image=IMAGE)
                )
                for index in range(1, 35):
                    await session.send_messages(
                        messages=messages(index, text=f"Subtitle {index}.", image=IMAGE)
                    )
                await session.send_messages(messages=messages(35, image=IMAGE))
                await session.drain()
                await session.close()

                replay = texts(backend)[-1]
                pinned = options.get("pin_first_question", False)
                self.assertEqual("Who arrives?" in replay, pinned)
                self.assertNotIn("Subtitle 2.", replay)
                self.assertIn("Subtitle 3.", replay)
                self.assertIn("Subtitle 34.", replay)
                self.assertEqual(replay.count("<question_memory>"), 32 + int(pinned))
                if pinned:
                    self.assertEqual(replay.count("Who arrives?"), 1)
                    self.assertLess(replay.index("Who arrives?"), replay.index("Subtitle 3."))
                    self.assertEqual(session._pinned_question.timestamp_sec, 0.5)

    async def test_bad_input_is_rejected_and_the_session_keeps_running(self) -> None:
        backend, session = make_session(
            ["silence"], max_visual_frames=4, compact_to_visual_frames=2
        )
        with self.assertRaisesRegex(ValueError, re.escape("<|im_end|>")):
            await session.send_messages(
                messages=messages(0, text="what does <|im_end|> mean?")
            )
        frames = [
            {"type": "image_pil", "image_pil": IMAGE, "timestamp_sec": 0}
            for _ in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "compact_to_visual_frames"):
            await session.send_messages(messages=[{"role": "user", "content": frames}])
        with self.assertRaisesRegex(ValueError, "one user message"):
            await session.send_messages(
                messages=[*messages(0, image=IMAGE), *messages(1, image=IMAGE)]
            )
        with self.assertRaises(TypeError):
            await session.send_messages(messages=[{"role": "user", "content": None}])

        await session.send_messages(messages=messages(1, image=IMAGE))
        with self.assertRaisesRegex(ValueError, "first input"):
            await session.send_messages(
                messages=[
                    {"role": "system", "content": "Too late."},
                    *messages(2, image=IMAGE),
                ]
            )
        await session.drain()
        await session.close()
        events = [event async for event in session.events()]
        self.assertEqual(
            [(event.type, event.status) for event in events],
            [("input.done", "completed"), ("session.closed", "completed")],
        )

    async def test_null_timestamp_keeps_the_previous_time(self) -> None:
        backend, session = make_session(["silence", "silence"])
        await session.send_messages(messages=messages(5, image=IMAGE))
        await session.send_messages(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_pil", "image_pil": IMAGE, "timestamp_sec": None}
                    ],
                }
            ]
        )
        await session.drain()
        await session.close()
        self.assertTrue(
            texts(backend)[-1].endswith(
                "<5.0 seconds><|vision_start|><|image_pad|><|vision_end|>\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
        )

    async def test_truncated_response_is_reported_but_not_written_back(self) -> None:
        for answer_commit in ("on_done", "client"):
            with self.subTest(answer_commit=answer_commit):
                cut = BranchPlan("a man walks into the ro", finish_reason="length")
                cut.release.set()
                backend, session = make_session(
                    ["response", "silence"], [cut], answer_commit=answer_commit
                )
                await session.send_messages(messages=messages(0, text="what happens?"))
                await session.drain()
                await asyncio.wait_for(cut.completed.wait(), 1)
                self.assertFalse(session.commit_answer(backend.branch_ids[0], "a man"))
                await session.send_messages(messages=messages(1, image=IMAGE))
                await session.drain()
                await session.close()

                events = [event async for event in session.events()]
                done = [event for event in events if event.type == "response.done"]
                self.assertEqual(
                    [(event.status, event.reason) for event in done],
                    [("completed", "length")],
                )
                self.assertNotIn("a man", texts(backend)[-1])

    async def test_client_commits_the_heard_part_of_an_answer(self) -> None:
        plan = BranchPlan("someone fell near the door")
        plan.release.set()
        backend, session = make_session(
            ["response", "silence", "silence"], [plan], answer_commit="client"
        )
        await session.send_messages(messages=messages(0, text="did anyone fall?"))
        await session.drain()
        await asyncio.wait_for(plan.completed.wait(), 1)
        await session.send_messages(messages=messages(1, image=IMAGE))
        await session.drain()

        response_id = backend.branch_ids[0]
        self.assertTrue(session.commit_answer(response_id, " someone fell "))
        self.assertFalse(session.commit_answer(response_id, "someone fell"))
        await session.send_messages(messages=messages(2, image=IMAGE))
        await session.drain()
        await session.close()

        self.assertNotIn("someone fell", texts(backend)[-2])
        self.assertIn('"content": "someone fell"}</answer_memory>', texts(backend)[-1])
        self.assertNotIn("near the door", texts(backend)[-1])
        events = [event async for event in session.events()]
        done = [e for e in events if e.type == "response.output_text.done"]
        self.assertEqual(
            [(e.text, e.control) for e in done],
            [("someone fell near the door", "response")],
        )

    async def test_empty_commit_writes_nothing_back(self) -> None:
        plan = BranchPlan("someone fell")
        plan.release.set()
        backend, session = make_session(
            ["response", "silence"], [plan], answer_commit="client"
        )
        await session.send_messages(messages=messages(0, text="did anyone fall?"))
        await session.drain()
        await asyncio.wait_for(plan.completed.wait(), 1)

        response_id = backend.branch_ids[0]
        with self.assertRaisesRegex(ValueError, "<silence>"):
            session.commit_answer(response_id, "heard <silence>")
        self.assertTrue(session.commit_answer(response_id, "  "))
        await session.send_messages(messages=messages(1, image=IMAGE))
        await session.drain()
        await session.close()
        self.assertNotIn("answer_memory", texts(backend)[-1])

    async def test_captions_commit_on_their_own_with_client_commit(self) -> None:
        caption = BranchPlan("a man walks in")
        caption.release.set()
        backend, session = make_session(
            ["caption", "silence"],
            [caption],
            renderer=make_ov21(),
            enable=["caption"],
            answer_commit="client",
        )
        await session.send_messages(messages=messages(0, image=IMAGE))
        await session.drain()
        await asyncio.wait_for(caption.completed.wait(), 1)
        await session.send_messages(messages=messages(1, image=IMAGE))
        await session.drain()
        await session.close()

        self.assertIn("a man walks in", texts(backend)[-1])
        self.assertFalse(session.commit_answer(backend.branch_ids[0], "a man"))

    async def test_caption_switch_opens_the_session_and_keeps_questions(
        self,
    ) -> None:
        answer = BranchPlan("a hat")
        answer.release.set()
        backend, session = make_session(
            ["silence", "response", "silence"],
            [answer],
            renderer=make_ov21(),
            enable=["caption"],
        )
        await session.send_messages(messages=messages(0, image=IMAGE))
        await session.drain()
        # Training writes "<caption_begin>\n<image>\n" as the first user turn.
        self.assertIn(
            "<|im_start|>user\n<caption_begin>\n<0.0 seconds>",
            texts(backend)[0],
        )
        # The switch adds captioning; it does not take questions away, and
        # <response> stays in allowed_token_ids.
        await session.send_messages(
            messages=messages(1, text="what is he holding?", image=IMAGE)
        )
        await session.drain()
        await asyncio.wait_for(answer.completed.wait(), 1)
        await session.send_messages(messages=messages(2, image=IMAGE))
        await session.drain()
        await session.close()
        self.assertIn("what is he holding?", texts(backend)[-1])
        self.assertIn("a hat", texts(backend)[-1])

    async def test_close_finishes_events_when_the_source_stopped_between_inputs(
        self,
    ) -> None:
        backend, session = make_session(["silence"], stop_source_after=1)
        await session.send_messages(messages=messages(0, image=IMAGE))
        await session.drain()
        await asyncio.wait_for(session.close(), 1)
        events = [event async for event in session.events()]
        self.assertEqual(
            (events[-1].type, events[-1].status), ("session.closed", "failed")
        )
        later = [event async for event in session.events()]
        self.assertEqual(later, [])


class ServiceContractTest(unittest.TestCase):
    def test_models_health_reports_served_name(self) -> None:
        response = asyncio.run(
            request(
                build_app(served_model_name="native-proactive"),
                "GET",
                "/v1/models",
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "native-proactive")

    def test_renderer_engine_args_are_cli_defaults(self) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.platforms.cpu import CpuPlatform

        # Parsing needs a platform; this checkout has no built one.
        with patch("vllm.platforms.current_platform", CpuPlatform()):
            ov21 = parse_args(["serve", "--model", "/ckpt/ov21", "--renderer", "ov21"])
            explicit = parse_args(
                [
                    "serve",
                    "model",
                    "--renderer",
                    "ov21",
                    "--mamba-cache-mode",
                    "align",
                    "--enable-prefix-caching",
                    "--async-scheduling",
                ]
            )
            univia = parse_args(
                [
                    "serve",
                    "/ckpt/univia",
                    "--renderer",
                    "ov_streaming.renderers.univia:UniVIARenderer",
                ]
            )

        # Existing-test compatibility: vLLM now resolves model paths in
        # AsyncEngineArgs.__post_init__. This fixture tests CLI argument
        # propagation, not checkpoint discovery or downloads.
        with patch(
            "vllm.engine.arg_utils.get_model_path",
            side_effect=lambda path, revision: path,
        ):
            engine_args = AsyncEngineArgs.from_cli_args(ov21)
        self.assertEqual(engine_args.model, "/ckpt/ov21")
        self.assertEqual(univia.model, "/ckpt/univia")
        self.assertEqual(
            engine_args.hf_overrides["architectures"],
            ["LlavaOnevision2Qwen35ForConditionalGeneration"],
        )
        self.assertEqual(engine_args.mamba_cache_mode, "none")
        self.assertFalse(engine_args.enable_prefix_caching)
        self.assertFalse(engine_args.async_scheduling)
        self.assertEqual(
            (explicit.mamba_cache_mode, explicit.enable_prefix_caching),
            ("align", True),
        )
        self.assertTrue(explicit.async_scheduling)
        self.assertEqual(
            univia.mm_processor_kwargs, {"min_pixels": 4096, "max_pixels": 131072}
        )
        self.assertFalse(univia.trust_remote_code)

    def test_visual_window_options_reach_session(self) -> None:
        registry = SessionRegistry(FakeBackend([]), make_univia(), 256)
        app = build_app(registry=registry)

        async def exercise() -> tuple[httpx.Response, httpx.Response]:
            response = await request(
                app,
                "POST",
                "/v1/ov/sessions",
                json={
                    "max_visual_frames": 60,
                    "compact_to_visual_frames": 48,
                    "pin_first_question": True,
                },
            )
            invalid = await request(
                app,
                "POST",
                "/v1/ov/sessions",
                json={
                    "max_visual_frames": 48,
                    "compact_to_visual_frames": 60,
                },
            )
            await request(
                app,
                "DELETE",
                f"/v1/ov/sessions/{response.json()['session_id']}",
            )
            return response, invalid

        response, invalid = asyncio.run(exercise())

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["max_visual_frames"], 60)
        self.assertEqual(response.json()["compact_to_visual_frames"], 48)
        self.assertTrue(response.json()["pin_first_question"])
        self.assertEqual(response.json()["answer_commit"], "on_done")
        self.assertEqual(invalid.status_code, 422)

    def test_session_beyond_engine_slots_is_rejected(self) -> None:
        app = build_app(registry=SessionRegistry(FakeBackend([]), make_univia(), 9))

        async def exercise() -> list[httpx.Response]:
            # 4 slots, 5 with a compaction source, then 4 more do not fit.
            responses = [
                await request(app, "POST", "/v1/ov/sessions", json=body)
                for body in ({}, {"max_visual_frames": 8}, {})
            ]
            await request(
                app, "DELETE", f"/v1/ov/sessions/{responses[0].json()['session_id']}"
            )
            responses.append(await request(app, "POST", "/v1/ov/sessions", json={}))
            return responses

        responses = asyncio.run(exercise())

        self.assertEqual(
            [response.status_code for response in responses], [201, 201, 503, 201]
        )

    def test_idle_sessions_are_reaped_unless_watched(self) -> None:
        registry = SessionRegistry(FakeBackend([]), make_univia(), 64)

        async def exercise() -> tuple[list[str], list[str], list[str]]:
            idle = await registry.create()
            watched = await registry.create()
            used = await registry.create()
            registry.watchers[watched.session_id] += 1
            for session_id in registry.last_seen:
                registry.last_seen[session_id] -= 100
            registry.get(used.session_id)
            reaped = await registry.reap(60)
            return reaped, [idle.session_id], sorted(registry.sessions)

        reaped, expected, left = asyncio.run(exercise())

        self.assertEqual(reaped, expected)
        self.assertEqual(len(left), 2)
        self.assertNotIn(expected[0], left)

    def test_closing_session_keeps_its_slots(self) -> None:
        univia = make_univia()
        plan = BranchPlan("It is boiling.")
        backend = FakeBackend([univia.control_ids["response"]], [plan])
        app = build_app(registry=SessionRegistry(backend, univia, 8))

        async def exercise() -> list[int]:
            created = await request(app, "POST", "/v1/ov/sessions", json={})
            session_id = created.json()["session_id"]
            await request(
                app,
                "POST",
                f"/v1/ov/sessions/{session_id}/messages",
                json={"messages": [{"role": "user", "content": "is it boiling?"}]},
            )
            await asyncio.wait_for(plan.add_started.wait(), 1)
            closing = asyncio.create_task(
                request(app, "POST", f"/v1/ov/sessions/{session_id}/close")
            )
            await asyncio.sleep(0.01)
            statuses = [
                (await request(app, "POST", "/v1/ov/sessions", json={})).status_code
                for _ in range(2)
            ]
            plan.release.set()
            await closing
            created = await request(app, "POST", "/v1/ov/sessions", json={})
            return [*statuses, created.status_code]

        self.assertEqual(asyncio.run(exercise()), [201, 503, 201])

    def test_commit_route_needs_a_waiting_answer(self) -> None:
        app = build_app(registry=SessionRegistry(FakeBackend([]), make_univia(), 256))

        async def exercise() -> tuple[httpx.Response, httpx.Response]:
            created = await request(
                app, "POST", "/v1/ov/sessions", json={"answer_commit": "client"}
            )
            session_id = created.json()["session_id"]
            commit = await request(
                app,
                "POST",
                f"/v1/ov/sessions/{session_id}/responses/missing/commit",
                json={"text": "heard"},
            )
            await request(app, "DELETE", f"/v1/ov/sessions/{session_id}")
            return created, commit

        created, commit = asyncio.run(exercise())

        self.assertEqual(created.json()["answer_commit"], "client")
        self.assertEqual(commit.status_code, 404)

    def test_decompression_bomb_is_rejected(self) -> None:
        app = build_app(registry=SessionRegistry(FakeBackend([]), make_univia(), 256))
        png = io.BytesIO()
        Image.new("RGB", (64, 64)).save(png, "PNG")
        url = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()

        async def exercise() -> httpx.Response:
            created = await request(app, "POST", "/v1/ov/sessions", json={})
            session_id = created.json()["session_id"]
            with patch("PIL.Image.MAX_IMAGE_PIXELS", 1000):
                response = await request(
                    app,
                    "POST",
                    f"/v1/ov/sessions/{session_id}/messages",
                    json={
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": url}}
                                ],
                            }
                        ]
                    },
                )
            await request(app, "DELETE", f"/v1/ov/sessions/{session_id}")
            return response

        self.assertEqual(asyncio.run(exercise()).status_code, 422)


async def request(
    app: Any,
    method: str,
    path: str,
    **kwargs: Any,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        return await client.request(method, path, **kwargs)


if __name__ == "__main__":
    unittest.main()
