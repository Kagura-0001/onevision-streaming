from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from ov_streaming.serve import SessionRegistry, build_app
from ov_streaming.verification import RoadInspectionVerification

from .fakes import BranchPlan, FakeBackend
from .test_renderers import IMAGE, make_ov21, make_univia
from .test_session import make_session, messages, request, texts


def inspection(controls, branches=(), **options):
    backend, session = make_session(
        controls,
        branches,
        renderer=make_ov21(),
        verification="road_inspection",
        **options,
    )
    backend.tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [ord(c) for c in text]
    )
    return backend, session


class VerificationPolicyTest(unittest.TestCase):
    def test_requires_explicit_complete_grounded_output_protocol(self):
        policy = RoadInspectionVerification()
        for renderer in (make_ov21(), make_univia()):
            marker = renderer.controls["response"]
            for raw, truncated, accepted, reason in (
                ("<silence>", False, False, "no_alert"),
                ("the road is damaged", False, False, "invalid_output"),
                (marker, False, False, "empty_answer"),
                (marker + "a gap", True, False, "truncated"),
                (marker + "a gap <silence>", False, False, "invalid_output"),
                (marker + "路面外缘缺失。", False, True, "accepted"),
            ):
                with self.subTest(marker=marker, raw=raw):
                    result = policy.parse(raw, renderer, truncated=truncated)
                    self.assertEqual(
                        (result.accepted, result.reason), (accepted, reason)
                    )
            suffix = renderer.render_verification_suffix(
                policy.instruction(renderer, 91)
            )
            self.assertTrue(suffix.startswith("<|im_end|>\n<|im_start|>user\n"))
            self.assertTrue(suffix.endswith("<|im_start|>assistant\n"))
            self.assertIn("91.000", suffix)


class VerificationSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_fork_does_not_block_frames_or_compaction_and_only_acceptance_commits(
        self,
    ):
        plan = BranchPlan("<response>路面外缘缺失。")
        backend, session = inspection(
            ["response", "silence", "silence", "silence"],
            [plan],
            max_visual_frames=2,
            compact_to_visual_frames=1,
        )
        self.addAsyncCleanup(session.abort)
        await session.send_messages(messages=messages(0, image=IMAGE, text="检查道路"))
        await session.drain()
        response_task = next(iter(session._responses.values())).task
        for t in (1, 2):
            await session.send_messages(messages=messages(t, image=IMAGE))
        await asyncio.wait_for(session.drain(), 1)
        self.assertFalse(plan.completed.is_set())
        self.assertEqual(len(backend.sources), 2)
        self.assertEqual(len(backend.branch_ids), 1)
        self.assertNotIn("复核视频", "".join(texts(backend)))
        self.assertNotIn("路面外缘缺失", "".join(texts(backend)))
        suffix = "".join(map(chr, backend.branch_suffix_token_ids[0]))
        self.assertIn("0.000", suffix)
        self.assertIn("复核视频", suffix)
        plan.release.set()
        await asyncio.wait_for(response_task, 1)
        await session.send_messages(messages=messages(3, image=IMAGE))
        await session.close()
        self.assertIn("路面外缘缺失", texts(backend)[-1])
        events = [e async for e in session.events()]
        outputs = [e for e in events if e.type == "response.output_text.done"]
        self.assertEqual(
            [(e.timestamp_sec, e.text) for e in outputs], [(0, "路面外缘缺失。")]
        )
        timing = outputs[0].metadata["timing"]
        self.assertGreaterEqual(
            timing["branch_complete_ms"], timing["branch_first_output_ms"]
        )
        self.assertGreaterEqual(
            timing["response_ready_ms"], timing["branch_complete_ms"]
        )
        self.assertTrue(all(v >= 0 for v in timing.values()))

    async def test_rejection_invalid_output_and_truncation_never_publish_or_commit(
        self,
    ):
        for payload, finish in (
            ("<silence>", "stop"),
            ("道路正常", "stop"),
            ("<response>有缺口", "length"),
            ("<response>", "stop"),
        ):
            with self.subTest(payload=payload, finish=finish):
                plan = BranchPlan(payload, finish_reason=finish)
                plan.release.set()
                backend, session = inspection(["response", "silence"], [plan])
                self.addAsyncCleanup(session.abort)
                await session.send_messages(messages=messages(0, image=IMAGE))
                await session.drain()
                await asyncio.wait_for(plan.completed.wait(), 1)
                await asyncio.sleep(0)
                await session.send_messages(messages=messages(1, image=IMAGE))
                await session.close()
                self.assertNotIn("answer_memory", texts(backend)[-1])
                self.assertFalse(session.commit_answer(backend.branch_ids[0], "bad"))
                events = [e async for e in session.events()]
                self.assertFalse(
                    any(e.type == "response.output_text.done" for e in events)
                )
                verdicts = [e for e in events if e.type == "verification.done"]
                self.assertEqual(len(verdicts), 1)
                self.assertFalse(verdicts[0].metadata["accepted"])
                self.assertEqual(verdicts[0].metadata["raw"], payload)

    async def test_pending_verification_bounds_work_but_later_trigger_can_alert(self):
        first, second = (
            BranchPlan("<silence>"),
            BranchPlan("<response>另一处路面缺失。"),
        )
        second.release.set()
        backend, session = inspection(["response"] * 3, [first, second])
        self.addAsyncCleanup(session.abort)
        await session.send_messages(messages=messages(0, image=IMAGE))
        await session.drain()
        response_task = next(iter(session._responses.values())).task
        await session.send_messages(messages=messages(1, image=IMAGE))
        await asyncio.wait_for(session.drain(), 1)
        self.assertEqual(len(backend.branch_ids), 1)
        first.release.set()
        await response_task
        await session.send_messages(messages=messages(2, image=IMAGE))
        await session.close()
        self.assertEqual(len(backend.branch_ids), 2)
        events = [e async for e in session.events()]
        busy = [e for e in events if e.metadata.get("verification_skipped")]
        self.assertEqual([e.timestamp_sec for e in busy], [1])
        output = [e for e in events if e.type == "response.output_text.done"]
        self.assertEqual([e.timestamp_sec for e in output], [2])

    async def test_timeout_aborts_branch_without_poisoning_source(self):
        backend, session = inspection(
            ["response", "silence"],
            [BranchPlan("<response>late")],
            verification_timeout=0.02,
        )
        self.addAsyncCleanup(session.abort)
        await session.send_messages(messages=messages(0, image=IMAGE))
        await session.drain()
        await asyncio.wait_for(next(iter(session._responses.values())).task, 1)
        self.assertEqual(backend.aborted, backend.branch_ids)
        await session.send_messages(messages=messages(1, image=IMAGE))
        await session.close()
        events = [e async for e in session.events()]
        self.assertFalse(any(e.type == "response.output_text.done" for e in events))
        done = [e for e in events if e.type == "response.done"]
        self.assertEqual(
            [(e.status, e.reason) for e in done],
            [("cancelled", "verification_timeout")],
        )
        self.assertEqual(events[-1].status, "completed")

    async def test_client_commit_and_caption_behavior_are_preserved(self):
        caption = BranchPlan("山路旁有河流。")
        answer = BranchPlan("<response>道路左缘缺失。")
        caption.release.set()
        answer.release.set()
        backend, session = inspection(
            ["caption", "response", "silence", "silence"],
            [caption, answer],
            answer_commit="client",
        )
        self.addAsyncCleanup(session.abort)
        for t, plan in ((0, caption), (1, answer)):
            await session.send_messages(messages=messages(t, image=IMAGE))
            await session.drain()
            await asyncio.wait_for(plan.completed.wait(), 1)
            await asyncio.sleep(0)
        await session.send_messages(messages=messages(2, image=IMAGE))
        await session.drain()
        self.assertIn("山路旁有河流", texts(backend)[-1])
        self.assertNotIn("道路左缘缺失", texts(backend)[-1])
        self.assertTrue(session.commit_answer(backend.branch_ids[-1], "道路左缘缺失。"))
        await session.send_messages(messages=messages(3, image=IMAGE))
        await session.close()
        self.assertIn("道路左缘缺失", texts(backend)[-1])
        self.assertEqual(
            backend.branch_suffix_token_ids[0],
            [session.renderer.control_ids["caption"]],
        )

    async def test_cancel_during_branch_admission_finishes_before_source_advances(self):
        submit = asyncio.Event()
        plan = BranchPlan("<response>never publish", submit=submit)
        backend, session = inspection(["response", "silence"], [plan])
        self.addAsyncCleanup(session.abort)
        self.addCleanup(submit.set)
        await session.send_messages(messages=messages(0, image=IMAGE))
        await plan.add_started.wait()
        cancelling = asyncio.create_task(session.cancel_response(backend.branch_ids[0]))
        await asyncio.sleep(0)
        await session.send_messages(messages=messages(1, image=IMAGE))
        self.assertEqual(len(backend.steps), 1)
        submit.set()
        await asyncio.wait_for(cancelling, 1)
        await session.close()
        self.assertEqual(len(backend.steps), 2)
        self.assertEqual(backend.aborted, backend.branch_ids)
        self.assertFalse(session._pending)
        events = [e async for e in session.events()]
        self.assertFalse(any(e.type == "response.output_text.done" for e in events))


class VerificationServiceTest(unittest.TestCase):
    def test_policy_is_explicitly_enabled_and_invalid_options_are_rejected(self):
        registry = SessionRegistry(FakeBackend([]), make_ov21(), 256)
        app = build_app(registry=registry)

        async def exercise():
            enabled = await request(
                app,
                "POST",
                "/v1/ov/sessions",
                json={
                    "verification": "road_inspection",
                    "verification_timeout": 4,
                },
            )
            self.assertEqual(enabled.status_code, 201)
            session = registry.get(enabled.json()["session_id"])
            self.assertEqual(session.verification, "road_inspection")
            self.assertEqual(session.verification_timeout, 4)
            for options in ({"verification": "unknown"}, {"verification_timeout": 0}):
                invalid = await request(app, "POST", "/v1/ov/sessions", json=options)
                self.assertEqual(invalid.status_code, 422)
            await registry.shutdown()

        asyncio.run(exercise())
