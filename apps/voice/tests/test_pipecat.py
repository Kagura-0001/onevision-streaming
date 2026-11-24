from __future__ import annotations

import asyncio
import subprocess
import sys
import unittest
from unittest.mock import AsyncMock, patch

from ov_streaming.types import Event

ANSWER = "The pot is boiling. Turn down the heat."


class FakeSession:
    def __init__(self, events: list[Event]) -> None:
        self.calls: list[tuple] = []
        self._events = events
        self._closed = asyncio.Event()

    async def __aenter__(self):
        return self

    async def send_messages(self, *, messages):
        self.calls.append(("send", messages[0]["content"]))
        return "input"

    async def commit_answer(self, response_id: str, text: str) -> bool:
        self.calls.append(("commit", response_id, text))
        return True

    async def cancel_response(self, response_id: str) -> bool:
        self.calls.append(("cancel", response_id))
        return True

    async def events(self):
        for event in self._events:
            yield event
        await self._closed.wait()

    async def close(self) -> None:
        self._closed.set()

    async def abort(self) -> None:
        self._closed.set()

    def sent(self) -> list[list[dict]]:
        return [call[1] for call in self.calls if call[0] == "send"]


class RefusedSession(FakeSession):
    async def __aenter__(self):
        raise RuntimeError("OV server returned 503: the engine has no free slots")


class FailedSession(FakeSession):
    async def events(self):
        raise RuntimeError("source engine stopped")
        yield


class FakeClient:
    def __init__(self, events: list[Event] = ()) -> None:
        self.ov = FakeSession(list(events))
        self.options = None

    def session(self, **options) -> FakeSession:
        self.options = options
        return self.ov


def answer_events(response_id: str, text: str, control: str = "response") -> list:
    return [
        Event("response.created", "s", response_id=response_id, control=control),
        Event(
            "response.output_text.done",
            "s",
            response_id=response_id,
            text=text,
            control=control,
        ),
        Event(
            "response.done",
            "s",
            response_id=response_id,
            status="completed",
            control=control,
        ),
    ]


class ImportBoundaryTest(unittest.TestCase):
    def test_voice_does_not_import_the_server(self) -> None:
        # Installed next to ov-streaming, the app could import the server; it
        # must reach OV only through the client.
        modules = ["ov_voice.answers", "ov_voice.pipecat", "ov_voice.__main__"]
        code = f"import sys, {', '.join(modules)}; print(*sys.modules)"
        output = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, check=True, text=True
        ).stdout
        imported = set(output.split())

        self.assertIn("ov_streaming.client", imported)
        for module in ("session", "backend", "renderers", "serve"):
            self.assertNotIn(f"ov_streaming.{module}", imported)


class OVProcessorTest(unittest.IsolatedAsyncioTestCase):
    async def run_processor(self, client, frames, **kwargs):
        from pipecat.tests.utils import run_test

        from ov_voice.pipecat import OVProcessor

        processor = OVProcessor(client, frame_interval=0.2, **kwargs)
        return await run_test(processor, frames_to_send=frames)

    def camera(self):
        from pipecat.frames.frames import UserImageRawFrame

        return UserImageRawFrame(image=bytes(12), size=(2, 2), format="RGB")

    def transcript(self, text):
        from pipecat.frames.frames import TranscriptionFrame

        return TranscriptionFrame(text, "", "")

    async def test_question_rides_on_the_next_frame(self) -> None:
        from pipecat.tests.utils import SleepFrame

        client = FakeClient()
        await self.run_processor(
            client,
            [
                self.camera(),
                self.transcript("is the pot boiling?"),
                self.camera(),
                SleepFrame(0.25),
                self.camera(),
                SleepFrame(0.3),
            ],
            max_visual_frames=96,
        )

        self.assertEqual(
            client.options,
            {
                "max_visual_frames": 96,
                "compact_to_visual_frames": None,
                "enable": [],
            },
        )
        first, second = client.ov.sent()
        self.assertEqual([part["type"] for part in first], ["image_url"])
        self.assertEqual([part["type"] for part in second], ["image_url", "text"])
        self.assertEqual(second[1]["text"], "is the pot boiling?")
        self.assertEqual(second[0]["image_url"]["url"].size, (2, 2))
        self.assertEqual(second[0]["timestamp_sec"], second[1]["timestamp_sec"])
        self.assertGreaterEqual(
            second[0]["timestamp_sec"] - first[0]["timestamp_sec"], 0.2
        )

    async def test_question_goes_alone_without_a_camera(self) -> None:
        client = FakeClient()
        await self.run_processor(client, [self.transcript("hello?")])

        [content] = client.ov.sent()
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["text"], "hello?")

    async def test_question_goes_alone_when_the_camera_stops(self) -> None:
        from pipecat.tests.utils import SleepFrame

        client = FakeClient()
        await self.run_processor(
            client,
            [
                self.camera(),
                SleepFrame(0.1),
                self.transcript("is it boiling?"),
                SleepFrame(0.6),
            ],
        )

        frame, question = client.ov.sent()
        self.assertEqual([part["type"] for part in question], ["text"])
        self.assertGreaterEqual(
            question[0]["timestamp_sec"] - frame[0]["timestamp_sec"], 0.4
        )

    async def test_session_failure_cancels_the_pipeline(self) -> None:
        from pipecat.frames.frames import CancelFrame, CancelWorkerFrame, ErrorFrame
        from pipecat.tests.utils import SleepFrame

        for session in (RefusedSession([]), FailedSession([])):
            with self.subTest(type(session).__name__):
                client = FakeClient()
                client.ov = session
                down, up = await self.run_processor(
                    client, [SleepFrame(0.05), self.camera()]
                )

                self.assertEqual(
                    [type(frame) for frame in up], [ErrorFrame, CancelWorkerFrame]
                )
                self.assertEqual([type(frame) for frame in down], [CancelFrame])
                self.assertEqual(session.sent(), [])

    async def test_responses_are_spoken_sentence_by_sentence(self) -> None:
        from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSSpeakFrame
        from pipecat.processors.frame_processor import FrameDirection
        from pipecat.tests.utils import SleepFrame, run_test

        from ov_voice.pipecat import OVProcessor

        client = FakeClient(
            [
                *answer_events("c1", "A pot on the stove.", control="caption"),
                *answer_events("r1", ANSWER),
            ]
        )
        down, _ = await run_test(
            OVProcessor(client),
            frames_to_send=[
                SleepFrame(0.05),
                BotStoppedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
                SleepFrame(0.05),
            ],
            frames_to_send_direction=FrameDirection.UPSTREAM,
        )

        spoken = [frame.text for frame in down if isinstance(frame, TTSSpeakFrame)]
        self.assertEqual(spoken, ["The pot is boiling.", "Turn down the heat."])

    async def test_the_time_prefix_is_not_spoken(self) -> None:
        from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSSpeakFrame
        from pipecat.processors.frame_processor import FrameDirection
        from pipecat.tests.utils import SleepFrame, run_test

        from ov_voice.pipecat import OVProcessor

        answer = "At 6.967 seconds: 这锅水还没有开。再等一会儿。"
        client = FakeClient(answer_events("r1", answer))
        down, _ = await run_test(
            OVProcessor(client),
            frames_to_send=[
                SleepFrame(0.05),
                BotStoppedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
                SleepFrame(0.05),
            ],
            frames_to_send_direction=FrameDirection.UPSTREAM,
        )

        spoken = [frame.text for frame in down if isinstance(frame, TTSSpeakFrame)]
        self.assertEqual(spoken, ["这锅水还没有开。", "再等一会儿。"])

    async def test_a_newer_answer_cuts_the_one_playing(self) -> None:
        from pipecat.frames.frames import InterruptionFrame, TTSSpeakFrame
        from pipecat.tests.utils import SleepFrame

        client = FakeClient(
            [*answer_events("r1", ANSWER), *answer_events("r2", "水开了。")]
        )
        down, _ = await self.run_processor(client, [SleepFrame(0.05)])

        spoken = [frame.text for frame in down if isinstance(frame, TTSSpeakFrame)]
        self.assertEqual(spoken[-1], "水开了。")
        self.assertTrue(any(isinstance(frame, InterruptionFrame) for frame in down))
        self.assertFalse(any(call[0] == "cancel" for call in client.ov.calls))

    async def test_barge_in_stops_the_answer_and_cancels_the_rest(self) -> None:
        from pipecat.frames.frames import (
            BotStoppedSpeakingFrame,
            InterruptionFrame,
            VADUserStartedSpeakingFrame,
        )
        from pipecat.tests.utils import SleepFrame

        client = FakeClient(
            [
                *answer_events("r1", ANSWER),
                Event("response.created", "s", response_id="r2", control="response"),
                Event("response.created", "s", response_id="c1", control="caption"),
            ]
        )
        down, up = await self.run_processor(
            client,
            [
                SleepFrame(0.05),
                BotStoppedSpeakingFrame(),
                VADUserStartedSpeakingFrame(),
                self.camera(),
            ],
        )

        self.assertTrue(any(isinstance(frame, InterruptionFrame) for frame in down))
        self.assertTrue(any(isinstance(frame, InterruptionFrame) for frame in up))
        calls = [call[:2] if call[0] == "send" else call for call in client.ov.calls]
        self.assertEqual(calls[0], ("cancel", "r2"))
        self.assertEqual(calls[1][0], "send")


class DelayedSession(FakeSession):
    """Delivers its events after a delay, once the frames have started."""

    def __init__(self, events, delay=0.1):
        super().__init__(events)
        self.delay = delay

    async def events(self):
        await asyncio.sleep(self.delay)
        async for event in super().events():
            yield event


class UserTurnTest(unittest.IsolatedAsyncioTestCase):
    """OVProcessor with user_turns, behind a UserTurnProcessor."""

    def transcript(self, text, finalized=True):
        from pipecat.frames.frames import TranscriptionFrame

        return TranscriptionFrame(text, "", "", finalized=finalized)

    async def test_transcripts_of_a_turn_are_one_question(self) -> None:
        from pipecat.frames.frames import (
            UserImageRawFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        )
        from pipecat.tests.utils import SleepFrame, run_test

        from ov_voice.pipecat import OVProcessor

        def camera():
            return UserImageRawFrame(image=bytes(12), size=(2, 2), format="RGB")

        client = FakeClient()
        await run_test(
            OVProcessor(client, frame_interval=0.1, user_turns=True),
            frames_to_send=[
                UserStartedSpeakingFrame(),
                self.transcript("这锅水", finalized=False),
                SleepFrame(0.15),
                # A frame during the turn carries none of it.
                camera(),
                self.transcript("开了吗？"),
                UserStoppedSpeakingFrame(),
                SleepFrame(0.15),
                camera(),
                SleepFrame(0.05),
            ],
        )

        first, second = client.ov.sent()
        self.assertEqual([part["type"] for part in first], ["image_url"])
        self.assertEqual([part["type"] for part in second], ["image_url", "text"])
        self.assertEqual(second[1]["text"], "这锅水开了吗？")

    async def test_an_answer_does_not_start_over_the_next_turn(self) -> None:
        """The next turn's start overtakes the queued end of the last one."""
        from pipecat.frames.frames import (
            TTSSpeakFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            VADUserStartedSpeakingFrame,
        )
        from pipecat.tests.utils import SleepFrame, run_test

        from ov_voice.pipecat import OVProcessor

        client = FakeClient()
        # The answer comes while the user is speaking the second turn.
        client.ov = DelayedSession(answer_events("r1", ANSWER), delay=0.3)
        down, _ = await run_test(
            OVProcessor(client, user_turns=True),
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                UserStartedSpeakingFrame(),
                SleepFrame(0.1),
                UserStoppedSpeakingFrame(),
                VADUserStartedSpeakingFrame(),
                UserStartedSpeakingFrame(),
                SleepFrame(0.4),
            ],
        )

        self.assertEqual([f for f in down if isinstance(f, TTSSpeakFrame)], [])

    async def test_answers_wait_for_the_end_of_the_turn(self) -> None:
        from pipecat.frames.frames import (
            TTSSpeakFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            VADUserStartedSpeakingFrame,
            VADUserStoppedSpeakingFrame,
        )
        from pipecat.tests.utils import SleepFrame, run_test

        from ov_voice.pipecat import OVProcessor

        for turn_ends in (False, True):
            with self.subTest(turn_ends=turn_ends):
                client = FakeClient()
                client.ov = DelayedSession(answer_events("r1", ANSWER))
                frames = [
                    VADUserStartedSpeakingFrame(),
                    UserStartedSpeakingFrame(),
                    # A pause inside the turn: the answer must not start.
                    VADUserStoppedSpeakingFrame(stop_secs=0.2),
                    SleepFrame(0.2),
                ]
                if turn_ends:
                    frames += [UserStoppedSpeakingFrame(), SleepFrame(0.05)]
                down, _ = await run_test(
                    OVProcessor(client, user_turns=True), frames_to_send=frames
                )

                spoken = [f.text for f in down if isinstance(f, TTSSpeakFrame)]
                expected = ["The pot is boiling.", "Turn down the heat."]
                self.assertEqual(spoken, expected if turn_ends else [])

    async def test_smart_turn_pipeline_joins_a_paused_question(self) -> None:
        """The real UserTurnProcessor ends the turn only after its last transcript."""
        from pipecat.audio.turn.base_turn_analyzer import (
            BaseTurnAnalyzer,
            BaseTurnParams,
            EndOfTurnState,
        )
        from pipecat.frames.frames import (
            STTMetadataFrame,
            VADUserStartedSpeakingFrame,
            VADUserStoppedSpeakingFrame,
        )
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.tests.utils import SleepFrame, run_test
        from pipecat.turns.user_start import VADUserTurnStartStrategy
        from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
        from pipecat.turns.user_turn_processor import UserTurnProcessor
        from pipecat.turns.user_turn_strategies import UserTurnStrategies

        from ov_voice.pipecat import OVProcessor

        class ScriptedTurn(BaseTurnAnalyzer):
            """Says the turn is incomplete at the first pause, complete after."""

            def __init__(self):
                super().__init__()
                self.pauses = 0

            @property
            def speech_triggered(self):
                return False

            @property
            def params(self):
                return BaseTurnParams()

            def append_audio(self, buffer, is_speech):
                return EndOfTurnState.INCOMPLETE

            async def analyze_end_of_turn(self):
                self.pauses += 1
                if self.pauses == 1:
                    return EndOfTurnState.INCOMPLETE, None
                return EndOfTurnState.COMPLETE, None

            def clear(self):
                pass

        strategies = UserTurnStrategies(
            start=[VADUserTurnStartStrategy(enable_interruptions=False)],
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=ScriptedTurn())],
        )
        client = FakeClient()
        await run_test(
            Pipeline(
                [
                    UserTurnProcessor(user_turn_strategies=strategies),
                    OVProcessor(client, user_turns=True),
                ]
            ),
            frames_to_send=[
                # What TranscriptionService announces: how long the turn may
                # wait for the transcript of the segment that ends it.
                STTMetadataFrame(service_name="asr", ttfs_p99_latency=1.0),
                VADUserStartedSpeakingFrame(),
                VADUserStoppedSpeakingFrame(stop_secs=0.2),
                self.transcript("这锅水"),
                SleepFrame(0.1),
                VADUserStartedSpeakingFrame(),
                VADUserStoppedSpeakingFrame(stop_secs=0.2),
                SleepFrame(0.05),
                # The second segment's transcript comes after its pause.
                self.transcript("开了吗？"),
                SleepFrame(0.1),
            ],
        )

        [content] = client.ov.sent()
        self.assertEqual([part["text"] for part in content], ["这锅水开了吗？"])


class PageMessageTest(unittest.IsolatedAsyncioTestCase):
    async def test_questions_and_ov_events_reach_the_page(self) -> None:
        from pipecat.frames.frames import TranscriptionFrame
        from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
        from pipecat.tests.utils import SleepFrame, run_test

        from ov_voice.pipecat import OVProcessor

        client = FakeClient()
        client.ov = DelayedSession(
            [Event("input.done", "s", input_id="input", control="silence")]
        )
        down, _ = await run_test(
            OVProcessor(client),
            frames_to_send=[TranscriptionFrame("开了吗？", "", ""), SleepFrame(0.2)],
        )

        question, event = [
            f.data for f in down if isinstance(f, RTVIServerMessageFrame)
        ]
        self.assertEqual(question["ov"], "question")
        self.assertEqual(question["text"], "开了吗？")
        self.assertEqual(event["ov"], "event")
        self.assertEqual(event["type"], "input.done")
        self.assertEqual(event["control"], "silence")
        self.assertGreaterEqual(event["latency_ms"], 0)


class TypedQuestionTest(unittest.IsolatedAsyncioTestCase):
    async def test_typed_text_goes_to_ov_as_a_question(self) -> None:
        from pipecat.processors.frameworks.rtvi.frames import RTVIClientMessageFrame
        from pipecat.tests.utils import SleepFrame, run_test

        from ov_voice.pipecat import OVProcessor

        client = FakeClient()
        await run_test(
            OVProcessor(client, user_turns=True),
            frames_to_send=[
                RTVIClientMessageFrame(
                    msg_id="1", type="ov-text", data={"text": " 锅里有什么？ "}
                ),
                RTVIClientMessageFrame(msg_id="2", type="ov-text", data={"text": ""}),
                RTVIClientMessageFrame(msg_id="3", type="other", data={"text": "x"}),
                SleepFrame(0.1),
            ],
        )

        [content] = client.ov.sent()
        self.assertEqual([part["text"] for part in content], ["锅里有什么？"])


class JoinTranscriptsTest(unittest.TestCase):
    def test_spaces_only_between_words(self) -> None:
        from ov_voice.pipecat import join_transcripts

        self.assertEqual(join_transcripts("这锅水", "开了吗？"), "这锅水开了吗？")
        self.assertEqual(
            join_transcripts("is the pot", "boiling?"), "is the pot boiling?"
        )
        self.assertEqual(join_transcripts("看这个", "iPhone"), "看这个iPhone")
        self.assertEqual(join_transcripts("", "hello"), "hello")


class TranscriptionServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_asks_for_json_without_a_language(self) -> None:
        from openai.types.audio import Transcription

        from ov_voice.pipecat import TranscriptionService

        service = TranscriptionService(
            api_key="EMPTY",
            base_url="http://asr/v1",
            settings=TranscriptionService.Settings(model="Qwen/Qwen3-ASR-1.7B"),
        )
        create = AsyncMock(return_value=Transcription(text="这锅水开了吗？"))
        with patch.object(service._client.audio.transcriptions, "create", create):
            await service._transcribe(b"RIFF")

        self.assertEqual(
            create.call_args.kwargs,
            {
                "file": ("audio.wav", b"RIFF", "audio/wav"),
                "model": "Qwen/Qwen3-ASR-1.7B",
                "response_format": "json",
            },
        )


if __name__ == "__main__":
    unittest.main()
