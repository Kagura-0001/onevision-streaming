from __future__ import annotations

import asyncio
import base64
import io
import itertools
import json
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ov_streaming import StreamingEngine, VideoFrame
from ov_streaming.codec import CODEC_WORKER_VERSION, CodecWindow
from ov_streaming.engine import AsyncStreamingEngine
from ov_streaming.renderers.base import Renderer
from ov_streaming.types import VisualFrame
from tests.fakes import BranchPlan, FakeBackend

SILENCE_ID = 10
RESPONSE_ID = 11
CAPTION_ID = 12


class _FakeTokenizer:
    vocab = {
        "<silence>": SILENCE_ID,
        "<response>": RESPONSE_ID,
        "<captioning>": CAPTION_ID,
        "<|im_end|>": 2,
    }

    def get_added_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        if text in self.vocab:
            return [self.vocab[text]]
        return list(range(len(text)))


class _FakeRenderer(Renderer):
    controls = {
        "silence": "<silence>",
        "response": "<response>",
        "caption": "<captioning>",
    }
    optional_controls = {"caption": "<caption_begin>\n"}
    memory_tags = {**Renderer.memory_tags, "caption": "answer_memory"}
    system_prompt = "You are a real-time streaming video assistant."
    timestep_millis = True

    def __init__(self):
        super().__init__(_FakeTokenizer(), SimpleNamespace())
        self.codec_frames = []

    def render_frame(self, image, timestamp_sec, *, frame_index=None, fps=None):
        return VisualFrame(f"<{timestamp_sec:.1f}s><img/>\n", {"size": image.size})

    def render_codec_frame(self, image, timestamp_sec, *, codec, frame_index=None, fps=None):
        self.codec_frames.append((image.size, timestamp_sec, fps, len(codec["patch_positions"])))
        return VisualFrame(f"<codec {timestamp_sec:.1f}s><img/>\n", {"size": image.size})

    def render_user(self, turn):
        frames = "".join(f.text for f in turn.frames)
        return (
            "".join(f"{self.render_memory(m)}\n" for m in turn.questions)
            + (frames or f"<{turn.timestamp_sec:.3f} seconds>\n")
            + "".join(f"{self.render_memory(m)}\n" for m in turn.memories)
        )


def _ready_plan(text: str, finish_reason: str = "stop") -> BranchPlan:
    plan = BranchPlan(payload=text, finish_reason=finish_reason)
    plan.release.set()
    return plan


def _step_texts(backend: FakeBackend) -> list[str]:
    """What each control step appended to the source prompt."""
    deltas, previous = [], ""
    for _, text, _ in backend.steps:
        deltas.append(text[len(previous):])
        previous = text
    return deltas


def _engine(backend: FakeBackend, renderer: Renderer | None = None) -> StreamingEngine:
    return StreamingEngine(
        "fake-model",
        _backend_instance=backend,
        _renderer_instance=renderer or _FakeRenderer(),
    )


def _frames(times):
    return [VideoFrame(Image.new("RGB", (48, 48)), t, round(t * 30), 30.0) for t in times]


def _window_line(timestamp_sec: float) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48)).save(buffer, format="JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    canvas = {
        "type": "image_url",
        "image_url": {"url": data_url},
        "timestamp_sec": timestamp_sec,
        "fps": 24.0,
        "source_size": [48, 48],
        "codec": {"layout": "row_major", "patch_size": 16, "patch_positions": [[0, 0, 0]] * 9},
    }
    return json.dumps(
        {
            "type": "window",
            "timestamp_sec": timestamp_sec,
            "messages": [{"role": "user", "content": [canvas, canvas]}],
            "worker_version": CODEC_WORKER_VERSION,
        }
    )


def _fake_codec_python(tmpdir: Path, body: str) -> str:
    """An executable that ignores ``-m ov_streaming.codec_worker`` and runs ``body``."""
    script = Path(tmpdir) / "fake_codec_python"
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    script.chmod(0o755)
    return str(script)


class PushInputTest(unittest.TestCase):
    def test_send_frame_with_question_and_close(self):
        backend = FakeBackend(
            controls=[SILENCE_ID, RESPONSE_ID],
            branches=[_ready_plan("The door opened.")],
        )
        with _engine(backend) as engine, engine.session(enable=[]) as session:
            image = Image.new("RGB", (48, 48))
            first = session.send_frame(image, timestamp_sec=0.0)
            second = session.send_frame(image, timestamp_sec=1.0, question="What happened?")
            rest = session.close()

        events = [*first, *second, *rest]
        self.assertEqual(
            [e.control for e in events if e.type == "input.done"], ["silence", "response"]
        )
        self.assertEqual(
            [e.text for e in events if e.type == "response.output_text.done"],
            ["The door opened."],
        )
        self.assertEqual(events[-1].type, "session.closed")
        # The question is stamped with its frame's time and precedes the image.
        step = _step_texts(backend)[1]
        self.assertLess(step.index("What happened?"), step.index("<1.0s>"))
        self.assertIn('"timestep": "00:00:01"', step)

    def test_send_question_is_a_text_only_turn(self):
        backend = FakeBackend(controls=[SILENCE_ID, SILENCE_ID])
        with _engine(backend) as engine, engine.session(enable=[]) as session:
            session.send_frame(Image.new("RGB", (48, 48)), timestamp_sec=2.0)
            session.send_question("Anyone there?", timestamp_sec=2.5)
            session.close()
        step = _step_texts(backend)[1]
        self.assertIn("<2.500 seconds>", step)
        self.assertIn('"timestep": "00:00:02.5"', step)

    def test_send_codec_window(self):
        backend = FakeBackend(controls=[SILENCE_ID])
        renderer = _FakeRenderer()
        image = Image.new("RGB", (48, 48))
        window = CodecWindow(
            timestamp_sec=1.0,
            fps=24.0,
            images=(image, image),
            positions=np.zeros((2, 3, 3, 3), dtype=np.int64),
            patch_size=16,
            strategy="rdo",
        )
        with _engine(backend, renderer) as engine, engine.session(enable=[]) as session:
            session.send_codec_window(window, question="Describe")
            session.close()
        self.assertEqual(renderer.codec_frames, [((48, 48), 1.0, 24.0, 9)] * 2)


class StreamFramesTest(unittest.TestCase):
    def test_question_between_frames_is_sent_at_its_own_time(self):
        backend = FakeBackend(
            controls=[SILENCE_ID, RESPONSE_ID, SILENCE_ID, SILENCE_ID, SILENCE_ID],
            branches=[_ready_plan("Water is boiling.")],
        )
        with _engine(backend) as engine, engine.session(enable=[]) as session:
            events = list(
                session.stream_frames(
                    _frames([0.0, 1.0, 2.0]),
                    questions=[(0.8, "Is it boiling?"), (2.0, "And now?")],
                )
            )

        steps = _step_texts(backend)
        self.assertEqual(len(steps), 4)
        # 0.8 s: a text-only turn before the 1 s frame, without that frame.
        self.assertIn("Is it boiling?", steps[1])
        self.assertIn("<0.800 seconds>", steps[1])
        self.assertNotIn("<1.0s>", steps[1])
        self.assertIn('"timestep": "00:00:00.8"', steps[1])
        # 2.0 s: same turn as the 2 s frame, ahead of the image.
        self.assertLess(steps[3].index("And now?"), steps[3].index("<2.0s>"))
        self.assertEqual(
            [e.text for e in events if e.type == "response.output_text.done"],
            ["Water is boiling."],
        )
        self.assertEqual(events[-1].type, "session.closed")

    def test_unbounded_source_is_consumed_lazily(self):
        backend = FakeBackend(controls=[SILENCE_ID] * 64)
        produced = []

        def camera():
            for index in itertools.count():
                produced.append(index)
                yield Image.new("RGB", (16, 16)), float(index)

        with _engine(backend) as engine, engine.session(enable=[]) as session:
            for event in session.stream_frames(camera()):
                if event.type == "input.done":
                    break
            session.close()
        # Leaving the iterator stops reading the camera.
        self.assertLess(len(produced), 16)

    def test_keep_session_open_between_streams(self):
        backend = FakeBackend(controls=[SILENCE_ID] * 4)
        with _engine(backend) as engine, engine.session(enable=[]) as session:
            first = list(session.stream_frames(_frames([0.0, 1.0]), close_when_done=False))
            second = list(session.stream_frames(_frames([2.0, 3.0]), close_when_done=False))
            rest = session.close()
        self.assertEqual([e.type for e in first].count("input.done"), 2)
        self.assertEqual([e.type for e in second].count("input.done"), 2)
        self.assertNotIn("session.closed", [e.type for e in first + second])
        self.assertEqual(rest[-1].type, "session.closed")

    def test_source_error_aborts_and_propagates(self):
        backend = FakeBackend(controls=[SILENCE_ID] * 4)

        def broken():
            yield Image.new("RGB", (16, 16)), 0.0
            raise OSError("camera unplugged")

        with _engine(backend) as engine:
            session = engine.session(enable=[])
            with self.assertRaisesRegex(OSError, "camera unplugged"):
                list(session.stream_frames(broken()))

    def test_async_api_with_async_source(self):
        backend = FakeBackend(controls=[SILENCE_ID] * 3)

        async def run():
            engine = AsyncStreamingEngine(
                "fake-model", _backend_instance=backend, _renderer_instance=_FakeRenderer()
            )
            session = engine.session(enable=[])

            async def frames():
                for frame in _frames([0.0, 1.0, 2.0]):
                    await asyncio.sleep(0)
                    yield frame

            return [event async for event in session.stream_frames(frames())]

        events = asyncio.run(run())
        self.assertEqual([e.type for e in events].count("input.done"), 3)
        self.assertEqual(events[-1].type, "session.closed")


class StreamVideoTest(unittest.TestCase):
    def test_frames_backend_samples_a_real_file(self):
        import cv2

        backend = FakeBackend(controls=[SILENCE_ID] * 8)
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "clip.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 32))
            for _ in range(10):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
            with _engine(backend) as engine, engine.session(enable=[]) as session:
                events = list(session.stream_video(video, fps=1.0))
        self.assertEqual([e.type for e in events].count("input.done"), 2)
        self.assertEqual([frames[0].data["size"] for _, _, frames in backend.steps], [(32, 32)] * 2)

    def test_missing_file_fails_before_streaming(self):
        backend = FakeBackend(controls=[])
        with _engine(backend) as engine, engine.session(enable=[]) as session:
            with self.assertRaises(FileNotFoundError):
                session.stream_video("/nonexistent/clip.mp4")

    def test_codec_backend_runs_the_worker_and_schedules_questions(self):
        backend = FakeBackend(
            controls=[SILENCE_ID, RESPONSE_ID, SILENCE_ID],
            branches=[_ready_plan("Seen codec.")],
        )
        renderer = _FakeRenderer()
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "clip.mp4"
            video.write_bytes(b"fake")
            argv_log = Path(tmpdir) / "argv.json"
            python = _fake_codec_python(
                tmpdir,
                f"""
                import json, sys
                open({str(argv_log)!r}, "w").write(json.dumps(sys.argv[1:]))
                print({_window_line(1.0)!r}, flush=True)
                print({_window_line(2.0)!r}, flush=True)
                print(json.dumps({{"type": "done", "stats": {{"windows": 2}}}}), flush=True)
                """,
            )
            with _engine(backend, renderer) as engine, engine.session(enable=[]) as session:
                events = list(
                    session.stream_video(
                        video,
                        video_backend="codec",
                        max_seconds=2.0,
                        codec_python=python,
                        questions=[
                            (0.0, "Standing question?"),
                            (1.5, "Mid-stream question?"),
                        ],
                    )
                )
            argv = json.loads(argv_log.read_text())

        self.assertEqual(
            argv,
            ["-m", "ov_streaming.codec_worker", "--video", str(video), "--max-seconds", "2.0"],
        )
        steps = _step_texts(backend)
        self.assertEqual(len(steps), 3)
        # Questions at or before the first visual window attach to turn 0 (no empty t=0 turn).
        self.assertIn("Standing question?", steps[0])
        self.assertIn('"timestep": "00:00:00"', steps[0])
        self.assertIn("<codec 1.0s>", steps[0])
        self.assertLess(steps[0].index("Standing question?"), steps[0].index("<codec 1.0s>"))
        self.assertNotIn("<0.000 seconds>", steps[0])
        # Questions between windows run causally at their own timestamp before the next window.
        self.assertIn("<1.500 seconds>", steps[1])
        self.assertIn("Mid-stream question?", steps[1])
        self.assertIn('"timestep": "00:00:01.5"', steps[1])
        self.assertIn("<codec 2.0s>", steps[2])
        self.assertEqual([t for _, t, _, _ in renderer.codec_frames], [1.0, 1.0, 2.0, 2.0])
        self.assertEqual(
            [e.text for e in events if e.type == "response.output_text.done"], ["Seen codec."]
        )

    def test_codec_worker_failure_reports_stderr(self):
        backend = FakeBackend(controls=[SILENCE_ID] * 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "clip.mp4"
            video.write_bytes(b"fake")
            python = _fake_codec_python(
                tmpdir,
                f"""
                import sys
                print({_window_line(1.0)!r}, flush=True)
                sys.stderr.write("decoder exploded\\n")
                sys.exit(3)
                """,
            )
            with _engine(backend) as engine:
                session = engine.session(enable=[])
                with self.assertRaisesRegex(RuntimeError, "exit 3.*decoder exploded"):
                    list(session.stream_video(video, video_backend="codec", codec_python=python))


class StreamH264Test(unittest.TestCase):
    def test_live_bytes_are_piped_to_the_worker(self):
        backend = FakeBackend(controls=[SILENCE_ID] * 3)
        chunks = [b"\x47" * 100, b"\x47" * 276, b"\x47" * 1000]
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "worker.json"
            python = _fake_codec_python(
                tmpdir,
                f"""
                import json, sys
                received = len(sys.stdin.buffer.read(376))
                print({_window_line(1.0)!r}, flush=True)
                received += len(sys.stdin.buffer.read())
                print({_window_line(2.0)!r}, flush=True)
                open({str(log)!r}, "w").write(json.dumps({{"argv": sys.argv[1:], "bytes": received}}))
                print(json.dumps({{"type": "done", "stats": {{"windows": 2}}}}), flush=True)
                """,
            )
            with _engine(backend) as engine, engine.session(enable=[]) as session:
                events = list(
                    session.stream_h264(
                        iter(chunks), fps=30.0, width=1920, height=1080, codec_python=python
                    )
                )
            record = json.loads(log.read_text())

        self.assertEqual(record["bytes"], 1376)
        self.assertEqual(
            record["argv"],
            [
                "-m", "ov_streaming.codec_worker",
                "--fps", "30.0", "--source-width", "1920", "--source-height", "1080",
            ],
        )
        self.assertEqual([e.type for e in events].count("input.done"), 2)
        self.assertEqual(events[-1].type, "session.closed")

    def test_upload_error_is_raised(self):
        backend = FakeBackend(controls=[SILENCE_ID] * 2)

        def chunks():
            yield b"\x47" * 376
            raise ConnectionAbortedError("camera link lost")

        with tempfile.TemporaryDirectory() as tmpdir:
            python = _fake_codec_python(
                tmpdir,
                """
                import sys
                sys.stdin.buffer.read()
                """,
            )
            with _engine(backend) as engine:
                session = engine.session(enable=[])
                with self.assertRaisesRegex(ConnectionAbortedError, "camera link lost"):
                    list(
                        session.stream_h264(
                            chunks(), fps=30.0, width=64, height=64, codec_python=python
                        )
                    )

    def test_early_break_terminates_worker_promptly(self):
        backend = FakeBackend(controls=[SILENCE_ID] * 10)

        def infinite_chunks():
            while True:
                yield b"\x47" * 1024
                time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            python = _fake_codec_python(
                tmpdir,
                f"""
                import sys, time
                sys.stdin.buffer.read(1024)
                print({_window_line(1.0)!r}, flush=True)
                time.sleep(60)
                """,
            )
            start = time.monotonic()
            with _engine(backend) as engine, engine.session(enable=[]) as session:
                seen_done = False
                for event in session.stream_h264(
                    infinite_chunks(),
                    fps=30.0,
                    width=1920,
                    height=1080,
                    codec_python=python,
                ):
                    if event.type == "input.done":
                        seen_done = True
                        break
                rest = session.close()
            elapsed = time.monotonic() - start

        self.assertTrue(seen_done)
        self.assertEqual(rest[-1].type, "session.closed")
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
