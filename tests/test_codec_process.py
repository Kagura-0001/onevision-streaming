from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ov_streaming.codec import CodecConfig
from ov_streaming.codec_process import (
    _isolated_import_root,
    codec_worker_command,
    codec_worker_env,
    resolve_codec_python,
)
from ov_streaming.media import iter_video_frames


class CodecWorkerCommandTest(unittest.TestCase):
    def test_default_file_command_matches_the_eval_cache_flags(self):
        command = codec_worker_command("py", CodecConfig(), video="clip.mp4")
        self.assertEqual(command, ["py", "-m", "ov_streaming.codec_worker", "--video", "clip.mp4"])

    def test_every_supported_field_is_forwarded(self):
        config = CodecConfig(
            strategy="pwindow",
            width=480,
            height=336,
            pixel_budget=313000,
            p_canvases=2,
            candidate_fps=12.0,
            max_candidates=12,
            decode_threads=2,
            max_seconds=30.0,
            paced=True,
        )
        command = codec_worker_command(
            "py", config, live_fps=29.97, source_size=(1280, 720)
        )
        self.assertEqual(
            command[3:],
            [
                "--fps", "29.97", "--source-width", "1280", "--source-height", "720",
                "--strategy", "pwindow",
                "--canvas-height", "336",
                "--canvas-width", "480",
                "--pixel-budget", "313000",
                "--p-canvases", "2",
                "--candidate-fps", "12.0",
                "--max-candidates", "12",
                "--decode-threads", "2",
                "--max-seconds", "30.0",
                "--paced",
            ],
        )

    def test_live_input_needs_source_size(self):
        with self.assertRaises(ValueError):
            codec_worker_command("py", CodecConfig(), live_fps=30.0)
        with self.assertRaises(ValueError):
            codec_worker_command("py", CodecConfig(), video="a.mp4", live_fps=30.0)

    def test_unsupported_lattice_is_rejected(self):
        with self.assertRaises(ValueError):
            codec_worker_command("py", CodecConfig(patch_size=14, pixel_budget=258048), video="a.mp4")

    def test_worker_sees_this_package_first(self):
        import ov_streaming

        pkg_dir = Path(ov_streaming.__file__).resolve().parent
        env = codec_worker_env({"PYTHONPATH": "/elsewhere"})
        entries = env["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(entries[1:], ["/elsewhere"])
        root = Path(entries[0])
        self.assertTrue((root / "ov_streaming").exists())
        self.assertEqual((root / "ov_streaming").resolve(), pkg_dir)
        if pkg_dir.parent.name in {"site-packages", "dist-packages"}:
            self.assertNotEqual(root, pkg_dir.parent)
        else:
            self.assertEqual(root, pkg_dir.parent)

    def test_wheel_site_packages_does_not_leak_sibling_packages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            site_packages = Path(tmpdir) / "site-packages"
            fake_pkg = site_packages / "ov_streaming"
            fake_pil = site_packages / "PIL"
            fake_pkg.mkdir(parents=True)
            fake_pil.mkdir(parents=True)
            (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
            (fake_pil / "__init__.py").write_text("", encoding="utf-8")

            env = codec_worker_env(
                {"PYTHONPATH": "/elsewhere"}, _package_dir=fake_pkg
            )
            entries = env["PYTHONPATH"].split(os.pathsep)
            self.assertEqual(entries[1:], ["/elsewhere"])
            shim = Path(entries[0])
            try:
                self.assertNotEqual(shim.resolve(), site_packages.resolve())
                self.assertEqual((shim / "ov_streaming").resolve(), fake_pkg.resolve())
                self.assertFalse((shim / "PIL").exists())
            finally:
                _isolated_import_root.cache_clear()
                shutil.rmtree(shim, ignore_errors=True)

    def test_codec_python_resolution(self):
        previous = os.environ.pop("OV_STREAMING_CODEC_PYTHON", None)
        try:
            self.assertEqual(resolve_codec_python(), sys.executable)
            os.environ["OV_STREAMING_CODEC_PYTHON"] = "/envs/py311/bin/python"
            self.assertEqual(resolve_codec_python(), "/envs/py311/bin/python")
            self.assertEqual(resolve_codec_python("/explicit"), "/explicit")
        finally:
            os.environ.pop("OV_STREAMING_CODEC_PYTHON", None)
            if previous is not None:
                os.environ["OV_STREAMING_CODEC_PYTHON"] = previous


class FrameSamplingTest(unittest.TestCase):
    def _write_video(self, path: Path, *, frames: int, fps: float) -> None:
        import cv2

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (16, 16))
        for index in range(frames):
            # Encode the frame index in the pixel value to check which frame was taken.
            writer.write(np.full((16, 16, 3), index * 10, dtype=np.uint8))
        writer.release()

    def test_fps_grid_takes_the_latest_frame_at_each_tick(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "clip.avi"
            self._write_video(video, frames=10, fps=4.0)  # 2.5 s
            sampled = list(iter_video_frames(video, fps=1.5))
        # Ticks 0, 0.667, 1.333, 2.0 take frames floor(t * 4) = 0, 2, 5, 8.
        self.assertEqual([round(f.timestamp_sec, 3) for f in sampled], [0.0, 0.667, 1.333, 2.0])
        self.assertEqual([f.frame_index for f in sampled], [0, 2, 5, 8])
        self.assertEqual({f.fps for f in sampled}, {4.0})
        pixel_values = [f.image.getpixel((8, 8))[0] for f in sampled]
        self.assertEqual([round(v / 10) for v in pixel_values], [0, 2, 5, 8])

    def test_max_seconds_is_inclusive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "clip.avi"
            self._write_video(video, frames=20, fps=5.0)  # 4 s
            sampled = list(iter_video_frames(video, fps=1.0, max_seconds=2.0))
        self.assertEqual([f.timestamp_sec for f in sampled], [0.0, 1.0, 2.0])
        self.assertEqual([f.frame_index for f in sampled], [0, 5, 10])


if __name__ == "__main__":
    unittest.main()
