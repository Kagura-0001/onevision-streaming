"""Frame inputs for frame-sampled streaming.

``iter_video_frames`` decodes a file or an OpenCV-readable live URL with one
decoder and one sampling rule, lmms-eval's ``fps_grid``: tick ``k / fps``
takes the latest decoded source frame at or before the tick. Source time is
``frame_index / source_fps`` (constant frame rate), as in the codec path.
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

EPS = 1e-6


@dataclass(frozen=True)
class VideoFrame:
    """One frame for the model.

    ``timestamp_sec`` is the stream time of the turn that carries the frame;
    ``frame_index`` and ``fps`` locate it in the source, and the renderer
    labels the frame ``frame_index / fps``. Without them the renderer labels
    it by ``timestamp_sec``.
    """

    image: Any
    timestamp_sec: float
    frame_index: int | None = None
    fps: float | None = None


def to_pil_image(image: Any) -> Image.Image:
    """Accept a PIL image, an RGB array, encoded bytes, or a file path."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image))) as opened:
            return opened.convert("RGB")
    if hasattr(image, "__array__"):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image).__name__}")


def iter_video_frames(
    source: str | Path,
    *,
    fps: float = 1.0,
    max_seconds: float = 0.0,
) -> Iterator[VideoFrame]:
    """Yield ``fps``-grid frames of a video file or live URL, decoded lazily."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError("max_seconds must be finite and non-negative")
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError(
            "frame sampling needs opencv-python-headless; install ov-streaming[transformers] "
            "or push frames yourself with stream_frames()"
        ) from error

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {source}")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or not 0 < source_fps <= 240:
            raise ValueError(
                f"video does not declare a usable frame rate ({source_fps}); "
                "push frames with explicit timestamps via stream_frames()"
            )

        tick_index = 0
        latest_index: int | None = None
        latest_bgr = None
        latest_image: Image.Image | None = None

        def tick_time() -> float:
            return tick_index / fps

        def within_limit(t: float) -> bool:
            return max_seconds == 0 or t <= max_seconds + EPS

        def sample() -> VideoFrame:
            nonlocal latest_image
            if latest_image is None:
                rgb = cv2.cvtColor(latest_bgr, cv2.COLOR_BGR2RGB)
                latest_image = Image.fromarray(rgb)
            return VideoFrame(latest_image, tick_time(), latest_index, source_fps)

        frame_index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            source_time = frame_index / source_fps
            if not within_limit(source_time):
                break
            # Ticks before this frame take the previous (latest) frame.
            while latest_index is not None and tick_time() < source_time - EPS:
                yield sample()
                tick_index += 1
            latest_index, latest_bgr, latest_image = frame_index, bgr, None
            if abs(source_time - tick_time()) <= EPS:
                yield sample()
                tick_index += 1
            frame_index += 1

        # After the last frame the grid continues to that frame's end.
        if latest_index is not None:
            media_end = (latest_index + 1) / source_fps
            while tick_time() < media_end - EPS and within_limit(tick_time()):
                yield sample()
                tick_index += 1
    finally:
        capture.release()


__all__ = ["VideoFrame", "iter_video_frames", "to_pil_image"]
