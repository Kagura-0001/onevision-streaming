"""Optional wheel-backed H.264 -> one-second I+P inputs for OV2.1.

No decoder or FFmpeg source is shipped here. Install a codec-video-prep wheel
with read_video_stream in the server-side codec gateway environment.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from PIL import Image

CODEC_SINGLE_P_CACHE_VERSION = "aspect48-pixel-budget-v1"
CODEC_WORKER_VERSION = "aspect48-multi-p-v2"
CODEC_PIXEL_BUDGETS = (100352, 258048, 313000)


def codec_canvas_size(source_width: int, source_height: int, pixel_budget: int,
                      factor: int = 48) -> tuple[int, int]:
    """Training's aspect-aware lattice sizing, with an explicit pixel budget.

    Return (width, height). Each side is floored onto the OV patch/block grid;
    neither source side is enlarged. The production mixed313k policy uses this
    scale/floor rule with 150000/225000/313000 budgets; evaluation selects R.
    """
    if any(type(v) is not int or v <= 0 for v in
           (source_width, source_height, pixel_budget, factor)):
        raise ValueError("source dimensions, pixel_budget and factor must be positive integers")
    if min(source_width, source_height) < factor or pixel_budget < factor * factor:
        raise ValueError("source or pixel budget cannot hold one codec lattice block without upscaling")
    scale = min(1.0, math.sqrt(pixel_budget / (source_width * source_height)))
    width = max(factor, int(source_width * scale) // factor * factor)
    height = max(factor, int(source_height * scale) // factor * factor)
    while width * height > pixel_budget:
        if width / source_width > height / source_height:
            width -= factor
        else:
            height -= factor
        if min(width, height) < factor:
            raise ValueError("pixel budget cannot fit this source aspect ratio on the codec lattice")
    return width, height


@dataclass(frozen=True)
class CodecConfig:
    strategy: Literal["rdo", "pwindow"] = "rdo"
    height: int | None = None
    width: int | None = None
    pixel_budget: int = 258048
    p_canvases: int = 1
    patch_size: int = 16
    block_size: int = 3
    candidate_fps: float = 24.0
    max_candidates: int = 24
    decode_threads: int = 1
    max_seconds: float = 0.0
    paced: bool = False

    def __post_init__(self):
        if self.strategy not in {"rdo", "pwindow"}:
            raise ValueError("codec strategy must be rdo or pwindow")
        if type(self.p_canvases) is not int or self.p_canvases not in (1, 2):
            raise ValueError("p_canvases must be 1 or 2")
        for name in (
            "pixel_budget",
            "patch_size",
            "block_size",
            "max_candidates",
            "decode_threads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        factor = self.patch_size * self.block_size
        if (self.height is None) != (self.width is None):
            raise ValueError("codec width and height must both be supplied or both inferred")
        if self.pixel_budget < factor * factor:
            raise ValueError("pixel_budget must hold at least one codec lattice block")
        if self.height is not None and (
            type(self.height) is not int or type(self.width) is not int
            or self.height <= 0 or self.width <= 0
            or self.height % factor or self.width % factor
        ):
            raise ValueError(
                "codec dimensions must be divisible by patch_size * block_size"
            )
        if self.height is not None and self.height * self.width > self.pixel_budget:
            raise ValueError("codec canvas exceeds pixel_budget")
        if self.decode_threads > 8 or self.max_candidates > 240:
            raise ValueError(
                "decode_threads <= 8 and max_candidates <= 240 are required"
            )
        if not math.isfinite(self.candidate_fps) or not 0 < self.candidate_fps <= 240:
            raise ValueError("candidate_fps must be in (0, 240]")
        if not math.isfinite(self.max_seconds) or self.max_seconds < 0:
            raise ValueError("max_seconds must be finite and non-negative")

    def for_source(self, width: int, height: int) -> CodecConfig:
        if self.width is None:
            target_width, target_height = codec_canvas_size(
                width, height, self.pixel_budget, self.patch_size * self.block_size,
            )
        else:
            target_width, target_height = self.width, self.height
            if target_width > width or target_height > height:
                raise ValueError("codec canvas must not upscale the native source")
        return replace(self, width=target_width, height=target_height)


@dataclass(frozen=True)
class CodecWindow:
    timestamp_sec: float
    fps: float
    images: tuple[Image.Image, ...]  # I followed by the configured P canvases
    positions: Any  # [1+P, H/patch, W/patch, 3], row-major source [frame,y,x]
    patch_size: int
    strategy: str

    def content(self) -> list[dict[str, Any]]:
        # Match the production training canvases. Encode once here so the
        # ordinary client cannot re-encode PIL input with default subsampling.
        def jpeg(image):
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=95,
                subsampling=0,
                optimize=False,
                progressive=False,
            )
            return buffer.getvalue()

        return [
            {
                "type": "image_url",
                "image_url": {"url": jpeg(image)},
                "timestamp_sec": self.timestamp_sec,
                "fps": self.fps,
                "source_size": [image.width, image.height],
                "codec": {
                    "layout": "row_major",
                    "patch_size": self.patch_size,
                    "patch_positions": positions.reshape(-1, 3).tolist(),
                },
            }
            for image, positions in zip(self.images, self.positions)
        ]


def codec_source_info(video: str) -> dict[str, Any]:
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,r_frame_rate,width,height",
                "-of",
                "json",
                video,
            ],
            text=True,
        )
    )
    stream = probe["streams"][0]
    if stream["codec_name"] != "h264":
        raise ValueError("this streaming codec wheel adapter currently requires H.264")
    # Production serializes source-frame positions with r_frame_rate rounded
    # to six decimals. Container avg_frame_rate can differ even for CFR input.
    a, b = map(int, stream["r_frame_rate"].split("/"))
    fps = round(a / b, 6) if b else 0
    if not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError("video must declare a source fps in (0, 240]")
    return {"fps": fps, "width": int(stream["width"]), "height": int(stream["height"])}


def _source_fps(video: str) -> float:
    return codec_source_info(video)["fps"]


def _dependencies():
    try:
        from codec_selector.core.config import BitcostReadinessConfig
        from codec_selector.plugins.packers.rdo_event_canvas_packer import (
            pack_rdo_event_candidates,
        )
        from codec_selector.plugins.scorers.bitcost import bitcost_items_to_score_maps
        from codec_selector.strategies import pwindow
        from codec_video_prep import cv_reader_fast
    except ImportError as error:
        raise RuntimeError(
            "install the codec-video-prep wheel in the codec gateway environment"
        ) from error
    if not hasattr(cv_reader_fast, "read_video_stream"):
        raise RuntimeError(
            "codec wheel lacks read_video_stream; use a streaming-capable wheel"
        )
    return (
        cv_reader_fast,
        BitcostReadinessConfig,
        bitcost_items_to_score_maps,
        pwindow,
        pack_rdo_event_candidates,
    )


def _prepare(records, cutoff, fps, config, dependencies):
    import numpy as np

    _, Config, score, pw, pack = dependencies
    c = config
    canvas_count = 1 + c.p_canvases
    p, b = c.patch_size, c.block_size
    gh, gw = c.height // p, c.width // p
    if len(records) > c.max_candidates:
        indices = np.linspace(0, len(records) - 1, c.max_candidates, dtype=int)
        records = [records[i] for i in indices]
    ids = [int(r["frame_idx"]) for r in records]
    anchor_pos = np.stack(
        np.broadcast_arrays(
            np.full((gh, gw), ids[0]), np.arange(gh)[:, None], np.arange(gw)[None, :]
        ),
        axis=-1,
    )
    anchor = records[0]["pixels"][..., ::-1].copy()
    cfg = Config(
        video="stream",
        out_dir="unused",
        grouping_mode="pwindow",
        patch=p,
        block_size=b,
        target_canvas=canvas_count,
        pwindow_segments=1,
        pwindow_windows=c.p_canvases,
        pwindow_anchor_policy="start",
        pwindow_order="anchor_first",
        pwindow_max_groups=45,
        pwindow_fill_canvas=True,
        pwindow_compress_filler_timestamps=False,
    )
    if len(ids) == 1:
        images = [anchor] + [anchor.copy() for _ in range(c.p_canvases)]
        positions = np.stack([anchor_pos] * canvas_count)
    else:
        maps = score(
            [r["bitcost"] for r in records], c.height, c.width, "sub", 99, True, "h264"
        )
        if c.strategy == "pwindow":
            plan = pw.PWindowStrategy().build_plan(ids, maps, cfg, gh, gw)
            _, _, images, _, source = pw.materialize_pwindow_plan(
                plan, ids, [r["pixels"] for r in records], p, b, gh, gw
            )
            coords = np.asarray(
                [
                    (y * b + dy, x * b + dx)
                    for y in range(gh // b)
                    for x in range(gw // b)
                    for dy in range(b)
                    for dx in range(b)
                ]
            )
            if len(images) != canvas_count:
                raise RuntimeError("pwindow did not produce the configured canvas count")
            positions = np.empty((canvas_count, gh, gw, 3), dtype=np.int64)
            positions[:, coords[:, 0], coords[:, 1]] = np.asarray(source[0]).reshape(
                canvas_count, gh * gw, 3
            )
        else:
            scores = pw._normalize_scores(
                [
                    pw._block_scores_from_score(m, p, b, cfg.pwindow_score_agg)
                    for m in maps
                ],
                cfg.pwindow_norm_low,
                cfg.pwindow_norm_high,
            )
            candidates = [
                {
                    "frame_id": fid,
                    "frame_index": i,
                    "group_id": 0,
                    "y": y,
                    "x": x,
                    "grid_h": gh,
                    "grid_w": gw,
                    "unit_type": "p_core",
                    "score": float(scores[i][y // b, x // b]) / (b * b),
                }
                for i, fid in enumerate(ids)
                for y in range(gh)
                for x in range(gw)
            ]
            canvases, _ = pack(
                candidates,
                canvas_shape=(gh, gw),
                block_size=b,
                global_max_canvas=c.p_canvases,
                max_canvas_per_group=c.p_canvases,
                anchor_canvas_count=0,
                strict_group_order=False,
                layout_order="gain",
                lambda_threshold=0.0,
                objective="absolute_gain",
            )
            if len(canvases) != c.p_canvases or any(
                len(canvas["units"]) != gh * gw // (b * b) for canvas in canvases
            ):
                raise RuntimeError("RDO did not fill the configured P canvases")
            images, positions = [anchor], [anchor_pos]
            lookup = {r["frame_idx"]: r["pixels"] for r in records}
            for canvas in canvases:
                units = sorted(
                    canvas["units"], key=lambda u: (u["frame_id"], u["by"], u["bx"])
                )
                rgb, pos = np.empty_like(anchor), np.empty_like(anchor_pos)
                for j, unit in enumerate(units):
                    dy, dx = divmod(j, gw // b)
                    sy, sx, fid = unit["by"] * b, unit["bx"] * b, unit["frame_id"]
                    rgb[dy * b * p : (dy + 1) * b * p, dx * b * p : (dx + 1) * b * p] = (
                        lookup[fid][sy * p : (sy + b) * p, sx * p : (sx + b) * p, ::-1]
                    )
                    yy, xx = np.indices((b, b))
                    pos[dy * b : (dy + 1) * b, dx * b : (dx + 1) * b] = np.stack(
                        [np.full((b, b), fid), sy + yy, sx + xx], axis=-1
                    )
                images.append(rgb)
                positions.append(pos)
            positions = np.stack(positions)
    return CodecWindow(
        cutoff,
        fps,
        tuple(Image.fromarray(im) for im in images),
        positions,
        p,
        c.strategy,
    )


def stream_codec_video(
    video: str | Path,
    consume: Callable[[CodecWindow], Any],
    *,
    config: CodecConfig | None = None,
    stop: threading.Event | None = None,
    source_fps: float | None = None,
    source_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Decode once; emit I plus the configured P canvases per second.

    The first input closes at one second (no synthetic tick-zero pair).
    EOF flushes the final partial second. Memory holds at most one second of
    sampled frames; consume runs synchronously and provides backpressure.
    """
    c = config or CodecConfig()
    video = str(video)
    # A live pipe cannot be probed separately: that would consume its bytes.
    # The gateway supplies the camera's declared CFR rate; decoded PTS are
    # still checked against it below. Files retain the production probe policy.
    if source_fps is None or source_size is None:
        if video == "pipe:0":
            raise ValueError("codec stdin requires declared source_fps and source_size")
        info = codec_source_info(video)
        if source_fps is None:
            source_fps = info["fps"]
        if source_size is None:
            source_size = (info["width"], info["height"])
    c = c.for_source(*source_size)
    dependencies = _dependencies()
    reader = dependencies[0]
    fps = source_fps
    if not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError("source_fps must be finite and in (0, 240]")
    records, cutoff, windows = [], 1.0, 0
    last_time, last_id = -1.0, -1
    last_record = None
    target_index = 0
    target_count = min(c.max_candidates, max(2, math.ceil(c.candidate_fps)))
    new_records = False

    def target_time():
        if target_count == 1:
            return cutoff - 1
        return cutoff - 1 + target_index / (target_count - 1)

    def capture():
        nonlocal target_index, new_records
        if last_record is None:
            raise ValueError("no predecessor frame for the codec sampling target")
        if not records or records[-1]["frame_idx"] != last_record["frame_idx"]:
            records.append(last_record)
        target_index += 1
        new_records = True

    def emit(end):
        nonlocal windows, new_records
        if records and new_records:
            consume(_prepare(records, end, fps, c, dependencies))
            windows += 1
            new_records = False

    def callback(record):
        nonlocal records, cutoff, last_time, last_id, last_record, target_index
        if stop is not None and stop.is_set():
            return False
        t, fid = float(record["source_time_s"]), int(record["frame_idx"])
        if not math.isfinite(t) or t < 0 or t <= last_time or fid <= last_id:
            raise ValueError(
                "codec frames must have strictly increasing source time and identity"
            )
        if c.max_seconds and t > c.max_seconds + 1e-9:
            return False
        if abs(fid / fps - t) > max(0.1, 2 / fps):
            raise ValueError("source frame/fps timeline drift; CFR input is required")
        if last_record is not None and t - last_time > 1.0 + 1e-9:
            raise ValueError("source timestamp gap exceeds one second")
        # Match production: inclusive linspace targets and the latest real
        # predecessor at each target, not the reader's next sampled frame.
        while t > target_time() + 1e-9:
            capture()
            if target_index == target_count:
                emit(cutoff)
                records = []
                cutoff += 1.0
                target_index = 0
        last_time, last_id, last_record = t, fid, record
        return True

    stats = reader.read_video_stream(
        video,
        callback,
        # Export every source frame (supported CFR input is <=240 fps).
        # Only the <=max_candidates predecessor records above are retained.
        sample_fps=240.0,
        out_w=c.width,
        out_h=c.height,
        # Native max_seconds stops before exporting a boundary frame. Allow
        # lookahead, then enforce the inclusive cutoff in our callback/EOF.
        max_seconds=c.max_seconds + 2 / fps if c.max_seconds else 0.0,
        export_bitcost=1,
        paced=int(c.paced),
        decode_threads=c.decode_threads,
        decode_mode="slice" if c.decode_threads == 1 else "frame",
    )
    if stop is None or not stop.is_set():
        end = float(stats.get("last_decoded_time_s", last_time)) + 1 / fps
        if c.max_seconds:
            end = min(end, c.max_seconds)
        end = min(cutoff, end)
        if last_record is not None:
            while target_index < target_count and target_time() <= end + 1e-9:
                capture()
        emit(end)
        if not windows:
            raise ValueError("codec decoder produced no frames")
    return {"windows": windows, "strategy": c.strategy, "fps": fps,
            "canvas": [c.width, c.height], "pixel_budget": c.pixel_budget,
            "p_canvases": c.p_canvases, "canvases_per_window": 1 + c.p_canvases,
            "worker_version": CODEC_WORKER_VERSION, "native": stats}


async def send_codec_video(session, video, *, config: CodecConfig | None = None):
    """Send I+P pairs over the regular HTTP client; keep reading session events."""
    loop, stop = asyncio.get_running_loop(), threading.Event()

    def consume(window):
        future = asyncio.run_coroutine_threadsafe(
            session.send_messages(
                messages=[{"role": "user", "content": window.content()}]
            ),
            loop,
        )
        while True:
            try:
                return future.result(timeout=0.1)
            except FutureTimeout:
                if stop.is_set():
                    future.cancel()
                    return

    worker = asyncio.create_task(
        asyncio.to_thread(stream_codec_video, video, consume, config=config, stop=stop)
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        stop.set()
        await worker
        raise
