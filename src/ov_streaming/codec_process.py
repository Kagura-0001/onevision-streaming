"""Run ``ov_streaming.codec_worker`` in a subprocess and read its I+P windows.

Every codec consumer goes through the same worker and JSON lines: lmms-eval's
per-video cache, the WebSocket codec gateway, and the in-process engine. A
file, an ffmpeg URL and the same H.264 bytes piped as MPEG-TS therefore
produce identical canvases.

The worker runs in its own interpreter because ``codec-video-prep`` only
ships CPython 3.10/3.11 wheels pinned to ``numpy<2``, which cannot share an
environment with vLLM. ``codec_python`` may point at such an environment;
this package's own source is put on its ``PYTHONPATH``.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import functools
import json
import os
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .codec import CODEC_WORKER_VERSION, CodecConfig

# One JSON line holds 2-3 base64 JPEG canvases plus their patch positions.
_LINE_LIMIT_BYTES = 32 * 1024 * 1024
_STDERR_TAIL_BYTES = 8192
_EXIT_GRACE_SEC = 5.0

# CodecConfig field -> codec_worker flag. Only fields that differ from the
# defaults are passed, so the default command matches lmms-eval's cache key.
_WORKER_FLAGS = {
    "strategy": "--strategy",
    "pixel_budget": "--pixel-budget",
    "p_canvases": "--p-canvases",
    "max_seconds": "--max-seconds",
    "width": "--canvas-width",
    "height": "--canvas-height",
    "candidate_fps": "--candidate-fps",
    "max_candidates": "--max-candidates",
    "decode_threads": "--decode-threads",
}
_UNSUPPORTED_FIELDS = ("patch_size", "block_size")


def resolve_codec_python(codec_python: str | os.PathLike[str] | None = None) -> str:
    """Explicit argument, then ``OV_STREAMING_CODEC_PYTHON``, then this interpreter."""
    if codec_python is not None:
        return str(codec_python)
    return os.environ.get("OV_STREAMING_CODEC_PYTHON") or sys.executable


def codec_worker_command(
    python: str,
    config: CodecConfig,
    *,
    video: str | None = None,
    live_fps: float | None = None,
    source_size: tuple[int, int] | None = None,
) -> list[str]:
    """Build the worker command for a file/URL (``video``) or stdin (``live_fps``)."""
    if (video is None) == (live_fps is None):
        raise ValueError("give exactly one of video or live_fps")
    if live_fps is not None and source_size is None:
        raise ValueError("live codec input requires source_size=(width, height)")

    defaults = CodecConfig()
    for name in _UNSUPPORTED_FIELDS:
        if getattr(config, name) != getattr(defaults, name):
            raise ValueError(f"codec_worker does not accept a custom {name}")

    command = [python, "-m", "ov_streaming.codec_worker"]
    if video is not None:
        command += ["--video", video]
    else:
        width, height = source_size
        command += [
            "--fps", str(live_fps),
            "--source-width", str(int(width)),
            "--source-height", str(int(height)),
        ]
    for field in fields(CodecConfig):
        flag = _WORKER_FLAGS.get(field.name)
        value = getattr(config, field.name)
        if flag is None or value is None or value == getattr(defaults, field.name):
            continue
        command += [flag, str(value)]
    if config.paced:
        command.append("--paced")
    return command


def _is_isolated_package_root(parent: Path, package_name: str) -> bool:
    """Return True if ``parent`` only holds ``package_name`` and its metadata."""
    if parent.name in {"site-packages", "dist-packages"}:
        return False
    dist_prefixes = (f"{package_name}-", f"{package_name}.")
    try:
        entries = list(parent.iterdir())
    except OSError:
        return False
    for entry in entries:
        name = entry.name
        if name == package_name or name == "__pycache__" or name.startswith("."):
            continue
        if name.startswith(dist_prefixes) and name.endswith((".dist-info", ".egg-info")):
            continue
        return False
    return True


@functools.lru_cache(maxsize=4)
def _isolated_import_root(package_dir: Path) -> Path:
    """Return a directory containing only ``package_dir`` so sibling deps don't leak."""
    parent = package_dir.parent
    if _is_isolated_package_root(parent, package_dir.name):
        return parent
    shim_dir = Path(tempfile.mkdtemp(prefix="ov_streaming_worker_"))
    os.symlink(package_dir, shim_dir / package_dir.name)
    atexit.register(shutil.rmtree, shim_dir, True)
    return shim_dir


def codec_worker_env(
    base: Mapping[str, str] | None = None,
    *,
    _package_dir: Path | None = None,
) -> dict[str, str]:
    """Environment whose PYTHONPATH starts with an isolated ``ov_streaming`` root."""
    env = dict(os.environ if base is None else base)
    pkg_dir = (_package_dir or Path(__file__).resolve().parent).resolve()
    import_root = str(_isolated_import_root(pkg_dir))
    existing = env.get("PYTHONPATH")
    if not existing:
        env["PYTHONPATH"] = import_root
    else:
        entries = existing.split(os.pathsep)
        if entries[0] != import_root:
            env["PYTHONPATH"] = import_root + os.pathsep + existing
    return env


@dataclass(frozen=True)
class CodecInput:
    """One closed second of codec input: I then P canvases, as content parts."""

    timestamp_sec: float
    parts: list[dict[str, Any]]


class CodecWorkerProcess:
    """A running codec worker. Feed stdin with ``write`` for live input."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._stderr_tail = bytearray()
        self._stderr_task = asyncio.create_task(self._collect_stderr())

    @classmethod
    async def start(cls, command: list[str], *, live: bool) -> CodecWorkerProcess:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE if live else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=codec_worker_env(),
            limit=_LINE_LIMIT_BYTES,
        )
        return cls(process)

    async def _collect_stderr(self) -> None:
        assert self._process.stderr is not None
        while chunk := await self._process.stderr.read(4096):
            self._stderr_tail.extend(chunk)
            del self._stderr_tail[:-_STDERR_TAIL_BYTES]

    def stderr_tail(self) -> str:
        return self._stderr_tail.decode("utf-8", errors="replace").strip()

    async def write(self, data: bytes) -> None:
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("this codec worker decodes a file; it has no stdin")
        stdin.write(data)
        await stdin.drain()

    async def end_input(self) -> None:
        stdin = self._process.stdin
        if stdin is not None and not stdin.is_closing():
            stdin.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await stdin.wait_closed()

    async def windows(self) -> AsyncIterator[CodecInput]:
        """Yield windows until the worker's ``done`` line; raise if it fails."""
        assert self._process.stdout is not None
        finished = False
        while True:
            line = await self._process.stdout.readline()
            if not line:
                break
            record = json.loads(line)
            record_type = record.get("type")
            if record_type == "done":
                finished = True
                continue
            if record_type != "window":
                raise RuntimeError(f"unexpected codec worker record: {record_type!r}")
            if record.get("worker_version") != CODEC_WORKER_VERSION:
                raise RuntimeError(
                    f"codec worker schema {record.get('worker_version')!r} "
                    f"does not match {CODEC_WORKER_VERSION!r}"
                )
            parts = record["messages"][0]["content"]
            yield CodecInput(float(record["timestamp_sec"]), list(parts))
        exit_code = await self._process.wait()
        await self._stderr_task
        if exit_code != 0 or not finished:
            raise RuntimeError(
                f"codec worker failed (exit {exit_code}): {self.stderr_tail()[-2000:]}"
            )

    async def aclose(self) -> None:
        """Stop the worker if it is still running and release its pipes."""
        if self._process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), _EXIT_GRACE_SEC)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self._process.kill()
                await self._process.wait()
        self._stderr_task.cancel()
        await asyncio.gather(self._stderr_task, return_exceptions=True)


__all__ = [
    "CodecInput",
    "CodecWorkerProcess",
    "codec_worker_command",
    "codec_worker_env",
    "resolve_codec_python",
]
