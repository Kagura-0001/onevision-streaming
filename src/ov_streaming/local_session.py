"""In-process streaming sessions over ``OVSession``.

Two ways to feed a session:

* Push one input at a time: ``send_frame``, ``send_codec_window``,
  ``send_question`` and ``send_messages``. A caller with its own capture
  loop uses these and reads ``events()`` in another task.
* Hand over a whole source and iterate its events:

  - ``stream_video(path_or_url, video_backend="frames" | "codec")``
  - ``stream_frames(frames)``: decoded frames from any producer, e.g. a camera
  - ``stream_h264(chunks, fps=..., width=..., height=...)``: live H.264 bytes
    (MPEG-TS, any chunk boundaries) turned into codec I+P windows

Codec input always goes through ``ov_streaming.codec_worker`` (see
``codec_process``), the same worker that builds lmms-eval's codec cache and
serves the codec gateway, so a file and the same stream piped live give
identical canvases.

Scheduled questions follow lmms-eval's streaming adapters: questions asked
at or before a session's first visual input join that first input ahead of its
images; later questions at ``t`` join the input stamped ``t``, ahead of its
images, or are sent as a causal text-only input at ``t`` before the next visual
input. Each text part carries ``t``, so the prompt shows when it was asked.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import threading
from collections import deque
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from .codec import CodecConfig, CodecWindow
from .codec_process import CodecWorkerProcess, codec_worker_command, resolve_codec_python
from .media import EPS, VideoFrame, iter_video_frames, to_pil_image
from .session import OVSession
from .types import Event

Questions = Sequence[tuple[float, str]]
# A timed input: its stream time and its visual content parts.
TimedInput = tuple[float, list[dict[str, Any]]]
RecordInput = Callable[[str], None]
Feed = Callable[[RecordInput], Awaitable[None]]

_FEED_DONE = object()
_EVENTS_DONE = object()
_ITER_END = object()
# After session.closed, how long the feeding task may take to finish closing.
_PRODUCER_GRACE_SEC = 5.0


# ---------------------------------------------------------------- content parts


def _decode_data_url(url: str) -> Image.Image:
    if ";base64," not in url:
        raise ValueError("image data URLs must be base64 encoded")
    _, encoded = url.split(",", 1)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as opened:
        return opened.convert("RGB")


def _normalize_part(part: Mapping[str, Any]) -> dict[str, Any]:
    """Convert ``image_url`` and raw images into the session's ``image_pil`` parts."""
    item = dict(part)
    part_type = item.get("type")
    if part_type == "image_pil":
        item["image_pil"] = to_pil_image(item["image_pil"])
    elif part_type == "image_url":
        image_url = item.pop("image_url")
        url = image_url["url"] if isinstance(image_url, Mapping) else image_url
        item["type"] = "image_pil"
        if isinstance(url, str) and url.startswith("data:"):
            item["image_pil"] = _decode_data_url(url)
        else:
            item["image_pil"] = to_pil_image(url)
    return item


def _normalize_message(message: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    content = normalized.get("content")
    if isinstance(content, list):
        normalized["content"] = [_normalize_part(part) for part in content]
    return normalized


def _text_part(text: str, timestamp_sec: float | None) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "text", "text": text}
    if timestamp_sec is not None:
        part["timestamp_sec"] = float(timestamp_sec)
    return part


def _frame_part(frame: VideoFrame) -> dict[str, Any]:
    part: dict[str, Any] = {
        "type": "image_pil",
        "image_pil": to_pil_image(frame.image),
        "timestamp_sec": float(frame.timestamp_sec),
    }
    if frame.frame_index is not None:
        part["frame_index"] = int(frame.frame_index)
    if frame.fps is not None:
        part["fps"] = float(frame.fps)
    return part


def _as_video_frame(item: Any) -> VideoFrame:
    if isinstance(item, VideoFrame):
        return item
    if isinstance(item, tuple) and 2 <= len(item) <= 4:
        return VideoFrame(*item)
    raise TypeError(
        "frames must be VideoFrame or (image, timestamp_sec[, frame_index[, fps]]) tuples"
    )


def _check_source(source: str | Path) -> str:
    """A local video must exist; anything with a scheme is passed to the decoder."""
    text = str(source)
    if "://" in text:
        return text
    if not Path(text).is_file():
        raise FileNotFoundError(f"video file not found: {text}")
    return text


# ------------------------------------------------------------------ scheduling


class _QuestionSchedule:
    """Questions ``(t, text)`` waiting for the stream to reach ``t``."""

    def __init__(self, questions: Questions | None) -> None:
        timed = [(float(t), str(text)) for t, text in (questions or ())]
        timed.sort(key=lambda question: question[0])
        self._pending = deque(timed)

    def pop_before(self, timestamp_sec: float) -> list[tuple[float, str]]:
        """Questions asked strictly before ``timestamp_sec``."""
        due = []
        while self._pending and self._pending[0][0] < timestamp_sec - EPS:
            due.append(self._pending.popleft())
        return due

    def pop_until(self, timestamp_sec: float) -> list[tuple[float, str]]:
        """Questions asked at or before ``timestamp_sec``."""
        due = []
        while self._pending and self._pending[0][0] <= timestamp_sec + EPS:
            due.append(self._pending.popleft())
        return due

    def pop_all(self) -> list[tuple[float, str]]:
        remaining = list(self._pending)
        self._pending.clear()
        return remaining


async def _aiterate(source: Iterable[Any] | AsyncIterable[Any]) -> AsyncIterator[Any]:
    """Iterate a sync or async iterable; blocking sync ``next`` runs in a thread."""
    if isinstance(source, AsyncIterable):
        async for item in source:
            yield item
        return
    iterator = iter(source)
    try:
        while True:
            item = await asyncio.to_thread(next, iterator, _ITER_END)
            if item is _ITER_END:
                return
            yield item
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            # A cancelled ``next`` may still be running in its thread; the
            # generator then closes itself when that call returns and it is freed.
            with contextlib.suppress(ValueError):
                await asyncio.to_thread(close)


async def _write_chunks(
    worker: CodecWorkerProcess, chunks: Iterable[bytes] | AsyncIterable[bytes]
) -> None:
    async with contextlib.aclosing(_aiterate(chunks)) as items:
        async for chunk in items:
            if chunk:
                await worker.write(bytes(chunk))
    await worker.end_input()


async def _run_with_upload(sender: asyncio.Task[Any], uploader: asyncio.Task[Any]) -> None:
    """Upload bytes while windows are sent; the windows decide the outcome.

    An upload error is raised unless it is a broken pipe, which means the
    worker exited and its own error (with stderr) surfaces from ``sender``.
    When the windows end first (``max_seconds`` or a worker failure) the rest
    of the upload is dropped.
    """
    try:
        done, _ = await asyncio.wait(
            {sender, uploader}, return_when=asyncio.FIRST_COMPLETED
        )
        if uploader in done and not uploader.cancelled():
            upload_error = uploader.exception()
            if upload_error is not None and not isinstance(
                upload_error, (BrokenPipeError, ConnectionResetError)
            ):
                raise upload_error
        await sender
    finally:
        for task in (sender, uploader):
            task.cancel()
        await asyncio.gather(sender, uploader, return_exceptions=True)


# --------------------------------------------------------------------- session


class AsyncLocalSession:
    """In-process streaming session directly backed by ``OVSession``."""

    def __init__(self, session: OVSession) -> None:
        self._session = session
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def raw_session(self) -> OVSession:
        return self._session

    async def __aenter__(self) -> AsyncLocalSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            await self.abort()
        else:
            await self.close()

    def memory_snapshot(self) -> dict[str, Any]:
        return self._session.memory_snapshot()

    # -- push one input -------------------------------------------------------

    async def send_messages(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        input_id: str | None = None,
    ) -> str:
        """Append OpenAI-style messages; images may be PIL, arrays, bytes or data URLs."""
        normalized = await asyncio.to_thread(
            lambda: tuple(_normalize_message(message) for message in messages)
        )
        return await self._session.send_messages(messages=normalized, input_id=input_id)

    async def send_frame(
        self,
        image: Any,
        *,
        timestamp_sec: float,
        frame_index: int | None = None,
        fps: float | None = None,
        question: str | None = None,
        system: str | None = None,
        input_id: str | None = None,
    ) -> str:
        parts = [] if question is None else [_text_part(question, timestamp_sec)]
        parts.append(_frame_part(VideoFrame(image, timestamp_sec, frame_index, fps)))
        return await self._send_user_parts(parts, system=system, input_id=input_id)

    async def send_codec_window(
        self,
        window: CodecWindow,
        *,
        question: str | None = None,
        system: str | None = None,
        input_id: str | None = None,
    ) -> str:
        parts = [] if question is None else [_text_part(question, window.timestamp_sec)]
        parts.extend(window.content())
        return await self._send_user_parts(parts, system=system, input_id=input_id)

    async def send_question(
        self,
        text: str,
        *,
        timestamp_sec: float | None = None,
        input_id: str | None = None,
    ) -> str:
        """Text-only input, e.g. a question during a live stream.

        Without ``timestamp_sec`` it is stamped with the latest input's time.
        """
        return await self._send_user_parts(
            [_text_part(text, timestamp_sec)], input_id=input_id
        )

    async def _send_user_parts(
        self,
        parts: list[dict[str, Any]],
        *,
        system: str | None = None,
        input_id: str | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": parts})
        return await self.send_messages(messages=messages, input_id=input_id)

    # -- events and lifecycle -------------------------------------------------

    async def events(self) -> AsyncIterator[Event]:
        """All session events. One consumer at a time; ``stream_*`` is one too."""
        async with contextlib.aclosing(self._session.events()) as events:
            async for event in events:
                yield event

    async def responses(self) -> AsyncIterator[Event]:
        async for event in self.events():
            if event.type == "response.output_text.done":
                yield event

    async def drain(self) -> None:
        await self._session.drain()

    async def cancel_response(self, response_id: str) -> bool:
        return await self._session.cancel_response(response_id)

    def commit_answer(self, response_id: str, text: str) -> bool:
        return self._session.commit_answer(response_id, text)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._session.close()

    async def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._session.abort()

    # -- stream a whole source ------------------------------------------------

    def stream_video(
        self,
        source: str | Path,
        *,
        video_backend: Literal["frames", "codec"] = "frames",
        fps: float = 1.0,
        max_seconds: float = 0.0,
        codec_config: CodecConfig | None = None,
        codec_python: str | None = None,
        questions: Questions | None = None,
        system: str | None = None,
        close_when_done: bool = True,
    ) -> AsyncIterator[Event]:
        """Stream a video file or URL and yield events as they arrive.

        ``frames`` samples ``fps`` frames per second (OpenCV). ``codec`` sends
        one I+P window per second built from the H.264 bitstream; ``fps`` does
        not apply to it.
        """
        source_ref = _check_source(source)
        if video_backend == "frames":
            if codec_config is not None or codec_python is not None:
                raise ValueError("codec_config and codec_python need video_backend='codec'")
            frames = iter_video_frames(source_ref, fps=fps, max_seconds=max_seconds)
            return self.stream_frames(
                frames,
                questions=questions,
                system=system,
                close_when_done=close_when_done,
            )
        if video_backend == "codec":
            config = codec_config or CodecConfig()
            if max_seconds:
                config = replace(config, max_seconds=max_seconds)
            command = codec_worker_command(
                resolve_codec_python(codec_python), config, video=source_ref
            )
            return self._stream_codec(
                command,
                chunks=None,
                questions=questions,
                system=system,
                close_when_done=close_when_done,
            )
        raise ValueError("video_backend must be 'frames' or 'codec'")

    def stream_frames(
        self,
        frames: Iterable[Any] | AsyncIterable[Any],
        *,
        questions: Questions | None = None,
        system: str | None = None,
        close_when_done: bool = True,
    ) -> AsyncIterator[Event]:
        """Stream frames from any producer and yield events.

        Items are ``VideoFrame`` or ``(image, timestamp_sec[, frame_index[, fps]])``.
        A sync iterable is read in a worker thread, so a blocking camera read
        does not stall the event loop.
        """

        async def timed_inputs() -> AsyncIterator[TimedInput]:
            async with contextlib.aclosing(_aiterate(frames)) as items:
                async for item in items:
                    frame = _as_video_frame(item)
                    part = await asyncio.to_thread(_frame_part, frame)
                    yield frame.timestamp_sec, [part]

        async def feed(record_input: RecordInput) -> None:
            await self._send_timed_inputs(timed_inputs(), questions, system, record_input)

        return self._drive(feed, close_when_done)

    def stream_h264(
        self,
        chunks: Iterable[bytes] | AsyncIterable[bytes],
        *,
        fps: float,
        width: int,
        height: int,
        codec_config: CodecConfig | None = None,
        codec_python: str | None = None,
        questions: Questions | None = None,
        system: str | None = None,
        close_when_done: bool = True,
    ) -> AsyncIterator[Event]:
        """Turn live H.264 bytes into codec I+P windows and yield events.

        ``chunks`` is the byte stream in MPEG-TS, split anywhere. ``fps`` is
        the declared constant frame rate and ``width``/``height`` the native
        size: a live pipe cannot be probed without consuming it.
        """
        command = codec_worker_command(
            resolve_codec_python(codec_python),
            codec_config or CodecConfig(),
            live_fps=fps,
            source_size=(width, height),
        )
        return self._stream_codec(
            command,
            chunks=chunks,
            questions=questions,
            system=system,
            close_when_done=close_when_done,
        )

    def _stream_codec(
        self,
        command: list[str],
        *,
        chunks: Iterable[bytes] | AsyncIterable[bytes] | None,
        questions: Questions | None,
        system: str | None,
        close_when_done: bool,
    ) -> AsyncIterator[Event]:
        async def feed(record_input: RecordInput) -> None:
            worker = await CodecWorkerProcess.start(command, live=chunks is not None)
            try:

                async def timed_inputs() -> AsyncIterator[TimedInput]:
                    async with contextlib.aclosing(worker.windows()) as windows:
                        async for window in windows:
                            yield window.timestamp_sec, window.parts

                sending = self._send_timed_inputs(
                    timed_inputs(), questions, system, record_input
                )
                if chunks is None:
                    await sending
                else:
                    await _run_with_upload(
                        asyncio.create_task(sending),
                        asyncio.create_task(_write_chunks(worker, chunks)),
                    )
            finally:
                await worker.aclose()

        return self._drive(feed, close_when_done)

    async def _send_timed_inputs(
        self,
        inputs: AsyncIterator[TimedInput],
        questions: Questions | None,
        system: str | None,
        record_input: RecordInput,
    ) -> None:
        schedule = _QuestionSchedule(questions)
        pending_system = system
        snapshot = self._session.memory_snapshot()
        has_history = bool(snapshot["turns"] or snapshot["archive"])

        async def send(parts: list[dict[str, Any]]) -> None:
            nonlocal pending_system, has_history
            input_id = await self._send_user_parts(parts, system=pending_system)
            pending_system = None
            has_history = True
            record_input(input_id)

        async with contextlib.aclosing(inputs) as timed:
            async for timestamp_sec, visual_parts in timed:
                if has_history:
                    for asked_at, text in schedule.pop_before(timestamp_sec):
                        await send([_text_part(text, asked_at)])
                due = [
                    _text_part(text, asked_at)
                    for asked_at, text in schedule.pop_until(timestamp_sec)
                ]
                await send([*due, *visual_parts])
        for asked_at, text in schedule.pop_all():
            await send([_text_part(text, asked_at)])

    async def _drive(self, feed: Feed, close_when_done: bool) -> AsyncIterator[Event]:
        """Run ``feed`` while yielding session events.

        With ``close_when_done`` the session is closed after the feed and the
        iterator ends at ``session.closed``. Otherwise it ends once every input
        the feed sent has its ``input.done``; responses still being written
        arrive through the next events consumer. A feed error aborts the
        session and is re-raised; leaving the iterator early only stops feeding.
        """
        mailbox: asyncio.Queue[Any] = asyncio.Queue()
        sent_input_ids: list[str] = []

        async def forward_events() -> None:
            try:
                async with contextlib.aclosing(self._session.events()) as events:
                    async for event in events:
                        mailbox.put_nowait(event)
            finally:
                mailbox.put_nowait(_EVENTS_DONE)

        async def produce() -> None:
            try:
                await feed(sent_input_ids.append)
                if close_when_done:
                    await self.close()
            except asyncio.CancelledError:
                raise
            except BaseException:
                await self.abort()
                raise
            finally:
                mailbox.put_nowait(_FEED_DONE)

        forwarder = asyncio.create_task(forward_events())
        producer = asyncio.create_task(produce())
        finished_input_ids: set[str] = set()
        feed_done = False
        try:
            while True:
                item = await mailbox.get()
                if item is _EVENTS_DONE:
                    break
                if item is _FEED_DONE:
                    feed_done = True
                else:
                    yield item
                    if item.type == "session.closed":
                        break
                    if item.type == "input.done" and item.input_id is not None:
                        finished_input_ids.add(item.input_id)
                if feed_done and not close_when_done:
                    feed_failed = producer.done() and (
                        producer.cancelled() or producer.exception() is not None
                    )
                    if feed_failed or finished_input_ids.issuperset(sent_input_ids):
                        break
            # Stop forwarding, then hand over events that were already forwarded.
            forwarder.cancel()
            await asyncio.gather(forwarder, return_exceptions=True)
            while not mailbox.empty():
                item = mailbox.get_nowait()
                if item is not _FEED_DONE and item is not _EVENTS_DONE:
                    yield item
            if not producer.done():
                await asyncio.wait({producer}, timeout=_PRODUCER_GRACE_SEC)
            if producer.done() and not producer.cancelled():
                producer.result()
        finally:
            for task in (producer, forwarder):
                if not task.done():
                    task.cancel()
            await asyncio.gather(producer, forwarder, return_exceptions=True)


# ----------------------------------------------------------- synchronous API


class _LoopThread:
    """Background asyncio event loop driving the synchronous wrappers."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10.0)
        self._loop.close()


async def _no_input(record_input: RecordInput) -> None:
    return None


class LocalSession:
    """Synchronous wrapper over ``AsyncLocalSession``.

    Single-input methods return the events up to that input's ``input.done``
    (``wait=False`` returns immediately; its events come with the next call).
    Streaming methods return a blocking iterator of events.
    """

    def __init__(self, loop: _LoopThread, inner: AsyncLocalSession) -> None:
        self._loop = loop
        self._inner = inner

    @property
    def session_id(self) -> str:
        return self._inner.session_id

    def __enter__(self) -> LocalSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()

    def memory_snapshot(self) -> dict[str, Any]:
        return self._inner.memory_snapshot()

    def _collect(self, stream: AsyncIterator[Event]) -> list[Event]:
        async def collect() -> list[Event]:
            async with contextlib.aclosing(stream) as events:
                return [event async for event in events]

        return self._loop.run(collect())

    def _iterate(self, stream: AsyncIterator[Event]) -> Iterator[Event]:
        finished = object()

        async def next_event() -> Any:
            try:
                return await anext(stream)
            except StopAsyncIteration:
                return finished

        try:
            while (item := self._loop.run(next_event())) is not finished:
                yield item
        finally:
            self._loop.run(stream.aclose())

    def _send_one(self, send: Callable[[], Awaitable[str]], wait: bool) -> list[Event]:
        if not wait:
            self._loop.run(send())
            return []

        async def feed(record_input: RecordInput) -> None:
            record_input(await send())

        return self._collect(self._inner._drive(feed, close_when_done=False))

    def send_messages(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        input_id: str | None = None,
        wait: bool = True,
    ) -> list[Event]:
        return self._send_one(
            lambda: self._inner.send_messages(messages=messages, input_id=input_id), wait
        )

    def send_frame(
        self,
        image: Any,
        *,
        timestamp_sec: float,
        frame_index: int | None = None,
        fps: float | None = None,
        question: str | None = None,
        system: str | None = None,
        input_id: str | None = None,
        wait: bool = True,
    ) -> list[Event]:
        return self._send_one(
            lambda: self._inner.send_frame(
                image,
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                fps=fps,
                question=question,
                system=system,
                input_id=input_id,
            ),
            wait,
        )

    def send_codec_window(
        self,
        window: CodecWindow,
        *,
        question: str | None = None,
        system: str | None = None,
        input_id: str | None = None,
        wait: bool = True,
    ) -> list[Event]:
        return self._send_one(
            lambda: self._inner.send_codec_window(
                window, question=question, system=system, input_id=input_id
            ),
            wait,
        )

    def send_question(
        self,
        text: str,
        *,
        timestamp_sec: float | None = None,
        input_id: str | None = None,
        wait: bool = True,
    ) -> list[Event]:
        return self._send_one(
            lambda: self._inner.send_question(
                text, timestamp_sec=timestamp_sec, input_id=input_id
            ),
            wait,
        )

    def stream_video(self, source: str | Path, **options: Any) -> Iterator[Event]:
        """Same options as ``AsyncLocalSession.stream_video``."""
        return self._iterate(self._inner.stream_video(source, **options))

    def stream_frames(self, frames: Iterable[Any], **options: Any) -> Iterator[Event]:
        """Same options as ``AsyncLocalSession.stream_frames``."""
        return self._iterate(self._inner.stream_frames(frames, **options))

    def stream_h264(self, chunks: Iterable[bytes], **options: Any) -> Iterator[Event]:
        """Same options as ``AsyncLocalSession.stream_h264``."""
        return self._iterate(self._inner.stream_h264(chunks, **options))

    def close(self) -> list[Event]:
        """Close the session and return the events not yet returned."""
        return self._collect(self._inner._drive(_no_input, close_when_done=True))

    def abort(self) -> None:
        self._loop.run(self._inner.abort())


__all__ = ["AsyncLocalSession", "LocalSession", "VideoFrame"]
