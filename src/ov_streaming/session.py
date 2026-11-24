from __future__ import annotations

import asyncio
import contextlib
import math
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .renderers import Renderer
from .ov21_protocol import escape_control_text
from .types import Event, Memory, Turn
from .verification import VerificationResult, load_verification

# Newest archived items of each memory tag that compaction keeps in the prompt.
_ARCHIVE_SIZE = 32
# Share of the context a compaction forced by its length fills with replayed
# turns; the rest is left for the archive, the new input and the answers.
_REPLAY_SHARE = 0.5


@dataclass
class _InputTiming:
    received: float
    queued: float
    started: float = 0.0
    control_ready: float = 0.0

    def metrics(self) -> dict[str, float]:
        return {
            "server_preprocess_ms": 1000 * (self.queued - self.received),
            "input_queue_ms": 1000 * (self.started - self.queued),
            "control_ms": 1000 * (self.control_ready - self.started),
        }


@dataclass
class _ResponseRun:
    response_id: str
    source_request_id: str
    control: str
    input_id: str
    timestamp_sec: float
    submitted: asyncio.Future[None]
    timing: _InputTiming
    suffix_token_ids: Sequence[int]
    verification: bool = False
    submitted_at: float | None = None
    first_output_at: float | None = None
    completed_at: float | None = None
    task: asyncio.Task[None] | None = None
    started: bool = False
    cancel_reason: str | None = None


class OVSession:
    """One persistent session with concurrent response branches."""

    def __init__(
        self,
        backend: Any,
        renderer: Renderer,
        session_id: str,
        *,
        enable: Sequence[str] | None = None,
        max_responses: int = 3,
        max_response_tokens: int | None = None,
        max_caption_tokens: int | None = None,
        caption_prompt: str | None = None,
        max_visual_frames: int | None = None,
        compact_to_visual_frames: int | None = None,
        pin_first_question: bool = False,
        answer_commit: Literal["on_done", "client"] = "on_done",
        verification: str | None = None,
        verification_timeout: float = 10.0,
    ) -> None:
        if max_responses < 1:
            raise ValueError("max_responses must be positive")
        if max_response_tokens is not None and (type(max_response_tokens) is not int or max_response_tokens < 1):
            raise ValueError("max_response_tokens must be a positive integer or None")
        if max_caption_tokens is not None and (
            type(max_caption_tokens) is not int or max_caption_tokens < 1
        ):
            raise ValueError("max_caption_tokens must be a positive integer or None")
        if caption_prompt is not None and (not isinstance(caption_prompt, str) or not caption_prompt.strip()):
            raise ValueError("caption_prompt must be non-empty text or None")
        if max_visual_frames is None:
            if compact_to_visual_frames is not None:
                raise ValueError("compact_to_visual_frames requires max_visual_frames")
        else:
            if max_visual_frames < 1:
                raise ValueError("max_visual_frames must be positive")
            if compact_to_visual_frames is None:
                compact_to_visual_frames = max_visual_frames
            if not 1 <= compact_to_visual_frames <= max_visual_frames:
                raise ValueError(
                    "compact_to_visual_frames must be between 1 and max_visual_frames"
                )
        self.backend = backend
        self.renderer = renderer
        self.session_id = session_id
        self.max_responses = max_responses
        self.max_response_tokens = renderer.max_response_tokens if max_response_tokens is None else max_response_tokens
        self.max_caption_tokens = max_caption_tokens
        self.caption_prompt = caption_prompt
        self.max_visual_frames = max_visual_frames
        self.compact_to_visual_frames = compact_to_visual_frames
        self.pin_first_question = pin_first_question
        self.answer_commit = answer_commit
        self.verification = verification
        self._verifier = load_verification(verification, renderer)
        if not math.isfinite(verification_timeout) or verification_timeout <= 0:
            raise ValueError("verification_timeout must be finite and positive")
        self.verification_timeout = verification_timeout

        # The engine's context, which the source and every branch must fit.
        self.max_model_len = getattr(backend, "max_model_len", None)
        if (
            max_caption_tokens is not None
            and self.max_model_len is not None
            and max_caption_tokens >= self.max_model_len
        ):
            raise ValueError("max_caption_tokens must be smaller than max_model_len")
        self.enable = None if enable is None else tuple(dict.fromkeys(enable))
        names, self._prefix = renderer.session_controls(self.enable)
        self._controls = {renderer.control_ids[name]: name for name in names}
        if caption_prompt is not None:
            opening = "<caption_begin>\n"
            if "caption" not in names or opening not in self._prefix:
                raise ValueError("caption_prompt requires enabled caption with a caption_begin opening")
            self._prefix = self._prefix.replace(opening, opening + escape_control_text(caption_prompt) + "\n", 1)
        self._cache_salt = uuid.uuid4().hex
        self._inputs: asyncio.Queue[tuple[str, Turn, _InputTiming] | None] = (
            asyncio.Queue()
        )
        self._events: asyncio.Queue[Event | None] = asyncio.Queue()
        self._responses: dict[str, _ResponseRun] = {}
        self._source: Any = None
        self._system: str | None = None
        # The control the source sampled last; the next append starts with it.
        self._control: str | None = None
        # Turns on the source, kept only when compaction may replay them.
        self._turns: list[Turn] = []
        self._source_tokens = 0
        # Tokens a frame takes beyond the text that stands for it.
        self._image_tokens = 0
        self._archive: list[Memory] = []
        self._pinned_question: Memory | None = None
        # Written back with the next turn, in completion order.
        self._pending: list[Memory] = []
        # Speaking time of each finished answer waiting for commit_answer().
        self._uncommitted: dict[str, float] = {}
        self._input_ids: set[str] = set()
        self._next_source_index = 0
        self._next_response_index = 0
        self._timestamp_sec = 0.0
        self._close_lock = asyncio.Lock()
        # Inputs are read off the event loop; this keeps them in arrival order.
        self._send_lock = asyncio.Lock()
        self._input_task: asyncio.Task[None] | None = None
        self._failure_cleanup_task: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None
        self._accepting = True
        self._closed = False
        self._event_consumer_active = False

    @property
    def compacts(self) -> bool:
        """Whether the session may open a second source to compact."""
        return self.max_visual_frames is not None or self.max_model_len is not None

    def memory_snapshot(self) -> dict[str, Any]:
        """Read the memory already written into this session, without draining branches.

        Pending completed branches have not reached a user turn yet. Keep them
        separate so an evaluation at a question boundary cannot silently treat
        them as earlier committed context. No frames, decoding, or state change
        are involved in this synchronous snapshot.
        """
        def item(memory: Memory) -> dict[str, Any]:
            return {"kind": memory.kind, "timestamp_sec": memory.timestamp_sec,
                    "text": memory.text}

        return {
            "session_id": self.session_id,
            "timestamp_sec": self._timestamp_sec,
            "archive": [item(memory) for memory in self._archive],
            "turns": [{"timestamp_sec": turn.timestamp_sec,
                       "questions": [item(memory) for memory in turn.questions],
                       "memories": [item(memory) for memory in turn.memories]}
                      for turn in self._turns],
            "pending": [item(memory) for memory in self._pending],
            "max_visual_frames": self.max_visual_frames,
            "compact_to_visual_frames": self.compact_to_visual_frames,
        }

    def _branch_max_tokens(self, control: str) -> int:
        if control == "caption" and self.max_caption_tokens is not None:
            return self.max_caption_tokens
        return self.max_response_tokens
    async def send_messages(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        input_id: str | None = None,
    ) -> str:
        received = time.perf_counter()
        input_id = input_id or uuid.uuid4().hex
        async with self._send_lock:
            self._raise_if_failed()
            if not self._accepting:
                raise RuntimeError("session is closing")
            if input_id in self._input_ids:
                raise ValueError(f"duplicate input_id: {input_id}")
            # Frames are processed in a worker thread, so a server's sessions
            # are not limited to one core between them.
            system, turn = await asyncio.to_thread(self._read_input, messages)
            self._raise_if_failed()
            if not self._accepting:
                raise RuntimeError("session is closing")
            if system is not None:
                self._system = system
            self._timestamp_sec = turn.timestamp_sec
            self._input_ids.add(input_id)
            self._ensure_started()
            self._inputs.put_nowait(
                (input_id, turn, _InputTiming(received, time.perf_counter()))
            )
            return input_id

    async def drain(self) -> None:
        if self._input_task is None:
            return
        drained = asyncio.create_task(self._inputs.join())
        try:
            done, _ = await asyncio.wait(
                {drained, self._input_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            drained.cancel()
            await asyncio.gather(drained, return_exceptions=True)
        self._raise_if_failed()
        if drained not in done:
            raise RuntimeError("session input worker stopped before drain")

    async def events(self) -> AsyncIterator[Event]:
        if self._event_consumer_active:
            raise RuntimeError("only one event consumer is allowed per session")
        self._event_consumer_active = True
        try:
            while (event := await self._events.get()) is not None:
                yield event
            # Leave the end marker for a later consumer.
            self._events.put_nowait(None)
        finally:
            self._event_consumer_active = False

    async def cancel_response(self, response_id: str) -> bool:
        response = self._responses.get(response_id)
        if response is None:
            return False
        self._request_response_cancel(response, "user")
        await self._wait_response(response)
        return True

    def commit_answer(self, response_id: str, text: str) -> bool:
        """Write back the part of a finished answer that the user received.

        Only answers held back by answer_commit="client" can be committed.
        An empty text writes nothing back.
        """
        if response_id not in self._uncommitted:
            return False
        text = text.strip()
        self.renderer.check_text(text)
        timestamp_sec = self._uncommitted.pop(response_id)
        if text:
            self._pending.append(Memory("response", timestamp_sec, text))
        return True

    async def close(self) -> None:
        """Stop input and wait for already-created responses."""
        async with self._close_lock:
            if self._closed:
                return
            self._accepting = False
            try:
                if self._failure_cleanup_task is None:
                    await self.drain()
                    if self._input_task is not None:
                        await self._inputs.put(None)
                        await self._input_task
                    await asyncio.gather(
                        *(run.task for run in self._responses.values()),
                        return_exceptions=True,
                    )
                    if self._source is not None:
                        await self._source.close()
            except Exception as error:
                self._record_failure(error)
            if self._failure_cleanup_task is not None:
                await asyncio.shield(self._failure_cleanup_task)
            self._finish_events("failed" if self._failure else "completed")

    async def abort(self) -> None:
        """Cancel all work immediately."""
        async with self._close_lock:
            if self._closed:
                return
            self._accepting = False
            if self._failure_cleanup_task is not None:
                await asyncio.shield(self._failure_cleanup_task)
                return
            if self._input_task is not None:
                self._input_task.cancel()
                await asyncio.gather(self._input_task, return_exceptions=True)
            await self._cancel_all_responses("session_aborted")
            await self._abort_source(self._source)
            self._finish_events("cancelled")

    def _read_input(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> tuple[str | None, Turn]:
        roles = [message.get("role") for message in messages]
        if roles == ["user"]:
            system = None
        elif roles == ["system", "user"] and not self._input_ids:
            system = messages[0]["content"]
            if not isinstance(system, str):
                raise ValueError("system content must be text")
            self.renderer.check_text(system)
        else:
            raise ValueError(
                "an input is one user message, after a system message only "
                "in the first input"
            )
        content = messages[-1]["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        timestamp_sec = self._timestamp_sec
        frames = []
        questions = []
        for part in content:
            if part.get("timestamp_sec") is not None:
                timestamp_sec = float(part["timestamp_sec"])
            if part["type"] == "image_pil":
                render = self.renderer.render_frame
                codec_options = {}
                if "codec" in part:
                    render = self.renderer.render_codec_frame
                    codec_options["codec"] = part["codec"]
                frame = render(
                    part["image_pil"],
                    timestamp_sec,
                    frame_index=part.get("frame_index"),
                    fps=part.get("fps"),
                    **codec_options,
                )
                frames.append(frame)
            else:
                text = part["text"].strip()
                self.renderer.check_text(text)
                questions.append(Memory("question", timestamp_sec, text))
        if (
            self.compact_to_visual_frames is not None
            and len(frames) > self.compact_to_visual_frames
        ):
            raise ValueError(
                "one input exceeds compact_to_visual_frames; split the input"
            )
        return system, Turn(timestamp_sec, tuple(frames), tuple(questions))

    def _ensure_started(self) -> None:
        if self._input_task is not None:
            return
        self._source = self._open_source()
        self._input_task = asyncio.create_task(
            self._run_inputs(), name=f"ov-inputs:{self.session_id}"
        )

    async def _run_inputs(self) -> None:
        try:
            while (item := await self._inputs.get()) is not None:
                try:
                    await self._process_input(*item)
                finally:
                    self._inputs.task_done()
            self._inputs.task_done()
        except Exception as error:
            self._record_failure(error)

    async def _process_input(
        self, input_id: str, turn: Turn, timing: _InputTiming
    ) -> None:
        timing.started = time.perf_counter()
        if self.pin_first_question and self._pinned_question is None and turn.questions:
            self._pinned_question = turn.questions[0]
        turn.memories = tuple(self._pending)
        self._pending.clear()
        append = None
        if self._control is not None:
            append = self.renderer.render_append(turn, self._control)
            if self.max_model_len is not None:
                turn.tokens = self._count_tokens(append, turn.frames)
        kept = self._select_kept_turns(turn)
        if kept is None:
            source = self._source
            frames = turn.frames
            if append is None:
                prompt = self.renderer.render_prompt(
                    [turn], system=self._system, prefix=self._prefix
                )
            else:
                prompt = append
        else:
            dropped = self._turns[: len(self._turns) - len(kept)]
            archive = self._archive_turns(dropped)
            kept = [*kept, turn]
            frames = tuple(frame for item in kept for frame in item.frames)
            prompt = self.renderer.render_prompt(
                kept, archive, self._system, prefix=self._prefix
            )
            source = self._open_source()

        try:
            token_id = await source.step(
                prompt, frames, allowed_token_ids=tuple(self._controls)
            )
            turn.control = self._controls[token_id]
            timing.control_ready = time.perf_counter()
            response_id = None
            verification_skipped = None
            if turn.control != "silence":
                if (
                    self._verifier is not None
                    and turn.control == "response"
                    and any(run.verification for run in self._responses.values())
                ):
                    # Bound GPU work without claiming these are the same event.
                    verification_skipped = "verification_busy"
                else:
                    response_id = await self._start_response(
                        turn, input_id, source.request_id, timing
                    )
        except BaseException:
            if source is not self._source:
                await self._abort_source(source)
            raise

        if self.max_model_len is not None:
            before = self._source_tokens if source is self._source else 0
            if frames:
                text = self.backend.count_tokens(prompt)
                image = source.num_tokens - before - text
                self._image_tokens = max(self._image_tokens, -(-image // len(frames)))
            if kept is None:
                turn.tokens = source.num_tokens - before
            self._source_tokens = source.num_tokens
        self._control = turn.control
        if kept is None:
            if self.compacts:
                self._turns.append(turn)
        else:
            previous = self._source
            self._source, self._turns, self._archive = source, kept, archive
            await self._abort_source(previous)
            self._emit(
                Event(
                    "context.compacted",
                    self.session_id,
                    input_id=input_id,
                    timestamp_sec=turn.timestamp_sec,
                    metadata={
                        "retained_turns": len(kept),
                        "retained_visual_frames": len(frames),
                    },
                )
            )
        if self.max_visual_frames is None and self.max_model_len is not None:
            self._archive_unreplayable_turns()
        self._emit(
            Event(
                "input.done",
                self.session_id,
                input_id=input_id,
                response_id=response_id,
                timestamp_sec=turn.timestamp_sec,
                status="completed",
                control=turn.control,
                metadata={
                    "timing": timing.metrics(),
                    **(
                        {"verification_skipped": verification_skipped}
                        if verification_skipped
                        else {}
                    ),
                },
            )
        )

    def _select_kept_turns(self, turn: Turn) -> list[Turn] | None:
        frames = len(turn.frames) + sum(len(item.frames) for item in self._turns)
        too_many_frames = (
            self.max_visual_frames is not None and frames > self.max_visual_frames
        )
        too_long = False
        if self.max_model_len is not None:
            # A branch's prompt is the source plus its control token, and its
            # answer has to fit after that.
            body_budget = self.max_response_tokens
            if "caption" in self._controls.values():
                body_budget = max(body_budget, self._branch_max_tokens("caption"))
            reserve = 1 + body_budget
            if self._verifier is not None:
                instruction = self._verifier.instruction(
                    self.renderer, turn.timestamp_sec
                )
                suffix = self.renderer.render_verification_suffix(instruction)
                reserve = max(
                    reserve,
                    self.backend.count_tokens(suffix) + self._verifier.max_tokens,
                )
            too_long = self._source_tokens + turn.tokens + reserve > self.max_model_len
        if not (too_many_frames or too_long):
            return None

        frame_room = math.inf
        if self.compact_to_visual_frames is not None:
            frame_room = self.compact_to_visual_frames - len(turn.frames)
        token_room = math.inf
        if self.max_model_len is not None:
            # The replay opens with the archive; this one, as if every turn
            # were dropped, is the largest it can be.
            opening = self.renderer.render_prompt(
                [turn], self._archive_turns(self._turns), self._system, self._prefix
            )
            replay_limit = self.max_model_len * _REPLAY_SHARE
            if (
                "caption" in self._controls.values()
                and self._branch_max_tokens("caption") > self.max_response_tokens
            ):
                replay_limit = min(replay_limit, self.max_model_len - reserve)
            token_room = replay_limit - self._count_tokens(opening, turn.frames)
        start = len(self._turns)
        for index in range(len(self._turns) - 1, -1, -1):
            frame_room -= len(self._turns[index].frames)
            token_room -= self._turns[index].tokens
            if frame_room < 0 or token_room < 0:
                break
            start = index
        return self._turns[start:]

    def _archive_unreplayable_turns(self) -> None:
        # Turns older than any compaction would replay go to the archive now,
        # so the session does not hold their frames until then.
        room = self.max_model_len * _REPLAY_SHARE
        start = len(self._turns)
        while start and room - self._turns[start - 1].tokens >= 0:
            start -= 1
            room -= self._turns[start].tokens
        if start:
            self._archive = self._archive_turns(self._turns[:start])
            del self._turns[:start]

    def _count_tokens(self, text: str, frames: Sequence[Any]) -> int:
        return self.backend.count_tokens(text) + len(frames) * self._image_tokens

    def _archive_turns(self, turns: Sequence[Turn]) -> list[Memory]:
        archive = [
            *self._archive,
            *(memory for turn in turns for memory in self.renderer.archive_turn(turn)),
        ]
        counts: Counter[str] = Counter()
        kept = []
        for memory in reversed(archive):
            # A standing question must outlive later timed text such as subtitles.
            # Keep its original position and spelling, outside the ordinary quota.
            if memory is self._pinned_question:
                kept.append(memory)
                continue
            tag = self.renderer.memory_tags[memory.kind]
            counts[tag] += 1
            if counts[tag] <= _ARCHIVE_SIZE:
                kept.append(memory)
        return kept[::-1]

    async def _start_response(
        self, turn: Turn, input_id: str, source_request_id: str, timing: _InputTiming
    ) -> str:
        verify = self._verifier is not None and turn.control == "response"
        suffix = [self.renderer.control_ids[turn.control]]
        if verify:
            instruction = self._verifier.instruction(self.renderer, turn.timestamp_sec)
            suffix = self.backend.tokenizer.encode(
                self.renderer.render_verification_suffix(instruction),
                add_special_tokens=False,
            )
        response_id = f"ov-response-{self.session_id}-{self._next_response_index:06d}"
        self._next_response_index += 1
        while len(self._responses) >= self.max_responses:
            oldest = next(iter(self._responses.values()))
            self._request_response_cancel(oldest, "capacity")
            await self._wait_response(oldest)

        run = _ResponseRun(
            response_id=response_id,
            source_request_id=source_request_id,
            control=turn.control,
            input_id=input_id,
            timestamp_sec=turn.timestamp_sec,
            submitted=asyncio.get_running_loop().create_future(),
            timing=timing,
            suffix_token_ids=suffix,
            verification=verify,
        )
        self._responses[response_id] = run
        self._emit(
            Event(
                "response.created",
                self.session_id,
                input_id=input_id,
                response_id=response_id,
                timestamp_sec=run.timestamp_sec,
                status="in_progress",
                control=run.control,
                metadata={
                    "verification": self.verification if run.verification else None,
                    **(
                        {
                            "max_tokens": (
                                self._verifier.max_tokens
                                if run.verification
                                else self._branch_max_tokens(run.control)
                            )
                        }
                        if self.max_caption_tokens is not None
                        else {}
                    ),
                },
            )
        )
        run.task = asyncio.create_task(
            self._run_response(run), name=f"ov-response:{response_id}"
        )
        # The engine must hold the branch before the next input resumes the
        # source, which drops the sampled control token.
        await run.submitted
        return response_id

    async def _run_response(self, run: _ResponseRun) -> None:
        finished = False
        text: list[str] = []
        finish_reason = None
        submission = asyncio.create_task(
            self.backend.add_branch(
                run.source_request_id,
                run.response_id,
                suffix_token_ids=run.suffix_token_ids,
                max_tokens=(
                    self._verifier.max_tokens
                    if run.verification
                    else self._branch_max_tokens(run.control)
                ),
            )
        )
        try:
            try:
                run.started = True
                if run.cancel_reason is not None:
                    raise asyncio.CancelledError
                reading = self._read_response(run, submission, text)
                if run.verification:
                    finish_reason = await asyncio.wait_for(
                        reading, timeout=self.verification_timeout
                    )
                else:
                    finish_reason = await reading
                finished = True
            finally:
                if not finished:
                    # ADD must finish before ABORT for the same request ID.
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await asyncio.shield(submission)
                    with contextlib.suppress(Exception):
                        await asyncio.shield(self.backend.abort(run.response_id))
            answer = "".join(text).strip()
            truncated = finish_reason == "length"
            if run.verification:
                result = self._verifier.parse(
                    answer, self.renderer, truncated=truncated
                )
                self._emit_verification(run, result, answer)
                if not result.accepted:
                    self._emit_response_done(
                        run, "completed", reason=f"verification_{result.reason}"
                    )
                    return
                answer = result.text
            # A cut-off answer still reaches the client but is never written back.
            if not truncated:
                if run.control == "response" and self.answer_commit == "client":
                    self._uncommitted[run.response_id] = run.timestamp_sec
                elif answer:
                    self._pending.append(Memory(run.control, run.timestamp_sec, answer))
            self._emit(
                Event(
                    "response.output_text.done",
                    self.session_id,
                    input_id=run.input_id,
                    response_id=run.response_id,
                    timestamp_sec=run.timestamp_sec,
                    text=answer,
                    control=run.control,
                    metadata={"timing": self._response_timing(run)},
                )
            )
            self._emit_response_done(
                run, "completed", reason="length" if truncated else None
            )
        except asyncio.TimeoutError as error:
            if not run.verification:
                if not run.submitted.done():
                    run.submitted.set_exception(error)
                self._emit_response_done(run, "failed", error="response timed out")
                return
            if not run.submitted.done():
                run.submitted.set_result(None)
            self._emit_verification(
                run, VerificationResult(False, "timeout"), "".join(text)
            )
            self._emit_response_done(run, "cancelled", reason="verification_timeout")
        except asyncio.CancelledError:
            if not run.submitted.done():
                run.submitted.set_result(None)
            self._emit_response_done(
                run, "cancelled", reason=run.cancel_reason or "cancelled"
            )
            raise
        except Exception as error:
            if not run.submitted.done():
                run.submitted.set_exception(error)
            self._emit_response_done(run, "failed", error=str(error))
        finally:
            self._responses.pop(run.response_id, None)

    async def _read_response(self, run, submission, text) -> str | None:
        branch = await asyncio.shield(submission)
        run.submitted_at = time.perf_counter()
        if not run.submitted.done():
            run.submitted.set_result(None)
        finish_reason = None
        async for output in branch.outputs():
            if output.text and run.first_output_at is None:
                run.first_output_at = time.perf_counter()
            text.append(output.text)
            finish_reason = output.finish_reason
        if run.verification and finish_reason not in {"stop", "length"}:
            raise RuntimeError("verification branch ended without a terminal result")
        run.completed_at = time.perf_counter()
        return finish_reason

    def _response_timing(self, run: _ResponseRun) -> dict[str, float]:
        result = run.timing.metrics()
        result["response_ready_ms"] = 1000 * (time.perf_counter() - run.timing.received)
        if run.submitted_at is not None:
            result["candidate_to_fork_ms"] = 1000 * (
                run.submitted_at - run.timing.control_ready
            )
            for name, value in (
                ("branch_first_output_ms", run.first_output_at),
                ("branch_complete_ms", run.completed_at),
            ):
                if value is not None:
                    result[name] = 1000 * (value - run.submitted_at)
        return result

    def _emit_verification(self, run, result: VerificationResult, raw: str) -> None:
        self._emit(
            Event(
                "verification.done",
                self.session_id,
                input_id=run.input_id,
                response_id=run.response_id,
                timestamp_sec=run.timestamp_sec,
                control=run.control,
                reason=result.reason,
                metadata={
                    "accepted": result.accepted,
                    "raw": raw,
                    "verification": self.verification,
                    "timing": self._response_timing(run),
                },
            )
        )

    def _request_response_cancel(self, run: _ResponseRun, reason: str) -> None:
        if run.cancel_reason is not None:
            return
        run.cancel_reason = reason
        # A task cancelled before it starts cannot run its cleanup.
        if run.started:
            run.task.cancel()

    async def _wait_response(self, run: _ResponseRun) -> None:
        await asyncio.shield(asyncio.gather(run.task, return_exceptions=True))

    async def _cancel_all_responses(self, reason: str) -> None:
        responses = list(self._responses.values())
        for response in responses:
            self._request_response_cancel(response, reason)
        await asyncio.gather(
            *(response.task for response in responses),
            return_exceptions=True,
        )

    def _open_source(self) -> Any:
        request_id = (
            f"ov-source-{self.session_id}-{self._next_source_index:06d}-"
            f"{uuid.uuid4().hex}"
        )
        self._next_source_index += 1
        return self.backend.open_source(request_id, cache_salt=self._cache_salt)

    async def _abort_source(self, source: Any) -> None:
        if source is not None:
            with contextlib.suppress(Exception):
                await source.abort()

    def _emit_response_done(
        self,
        run: _ResponseRun,
        status: str,
        *,
        reason: str | None = None,
        error: str | None = None,
    ) -> None:
        self._emit(
            Event(
                "response.done",
                self.session_id,
                input_id=run.input_id,
                response_id=run.response_id,
                timestamp_sec=run.timestamp_sec,
                status=status,
                reason=reason,
                error=error,
                control=run.control,
                metadata={"timing": self._response_timing(run)},
            )
        )

    def _record_failure(self, error: BaseException) -> None:
        if self._failure is not None:
            return
        self._failure = error
        self._accepting = False
        self._emit(Event("error", self.session_id, error=str(error)))
        self._failure_cleanup_task = asyncio.create_task(
            self._cleanup_after_failure(),
            name=f"ov-failure:{self.session_id}",
        )

    async def _cleanup_after_failure(self) -> None:
        if self._input_task is not None:
            self._input_task.cancel()
            await asyncio.gather(self._input_task, return_exceptions=True)
        await self._cancel_all_responses("session_error")
        await self._abort_source(self._source)
        self._finish_events("failed")

    def _finish_events(self, status: str) -> None:
        if self._closed:
            return
        self._closed = True
        self._events.put_nowait(Event("session.closed", self.session_id, status=status))
        self._events.put_nowait(None)

    def _emit(self, event: Event) -> None:
        if not self._closed:
            self._events.put_nowait(event)

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("session failed") from self._failure
