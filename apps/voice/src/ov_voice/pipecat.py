"""OV in a Pipecat pipeline.

transport.input() -> VADProcessor -> TranscriptionService -> UserTurnProcessor
    -> OVProcessor -> SpeechService -> transport.output()

OVProcessor reaches OVStreaming only through ov_streaming.client. The two
services are Pipecat's OpenAI services adjusted for vLLM's ASR and for speech
servers that name their own voices, such as vLLM-Omni. The UserTurnProcessor
decides when the user has finished a turn; without one, every transcript is a
turn of its own.
"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from loguru import logger
from openai.types.audio import Transcription
from ov_streaming.client import ClientSession, OVClient
from PIL import Image
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    CancelWorkerFrame,
    DataFrame,
    EndFrame,
    Frame,
    InputImageRawFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import (
    RTVIClientMessageFrame,
    RTVIServerMessageFrame,
)
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.whisper.base_stt import BaseWhisperSTTService
from pipecat.utils.string import match_endofsentence

from ov_voice.answers import Action, Interrupt, Speak, SpokenAnswers


class TranscriptionService(BaseWhisperSTTService):
    """Transcribes each utterance the VAD cuts out at /v1/audio/transcriptions.

    OpenAISTTService always sends a language, which makes Qwen3-ASR transcribe
    as that language; this sends none, so the model detects it. It asks for
    json because vLLM refuses verbose_json for models without segment
    timestamps, Qwen3-ASR among them.
    """

    async def _transcribe(self, audio: bytes) -> Transcription:
        return await self._client.audio.transcriptions.create(
            file=("audio.wav", audio, "audio/wav"),
            model=self._settings.model,
            response_format="json",
        )


class SpeechService(OpenAITTSService):
    """Speaks through an OpenAI-compatible /v1/audio/speech with its own voices.

    OpenAITTSService refuses voices other than OpenAI's. This sends the voice
    as given and asks for PCM, 24 kHz from OpenAI and vLLM-Omni alike, streamed
    as it is decoded.
    """

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async with self._client.audio.speech.with_streaming_response.create(
            input=text,
            model=self._settings.model,
            voice=self._settings.voice,
            response_format="pcm",
            stream_format="audio",
        ) as response:
            await self.start_tts_usage_metrics(text)
            async for chunk in response.iter_bytes(self.chunk_size):
                await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(
                    chunk, self.sample_rate, 1, context_id=context_id
                )


@dataclass
class _TurnEnded(DataFrame):
    """Queued by OVProcessor to itself, behind the transcripts of a turn.

    It carries the turn's number and times, taken when the turn ended, since
    the next turn may have started by the time this frame is handled.
    """

    number: int = 0
    # Wall-clock times: when the user stopped speaking, when the turn ended.
    speech_end: float | None = None
    ended: float = 0.0


@dataclass
class _Question:
    """What goes to OV with the next input: the turns finished since the last."""

    text: str = ""
    turns: int = 0
    segments: int = 0
    # Times of the latest turn in it.
    speech_end: float | None = None
    ended: float = 0.0


def _log_question(question: _Question, *, with_frame: bool) -> None:
    # The times are those of the question's latest turn. speech_end->turn_end
    # covers VAD, turn detection and ASR; turn_end->sent covers the wait for a
    # camera frame and the request to OV.
    sent = time.time()
    speech_end = question.speech_end or question.ended

    def ms(start: float, end: float) -> int:
        return round(max(0.0, end - start) * 1000)

    logger.info(
        f"OV question: speech_end->sent={ms(speech_end, sent)}ms "
        f"speech_end->turn_end={ms(speech_end, question.ended)}ms "
        f"turn_end->sent={ms(question.ended, sent)}ms "
        f"turns={question.turns} segments={question.segments} "
        f"with_frame={with_frame} text={question.text!r}"
    )


def join_transcripts(before: str, after: str) -> str:
    """Join two pieces of one utterance, with a space only between words."""
    widths = {unicodedata.east_asian_width(c) for c in before[-1:] + after[:1]}
    if before and after and not widths & {"W", "F"}:
        return f"{before} {after}"
    return before + after


# OV2.1 starts an answer with the time it answers at, "At 6.967 seconds: ...".
# That stays in the memory written back, as in training, but is not spoken.
_TIME_PREFIX = re.compile(r"\s*At \d+(?:\.\d+)? seconds?:\s*")


def find_sentence_ends(text: str) -> list[int]:
    ends = []
    start = 0
    while end := match_endofsentence(text[start:]):
        start += end
        ends.append(start)
    if text[start:].strip():
        ends.append(len(text))
    return ends


class OVProcessor(FrameProcessor):
    """Feeds the camera and the user's questions to OV and speaks its answers.

    At most one camera frame per frame_interval seconds goes in as an input.
    A question cannot be taken back once appended, so it goes in only when the
    user's turn is over. With user_turns, a UserTurnProcessor upstream decides
    that: the transcripts of a turn, one per VAD segment, are held and joined
    until its UserStoppedSpeakingFrame. Without it, each transcript is a turn.
    A finished question rides on the next frame, or goes in alone when no
    frame has come for two intervals or none comes within two.

    Only responses are spoken, one TTSSpeakFrame per sentence, so the output
    transport sends a BotStoppedSpeakingFrame upstream as each sentence
    finishes playing. Captions are never spoken. OV remembers every answer as
    soon as it is written; SpokenAnswers decides what is spoken: a newer answer
    cuts the one playing, and the user starting to speak stops the answer and
    cancels the answers still generating. The next answer waits until the
    user's turn is over. When OV refuses the session or the session fails, the
    pipeline is cancelled.
    """

    def __init__(
        self,
        client: OVClient,
        *,
        frame_interval: float = 1.0,
        max_visual_frames: int | None = None,
        compact_to_visual_frames: int | None = None,
        user_turns: bool = False,
        enable: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._client = client
        self._frame_interval = frame_interval
        self._max_visual_frames = max_visual_frames
        self._compact_to_visual_frames = compact_to_visual_frames
        self._user_turns = user_turns
        self._enable = list(enable or [])
        # When each input was sent, until OV reports it done.
        self._sent_at: dict[str, float] = {}
        self._answers = SpokenAnswers()
        # Policy calls and their actions run one at a time, so an interruption
        # never lands between the sentences of an answer being queued.
        self._lock = asyncio.Lock()
        self._session: ClientSession | None = None
        self._generating: set[str] = set()
        # Transcripts of the turn in progress, and the finished question.
        self._heard: list[str] = []
        self._question = _Question()
        self._question_timer: asyncio.Task | None = None
        # The number of the latest user turn, and when the user last stopped.
        self._user_turn = 0
        self._speech_end: float | None = None
        self._frame_time: float | None = None
        self._turn = 0
        # Text ends of the sentences of the answer being spoken, not yet played.
        self._sentence_ends: deque[int] = deque()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._open_session()
            return
        if isinstance(frame, CancelFrame):
            await self.push_frame(frame, direction)
            await self._stop_question_timer()
            await self._session.abort()
            return
        if isinstance(frame, EndFrame):
            await self._stop_question_timer()
            await self._session.close()
        elif isinstance(frame, InputImageRawFrame):
            await self._send_frame(frame)
        elif isinstance(frame, TranscriptionFrame):
            if frame.text.strip():
                self._heard.append(frame.text.strip())
            if not self._user_turns:
                await self._finish_turn(_TurnEnded(speech_end=self._speech_end))
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            # Any speech stops the answer at once, even a pause in a turn.
            async with self._lock:
                await self._run(self._answers.user_started())
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speech_end = frame.timestamp - frame.stop_secs
            if not self._user_turns:
                async with self._lock:
                    await self._run(self._answers.user_stopped())
        elif isinstance(frame, UserStartedSpeakingFrame) and self._user_turns:
            self._user_turn += 1
        elif isinstance(frame, UserStoppedSpeakingFrame) and self._user_turns:
            # A system frame overtakes the data frames queued before it, among
            # them the turn's last transcript. A data frame queued now comes
            # after that transcript, so the turn ends when this one is handled.
            await self.queue_frame(
                _TurnEnded(
                    number=self._user_turn,
                    speech_end=self._speech_end,
                    ended=time.time(),
                )
            )
        elif isinstance(frame, RTVIClientMessageFrame) and frame.type == "ov-text":
            # A question typed on the page: a whole turn, without speech.
            text = str((frame.data or {}).get("text", "")).strip()
            if text:
                # Angle brackets could spell a control token, which OV refuses.
                text = text.replace("<", "＜").replace(">", "＞")
                await self._finish_turn(_TurnEnded(ended=time.time()), [text])
            return
        elif isinstance(frame, _TurnEnded):
            await self._finish_turn(frame)
            # The next turn's start, a system frame too, may have overtaken
            # this one; then the user is speaking again and answers wait.
            if frame.number == self._user_turn:
                async with self._lock:
                    await self._run(self._answers.user_stopped())
            return
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._finish_sentence()
        await self.push_frame(frame, direction)

    async def _open_session(self) -> None:
        self._session = self._client.session(
            max_visual_frames=self._max_visual_frames,
            compact_to_visual_frames=self._compact_to_visual_frames,
            enable=self._enable,
        )
        try:
            await self._session.__aenter__()
        except Exception as error:
            await self._cancel_pipeline(error)
            return
        self.create_task(self._read_events())

    async def _send_frame(self, frame: InputImageRawFrame) -> None:
        now = self._now()
        if (
            self._frame_time is not None
            and now - self._frame_time < self._frame_interval
        ):
            return
        self._frame_time = now
        await self._send_input(
            now, Image.frombytes(frame.format, frame.size, frame.image)
        )

    async def _finish_turn(
        self, turn: _TurnEnded, parts: list[str] | None = None
    ) -> None:
        """Add a turn to the question and send it: the transcripts heard, or typed text."""
        heard = self._heard if parts is None else parts
        if not heard:
            return
        question = self._question
        for part in heard:
            question.text = join_transcripts(question.text, part)
        question.turns += 1
        question.segments += len(heard)
        question.speech_end = turn.speech_end
        question.ended = turn.ended or time.time()
        if parts is None:
            self._heard.clear()
        now = self._now()
        if (
            self._frame_time is None
            or now - self._frame_time > 2 * self._frame_interval
        ):
            await self._send_input(now)
        elif self._question_timer is None:
            self._question_timer = self.create_task(self._send_held_question())

    async def _send_held_question(self) -> None:
        await asyncio.sleep(2 * self._frame_interval)
        self._question_timer = None
        if self._question.text:
            await self._send_input(self._now())

    async def _stop_question_timer(self) -> None:
        if self._question_timer is not None:
            await self.cancel_task(self._question_timer)

    async def _send_input(
        self, timestamp_sec: float, image: Image.Image | None = None
    ) -> None:
        async with self._lock:
            content = []
            if image is not None:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image},
                        "timestamp_sec": timestamp_sec,
                    }
                )
            question = self._question
            if question.text:
                content.append(
                    {
                        "type": "text",
                        "text": question.text,
                        "timestamp_sec": timestamp_sec,
                    }
                )
                self._question = _Question()
            input_id = await self._session.send_messages(
                messages=[{"role": "user", "content": content}]
            )
            self._sent_at[input_id] = time.time()
            if question.text:
                _log_question(question, with_frame=image is not None)
                await self._notify(
                    "question",
                    input_id=input_id,
                    text=question.text,
                    timestamp_sec=timestamp_sec,
                    segments=question.segments,
                )

    async def _read_events(self) -> None:
        try:
            async for event in self._session.events():
                await self._report(event)
                if event.control != "response":
                    continue
                if event.type == "response.created":
                    self._generating.add(event.response_id)
                elif event.type == "response.done":
                    self._generating.discard(event.response_id)
                elif event.type == "response.output_text.done" and event.text:
                    async with self._lock:
                        await self._run(self._answers.answer(event.text))
        except Exception as error:
            await self._cancel_pipeline(error)

    async def _cancel_pipeline(self, error: Exception) -> None:
        await self.push_error(f"OV session failed: {error}", exception=error)
        await self.push_frame(CancelWorkerFrame(), FrameDirection.UPSTREAM)

    async def _finish_sentence(self) -> None:
        async with self._lock:
            if not self._sentence_ends:
                return
            self._sentence_ends.popleft()
            if not self._sentence_ends:
                await self._run(self._answers.spoken(self._turn))

    async def _report(self, event) -> None:
        """Pass an OV event on to the page, with the input's round trip."""
        data = event.to_dict()
        if event.type == "input.done":
            sent = self._sent_at.pop(event.input_id, None)
            if sent is not None:
                data["latency_ms"] = round((time.time() - sent) * 1000)
        await self._notify("event", **data)

    async def _notify(self, kind: str, **data) -> None:
        # An RTVI server message, so a Pipecat client gets it as a ServerMessage
        # event; the page shows the session from these.
        await self.push_frame(RTVIServerMessageFrame(data={"ov": kind, **data}))

    async def _run(self, actions: list[Action]) -> None:
        for action in actions:
            if isinstance(action, Speak):
                self._turn = action.turn
                prefix = _TIME_PREFIX.match(action.text)
                start = prefix.end() if prefix else 0
                base = start
                for end in find_sentence_ends(action.text[base:]):
                    end += base
                    self._sentence_ends.append(end)
                    sentence = action.text[start:end].strip()
                    await self.push_frame(
                        TTSSpeakFrame(sentence, append_to_context=False)
                    )
                    start = end
            elif isinstance(action, Interrupt):
                self._sentence_ends.clear()
                await self.broadcast_interruption()
                if action.by_user:
                    for response_id in list(self._generating):
                        await self._session.cancel_response(response_id)

    def _now(self) -> float:
        return self.get_clock().get_time() / 1e9
