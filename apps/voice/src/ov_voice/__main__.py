"""Talk to an OV session from a browser.

python -m ov_voice -t webrtc --ov-url http://127.0.0.1:8011

Pipecat's development runner serves http://localhost:7860 and runs bot() for
each browser that connects. /live/ is this app's page, which shows what OV
decides and says; /client/ is Pipecat's prebuilt page.

Several OV servers, each on its own GPU, serve more people at once:
--ov-url takes a comma-separated list, and each browser gets the server with
the fewest sessions. /start answers 503 once every server holds
--max-sessions-per-ov sessions or all of them hold --max-sessions.
"""

import argparse
import contextlib
from collections.abc import Iterator
from pathlib import Path

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ov_streaming.client import OVClient
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.runner.run import app, main
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.base_transport import TransportParams, VideoInSourceParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from ov_voice.pipecat import OVProcessor, SpeechService, TranscriptionService

# VAD silence that ends a speech segment, per turn detection. Smart Turn wants
# short segments and decides itself whether a pause ends the turn; with
# silence alone, a segment is a question, so a breath must not end it.
VAD_STOP_SECS = {"smart-turn": 0.2, "silence": 0.8}


def turn_processors(turn_detection: str) -> list:
    """The processors between ASR and OV that decide when a turn is over."""
    if turn_detection == "silence":
        return []
    # OVProcessor interrupts the answer itself, on the VAD, so the turn
    # processor only reports when turns start and stop.
    strategies = UserTurnStrategies(
        start=[
            VADUserTurnStartStrategy(enable_interruptions=False),
            TranscriptionUserTurnStartStrategy(enable_interruptions=False),
        ],
        stop=[
            TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())
        ],
    )
    return [UserTurnProcessor(user_turn_strategies=strategies)]


class OVPool:
    """The OV servers, and how many sessions each one holds."""

    def __init__(
        self, urls: list[str], per_server: int | None, total: int | None
    ) -> None:
        self.sessions = dict.fromkeys(urls, 0)
        self.per_server = per_server
        self.total = total

    def full(self) -> bool:
        if self.total is not None and sum(self.sessions.values()) >= self.total:
            return True
        # The least loaded server is full only when every server is.
        return self.per_server is not None and (
            min(self.sessions.values()) >= self.per_server
        )

    @contextlib.contextmanager
    def session(self) -> Iterator[str | None]:
        """The least loaded server, or None when every one is full."""
        if self.full():
            yield None
            return
        url = min(self.sessions, key=self.sessions.__getitem__)
        self.sessions[url] += 1
        try:
            yield url
        finally:
            self.sessions[url] -= 1


app.mount("/live", StaticFiles(directory=Path(__file__).with_name("web"), html=True))


@app.middleware("http")
async def turn_away_when_full(request: Request, call_next):
    if request.url.path == "/start" and app.state.pool.full():
        return JSONResponse({"detail": "OV 已满员，请稍后再试"}, status_code=503)
    return await call_next(request)


@app.get("/model")
async def model() -> dict:
    """The model OV serves, for the page's header."""
    url = next(iter(app.state.pool.sessions))
    async with httpx.AsyncClient(timeout=5) as client:
        reply = await client.get(f"{url}/v1/models")
    return {"model": reply.json()["data"][0]["id"]}


@app.get("/load")
async def load() -> dict:
    """Sessions per OV server."""
    pool = app.state.pool
    return {
        "max_sessions": pool.total,
        "max_sessions_per_ov": pool.per_server,
        "sessions": pool.sessions,
    }


async def bot(runner_args: SmallWebRTCRunnerArguments) -> None:
    # Two browsers can pass /start at once for the last free place.
    with app.state.pool.session() as ov_url:
        if ov_url is not None:
            await run_bot(runner_args, ov_url)


async def run_bot(runner_args: SmallWebRTCRunnerArguments, ov_url: str) -> None:
    args = runner_args.cli_args
    stop_secs = VAD_STOP_SECS[args.turn_detection]
    transport = SmallWebRTCTransport(
        runner_args.webrtc_connection,
        TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=True,
            video_in_sources={"camera": VideoInSourceParams()},
        ),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=stop_secs))
            ),
            TranscriptionService(
                api_key="EMPTY",
                base_url=args.asr_url,
                settings=TranscriptionService.Settings(model=args.asr_model),
            ),
            *turn_processors(args.turn_detection),
            OVProcessor(
                OVClient(ov_url),
                frame_interval=args.frame_interval,
                max_visual_frames=args.max_visual_frames,
                compact_to_visual_frames=args.compact_to_visual_frames,
                user_turns=args.turn_detection != "silence",
                # Captions are always on; the page decides whether to show them.
                enable=[] if args.no_captions else ["caption"],
            ),
            SpeechService(
                api_key="EMPTY",
                base_url=args.tts_url,
                settings=SpeechService.Settings(
                    model=args.tts_model, voice=args.tts_voice
                ),
            ),
            transport.output(),
        ]
    )
    # OV may watch for a long time without anyone speaking.
    worker = PipelineWorker(pipeline, idle_timeout_secs=None)
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ov-url",
        default="http://127.0.0.1:8011",
        help="one OV server, or several separated by commas",
    )
    parser.add_argument(
        "--max-sessions-per-ov",
        type=int,
        help="sessions one OV server takes; unlimited if not set",
    )
    parser.add_argument(
        "--max-sessions", type=int, help="sessions in all; unlimited if not set"
    )
    parser.add_argument("--asr-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--tts-url", default="http://127.0.0.1:8091/v1")
    parser.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--tts-voice", default="vivian")
    parser.add_argument("--frame-interval", type=float, default=1.0)
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help="do not let OV caption the camera (captions are never spoken)",
    )
    parser.add_argument("--max-visual-frames", type=int)
    parser.add_argument("--compact-to-visual-frames", type=int)
    parser.add_argument(
        "--turn-detection",
        choices=sorted(VAD_STOP_SECS),
        default="smart-turn",
        help="smart-turn: Smart Turn v3 decides when a turn ends; "
        "silence: a VAD pause ends it (the behaviour before Smart Turn)",
    )
    known = parser.parse_known_args()[0]
    urls = [url.strip().rstrip("/") for url in known.ov_url.split(",") if url.strip()]
    app.state.pool = OVPool(urls, known.max_sessions_per_ov, known.max_sessions)
    main(parser)
