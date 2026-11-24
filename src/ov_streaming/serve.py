from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import os
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .renderers import Renderer, load_renderer
from .session import OVSession

_MAX_IMAGE_BYTES = 24 * 1024 * 1024
_IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}


class CreateSessionBody(BaseModel):
    enable: list[str] | None = None
    max_responses: int = Field(default=3, ge=1, le=32)
    max_response_tokens: int | None = Field(default=None, ge=1, strict=True)
    max_caption_tokens: int | None = Field(default=None, ge=1, strict=True)
    caption_prompt: str | None = Field(default=None, strict=True)
    max_visual_frames: int | None = Field(default=None, ge=1)
    compact_to_visual_frames: int | None = Field(default=None, ge=1)
    pin_first_question: bool = False
    answer_commit: Literal["on_done", "client"] = "on_done"
    verification: Literal["road_inspection"] | None = None
    verification_timeout: float = Field(default=10.0, gt=0, allow_inf_nan=False)


class CommitBody(BaseModel):
    text: str


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class MessagesBody(BaseModel):
    input_id: str | None = None
    messages: list[Message] = Field(min_length=1)


class SessionRegistry:
    def __init__(self, backend: Any, renderer: Renderer, max_num_seqs: int) -> None:
        self.backend = backend
        self.renderer = renderer
        self.max_num_seqs = max_num_seqs
        self.sessions: dict[str, OVSession] = {}
        self.closing: set[OVSession] = set()
        # When a client last used each session, and its open event streams.
        self.last_seen: dict[str, float] = {}
        self.watchers: Counter[str] = Counter()

    async def create(self, **options: Any) -> OVSession:
        session_id = uuid.uuid4().hex
        session = OVSession(self.backend, self.renderer, session_id, **options)
        # Paused sources hold engine slots, so a session reserves its source,
        # every branch, and the source a compaction opens next to it.
        # Closing sessions hold theirs until they finish.
        sessions = [*self.sessions.values(), *self.closing, session]
        slots = sum(
            item.max_responses + (2 if item.compacts else 1) for item in sessions
        )
        if slots > self.max_num_seqs:
            raise RuntimeError("the engine has no free request slots")
        self.sessions[session_id] = session
        self.last_seen[session_id] = time.monotonic()
        return session

    def get(self, session_id: str) -> OVSession:
        session = self.sessions[session_id]
        self.last_seen[session_id] = time.monotonic()
        return session

    async def reap(self, idle_secs: float) -> list[str]:
        """Abort sessions no client has used for idle_secs.

        A client that dies without closing its session would otherwise hold
        the session's engine slots until the server stops. A session with an
        open event stream is in use.
        """
        now = time.monotonic()
        idle = [
            session_id
            for session_id in self.sessions
            if not self.watchers[session_id]
            and now - self.last_seen[session_id] > idle_secs
        ]
        for session_id in idle:
            await self.remove(session_id, abort=True)
        return idle

    async def remove(self, session_id: str, *, abort: bool) -> bool:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        self.last_seen.pop(session_id, None)
        self.watchers.pop(session_id, None)
        self.closing.add(session)
        try:
            if abort:
                await session.abort()
            else:
                await session.close()
        finally:
            self.closing.discard(session)
        return True

    async def shutdown(self) -> None:
        sessions = list(self.sessions.values())
        self.sessions.clear()
        await asyncio.gather(
            *(session.abort() for session in sessions),
            return_exceptions=True,
        )


def build_app(
    registry: SessionRegistry | None = None,
    lifespan: Any = None,
    served_model_name: str = "ov-streaming",
) -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="OVStreaming", lifespan=lifespan)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "owned_by": "ov-streaming",
                }
            ],
        }

    def sessions() -> SessionRegistry:
        return registry if registry is not None else app.state.registry

    def session(session_id: str) -> OVSession:
        try:
            return sessions().get(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="unknown session") from error

    @app.get("/v1/ov/sessions/{session_id}/memory")
    async def memory_snapshot(session_id: str) -> dict[str, Any]:
        return session(session_id).memory_snapshot()

    @app.post("/v1/ov/sessions", status_code=201)
    async def create_session(body: CreateSessionBody) -> dict[str, Any]:
        try:
            created = await sessions().create(**body.model_dump())
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "session_id": created.session_id,
            "events_url": f"/v1/ov/sessions/{created.session_id}/events",
            "enable": None if created.enable is None else list(created.enable),
            "max_responses": created.max_responses,
            "max_response_tokens": created.max_response_tokens,
            **(
                {"max_caption_tokens": created.max_caption_tokens}
                if created.max_caption_tokens is not None else {}
            ),
            **({"caption_prompt": created.caption_prompt} if created.caption_prompt is not None else {}),
            "max_visual_frames": created.max_visual_frames,
            "compact_to_visual_frames": created.compact_to_visual_frames,
            "pin_first_question": created.pin_first_question,
            "answer_commit": created.answer_commit,
            "verification": created.verification,
            "verification_timeout": created.verification_timeout,
        }

    @app.post("/v1/ov/sessions/{session_id}/messages", status_code=202)
    async def send_messages(
        session_id: str,
        body: MessagesBody,
    ) -> dict[str, Any]:
        active_session = session(session_id)
        try:
            # Decoding images off the event loop keeps other sessions moving.
            messages = await asyncio.to_thread(
                lambda: tuple(
                    _decode_message(message.model_dump()) for message in body.messages
                )
            )
            input_id = await active_session.send_messages(
                input_id=body.input_id, messages=messages
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"session_id": session_id, "input_id": input_id, "status": "accepted"}

    @app.get("/v1/ov/sessions/{session_id}/events")
    async def stream_events(session_id: str) -> StreamingResponse:
        active_session = session(session_id)

        async def stream() -> AsyncIterator[str]:
            registry = sessions()
            registry.watchers[session_id] += 1
            try:
                async for event in active_session.events():
                    yield f"data: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"
            finally:
                if session_id in registry.sessions:
                    registry.watchers[session_id] -= 1
                    registry.last_seen[session_id] = time.monotonic()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/v1/ov/sessions/{session_id}/close")
    async def close_session(session_id: str) -> dict[str, Any]:
        if not await sessions().remove(session_id, abort=False):
            raise HTTPException(status_code=404, detail="unknown session")
        return {"session_id": session_id, "closed": True}

    @app.delete("/v1/ov/sessions/{session_id}")
    async def abort_session(session_id: str) -> dict[str, Any]:
        if not await sessions().remove(session_id, abort=True):
            raise HTTPException(status_code=404, detail="unknown session")
        return {"session_id": session_id, "aborted": True}

    @app.delete("/v1/ov/sessions/{session_id}/responses/{response_id}")
    async def cancel_response(session_id: str, response_id: str) -> dict[str, Any]:
        if not await session(session_id).cancel_response(response_id):
            raise HTTPException(status_code=404, detail="unknown active response")
        return {"response_id": response_id, "cancelled": True}

    @app.post("/v1/ov/sessions/{session_id}/responses/{response_id}/commit")
    async def commit_answer(
        session_id: str, response_id: str, body: CommitBody
    ) -> dict[str, Any]:
        try:
            found = session(session_id).commit_answer(response_id, body.text)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not found:
            raise HTTPException(status_code=404, detail="unknown uncommitted answer")
        return {"response_id": response_id, "committed": True}

    return app


def _decode_message(message: dict[str, Any]) -> dict[str, Any]:
    value = dict(message)
    content = value.get("content")
    if isinstance(content, list):
        if not content:
            raise ValueError("message content parts must not be empty")
        value["content"] = [_decode_content_part(part) for part in content]
    elif not content.strip():
        raise ValueError("message content must not be empty")
    return value


def _decode_content_part(part: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(part)
    part_type = value.get("type")
    if part_type == "text":
        text = value.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text content requires non-empty text")
        return value
    if part_type != "image_url":
        raise ValueError(f"unsupported content type: {part_type!r}")
    image_url = value.pop("image_url", None)
    if not isinstance(image_url, Mapping) or not isinstance(image_url.get("url"), str):
        raise ValueError("image_url content requires image_url.url")
    value["type"] = "image_pil"
    value["image_pil"] = _decode_image(image_url["url"])
    return value


def _decode_image(value: str) -> Image.Image:
    if not value.startswith("data:") or ";base64," not in value:
        raise ValueError("image_url supports only base64 data URLs")
    header, encoded = value.split(",", 1)
    media_type = header[5:].split(";", 1)[0]
    if media_type not in _IMAGE_MEDIA_TYPES:
        raise ValueError("image must be JPEG, PNG, or WebP")
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("invalid base64 image") from error
    if not data or len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("image must be between 1 byte and 24 MiB")
    try:
        with Image.open(io.BytesIO(data)) as source:
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("invalid image data") from error


def parse_engine_args(
    parser: argparse.ArgumentParser,
    engine_parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    renderer_class = load_renderer(parser.parse_args(argv).renderer)
    # Branching from a paused source is untested with async scheduling.
    engine_parser.set_defaults(async_scheduling=False, **renderer_class.engine_args)
    args = parser.parse_args(argv)
    # A dict flag on the command line adds its keys to the renderer's dict.
    for name, value in renderer_class.engine_args.items():
        if isinstance(value, dict):
            setattr(args, name, value | getattr(args, name))
    args.renderer_class = renderer_class
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(prog="ov-streaming")
    serve = parser.add_subparsers(dest="command", required=True).add_parser("serve")
    serve.add_argument("model_tag", nargs="?", help="the model to serve")
    serve.add_argument(
        "--renderer",
        required=True,
        help="univia, ov21, or module:Class of a Renderer subclass",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8011)
    serve.add_argument(
        "--session-idle-timeout",
        type=float,
        default=300,
        help="abort a session no client has used for this many seconds; 0 never",
    )
    AsyncEngineArgs.add_cli_args(serve)
    args = parse_engine_args(parser, serve, argv)
    # FlexibleArgumentParser moves --model after "serve" into model_tag.
    args.model = args.model_tag or args.model
    return args


def main() -> None:
    # Branching with suffix tokens needs the V1 model runner.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    from .vllm_plugin import register as register_vllm_plugin

    register_vllm_plugin()
    args = parse_args()

    import uvicorn
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.tokenizers import cached_tokenizer_from_config
    from vllm.v1.engine.async_llm import AsyncLLM

    from .backend import VLLMBackend

    engine_args = AsyncEngineArgs.from_cli_args(args)
    model_config = engine_args.create_model_config()
    renderer = args.renderer_class(
        cached_tokenizer_from_config(model_config), model_config
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        engine = AsyncLLM.from_engine_args(engine_args)
        registry = SessionRegistry(
            VLLMBackend(engine),
            renderer,
            engine.vllm_config.scheduler_config.max_num_seqs,
        )
        app.state.registry = registry

        async def reap_idle_sessions() -> None:
            while True:
                await asyncio.sleep(min(30, args.session_idle_timeout))
                await registry.reap(args.session_idle_timeout)

        reaper = None
        if args.session_idle_timeout > 0:
            reaper = asyncio.create_task(reap_idle_sessions())
        try:
            yield
        finally:
            if reaper is not None:
                reaper.cancel()
                await asyncio.gather(reaper, return_exceptions=True)
            await registry.shutdown()
            engine.shutdown()

    uvicorn.run(
        build_app(lifespan=lifespan, served_model_name=model_config.served_model_name),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
