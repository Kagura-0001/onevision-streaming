"""Server-side H.264/MPEG-TS WebSocket ingress; no vLLM dependency."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import uuid
from typing import Literal

import anyio
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from .client import OVClient
from .codec import CodecConfig
from .codec_process import codec_worker_command, codec_worker_env

MAX_CHUNK = 64 * 1024


class StreamStart(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    type: Literal["start"]
    format: Literal["mpegts"] = "mpegts"
    fps: float = Field(gt=0, le=240)
    # Native size: the decoder cannot probe a live pipe without consuming it.
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    strategy: Literal["rdo", "pwindow"] = "rdo"
    caption: bool = True


async def spawn_decoder(start):
    command = codec_worker_command(
        sys.executable,
        CodecConfig(strategy=start.strategy),
        live_fps=start.fps,
        source_size=(start.width, start.height),
    )
    return await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=codec_worker_env(),
        limit=8 * 1024 * 1024,
    )


def build_gateway(
    upstream="http://127.0.0.1:8011",
    *,
    max_streams=2,
    idle_timeout=30.0,
    processing_timeout=120.0,
    client_factory=OVClient,
    worker_factory=spawn_decoder,
):
    app = FastAPI(title="OVStreaming codec gateway")
    active = set()

    @app.get("/health")
    async def health():
        return {"active_streams": len(active), "max_streams": max_streams}

    @app.websocket("/v1/codec/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        token = object()
        if len(active) >= max_streams:
            await ws.send_json({"type": "error", "error": "codec capacity reached"})
            await ws.close(code=1013)
            return
        active.add(token)
        worker = session = None
        tasks = []
        ended = closing = closed_cleanly = False
        send_lock = asyncio.Lock()
        stderr_tail = bytearray()
        pending_inputs = {}

        async def send(value):
            async with send_lock:
                await asyncio.wait_for(ws.send_json(value), processing_timeout)

        async def submit(messages):
            # HTTP 202 only enqueues input. Wait for the model's control step
            # so a slow GPU also backpressures the camera through both pipes.
            input_id = uuid.uuid4().hex
            done = asyncio.get_running_loop().create_future()
            pending_inputs[input_id] = done
            try:
                await session.send_messages(messages=messages, input_id=input_id)
                await asyncio.wait_for(done, processing_timeout)
                return input_id
            finally:
                pending_inputs.pop(input_id, None)

        try:
            start = StreamStart.model_validate_json(
                await asyncio.wait_for(ws.receive_text(), idle_timeout)
            )
            session = client_factory(upstream, timeout=processing_timeout).session(
                enable=["caption"] if start.caption else None,
                max_visual_frames=96,
                compact_to_visual_frames=64,
            )
            await session.__aenter__()
            worker = await worker_factory(start)
            await send(
                {
                    "type": "ready",
                    "session_id": session.session_id,
                    "format": "mpegts",
                    "max_chunk_bytes": MAX_CHUNK,
                }
            )

            async def receive():
                nonlocal ended
                received = 0
                prefix = bytearray()
                validated = False
                while True:
                    message = await asyncio.wait_for(ws.receive(), idle_timeout)
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect(message.get("code", 1000))
                    data = message.get("bytes")
                    if data is not None:
                        if not data or len(data) > MAX_CHUNK:
                            raise ValueError(
                                "binary chunks must contain 1..65536 bytes"
                            )
                        received += len(data)
                        if not validated:
                            prefix.extend(data)
                            if len(prefix) >= 376:
                                if prefix[0] != 0x47 or prefix[188] != 0x47:
                                    raise ValueError(
                                        "expected H.264 in 188-byte MPEG-TS packets"
                                    )
                                data = bytes(prefix)
                                prefix.clear()
                                validated = True
                            else:
                                data = b""
                        worker.stdin.write(data)
                        await asyncio.wait_for(worker.stdin.drain(), processing_timeout)
                        await send({"type": "upload.ack", "bytes_received": received})
                        continue
                    command = json.loads(message.get("text") or "")
                    if command.get("type") == "end":
                        if not validated:
                            raise ValueError("empty or incomplete MPEG-TS stream")
                        ended = True
                        worker.stdin.close()
                        return
                    if command.get("type") == "question":
                        text = command.get("text")
                        if (
                            not isinstance(text, str)
                            or not text.strip()
                            or len(text) > 8192
                        ):
                            raise ValueError("question requires 1..8192 characters")
                        input_id = await submit([{"role": "user", "content": text}])
                        await send({"type": "question.accepted", "input_id": input_id})
                        continue
                    raise ValueError("expected binary video, question, or end")

            async def decoded():
                nonlocal closing, closed_cleanly
                done = False
                while True:
                    line = await asyncio.wait_for(
                        worker.stdout.readline(), processing_timeout
                    )
                    if not line:
                        break
                    row = json.loads(line)
                    if row["type"] == "window":
                        input_id = await submit(row.pop("messages"))
                        await send(dict(row, type="codec.input", input_id=input_id))
                    elif row["type"] == "done":
                        done = True
                        await send({"type": "codec.done", "stats": row["stats"]})
                    else:
                        raise RuntimeError("invalid decoder response")
                code = await asyncio.wait_for(worker.wait(), processing_timeout)
                if code or not done or not ended:
                    raise RuntimeError(
                        "decoder failed or ended before end: "
                        + stderr_tail.decode(errors="replace")[-2000:]
                    )
                closing = True
                await asyncio.wait_for(session.close(), processing_timeout)
                closed_cleanly = True

            async def events():
                async for event in session.events():
                    if event.type == "input.done":
                        done = pending_inputs.get(event.input_id)
                        if done is not None and not done.done():
                            done.set_result(None)
                    await send(event.to_dict())
                if not closing:
                    raise RuntimeError("model event stream ended unexpectedly")

            async def stderr():
                while chunk := await worker.stderr.read(4096):
                    stderr_tail.extend(chunk)
                    del stderr_tail[:-8192]

            tasks = [
                asyncio.create_task(f()) for f in (receive, decoded, events, stderr)
            ]
            pending = set(tasks)
            while pending:
                completed, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in completed:
                    task.result()
            await send({"type": "stream.closed"})
            await ws.close()
        except WebSocketDisconnect:
            pass
        except Exception as error:
            with contextlib.suppress(Exception):
                await send({"type": "error", "error": str(error)})
                await ws.close(code=1011)
        finally:
            with anyio.CancelScope(shield=True):
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if worker is not None:
                    # Terminate even when native probing is blocked; a Python
                    # cancellation flag cannot interrupt avformat_find_stream_info.
                    if worker.returncode is None:
                        with contextlib.suppress(ProcessLookupError):
                            worker.terminate()
                        try:
                            await asyncio.wait_for(worker.wait(), 5)
                        except asyncio.TimeoutError:
                            with contextlib.suppress(ProcessLookupError):
                                worker.kill()
                            await worker.wait()
                if session is not None:
                    # ClientSession marks itself closed before issuing close.
                    # If that request is cancelled/fails, abort() is a no-op;
                    # explicitly delete the upstream session to avoid a leak.
                    if closing and not closed_cleanly and session.session_id:
                        with contextlib.suppress(Exception):
                            async with httpx.AsyncClient(
                                base_url=upstream, timeout=10
                            ) as http:
                                await http.delete(
                                    f"/v1/ov/sessions/{session.session_id}"
                                )
                    with contextlib.suppress(Exception):
                        await session.abort()
                active.discard(token)

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8011")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--max-streams", type=int, default=2)
    args = parser.parse_args()
    if args.max_streams < 1:
        parser.error("--max-streams must be positive")
    uvicorn.run(
        build_gateway(args.upstream, max_streams=args.max_streams),
        host=args.host,
        port=args.port,
        ws_max_size=MAX_CHUNK,
        ws_max_queue=4,
    )


if __name__ == "__main__":
    main()
