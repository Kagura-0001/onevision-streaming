"""The example uploader must exit even when the camera pipe is idle."""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("websockets.asyncio.server")
from websockets.asyncio.server import serve


def test_gateway_error_cancels_idle_camera_read(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "upload_h264", Path(__file__).parents[1] / "scripts/upload_h264.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def exercise():
        async def handler(ws):
            await ws.recv()
            await ws.send(json.dumps({"type": "ready"}))
            await asyncio.sleep(0.05)
            await ws.send(json.dumps({"type": "error", "error": "model disconnected"}))

        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "rb", buffering=0) as camera:
            monkeypatch.setattr(module.sys, "stdin", SimpleNamespace(buffer=camera))
            try:
                async with serve(handler, "127.0.0.1", 0) as server:
                    port = server.sockets[0].getsockname()[1]
                    args = SimpleNamespace(
                        replay=None,
                        url=f"ws://127.0.0.1:{port}",
                        fps=30,
                        width=1920,
                        height=1080,
                        strategy="rdo",
                        caption=True,
                        timeout=2,
                    )
                    with pytest.raises(RuntimeError, match="model disconnected"):
                        await asyncio.wait_for(module.main(args), 5)
            finally:
                os.close(write_fd)

    asyncio.run(exercise())
