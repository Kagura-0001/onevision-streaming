"""Both decode locations stay available through the original video entry point."""

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/stream_codec_video.py"


def load_script():
    spec = importlib.util.spec_from_file_location("stream_codec_video", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("explicit,strategy", [(False, "rdo"), (True, "pwindow")])
def test_original_client_path_is_preserved(monkeypatch, explicit, strategy):
    import ov_streaming
    import ov_streaming.codec as codec

    module = load_script()
    seen = {}

    class Session:
        async def __aenter__(self):
            self.closed = asyncio.Event()
            return self

        async def __aexit__(self, *args):
            pass

        async def events(self):
            await self.closed.wait()
            if False:
                yield

        async def close(self):
            self.closed.set()

    class Client:
        def __init__(self, url, **kwargs):
            seen["url"] = url

        def session(self, **kwargs):
            return Session()

    async def send(session, video, *, config):
        seen.update(video=video, config=config)
        return {"windows": 1}

    monkeypatch.setattr(ov_streaming, "OVClient", Client)
    monkeypatch.setattr(codec, "send_codec_video", send)
    argv = ["nba.mp4", "--strategy", strategy, "--max-seconds", "9", "--paced"]
    if explicit:
        argv += ["--codec-location", "client", "--decode-threads", "2"]
    asyncio.run(module.main(module.parse_args(argv)))
    assert seen["url"] == "http://127.0.0.1:8011"
    assert seen["video"] == "nba.mp4"
    assert seen["config"].strategy == strategy
    assert seen["config"].max_seconds == 9 and seen["config"].paced
    assert seen["config"].decode_threads == (2 if explicit else 1)


@pytest.mark.parametrize(
    "argv",
    [
        ["nba.mp4", "--codec-location", "server"],
        ["nba.mp4", "--codec-location", "server", "--fps", "nan"],
        [
            "nba.mp4",
            "--codec-location",
            "server",
            "--fps",
            "30",
            "--url",
            "http://model:8011",
        ],
        ["nba.mp4", "--url", "ws://gateway:8012/v1/codec/stream"],
        ["nba.mp4", "--codec-location", "server", "--fps", "30", "--max-seconds", "9"],
        [
            "nba.mp4",
            "--codec-location",
            "server",
            "--fps",
            "30",
            "--decode-threads",
            "2",
        ],
    ],
)
def test_wrong_endpoint_or_location_options_fail_before_loading_codec(argv):
    with pytest.raises(SystemExit):
        load_script().parse_args(argv)


@pytest.mark.parametrize("strategy,stdin_arg", [("rdo", []), ("pwindow", ["-"])])
def test_server_cli_uploads_and_receives_without_decode_dependencies(
    tmp_path, strategy, stdin_arg
):
    pytest.importorskip("websockets.asyncio.server")
    from websockets.asyncio.server import serve

    # Guard the subprocess, even when the test interpreter itself has all
    # optional packages installed. A thin device must not import any of them.
    (tmp_path / "sitecustomize.py").write_text("""
import sys
class NoDecode:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'PIL','numpy','codec_video_prep','codec_selector','ov_streaming'}:
   raise RuntimeError('decode dependency imported by upload-only client: '+fullname)
sys.meta_path.insert(0, NoDecode())
""")
    data = (b"\x47" + b"\x00" * 187) * 2
    received = bytearray()

    async def exercise():
        async def handler(ws):
            start = json.loads(await ws.recv())
            assert start == dict(
                type="start",
                format="mpegts",
                fps=30,
                width=1920,
                height=1080,
                strategy=strategy,
                caption=True,
            )
            await ws.send('{"type":"ready"}')
            while True:
                row = await ws.recv()
                if isinstance(row, bytes):
                    received.extend(row)
                    await ws.send(
                        json.dumps(
                            dict(type="upload.ack", bytes_received=len(received))
                        )
                    )
                else:
                    assert json.loads(row)["type"] == "end"
                    await ws.send(
                        '{"type":"response.output_text.done","text":"result"}'
                    )
                    await ws.send('{"type":"stream.closed"}')
                    return

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(SCRIPT),
                *stdin_arg,
                "--codec-location",
                "server",
                "--fps",
                "30",
                "--width",
                "1920",
                "--height",
                "1080",
                "--strategy",
                strategy,
                "--caption",
                "--url",
                f"ws://127.0.0.1:{port}/v1/codec/stream",
                env=dict(os.environ, PYTHONPATH=str(tmp_path)),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(data), 10)
                assert process.returncode == 0, stderr.decode()
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            events = [json.loads(line) for line in stdout.splitlines()]
            assert events[-1]["type"] == "stream.closed"
            assert any(x.get("text") == "result" for x in events)

    asyncio.run(exercise())
    assert received == data
