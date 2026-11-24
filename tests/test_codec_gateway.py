import asyncio
import json
import sys
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ov_streaming.codec_gateway import MAX_CHUNK, build_gateway

TS = (b"\x47" + b"\x00" * 187) * 2
START = {"type": "start", "fps": 30, "width": 1920, "height": 1080}
WINDOW = dict(
    type="window",
    timestamp_sec=1.0,
    positions_sha256="positions",
    jpeg_sha256=["i", "p"],
    messages=[{"role": "user", "content": "window"}],
)
SCRIPT = """
import sys, json
sys.stdin.buffer.read(376)
print(%r, flush=True)
while sys.stdin.buffer.read(1024): pass
print(json.dumps({'type':'done','stats':{'windows':1}}), flush=True)
""" % json.dumps(WINDOW)


class Session:
    def __init__(self):
        self.session_id = "test"
        self.messages = []
        self.queue = asyncio.Queue()
        self.aborted = self.closed = False
        self._http = None

    async def __aenter__(self):
        return self

    async def send_messages(self, *, messages, input_id):
        self.messages.append(messages)
        await self.queue.put({"type": "input.done", "input_id": input_id})
        return input_id

    async def events(self):
        while (row := await self.queue.get()) is not None:
            yield SimpleNamespace(**row, to_dict=lambda: row)

    async def close(self):
        self.closed = True
        await self.queue.put({"type": "session.closed"})
        await self.queue.put(None)

    async def abort(self):
        self.aborted = True


def setup(script=SCRIPT, **options):
    sessions, processes, configs = [], [], []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def session(self, **kwargs):
            configs.append(kwargs)
            item = Session()
            sessions.append(item)
            return item

    async def worker(start):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(process)
        return process

    return (
        TestClient(
            build_gateway(client_factory=Client, worker_factory=worker, **options)
        ),
        sessions,
        processes,
        configs,
    )


def receive_until(ws, kind):
    rows = []
    while True:
        row = ws.receive_json()
        rows.append(row)
        if row["type"] == kind:
            return rows
        assert row["type"] != "error", row


def test_live_window_before_eof_and_results_on_same_connection():
    client, sessions, processes, configs = setup()
    with client, client.websocket_connect("/v1/codec/stream") as ws:
        ws.send_json(START)
        assert ws.receive_json()["type"] == "ready"
        # Arbitrary byte boundaries, not one TS packet per WebSocket message.
        ws.send_bytes(TS[:1])
        assert ws.receive_json()["bytes_received"] == 1
        ws.send_bytes(TS[1:])
        rows = receive_until(ws, "codec.input")
        assert rows[-1]["timestamp_sec"] == 1
        assert not sessions[0].closed  # produced before upload end
        ws.send_json({"type": "question", "text": "What happened?"})
        receive_until(ws, "question.accepted")
        ws.send_json({"type": "end"})
        rows = receive_until(ws, "stream.closed")
        assert any(r["type"] == "codec.done" for r in rows)
        assert any(r["type"] == "session.closed" for r in rows)
    assert sessions[0].closed and processes[0].returncode == 0
    assert sessions[0].messages[0] == WINDOW["messages"]
    assert sessions[0].messages[1][0]["content"] == "What happened?"
    assert configs[0]["enable"] == ["caption"]


@pytest.mark.parametrize(
    "start",
    [
        dict(START, fps=0),
        dict(START, fps=float("nan")),
        dict(START, strategy="other"),
        dict(START, format="h264"),
        dict(START, url="/etc/passwd"),
        {key: value for key, value in START.items() if key != "width"},
    ],
)
def test_invalid_start_does_not_spawn_decoder(start):
    client, sessions, processes, _ = setup()
    with client, client.websocket_connect("/v1/codec/stream") as ws:
        ws.send_text(json.dumps(start))
        assert ws.receive_json()["type"] == "error"
    assert not sessions and not processes


@pytest.mark.parametrize(
    "payload",
    [b"x" * 376, b"x" * (MAX_CHUNK + 1), b""],
    ids=["wrong-format", "oversize", "empty"],
)
def test_invalid_video_aborts_owned_resources(payload):
    client, sessions, processes, _ = setup()
    with client, client.websocket_connect("/v1/codec/stream") as ws:
        ws.send_json(START)
        ws.receive_json()
        ws.send_bytes(payload)
        assert ws.receive_json()["type"] == "error"
    assert sessions[0].aborted
    assert processes[0].returncode is not None


def test_disconnect_terminates_blocked_probe_and_releases_capacity():
    client, sessions, processes, _ = setup(
        "import time; time.sleep(600)", max_streams=1
    )
    with client:
        with client.websocket_connect("/v1/codec/stream") as ws:
            ws.send_json(START)
            ws.receive_json()
            with client.websocket_connect("/v1/codec/stream") as other:
                assert other.receive_json()["type"] == "error"
        deadline = time.monotonic() + 3
        while (
            client.get("/health").json()["active_streams"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert client.get("/health").json()["active_streams"] == 0
    assert processes[0].returncode is not None and sessions[0].aborted


def test_decoder_error_is_reported_and_reaped():
    client, sessions, processes, _ = setup('raise RuntimeError("missing PTS")')
    with client, client.websocket_connect("/v1/codec/stream") as ws:
        ws.send_json(START)
        ws.receive_json()
        row = ws.receive_json()
        assert row["type"] == "error" and "missing PTS" in row["error"]
    assert sessions[0].aborted and processes[0].returncode != 0


def test_idle_upload_times_out_and_reaps_decoder():
    client, sessions, processes, _ = setup(
        "import time; time.sleep(600)", idle_timeout=0.1
    )
    with client, client.websocket_connect("/v1/codec/stream") as ws:
        ws.send_json(START)
        ws.receive_json()
        assert ws.receive_json()["type"] == "error"
    assert sessions[0].aborted and processes[0].returncode is not None


def test_slow_model_control_step_times_out_and_aborts(monkeypatch):
    async def enqueue_only(self, *, messages, input_id):
        self.messages.append(messages)
        return input_id

    monkeypatch.setattr(Session, "send_messages", enqueue_only)
    client, sessions, processes, _ = setup(processing_timeout=0.15)
    with client, client.websocket_connect("/v1/codec/stream") as ws:
        ws.send_json(START)
        ws.receive_json()
        ws.send_bytes(TS)
        receive_until(ws, "upload.ack")
        assert ws.receive_json()["type"] == "error"
    assert sessions[0].aborted and processes[0].returncode is not None


def test_failed_graceful_close_explicitly_deletes_upstream(monkeypatch):
    deleted = []

    async def fail_close(self):
        raise RuntimeError("close failed")

    class HTTP:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def delete(self, path):
            deleted.append(path)

    monkeypatch.setattr(Session, "close", fail_close)
    monkeypatch.setattr("ov_streaming.codec_gateway.httpx.AsyncClient", HTTP)
    client, sessions, processes, _ = setup()
    with client, client.websocket_connect("/v1/codec/stream") as ws:
        ws.send_json(START)
        ws.receive_json()
        ws.send_bytes(TS)
        receive_until(ws, "codec.input")
        ws.send_json({"type": "end"})
        while (row := ws.receive_json())["type"] != "error":
            pass
        assert "close failed" in row["error"]
    assert deleted == ["/v1/ov/sessions/test"]
    assert sessions[0].aborted
