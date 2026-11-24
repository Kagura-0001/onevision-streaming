from ov_streaming.session import OVSession
from ov_streaming.types import Memory, Turn


def test_snapshot_keeps_native_locations_and_pending_separate():
    session = OVSession.__new__(OVSession)
    session.session_id = "snapshot-test"
    session._timestamp_sec = 42.0
    session.max_visual_frames = 32
    session.compact_to_visual_frames = 32
    session._archive = [Memory("caption", 1.0, "older event")]
    session._turns = [Turn(41.0, memories=(Memory("caption", 39.0, "written"),))]
    session._pending = [Memory("caption", 40.0, "not written yet")]

    result = session.memory_snapshot()

    assert result["archive"] == [{"kind": "caption", "timestamp_sec": 1.0, "text": "older event"}]
    assert result["turns"][0]["timestamp_sec"] == 41.0
    assert result["turns"][0]["memories"][0]["timestamp_sec"] == 39.0
    assert result["pending"][0]["text"] == "not written yet"
    assert len(session._pending) == 1  # Reading never flushes or drains.
    result["turns"][0]["memories"][0]["text"] = "changed copy"
    assert session._turns[0].memories[0].text == "written"


def test_memory_snapshot_http_is_read_only_and_unknown_session_is_404():
    from fastapi.testclient import TestClient
    from ov_streaming.serve import build_app

    class Registry:
        def get(self, session_id):
            if session_id != "known":
                raise KeyError(session_id)
            return self

        def memory_snapshot(self):
            return {"archive": [], "turns": [], "pending": []}

    with TestClient(build_app(registry=Registry())) as client:
        assert client.get("/v1/ov/sessions/known/memory").json() == {
            "archive": [], "turns": [], "pending": []}
        assert client.get("/v1/ov/sessions/missing/memory").status_code == 404
