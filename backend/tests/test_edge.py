"""Edge GPU processing: placement, assignment, ingest and its safety rules.

The full two-process path (a real server plus a real tools/gpu_worker.py) is
exercised by tools/test_edge_e2e.py; these tests pin the server-side rules.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

# Same scratch setup as test_api.py; setdefault so whichever module the suite
# imports first decides, and the app is configured exactly once.
_TMP = Path(tempfile.mkdtemp(prefix="sentinel_edge_test_"))
os.environ.setdefault("SENTINEL_DATABASE_URL", f"sqlite:///{(_TMP / 'test.db').as_posix()}")
os.environ.setdefault("SENTINEL_STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("SENTINEL_CONFIG_DIR", str(_TMP / "config"))
os.environ.setdefault("SENTINEL_VECTOR_DIR", str(_TMP / "vectors"))
os.environ.setdefault("SENTINEL_ADMIN_PASSWORD", "test-admin-password")
os.environ.setdefault("SENTINEL_ADMIN_USERNAME", "admin")
os.environ.setdefault("SENTINEL_VECTOR_BACKEND", "memory")

from fastapi.testclient import TestClient  # noqa: E402

from app.edge.hub import get_hub  # noqa: E402
from app.main import app  # noqa: E402
from app.pipeline.runner import CameraRuntime  # noqa: E402

JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def admin(client):
    r = client.post("/api/auth/login", json={"username": os.environ["SENTINEL_ADMIN_USERNAME"],
                                             "password": os.environ["SENTINEL_ADMIN_PASSWORD"]})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def node_auth(client, admin):
    client.post("/api/auth/users", headers=admin, json={
        "username": "gpu-node", "password": "gpu-node-password", "role": "edge_node",
        "full_name": "Test GPU node"})
    r = client.post("/api/auth/login", json={"username": "gpu-node", "password": "gpu-node-password"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _register(client, auth, node_id="NODE_TEST_A", device="cuda:0", max_cameras=2):
    r = client.post("/api/edge/nodes/register", headers=auth, json={
        "node_id": node_id, "name": f"Test {node_id}", "device": device,
        "gpu_name": "Test GPU" if device.startswith("cuda") else "none",
        "total_memory_mb": 16000, "max_cameras": max_cameras})
    assert r.status_code == 200, r.text
    return node_id


def _camera(client, admin, name, meta=None, source_type="synthetic"):
    r = client.post("/api/cameras", headers=admin, json={
        "name": name, "source_type": source_type, "source_uri": "0" if source_type == "webcam" else "",
        "width": 640, "height": 360, "fps": 10, "meta": meta or {}})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _rt(camera_id="CAM_X", source_type="rtsp", **options):
    return CameraRuntime(camera_id=camera_id, name="x", source_type=source_type,
                         source_uri="rtsp://example/stream", options=options)


# ------------------------------------------------------------------ roles
def test_edge_node_role_is_least_privilege(client, node_auth):
    assert client.get("/api/incidents", headers=node_auth).status_code == 403
    assert client.get("/api/cameras", headers=node_auth).status_code in (200, 403)
    assert client.get("/api/watchlist", headers=node_auth).status_code == 403
    assert client.post("/api/cameras", headers=node_auth, json={"name": "x"}).status_code == 403


def test_other_roles_cannot_act_as_a_node(client, admin):
    client.post("/api/auth/users", headers=admin, json={
        "username": "just-a-viewer", "password": "viewer-password-1", "role": "viewer"})
    token = client.post("/api/auth/login", json={"username": "just-a-viewer",
                                                  "password": "viewer-password-1"}).json()["access_token"]
    r = client.post("/api/edge/nodes/register", headers={"Authorization": f"Bearer {token}"},
                    json={"node_id": "NODE_EVIL", "device": "cuda:0"})
    assert r.status_code == 403


# -------------------------------------------------------------- placement
def test_placement_rules(client, node_auth):
    hub = get_hub()
    # server: never leaves this machine, even with nodes online
    _register(client, node_auth, "NODE_PLACE", max_cameras=1)
    assert hub.choose_node(_rt(processing_node="server")) is None
    # auto: a GPU node when one has capacity
    chosen = hub.choose_node(_rt(processing_node="auto"))
    assert chosen is not None and chosen.has_gpu
    # auto never moves a webcam index: "0" means the webcam on the machine running it
    assert hub.choose_node(_rt(source_type="webcam", processing_node="auto")) is None
    # pinned to an unknown node is an error, not a silent fallback
    with pytest.raises(RuntimeError):
        hub.choose_node(_rt(processing_node="NODE_DOES_NOT_EXIST"))
    hub.drop("NODE_PLACE")


def test_edge_placement_refuses_without_a_node(client, admin):
    hub = get_hub()
    for n in list(hub.status()["nodes"]):
        hub.drop(n["node_id"])
    with pytest.raises(RuntimeError, match="no GPU edge node"):
        hub.choose_node(_rt(processing_node="edge"))
    cam = _camera(client, admin, "Needs a GPU", meta={"processing_node": "edge"})
    r = client.post(f"/api/cameras/{cam}/start", headers=admin)
    assert r.status_code == 409, r.text


def test_cpu_only_nodes_are_not_chosen_automatically(client, node_auth):
    hub = get_hub()
    _register(client, node_auth, "NODE_CPU", device="cpu")
    assert all(n.node_id != "NODE_CPU" for n in [hub.choose_node(_rt(processing_node="auto"))] if n)
    hub.drop("NODE_CPU")


# ----------------------------------------------------- assignment + ingest
@pytest.fixture(scope="module")
def assigned(client, admin, node_auth):
    node_id = _register(client, node_auth, "NODE_TEST_A")
    cam = _camera(client, admin, "Edge Camera", meta={"processing_node": node_id})
    r = client.post(f"/api/cameras/{cam}/start", headers=admin)
    assert r.status_code == 200, r.text
    assert r.json()["processing_node"]["node_id"] == node_id
    yield node_id, cam
    client.post(f"/api/cameras/{cam}/stop", headers=admin)


def test_node_receives_its_assignment(client, node_auth, assigned):
    node_id, cam = assigned
    r = client.get(f"/api/edge/nodes/{node_id}/assignments", headers=node_auth)
    assert r.status_code == 200
    data = r.json()
    spec = next(c for c in data["cameras"] if c["runtime"]["camera_id"] == cam)
    assert spec["runtime"]["source_type"] == "synthetic"
    assert spec["config_hash"] and data["policy_version"] and data["watchlist_version"]


def test_camera_reports_connecting_until_the_node_speaks(client, admin, assigned):
    _node, cam = assigned
    rt = client.get(f"/api/cameras/{cam}", headers=admin).json()["runtime"]
    assert rt["status"] in ("connecting", "online")
    assert rt["processing_node"]["online"] is True


def test_ingest_updates_status_tracks_findings_and_incidents(client, admin, node_auth, assigned):
    node_id, cam = assigned
    incident_id = f"INC_EDGE_{int(time.time() * 1000)}"
    batch = {
        "statuses": [{"camera_id": cam, "name": "Edge Camera", "status": "online",
                      "measured_fps": 9.8, "frame_index": 420, "track_count": 2,
                      "inference_ms": 11.0, "device": "cuda:0"}],
        "results": [{"camera_id": cam, "frame_index": 420, "measured_fps": 9.8,
                     "timestamp": "2026-09-20T00:00:00+00:00",
                     "tracks": [{"track_id": 7, "class_name": "person", "bbox": [1, 2, 30, 90],
                                 "speed": 1.0, "age_frames": 30, "duration_seconds": 3.0}],
                     "crowd": {"count": 2, "density": 0.1, "density_band": "low"}}],
        "events": [{"topic": "finding", "severity": "medium",
                    "payload": {"camera_id": cam, "behavior": "loitering", "confidence": 0.8,
                                "severity": "medium", "explanation": "edge test"}}],
        "incidents": [{"id": incident_id, "camera_id": cam, "title": "Edge-raised incident",
                       "event_type": "loitering", "severity": "medium", "risk_score": 55,
                       "is_new": True, "started_at": "2026-09-20T00:00:00+00:00",
                       "last_update_at": "2026-09-20T00:00:01+00:00", "timeline": []}],
    }
    r = client.post(f"/api/edge/nodes/{node_id}/ingest", headers=node_auth, json=batch)
    assert r.status_code == 200, r.text
    assert r.json() == {"accepted": 4, "rejected": 0}

    rt = client.get(f"/api/cameras/{cam}", headers=admin).json()["runtime"]
    assert rt["status"] == "online" and rt["frame_index"] == 420
    tracks = client.get(f"/api/cameras/{cam}/tracks", headers=admin).json()["tracks"]
    assert tracks and tracks[0]["track_id"] == 7
    assert client.get(f"/api/cameras/{cam}/crowd", headers=admin).json()["count"] == 2

    stored = client.get(f"/api/incidents/{incident_id}", headers=admin)
    assert stored.status_code == 200, stored.text
    assert stored.json()["title"] == "Edge-raised incident"

    events = client.get("/api/system/events", headers=admin,
                        params={"topic": "finding", "limit": 20}).json()["items"]
    assert any(e["source"] == f"edge:{node_id}" for e in events)


def test_a_node_cannot_report_for_cameras_it_does_not_run(client, admin, node_auth, assigned):
    node_id, _cam = assigned
    other = _camera(client, admin, "Not Yours")
    batch = {"statuses": [{"camera_id": other, "status": "online"}],
             "events": [{"topic": "finding", "payload": {"camera_id": other, "behavior": "x"}}],
             "incidents": [{"id": "INC_FORGED", "camera_id": other, "title": "forged"}]}
    r = client.post(f"/api/edge/nodes/{node_id}/ingest", headers=node_auth, json=batch)
    assert r.json() == {"accepted": 0, "rejected": 3}
    assert client.get("/api/incidents/INC_FORGED", headers=admin).status_code == 404


def test_preview_frames(client, admin, node_auth, assigned):
    node_id, cam = assigned
    url = f"/api/edge/nodes/{node_id}/frames/{cam}"
    assert client.put(url, headers={**node_auth, "Content-Type": "image/jpeg"},
                      content=b"not a jpeg").status_code == 415
    ok = client.put(url, headers={**node_auth, "Content-Type": "image/jpeg"}, content=JPEG)
    assert ok.status_code == 200
    snap = client.get(f"/api/cameras/{cam}/snapshot", headers=admin)
    assert snap.status_code == 200 and snap.content == JPEG
    other = _camera(client, admin, "Frame Target Not Assigned")
    r = client.put(f"/api/edge/nodes/{node_id}/frames/{other}",
                   headers={**node_auth, "Content-Type": "image/jpeg"}, content=JPEG)
    assert r.status_code == 409


def test_silent_node_is_reported_degraded_not_healthy(client, admin, assigned):
    node_id, cam = assigned
    node = get_hub().node(node_id)
    saved = node.last_seen
    node.last_seen = time.time() - 3600
    try:
        rt = client.get(f"/api/cameras/{cam}", headers=admin).json()["runtime"]
        assert rt["status"] == "degraded"
        assert "has not reported" in rt["status_detail"]
    finally:
        node.last_seen = saved


def test_unknown_node_is_told_to_register(client, node_auth):
    r = client.get("/api/edge/nodes/NODE_NEVER_SEEN/assignments", headers=node_auth)
    assert r.status_code == 404


def test_node_capacity_is_respected(client, node_auth):
    hub = get_hub()
    _register(client, node_auth, "NODE_TINY", max_cameras=1)
    first = hub.choose_node(_rt("CAM_T1", processing_node="NODE_TINY"))
    hub.assign(_rt("CAM_T1", processing_node="NODE_TINY"), first)
    with pytest.raises(RuntimeError, match="camera limit"):
        hub.choose_node(_rt("CAM_T2", processing_node="NODE_TINY"))
    hub.unassign("CAM_T1")
    hub.drop("NODE_TINY")


def test_stop_releases_the_camera_from_the_node(client, admin, node_auth):
    node_id = _register(client, node_auth, "NODE_TEST_B")
    cam = _camera(client, admin, "Short-lived", meta={"processing_node": node_id})
    assert client.post(f"/api/cameras/{cam}/start", headers=admin).status_code == 200
    client.post(f"/api/cameras/{cam}/stop", headers=admin)
    data = client.get(f"/api/edge/nodes/{node_id}/assignments", headers=node_auth).json()
    assert all(c["runtime"]["camera_id"] != cam for c in data["cameras"])


# ------------------------------------------------------------- sync feeds
def test_policy_and_watchlist_sync(client, node_auth, assigned):
    node_id, _ = assigned
    pol = client.get(f"/api/edge/nodes/{node_id}/policy", headers=node_auth).json()
    assert "threat_weights" in pol["policy"] and pol["version"]
    wl = client.get(f"/api/edge/nodes/{node_id}/watchlist", headers=node_auth).json()
    assert isinstance(wl["subjects"], list) and wl["version"]


def test_model_download_cannot_escape_the_models_folder(client, node_auth):
    for path in ("../sentinel.db", "..\\..\\backend\\.env", "/etc/passwd"):
        r = client.get("/api/edge/models/file", headers=node_auth, params={"path": path})
        assert r.status_code == 404, path
    assert "files" in client.get("/api/edge/models", headers=node_auth).json()


# ---------------------------------------------------------- node-side unit
def test_node_outbox_restores_unsent_batches_in_order():
    from app.edge.node import Outbox

    box = Outbox()
    for i in range(3):
        box.put("events", {"i": i})
    batch = box.take()
    assert [e["i"] for e in batch["events"]] == [0, 1, 2]
    box.put("events", {"i": 3})
    box.restore(batch)                               # the send failed
    assert [e["i"] for e in box.take()["events"]] == [0, 1, 2, 3]


def test_locateanything_output_parser():
    from app.vision.detector import parse_locateanything

    text = ("<ref>person</ref><box><292><145><658><864></box><ref>backpack</ref>"
            "<box><335><245><577><542></box><ref>car</ref><box><277><217><306><241></box>"
            "<box><656><223><998><745></box><box><900><10><100><20></box><|im_end|>")
    out = list(parse_locateanything(text, 480, 640))
    assert [label for label, _ in out] == ["person", "backpack", "car", "car"]   # degenerate dropped
    label, (x1, y1, x2, y2) = out[0]
    assert abs(x1 - 140.16) < 0.01 and abs(y2 - 552.96) < 0.01


def test_locateanything_degenerate_box_chain_is_dropped():
    """Real runaway output: one true person box, then a chain sliding along a row."""
    from app.vision.detector import parse_locateanything

    chain = "".join(f"<box><{x}><214><{x + 11}><222></box>" for x in range(292, 600, 11))
    text = "<ref>person</ref><box><292><145><658><864></box>" + chain
    out = list(parse_locateanything(text, 480, 640))
    assert len(out) == 1, f"kept {len(out)} boxes"
    # a few genuinely side-by-side objects must survive
    shelf = "".join(f"<box><{x}><300><{x + 50}><400></box>" for x in (100, 150, 200))
    assert len(list(parse_locateanything("<ref>bottle</ref>" + shelf, 1000, 1000))) == 3
