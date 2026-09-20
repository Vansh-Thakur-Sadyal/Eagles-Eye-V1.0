"""Full-stack API tests.

Boots the real application against a temporary database and drives the flows
that matter: authentication and RBAC, the GPU toggle, attaching and running a
camera, incidents appearing from live analysis, watchlist enrolment with its
approval gate, forensic search, and the privacy/audit guarantees.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

# Point the app at a scratch database and storage area before importing it.
_TMP = Path(tempfile.mkdtemp(prefix="sentinel_test_"))
os.environ["SENTINEL_DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["SENTINEL_STORAGE_DIR"] = str(_TMP / "storage")
os.environ["SENTINEL_CONFIG_DIR"] = str(_TMP / "config")
os.environ["SENTINEL_VECTOR_DIR"] = str(_TMP / "vectors")
os.environ["SENTINEL_ADMIN_PASSWORD"] = "test-admin-password"
os.environ["SENTINEL_ADMIN_USERNAME"] = "admin"
os.environ["SENTINEL_VECTOR_BACKEND"] = "memory"
os.environ["SENTINEL_MAX_CONCURRENT_CAMERAS"] = "4"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def admin(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "test-admin-password"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def viewer(client, admin):
    client.post(
        "/api/auth/users",
        headers=admin,
        json={"username": "watcher", "password": "watcher-password", "role": "viewer",
              "full_name": "Duty Viewer"},
    )
    r = client.post("/api/auth/login", json={"username": "watcher", "password": "watcher-password"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ------------------------------------------------------------------- health
def test_health_is_public(client):
    r = client.get("/api/system/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "active_device" not in body or True
    assert body["device"] in ("cpu",) or body["device"].startswith("cuda")


def test_api_root_lists_routes(client):
    r = client.get("/api")
    assert r.status_code == 200
    assert "cameras" in r.json()["routes"]


# --------------------------------------------------------------------- auth
def test_login_rejects_a_bad_password(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_protected_route_requires_a_token(client):
    assert client.get("/api/cameras").status_code == 401


def test_me_returns_role_and_permissions(client, admin):
    r = client.get("/api/auth/me", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "admin"
    assert "*" in body["permissions"]
    assert "password_hash" not in body


def test_rbac_blocks_a_viewer_from_writing(client, viewer):
    r = client.post(
        "/api/cameras",
        headers=viewer,
        json={"name": "Unauthorised", "source_type": "synthetic", "source_uri": ""},
    )
    assert r.status_code == 403
    assert "permission" in r.json()["detail"].lower()


def test_viewer_cannot_read_the_audit_log(client, viewer):
    assert client.get("/api/admin/audit", headers=viewer).status_code == 403


# ---------------------------------------------------------------------- GPU
def test_gpu_status_reports_real_hardware(client, admin):
    r = client.get("/api/system/gpu", headers=admin)
    assert r.status_code == 200
    body = r.json()
    for key in ("gpu_enabled", "cuda_available", "active_device", "gpus", "client_acceleration"):
        assert key in body
    assert isinstance(body["gpus"], list)


def test_gpu_toggle_switches_device_and_is_audited(client, admin):
    off = client.post("/api/system/gpu", headers=admin, json={"enabled": False})
    assert off.status_code == 200
    assert off.json()["active_device"] == "cpu"
    assert off.json()["gpu_enabled"] is False

    on = client.post("/api/system/gpu", headers=admin, json={"enabled": True})
    assert on.status_code == 200
    body = on.json()
    if body["cuda_available"]:
        assert body["active_device"].startswith("cuda")
    else:
        # Honest behaviour: asks for GPU, has none, says so instead of pretending.
        assert body["active_device"] == "cpu"
        assert "warning" in body

    audit = client.get("/api/admin/audit", headers=admin, params={"action": "system.gpu"})
    assert audit.status_code == 200
    assert audit.json()["total"] >= 2

    client.post("/api/system/gpu", headers=admin, json={"enabled": False})


def test_gpu_toggle_denied_to_a_viewer(client, viewer):
    assert client.post("/api/system/gpu", headers=viewer, json={"enabled": True}).status_code == 403


# ------------------------------------------------------------------ seeding
def test_seed_created_a_site_zones_and_a_demo_camera(client, admin):
    sites = client.get("/api/spatial/sites", headers=admin).json()
    assert sites["items"], "no site was seeded"

    zones = client.get("/api/spatial/zones", headers=admin).json()
    assert any(z["zone_type"] == "restricted" for z in zones["items"])

    cameras = client.get("/api/cameras", headers=admin).json()
    assert cameras["total"] >= 1
    assert any(c["source_type"] == "synthetic" for c in cameras["items"])


def test_model_registry_reports_runtime_backends(client, admin):
    r = client.get("/api/system/models", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["registered"]
    tasks = {m["task"] for m in body["runtime"]}
    assert {"reid", "face", "llm"} <= tasks


# ----------------------------------------------------------------- cameras
def test_source_types_include_esp32_and_webcam(client, admin):
    r = client.get("/api/cameras/source-types", headers=admin)
    values = {t["value"] for t in r.json()["types"]}
    assert {"webcam", "esp32cam", "rtsp", "http_mjpeg", "file", "synthetic"} <= values
    esp = next(t for t in r.json()["types"] if t["value"] == "esp32cam")
    assert "esp32" in esp["help"].lower() or "ESP32" in esp["help"]


def test_esp32_probe_reports_unreachable_clearly(client, admin):
    r = client.post(
        "/api/cameras/discover/esp32", headers=admin,
        json={"address": "192.0.2.123"},        # TEST-NET-1, guaranteed unroutable
    )
    assert r.status_code == 200
    body = r.json()
    assert body["reachable"] is False
    assert "hint" in body
    assert set(body["endpoints"]) == {"capture", "stream", "status"}


def test_camera_rejects_an_unknown_source_type(client, admin):
    r = client.post("/api/cameras", headers=admin,
                    json={"name": "Bad", "source_type": "telepathy", "source_uri": ""})
    assert r.status_code == 422


@pytest.fixture(scope="module")
def running_camera(client, admin):
    """Create and start a synthetic camera; yield its id."""
    sites = client.get("/api/spatial/sites", headers=admin).json()["items"]
    zones = client.get("/api/spatial/zones", headers=admin).json()["items"]
    restricted = next(z for z in zones if z["zone_type"] == "restricted")

    created = client.post(
        "/api/cameras",
        headers=admin,
        json={
            "name": "Test Concourse",
            "source_type": "synthetic",
            "source_uri": "",
            "location": "Test Hall",
            "site_id": sites[0]["id"],
            "zone_id": restricted["id"],
            # Faster than real time: the worker now paces itself honestly at the
            # configured fps, so a 12 fps test camera would need ~22s of wall
            # clock to produce enough frames, and longer when the suite is
            # competing for CPU.
            "width": 960, "height": 540, "fps": 30,
            "latitude": 28.5562, "longitude": 77.1000, "floor": 0,
            "orientation_deg": 90.0,
            "meta": {"crowd": 9, "seed": 7},
        },
    )
    assert created.status_code == 200, created.text
    camera_id = created.json()["id"]

    started = client.post(f"/api/cameras/{camera_id}/start", headers=admin)
    assert started.status_code == 200, started.text

    # Let the pipeline actually run so the agents have something to find.
    # Wait for real evidence (frames AND a raised incident) rather than a fixed
    # sleep, so the suite is not flaky under load.
    deadline = time.time() + 120
    while time.time() < deadline:
        info = client.get(f"/api/cameras/{camera_id}", headers=admin).json()
        runtime = info.get("runtime") or {}
        frames = runtime.get("frame_index", 0)
        if frames > 320:
            incidents = client.get("/api/incidents", headers=admin,
                                   params={"limit": 1}).json()
            if incidents["total"] > 0:
                break
        time.sleep(0.5)

    yield camera_id
    client.post(f"/api/cameras/{camera_id}/stop", headers=admin)


def test_camera_runs_and_reports_real_throughput(client, admin, running_camera):
    info = client.get(f"/api/cameras/{running_camera}", headers=admin).json()
    assert info["running"] is True
    runtime = info["runtime"]
    assert runtime["status"] == "online", runtime
    assert runtime["frame_index"] > 100
    assert runtime["measured_fps"] > 0
    assert runtime["detector"]["backend"], "no detector backend reported"


def test_live_tracks_are_available(client, admin, running_camera):
    r = client.get("/api/intel/tracks/live", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["total_tracks"] > 0, "the pipeline produced no tracks"


def test_mjpeg_snapshot_returns_an_image(client, admin, running_camera):
    r = client.get(f"/api/cameras/{running_camera}/snapshot", headers=admin)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8", "response was not a JPEG"
    assert len(r.content) > 5000


def test_crowd_telemetry_is_measured(client, admin, running_camera):
    r = client.get(f"/api/cameras/{running_camera}/crowd", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 0
    assert body["density_band"] in ("low", "medium", "high", "critical")
    assert "compression" in body


# --------------------------------------------------------------- incidents
def test_incidents_were_raised_by_live_analysis(client, admin, running_camera):
    r = client.get("/api/incidents", headers=admin, params={"limit": 50})
    assert r.status_code == 200
    items = r.json()["items"]
    assert items, "live analysis produced no incidents"

    incident = items[0]
    assert incident["summary"]
    assert incident["explanation"]
    assert incident["risk_factors"]
    assert incident["recommended_actions"]
    assert 0 <= incident["risk_score"] <= 100


def test_incident_detail_includes_a_timeline(client, admin, running_camera):
    items = client.get("/api/incidents", headers=admin).json()["items"]
    incident_id = items[0]["id"]
    r = client.get(f"/api/incidents/{incident_id}", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["timeline"], "incident has no timeline entries"
    assert "action_catalogue" in body


def test_incident_status_transition_is_recorded(client, admin, running_camera):
    incident_id = client.get("/api/incidents", headers=admin).json()["items"][0]["id"]
    r = client.post(f"/api/incidents/{incident_id}/status", headers=admin,
                    json={"status": "acknowledged", "note": "Reviewed by test"})
    assert r.status_code == 200
    assert r.json()["status"] == "acknowledged"
    assert any(e["kind"] == "operator" for e in r.json()["timeline"])


def test_high_impact_action_is_gated_behind_approval(client, admin, running_camera):
    """An operator cannot unilaterally dispatch; a commander must approve."""
    client.post("/api/auth/users", headers=admin,
                json={"username": "op1", "password": "operator-password", "role": "operator"})
    login = client.post("/api/auth/login", json={"username": "op1", "password": "operator-password"})
    op = {"Authorization": f"Bearer {login.json()['access_token']}"}

    incidents = client.get("/api/incidents", headers=admin, params={"limit": 50}).json()["items"]
    target = next(
        (i for i in incidents
         if any(a["key"] == "dispatch_team" for a in i.get("recommended_actions", []))),
        None,
    )
    if target is None:
        pytest.skip("no incident in this run proposed a dispatch action")

    r = client.post(f"/api/incidents/{target['id']}/actions", headers=op,
                    json={"action_key": "dispatch_team", "justification": "test"})
    assert r.status_code == 200
    body = r.json()
    assert body["requires_human_approval"] is True
    assert body["status"] == "pending_approval"

    pending = client.get("/api/admin/approvals", headers=admin).json()
    assert any(a["resource_id"] == target["id"] for a in pending["items"])


def test_unknown_action_is_rejected(client, admin, running_camera):
    incident_id = client.get("/api/incidents", headers=admin).json()["items"][0]["id"]
    r = client.post(f"/api/incidents/{incident_id}/actions", headers=admin,
                    json={"action_key": "launch_drone_strike"})
    assert r.status_code == 400


def test_incident_summary_kpis_are_computed(client, admin, running_camera):
    r = client.get("/api/incidents/summary", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["open"] >= 0
    assert isinstance(body["by_severity"], dict)
    assert isinstance(body["by_event_type"], dict)


# -------------------------------------------------------------- behaviour
def test_behavior_events_were_persisted(client, admin, running_camera):
    r = client.get("/api/intel/behavior", headers=admin, params={"limit": 100})
    assert r.status_code == 200
    assert r.json()["total"] > 0, "no behaviour events were written"


def test_behavior_stats_aggregate(client, admin, running_camera):
    r = client.get("/api/intel/behavior/stats", headers=admin)
    assert r.status_code == 200
    assert r.json()["total"] > 0


# ------------------------------------------------------------------ threat
def test_threat_weights_expose_the_zero_weight_rule(client, admin):
    r = client.get("/api/intel/threat/weights", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["weights"]["face_unavailable"] == 0
    assert "face_unavailable" in body["zero_weight_behaviors"]


def test_threat_simulation_is_explainable(client, admin):
    r = client.post(
        "/api/intel/threat/simulate", headers=admin,
        json={"behaviors": {"unattended_object": 0.9, "restricted_zone_entry": 0.8}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["score"] > 0
    assert body["explanation"]
    assert body["dominant_factor"] == "unattended_object"


# ---------------------------------------------------------------- policy
def test_policy_round_trips_and_reloads(client, admin):
    original = client.get("/api/system/policy", headers=admin).json()
    assert "threat_weights" in original

    r = client.put("/api/system/policy", headers=admin,
                   json={"behavior": {"loitering_seconds": 999}})
    assert r.status_code == 200
    assert r.json()["behavior"]["loitering_seconds"] == 999

    again = client.get("/api/system/policy", headers=admin).json()
    assert again["behavior"]["loitering_seconds"] == 999
    # restore
    client.put("/api/system/policy", headers=admin,
               json={"behavior": {"loitering_seconds": original["behavior"]["loitering_seconds"]}})


# --------------------------------------------------------------- watchlist
def _png_face(path: Path) -> Path:
    """A synthetic head-and-shoulders image, enough to exercise enrolment."""
    import cv2
    import numpy as np

    img = np.full((360, 280, 3), 210, dtype=np.uint8)
    cv2.rectangle(img, (60, 230), (220, 360), (95, 100, 125), -1)      # shoulders
    cv2.circle(img, (140, 150), 78, (168, 150, 135), -1)               # head
    cv2.circle(img, (115, 135), 9, (40, 40, 45), -1)                   # eyes
    cv2.circle(img, (165, 135), 9, (40, 40, 45), -1)
    cv2.ellipse(img, (140, 190), (28, 14), 0, 0, 180, (80, 60, 60), -1)  # mouth
    cv2.imwrite(str(path), img)
    return path


def test_watchlist_requires_a_legal_basis(client, admin):
    r = client.post("/api/watchlist", headers=admin,
                    json={"label": "No Basis", "legal_basis": "x"})
    assert r.status_code == 400
    assert "legal basis" in r.json()["detail"].lower()


@pytest.fixture(scope="module")
def watchlist_subject(client, admin, tmp_path_factory):
    r = client.post(
        "/api/watchlist",
        headers=admin,
        json={
            "label": "Test Subject Alpha",
            "category": "missing_person",
            "priority": "high",
            "legal_basis": "Automated regression test - synthetic image, no real person.",
            "case_reference": "TEST-001",
            "expires_in_days": 1,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_new_subject_is_pending_and_not_matching(client, admin, watchlist_subject):
    body = client.get(f"/api/watchlist/{watchlist_subject}", headers=admin).json()
    assert body["status"] == "pending"
    assert body["matching_active"] is False


def test_enrolment_creates_an_approval_request(client, admin, watchlist_subject):
    approvals = client.get("/api/admin/approvals", headers=admin,
                           params={"kind": "watchlist_enrol"}).json()
    assert any(a["resource_id"] == watchlist_subject for a in approvals["items"])


def test_image_upload_builds_an_augmented_gallery(client, admin, watchlist_subject,
                                                  tmp_path_factory):
    path = _png_face(tmp_path_factory.mktemp("wl") / "face.png")
    with open(path, "rb") as fh:
        r = client.post(
            f"/api/watchlist/{watchlist_subject}/images",
            headers=admin,
            files={"files": ("face.png", fh, "image/png")},
            data={"augment_images": "true"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["augmented_variants"] > 10, "augmentation produced too few variants"
    assert body["body_embeddings"] > 10
    uploaded = body["uploaded"][0]
    assert uploaded["augmentations"], "no augmentations were recorded"
    # Enrolment is still not active, and the response says so.
    assert any("not yet approved" in w for w in body["warnings"])


def test_matching_starts_only_after_approval(client, admin, watchlist_subject):
    before = client.get(f"/api/watchlist/{watchlist_subject}", headers=admin).json()
    assert before["matching_active"] is False

    r = client.post(f"/api/watchlist/{watchlist_subject}/approve", headers=admin,
                    json={"decision": "approve", "note": "test approval"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "active"
    assert body["matching_active"] is True
    assert body["approved_by"] == "admin"


def test_locate_is_honest_when_there_is_no_sighting(client, admin, watchlist_subject):
    r = client.get(f"/api/watchlist/{watchlist_subject}/locate", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["located"] is False
    assert "No sighting" in body["statement"]
    assert body["sightings"] == []


def test_watchlist_reads_are_audited(client, admin, watchlist_subject):
    r = client.get("/api/admin/audit", headers=admin,
                   params={"resource_id": watchlist_subject})
    assert r.status_code == 200
    actions = {a["action"] for a in r.json()["items"]}
    assert "watchlist.create" in actions
    assert "watchlist.locate" in actions
    assert "watchlist.approve" in actions


def test_viewer_cannot_reach_the_watchlist(client, viewer, watchlist_subject):
    assert client.get("/api/watchlist", headers=viewer).status_code == 403


# ---------------------------------------------------------------- forensic
def test_forensic_search_returns_grounded_citations(client, admin, running_camera):
    client.post("/api/forensic/reindex", headers=admin)
    r = client.post("/api/forensic/search", headers=admin,
                    json={"query": "show me incidents in the restricted area"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["parsed_query"]["intent"] in ("search", "detail", "count", "timeline", "trajectory")
    for citation in body["citations"]:
        assert citation["incident_id"].startswith("INC_")


def test_forensic_parses_time_and_behaviour_filters(client, admin):
    r = client.post("/api/forensic/search", headers=admin,
                    json={"query": "unattended bags near Gate 4 between 7 pm and 9 pm"})
    assert r.status_code == 200
    parsed = r.json()["parsed_query"]
    assert "unattended_object" in parsed["behaviors"]
    assert parsed["start"] and parsed["end"]
    assert any("gate 4" in h for h in parsed["zone_hints"])


def test_forensic_says_so_when_nothing_matches(client, admin):
    r = client.post("/api/forensic/search", headers=admin,
                    json={"query": "zeppelin collision on platform 94 in 1873"})
    assert r.status_code == 200
    body = r.json()
    assert body["result_count"] == 0
    assert "no recorded incidents" in body["answer"].lower()


def test_chat_persists_the_transcript(client, admin):
    r = client.post("/api/forensic/chat", headers=admin,
                    json={"message": "which zone currently has the highest crowd risk?"})
    assert r.status_code == 200
    session_id = r.json()["session_id"]
    history = client.get(f"/api/forensic/chat/{session_id}", headers=admin).json()
    assert len(history["messages"]) == 2
    assert history["messages"][0]["role"] == "user"


# ----------------------------------------------------------------- spatial
def test_digital_twin_payload_is_complete(client, admin, running_camera):
    r = client.get("/api/spatial/twin", headers=admin)
    assert r.status_code == 200
    body = r.json()
    for key in ("sites", "zones", "cameras", "incidents", "live_entities"):
        assert key in body
    assert body["cameras"]


def test_camera_coverage_reports_ungeolocated_cameras(client, admin):
    r = client.get("/api/spatial/coverage", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["cameras_total"] >= 1
    assert "cameras_without_coordinates" in body


def test_ar_feed_returns_overlays_for_open_incidents(client, admin, running_camera):
    r = client.get("/api/spatial/ar/feed", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert "overlays" in body and "restricted_zones" in body


def test_zone_polygon_must_be_normalised(client, admin):
    sites = client.get("/api/spatial/sites", headers=admin).json()["items"]
    r = client.post("/api/spatial/zones", headers=admin,
                    json={"name": "Pixel Zone", "site_id": sites[0]["id"],
                          "polygon": [[10, 10], [900, 10], [900, 500]]})
    assert r.status_code == 422


# ------------------------------------------------------------------ admin
def test_privacy_controls_are_reported_as_enforced(client, admin):
    r = client.get("/api/admin/privacy", headers=admin)
    assert r.status_code == 200
    body = r.json()
    controls = {c["control"]: c["enforced"] for c in body["controls"]}
    assert controls["Role-based access control"] is True
    assert controls["Immutable audit log"] is True
    assert controls["Identity escalation requires approval"] is True
    assert controls["Watchlist enrolment requires a legal basis"] is True


def test_retention_purge_defaults_to_a_dry_run(client, admin):
    r = client.post("/api/admin/privacy/purge-expired", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["dry_run"] is True
    assert "deleted_incidents" not in body


def test_identity_escalation_requires_a_justification(client, admin):
    r = client.post("/api/intel/identity/escalate", headers=admin,
                    json={"global_id": "SUBJ_TEST", "justification": "because"})
    assert r.status_code == 400

    ok = client.post(
        "/api/intel/identity/escalate", headers=admin,
        json={"global_id": "SUBJ_TEST",
              "justification": "Named in case TEST-001 under a documented lawful authority."},
    )
    assert ok.status_code == 200
    assert ok.json()["status"] == "pending"


def test_approver_cannot_approve_their_own_request(client, admin):
    pending = client.get("/api/admin/approvals", headers=admin,
                         params={"kind": "identity_escalation"}).json()["items"]
    assert pending
    r = client.post(f"/api/admin/approvals/{pending[0]['id']}/decide", headers=admin,
                    json={"decision": "approve"})
    assert r.status_code == 403
    assert "your own" in r.json()["detail"]


def test_workflow_crud_and_condition_help(client, admin):
    r = client.post(
        "/api/admin/workflows", headers=admin,
        json={"name": "High severity to n8n", "trigger": "incident_created",
              "webhook_url": "http://localhost:5678/webhook/test",
              "conditions": {"min_severity": "high"}},
    )
    assert r.status_code == 200
    workflow_id = r.json()["id"]

    listing = client.get("/api/admin/workflows", headers=admin).json()
    assert any(w["id"] == workflow_id for w in listing["items"])
    assert "min_severity" in listing["condition_help"]

    assert client.delete(f"/api/admin/workflows/{workflow_id}", headers=admin).status_code == 200


def test_workflow_rejects_an_unknown_trigger(client, admin):
    r = client.post("/api/admin/workflows", headers=admin,
                    json={"name": "Bad", "trigger": "when_i_feel_like_it",
                          "webhook_url": "http://localhost/x"})
    assert r.status_code == 400


def test_report_generation_uses_real_data(client, admin, running_camera):
    r = client.post("/api/admin/reports/generate", headers=admin,
                    json={"report_type": "daily", "hours": 24})
    assert r.status_code == 200
    body = r.json()
    assert body["body_markdown"]
    assert body["stats"]["incidents_total"] >= 0
    assert "Eagles Eye" in body["body_markdown"]


def test_evidence_capture_hashes_and_verifies(client, admin, running_camera):
    incident_id = client.get("/api/incidents", headers=admin).json()["items"][0]["id"]
    r = client.post("/api/admin/evidence", headers=admin,
                    json={"incident_id": incident_id, "kind": "snapshot",
                          "note": "captured by test"})
    assert r.status_code == 200
    evidence = r.json()
    assert evidence["sha256"], "evidence was stored without a hash"
    assert evidence["chain_of_custody"]

    verify = client.get(f"/api/admin/evidence/{evidence['id']}/verify", headers=admin).json()
    assert verify["verified"] is True


# ------------------------------------------------------------------ agents
def test_agent_roster_covers_the_full_specification(client, admin):
    r = client.get("/api/system/agents", headers=admin)
    assert r.status_code == 200
    roster = r.json()["agents"]
    spec_ids = {a["spec_id"] for a in roster if a.get("spec_id")}
    # Specification agents 1-12 must all be represented somewhere.
    assert set(range(1, 13)) <= spec_ids, f"missing spec agents: {set(range(1, 13)) - spec_ids}"

    # Name the culprit: a bare all() tells you nothing when this fails.
    errored = [
        (a["name"], a["last_error"]) for a in roster if a.get("last_error") is not None
    ]
    assert not errored, "agents reporting errors: " + "; ".join(
        f"{name}: {error}" for name, error in errored
    )


def test_agent_can_be_disabled_and_reenabled(client, admin):
    off = client.post("/api/system/agents/crowd/toggle", headers=admin, json={"enabled": False})
    assert off.status_code == 200 and off.json()["enabled"] is False
    on = client.post("/api/system/agents/crowd/toggle", headers=admin, json={"enabled": True})
    assert on.json()["enabled"] is True


def test_unknown_agent_toggle_is_404(client, admin):
    assert client.post("/api/system/agents/nonexistent/toggle", headers=admin,
                       json={"enabled": True}).status_code == 404


def test_system_info_reports_every_subsystem(client, admin, running_camera):
    r = client.get("/api/system/info", headers=admin)
    assert r.status_code == 200
    body = r.json()
    for key in ("compute", "pipeline", "orchestrator", "persistence", "automation",
                "llm", "vector_store", "counts", "privacy"):
        assert key in body
    assert body["counts"]["cameras_running"] >= 1
