#!/usr/bin/env python
"""End-to-end check of edge GPU processing with two real processes.

    python tools/test_edge_e2e.py [--node-device auto|cpu] [--keep]

1. starts a Sentinel server (the "VPS") on a scratch database, with
   processing_placement=edge so it may not process cameras itself;
2. creates an edge_node account and starts tools/gpu_worker.py against it;
3. starts a synthetic camera on the server and verifies that it is assigned
   to the node, runs there, and that status, previews, tracks, findings and
   incidents all arrive back at the server;
4. stops the camera from the server and verifies the node lets it go.

Exit code 0 only if every check passes. Writes training/runs/edge_e2e.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
PY = str(BACKEND / ".venv" / "Scripts" / "python.exe") if os.name == "nt" \
    else str(BACKEND / ".venv" / "bin" / "python")
PORT = 8099
API = f"http://127.0.0.1:{PORT}"
ADMIN_PW = "e2e-admin-password"
NODE_PW = "e2e-node-password"


def wait_for(fn, timeout: float, what: str, interval: float = 1.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            last = fn()
            if last:
                return last
        except Exception as exc:          # server still starting, etc.
            last = exc
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {what} (last: {last!r})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-device", default="cpu", help="device_mode for the node")
    parser.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="sentinel_edge_e2e_"))
    server_dir, node_dir = tmp / "server", tmp / "node"
    (server_dir / "storage" / "models" / "anomaly").mkdir(parents=True)
    # One real promoted model so model sync is exercised end to end.
    src_model = ROOT / "storage" / "models" / "anomaly" / "anomaly_head.pt"
    if src_model.exists():
        shutil.copy2(src_model, server_dir / "storage" / "models" / "anomaly" / src_model.name)

    server_env = {**os.environ,
                  "SENTINEL_DATABASE_URL": f"sqlite:///{(server_dir / 'server.db').as_posix()}",
                  "SENTINEL_STORAGE_DIR": str(server_dir / "storage"),
                  "SENTINEL_CONFIG_DIR": str(server_dir / "config"),
                  "SENTINEL_VECTOR_DIR": str(server_dir / "vectors"),
                  "SENTINEL_VECTOR_BACKEND": "memory",
                  "SENTINEL_ADMIN_USERNAME": "admin",
                  "SENTINEL_ADMIN_PASSWORD": ADMIN_PW,
                  "SENTINEL_PROCESSING_PLACEMENT": "edge",
                  "SENTINEL_DEVICE_MODE": "cpu",          # the "VPS" has no GPU
                  "SENTINEL_N8N_ENABLED": "false"}
    node_env = {**os.environ, "SENTINEL_EDGE_PASSWORD": NODE_PW,
                "SENTINEL_DEVICE_MODE": args.node_device}

    report = {"checks": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    server = node = None
    server_log = open(tmp / "server.log", "w")
    node_log = open(tmp / "node.log", "w")

    def check(name: str, ok: bool, detail=None) -> None:
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
        if not ok:
            raise AssertionError(name)

    try:
        print("starting server (the 'VPS') ...")
        server = subprocess.Popen([PY, "-m", "uvicorn", "app.main:app", "--port", str(PORT),
                                   "--log-level", "warning"], cwd=BACKEND, env=server_env,
                                  stdout=server_log, stderr=subprocess.STDOUT)
        wait_for(lambda: httpx.get(f"{API}/api/system/health", timeout=3).status_code == 200,
                 180, "server health")
        c = httpx.Client(base_url=API, timeout=30)
        token = c.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PW}
                       ).json()["access_token"]
        c.headers["Authorization"] = f"Bearer {token}"

        r = c.post("/api/auth/users", json={"username": "edgebot", "password": NODE_PW,
                                            "role": "edge_node", "full_name": "E2E GPU node"})
        check("edge_node account created", r.status_code == 200, r.status_code)

        # Least privilege: the node account cannot read incidents.
        nt = httpx.post(f"{API}/api/auth/login",
                        json={"username": "edgebot", "password": NODE_PW}).json()["access_token"]
        denied = httpx.get(f"{API}/api/incidents", headers={"Authorization": f"Bearer {nt}"})
        check("edge_node role cannot read incidents", denied.status_code == 403,
              denied.status_code)

        sites = c.get("/api/spatial/sites").json()["items"]
        zones = c.get("/api/spatial/zones").json()["items"]
        restricted = next(z for z in zones if z["zone_type"] == "restricted")
        cam = c.post("/api/cameras", json={
            "name": "E2E Concourse", "source_type": "synthetic", "source_uri": "",
            "location": "Edge test hall", "site_id": sites[0]["id"], "zone_id": restricted["id"],
            "width": 960, "height": 540, "fps": 15, "meta": {"crowd": 9, "seed": 7},
        }).json()
        camera_id = cam["id"]

        r = c.post(f"/api/cameras/{camera_id}/start")
        check("placement=edge refuses to start with no node online", r.status_code == 409,
              r.json().get("detail"))

        print("starting edge node ...")
        node = subprocess.Popen([PY, str(ROOT / "tools" / "gpu_worker.py"), "--api", API,
                                 "--username", "edgebot", "--name", "E2E node",
                                 "--data-dir", str(node_dir), "--max-cameras", "2"],
                                cwd=ROOT, env=node_env, stdout=node_log, stderr=subprocess.STDOUT)
        nodes = wait_for(lambda: [n for n in c.get("/api/edge/nodes").json()["nodes"]
                                  if n["online"]], 240, "node registration", 2)
        check("node registered and online", True, f"{nodes[0]['name']} / {nodes[0]['gpu_name']}")

        synced = node_dir / "storage" / "models" / "anomaly" / "anomaly_head.pt"
        if src_model.exists():
            check("promoted model synced to node", synced.is_file() and
                  synced.stat().st_size == src_model.stat().st_size, synced.name)

        r = c.post(f"/api/cameras/{camera_id}/start")
        check("camera start accepted", r.status_code == 200, r.json().get("status"))
        placed = r.json().get("processing_node") or {}
        check("camera placed on the edge node", placed.get("node_id") == nodes[0]["node_id"],
              placed.get("name"))

        info = wait_for(lambda: (lambda d: d if (d.get("runtime") or {}).get("status") == "online"
                                 and d["runtime"].get("frame_index", 0) > 30 else None)(
            c.get(f"/api/cameras/{camera_id}").json()), 180, "camera online via node", 2)
        rt = info["runtime"]
        check("server sees node-processed frames", rt["frame_index"] > 30,
              f"frame {rt['frame_index']}, {rt.get('measured_fps')} fps, detector "
              f"{(rt.get('detector') or {}).get('backend')}")

        jpeg = wait_for(lambda: (lambda r: r.content if r.status_code == 200 else None)(
            c.get(f"/api/cameras/{camera_id}/snapshot")), 60, "preview frame", 2)
        check("preview JPEG relayed to server", jpeg[:2] == b"\xff\xd8", f"{len(jpeg)} bytes")

        tracks = wait_for(lambda: c.get(f"/api/cameras/{camera_id}/tracks").json()["tracks"],
                          60, "tracks", 2)
        check("tracks relayed to server", len(tracks) > 0, f"{len(tracks)} tracks")

        incidents = wait_for(lambda: (lambda d: d if d["total"] > 0 else None)(
            c.get("/api/incidents", params={"limit": 5}).json()), 240, "an incident", 3)
        first = incidents["items"][0]
        check("incident raised on node stored on server", first.get("camera_id") == camera_id,
              f"{incidents['total']} incident(s), first: {first.get('title')}")

        events = c.get("/api/system/events", params={"topic": "finding", "limit": 50}).json()
        items = events if isinstance(events, list) else events.get("events", events.get("items", []))
        from_edge = [e for e in items if str(e.get("source", "")).startswith("edge:")]
        check("findings republished from the node", len(from_edge) > 0, f"{len(from_edge)}")

        r = c.post(f"/api/cameras/{camera_id}/stop")
        check("camera stop accepted", r.status_code == 200)
        wait_for(lambda: all(camera_id not in n["cameras"]
                             for n in c.get("/api/edge/nodes").json()["nodes"]), 30,
                 "unassignment", 1)
        check("node released the camera", True)

        report["result"] = "pass"
        print("\nALL CHECKS PASSED")
        return 0
    except Exception as exc:
        report["result"] = "fail"
        report["error"] = repr(exc)
        print(f"\nFAILED: {exc!r}\n  logs in {tmp}")
        args.keep = True
        return 1
    finally:
        for proc in (node, server):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
        server_log.close()
        node_log.close()
        out = ROOT / "training" / "runs" / "edge_e2e.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        report["scratch_dir"] = str(tmp) if args.keep else None
        out.write_text(json.dumps(report, indent=2))
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
