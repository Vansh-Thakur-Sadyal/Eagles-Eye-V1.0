#!/usr/bin/env python
"""Sentinel AI edge GPU node - process cameras on THIS machine's GPU.

Run it on any PC with an NVIDIA GPU. It connects out to a Sentinel server (for
example the dashboard on your VPS), takes the cameras the server assigns to
it, and runs the full pipeline here: detection, tracking, Re-ID, face
matching, crowd density, video-event and behaviour agents. The server receives
only what the agents conclude (incidents, findings, tracks, status) plus JPEG
previews while someone is watching - raw video never has to leave the site.

    python tools/gpu_worker.py --api https://sentinel.example.com \\
        --username gpu-node-1 --name "Control room RTX 5070 Ti"

The account should have the ``edge_node`` role (create it in Settings > Users).
The password is read from SENTINEL_EDGE_PASSWORD, or prompted for; passing
--password works too but leaves it visible in the process list.

Outbound HTTPS only: the node polls the server, so it works from behind NAT
and firewalls with no port forwarding. Stop with Ctrl-C; it hands its cameras
back on the way out. See docs/EDGE_GPU.md.
"""
from __future__ import annotations

import argparse
import getpass
import logging
import os
import signal
import socket
import sys
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Process Sentinel cameras on this machine's GPU")
    parser.add_argument("--api", required=True, help="Sentinel server base URL")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=None,
                        help="prefer SENTINEL_EDGE_PASSWORD or the interactive prompt")
    parser.add_argument("--name", default=None, help="display name for this node")
    parser.add_argument("--node-id", default=None, help="stable id (default: saved per data dir)")
    parser.add_argument("--max-cameras", type=int, default=4)
    parser.add_argument("--data-dir", default=str(Path.home() / ".sentinel_edge"),
                        help="node-local models, config and scratch database")
    parser.add_argument("--no-model-sync", action="store_true",
                        help="use the models already in --data-dir/storage/models")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    log = logging.getLogger("sentinel.edge")

    password = args.password or os.environ.get("SENTINEL_EDGE_PASSWORD")
    if not password:
        password = getpass.getpass(f"password for {args.username}@{args.api}: ")

    data = Path(args.data_dir).expanduser().resolve()
    (data / "storage" / "models").mkdir(parents=True, exist_ok=True)
    (data / "config").mkdir(parents=True, exist_ok=True)
    id_file = data / "node_id"
    node_id = args.node_id or (id_file.read_text().strip() if id_file.exists() else None)
    if not node_id:
        node_id = f"NODE_{uuid.uuid4().hex[:10].upper()}"
    id_file.write_text(node_id)

    # Point the pipeline at node-local state BEFORE any app module reads settings.
    os.environ.update({
        "SENTINEL_STORAGE_DIR": str(data / "storage"),
        "SENTINEL_CONFIG_DIR": str(data / "config"),
        "SENTINEL_VECTOR_DIR": str(data / "vectors"),
        "SENTINEL_DATABASE_URL": f"sqlite:///{(data / 'node.db').as_posix()}",
        "SENTINEL_VECTOR_BACKEND": "memory",
        "SENTINEL_PROCESSING_PLACEMENT": "server",   # this node IS where work runs
        "SENTINEL_N8N_ENABLED": "false",              # the server owns automation
        "SENTINEL_MAX_CONCURRENT_CAMERAS": str(args.max_cameras),
    })

    from app.edge.node import EdgeNode, probe_gpu

    gpu = probe_gpu()
    name = args.name or f"{socket.gethostname()} ({gpu['gpu_name']})"
    print("=" * 72)
    print("  Sentinel AI edge GPU node")
    print(f"  server    : {args.api}")
    print(f"  node      : {name}  [{node_id}]")
    print(f"  gpu       : {gpu['gpu_name']} ({gpu['total_memory_mb']} MB), device {gpu['device']}")
    print(f"  torch     : {gpu['torch'] or 'not installed'} (CUDA {gpu['cuda'] or 'n/a'})")
    print(f"  data dir  : {data}")
    print("=" * 72)
    if gpu["device"] == "cpu":
        print("\nWARNING: torch cannot see a CUDA GPU here. The node will work, on the CPU,")
        print("which defeats the point. Install a CUDA build of torch - see docs/INSTALL.md.\n")

    node = EdgeNode(args.api, args.username, password, node_id=node_id, name=name,
                    max_cameras=args.max_cameras)
    try:
        node.login()
    except Exception as exc:
        print(f"sign-in failed: {exc}")
        return 1

    if not args.no_model_sync:
        try:
            synced = node.sync_models(data / "storage" / "models")
            log.info("models: %d downloaded, %d already current", synced["fetched"],
                     synced["up_to_date"])
        except Exception as exc:
            log.warning("model sync failed (%s) - continuing with local models", exc)

    from app.db import init_db

    init_db()
    node.attach()          # builds the orchestrator: every agent loads its models here
    node.register()

    stop = threading.Event()

    def _stop(_signum, _frame):
        print("\nstopping - handing cameras back to the server ...")
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    try:
        signal.signal(signal.SIGTERM, _stop)
    except (AttributeError, ValueError):
        pass

    log.info("ready - waiting for camera assignments (Ctrl-C to stop)")
    node.run(stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
