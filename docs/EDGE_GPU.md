# Using everyone's own GPU (edge nodes)

A dashboard hosted on a VPS usually has no GPU, and a browser cannot run CUDA.
So Sentinel sends the heavy work to **edge nodes**: any PC with an NVIDIA GPU
that runs `tools/gpu_worker.py`. Each node runs the complete, unchanged
pipeline on its own GPU (detection, tracking, Re-ID, face matching, crowd
density, video-event and behaviour agents). The VPS only receives what the
agents concluded:

```
 camera(s) ──► [ operator PC + GPU ]  ──(HTTPS, outbound only)──►  [ VPS dashboard ]
               tools/gpu_worker.py        incidents, findings,        stores, alerts,
               full pipeline on CUDA      tracks, status, previews    shows everyone
```

- **Video stays on site.** Only results, plus JPEG previews while someone is
  watching (up to 8 fps, one every 5 s otherwise), go to the VPS.
- **No port forwarding.** Nodes connect *out* and poll, so they work behind
  NAT and office firewalls.
- **Everything else keeps working.** On the VPS, a camera running on a node
  looks like a local camera: live view, snapshots, tracks, crowd telemetry,
  incidents, evidence capture, n8n automation and forensic search.

## Set up

On the VPS, in `backend/.env`:

```
SENTINEL_PROCESSING_PLACEMENT=edge   # never process video on the VPS itself
```

| value | meaning |
|---|---|
| `auto` | use an online GPU node if there is one, otherwise process here (default) |
| `server` | always process on the machine serving the API |
| `edge` | only on edge nodes - a camera refuses to start if no GPU node is online |

Then create an account for each node in **Settings → Users** with the role
`edge_node`. That role can take camera assignments, report results and
mirror the policy and watchlist gallery. It **cannot** read incidents, cases
or users, or change anything, so a leaked node password exposes very little.

On each GPU machine (a Sentinel checkout with `backend/.venv` installed as in
[INSTALL.md](INSTALL.md), including the CUDA build of torch):

```
set SENTINEL_EDGE_PASSWORD=...            (PowerShell: $env:SENTINEL_EDGE_PASSWORD="...")
python tools/gpu_worker.py --api https://your-vps.example.com --username gpu-node-1 --name "Control room 5070 Ti"
```

On start the node:
1. downloads the **promoted models** from the VPS (checksum-verified), so every
   GPU runs exactly the models that passed evaluation;
2. registers its GPU and capacity (`--max-cameras`, default 4);
3. mirrors the **policy** and the **watchlist gallery** (embeddings only, never
   photographs), and re-syncs whenever either changes on the VPS;
4. waits for camera assignments.

Stop it with Ctrl-C. Its cameras show as *degraded* on the dashboard (never
silently healthy) until it returns or they are moved.

## Choosing where a camera runs

Each camera has a **Processed on** setting (camera detail panel, or when
attaching one): default, `auto`, `server`, `edge`, or a specific node.

- **Pin webcams and ESP32-CAMs to the node they are physically attached to /
  on the same LAN as.** A webcam index like `0` means "the webcam on whatever
  machine runs this camera", so `auto` never moves a webcam off the server.
  Pin it to the node explicitly.
- RTSP/HTTP cameras reachable from several nodes can use `auto`/`edge`: the
  least-loaded GPU node takes them.
- Changing it takes effect the next time the camera starts.

**System Health → Edge GPU nodes** lists every node with its GPU, load,
last contact and how much it has reported.

## Security

- Nodes only report on cameras assigned to them. Anything else is rejected
  (a node cannot inject incidents for another site's camera).
- Preview uploads must be JPEG, at most 3 MB.
- Model downloads are confined to `storage/models/`.
- Use HTTPS for the VPS. Node tokens are ordinary JWTs and are refreshed
  automatically.

## How it was verified

- `backend/tests/test_edge.py`: 18 tests covering placement rules, role
  limits, assignment, ingest, rejection of other nodes' cameras, preview
  validation, stale-node reporting, capacity, model-path confinement and the
  offline outbox.
- `tools/test_edge_e2e.py`: two real processes, a server with
  `placement=edge` and a separate `gpu_worker.py`. It checks that the server
  refuses to start a camera with no node; that the node registers (it saw the
  RTX 5070 Ti), syncs a promoted model and takes the camera; that frames,
  previews, tracks, findings and an incident all reach the server; and that
  stopping the camera releases it. All 14 checks pass
  (`training/runs/edge_e2e.json`).

## What it does not do

- **A browser tab does not become a GPU worker.** Browsers can't run CUDA,
  and WebGPU can't run this PyTorch pipeline. A person contributes their GPU
  by running the node program, not by opening the dashboard.
- One camera runs on one node at a time. Nodes do not split a single stream.
- If a node goes offline, its cameras are not moved automatically. They show
  as degraded and an operator restarts them (moving them to another node if
  their placement allows it).
