# Architecture

```
CCTV / IoT
    │
    ▼
┌──────────────────────────────┐
│ CAPTURE          capture.py  │  webcam · ESP32-CAM · RTSP · MJPEG · file · synthetic
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ PERCEPTION                   │  Agent 1  detector.py   YOLO11 / LocateAnything-3B / motion fallback
│                              │  Agent 2  tracker.py    ByteTrack, two-stage association
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ SPATIO-TEMPORAL WORLD MODEL  │  TrackState: trajectory, velocity, zone, embedding, face visibility
└──────────────┬───────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR            orchestrator.py                      │
│   3  reid          cross-camera association (confidence)     │
│   4  appearance    robustness to clothing/lighting/pose      │
│   5  occlusion     face observability (never a risk signal)  │
│   6  behavior      loitering, counter-flow, running, zones   │
│   7  relationship  persistent following patterns             │
│   8  crowd         density, flow, compression, surge         │
│   9  object        bag association, unattended escalation    │
│  12  spatial       world coordinates for twin / AR           │
│  13  watchlist     authorised person-of-interest matching    │
│  14  privacy       redaction, anonymity, escalation gating   │
└──────────────┬──────────────────────────────────────────────┘
               ▼
        10  threat.py        explainable saturating risk score
               ▼
        11  commander.py     narration + ranked recommendations
               ▼
   ┌───────────┼───────────────┬──────────────┐
   ▼           ▼               ▼              ▼
persistence   events bus   automation      12 forensic.py
 (database)   (WebSocket)   (n8n)          (RAG + conversational)
   │              │             │              │
   └──────────────┴──────┬──────┴──────────────┘
                         ▼
              React dashboard · Digital Twin · AR
                         ▼
                  HUMAN OPERATOR decides
```

## Why it is shaped this way

**One agent, one competence.** Each agent reads the shared world model and
emits `Finding` objects with a confidence and an explanation. A failing agent
is caught by `Agent.run()` and never stops a frame — one bad module degrades
one signal, not the platform.

**Agents consume trajectories, not pixels.** Only agents 1 and 2 touch image
data. Everything above them works on `TrackState` geometry. This is why the
generated trajectory datasets are legitimate training and evaluation data for
the reasoning layer, and why `tools/evaluate_agents.py` can score the agents
without a detector in the loop.

**Correlation, not duplication.** Findings that share a camera and overlap in
time are absorbed into one open incident, which is re-scored and re-narrated
rather than re-raised. Without this, one abandoned bag produces an alert per
frame.

**Saturating risk.** The score is `100 · (1 − e^(−Σ/55))`, so five weak signals
cannot add up to a critical incident. Each behaviour contributes
`weight × confidence × recency`, and only the strongest instance of a behaviour
counts.

## Graceful degradation

Every heavy dependency is optional, and the substitute reports itself honestly
in the UI rather than pretending:

| Missing | Substitute | Honest cost |
|---|---|---|
| torch / ultralytics | MOG2 motion detector | Blob detection only; reports `fallback-motion` |
| torchreid | Striped HSV histogram + edge density | Works short-horizon; weaker across long gaps |
| insightface | OpenCV Haar + handcrafted descriptor | Detects fewer faces; watchlist leans on body, which is capped lower |
| chromadb | In-process TF-IDF inverted index | Exact keyword retrieval, no semantic paraphrase |
| LLM provider | Deterministic templates over the same evidence | Plainer wording; cannot hallucinate |

This is deliberate: the platform must be demonstrable and testable on a clean
checkout, and an operator must never be shown a confident number produced by a
fallback they did not know was running.

## Threads and data flow

- **One thread per camera** (`CameraWorker`). Grab → detect → track → agents →
  publish. Sleeps to hold the configured fps.
- **Event bus** is thread-safe on publish and hands events to the API event
  loop via `call_soon_threadsafe`, so worker threads never touch asyncio.
- **Persistence** runs on the worker thread with its own short-lived sessions,
  never a request-scoped one. Track rows are written on a cadence, not per
  frame.
- **Automation** is a bounded queue drained by its own thread with retries, so
  a slow n8n can never stall a camera.
- **Rendering** only happens while something is consuming the stream, with a
  grace window that keeps a recent still warm for snapshots and evidence.

## Where the guarantees live

| Guarantee | Enforced in |
|---|---|
| Covered face never raises risk | `threat.py` `ZERO_WEIGHT_BEHAVIORS`, weight 0 in policy |
| Contents never inferred | `objects.py` has no such path; dataset validator fails on contents labels |
| Matches are confidences | `identity.py`, `watchlist.py` — body-only capped at 0.72 |
| AI recommends, human decides | `commander.py` `ACTION_CATALOGUE.requires_human_approval` |
| Anonymous by default | `privacy.py`; identity escalation creates an `ApprovalRequest` |
| Bystanders redacted before transmission | `privacy.redact_frame` in the worker, before JPEG encode |
| Every sensitive read audited | `deps.record()` on each route → `AuditLog` |
| Watchlist needs basis + approval | `WatchlistSubject.status`, `/approve` endpoint |

## Compute placement

Three places inference can run, selected at runtime:

1. **Server** — torch models on the API host. The dashboard toggle moves every
   registered model between CUDA and CPU live, with no restart.
2. **Remote worker** — `tools/gpu_worker.py` registers a machine's GPU. This is
   how a CPU-only VPS reaches a discrete GPU elsewhere.
3. **Client** — WebGPU in the browser, for twin and heatmap rendering only. It
   requests the high-performance adapter, which selects a discrete GPU with no
   permission prompt. A browser cannot run CUDA models, so this never covers
   detection.

## Extending it

**A new agent**: subclass `Agent`, implement `process(ctx) -> List[Finding]`,
register it in `Orchestrator.__init__`. It appears in the roster, gets a
toggle, and its findings flow into scoring automatically.

**A new behaviour**: add a weight to `threat_weights` in policy, a title in
`commander.TITLES`, a playbook entry in `commander.PLAYBOOK`, and a label in
the frontend's `format.ts`. Nothing else needs to know.

**A new camera source**: subclass `FrameSource`, add it to `SOURCE_TYPES`. The
attach dialog picks it up from `/api/cameras/source-types`.
