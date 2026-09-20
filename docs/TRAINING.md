# Training

> **Measured results of the local training run are in [TRAINING_RESULTS.md](TRAINING_RESULTS.md).**

## Local RTX 5070 Ti, or rent an RTX 6000 Pro 96 GB?

**Train locally. Rent only for one specific job, and probably not even that.**

The short reason: 16 GB is enough VRAM for *every model in this stack*. What a
96 GB card buys you here is wall-clock time, not capability — and you would pay
for that speed twice, once in rental and once in uploading 675 GB of data to a
machine that does not have it yet.

### What actually needs training

Going through your download list honestly, most of it does not need a big card:

| Dataset | Size | What you train | VRAM needed | Verdict |
|---|---|---|---|---|
| **Open Images V7** | 561 GB | Detector fine-tune | 8–14 GB | **Local**, on a curated subset — see below |
| **XD-Violence** | 86 GB | Violence classifier | 6–10 GB | **Local** — feature extraction, then a small head |
| **UCF-Crime** | 13 GB | Anomaly (MIL) head | 4–8 GB | **Local** — same pipeline |
| **MOT17 / MOT20** | 10 GB | *nothing* | — | ByteTrack is training-free. Use these to **evaluate** |
| **ShanghaiTech** | 4.3 GB | Anomaly / crowd | 6–10 GB | **Local** |
| **UCF-QNRF** | 1–3 GB | Crowd counting | 10–14 GB | **Local**, patch-based |
| **Market-1501** | 0.15 GB | OSNet Re-ID | 6–8 GB | **Local** — trains in about an hour |

Two things fall out of that table immediately:

**1. MOT17 and MOT20 are not training data for you.** ByteTrack is an
association algorithm, not a learned model — Sentinel's implementation in
`backend/app/vision/tracker.py` has no weights. Those datasets are how you
*measure* tracking quality (IDF1, MOTA), which is valuable, but there is
nothing to train.

**2. You do not need all 561 GB of Open Images.** YOLO11's COCO pretraining
already covers all eleven classes Sentinel uses: person, backpack, handbag,
suitcase, bicycle, motorcycle, car, bus, truck, bottle, laptop. You are
fine-tuning for *domain* (overhead CCTV angles, poor light, small distant
subjects), not for new classes. A curated 50k–100k image subset of those
classes gets you most of the benefit for ~5% of the compute.

### Measured on this machine

With the stack installed (torch 2.11.0+cu128, ultralytics 8.4.155), YOLO11m at
640px on an RTX 5070 Ti:

| | ms/frame | fps |
|---|---|---|
| GPU (fp16) | 7.5 | 133 |
| CPU | 54.1 | 18.5 |

A 1280x720 webcam through the full pipeline (detect + track + all agents) runs
at 9.5 ms inference and holds the configured 12 fps with headroom. Toggling the
GPU off in the dashboard moves it to 34.9 ms without dropping a frame.

That is the baseline to compare any training decision against.

### Rough time estimates on your 5070 Ti

Approximate, and they depend heavily on disk speed and image size:

| Job | Data | Estimate |
|---|---|---|
| YOLO11m fine-tune, 640px, 100 epochs | 50k curated images | ~12–20 h (overnight) |
| YOLO11m fine-tune, 1280px, 100 epochs | 50k curated images | ~40–60 h |
| OSNet Re-ID | Market-1501 | ~1–2 h |
| I3D/CLIP feature extraction | UCF-Crime (13 GB) | ~3–6 h |
| MIL anomaly head on those features | — | minutes |
| Feature extraction | XD-Violence (86 GB) | ~1–2 days, mostly I/O |
| Crowd counting | UCF-QNRF | ~4–8 h |

Everything on that list is an overnight or weekend job. None of it is blocked
by 16 GB.

### When renting is actually worth it

Rent the RTX 6000 Pro if **one** of these is true:

- You decide the curated subset genuinely is not accurate enough and you want a
  full-scale Open Images run. On 1.74M images that is roughly 20+ days locally
  versus a few days rented — that is a real difference.
- You want to train at **1280px**. This is the strongest argument for renting,
  and it is a real one for your use case: surveillance cameras see people and
  bags as small distant objects, and higher input resolution helps small-object
  recall more than a bigger model does. 96 GB lets you run 1280px with a large
  batch comfortably.
- You are running a hyperparameter sweep and want several configurations in
  parallel.

If you rent, use the VRAM for **resolution and batch size**, not for a larger
model. `yolo11m` at 1280px will serve you better than `yolo11x` at 640px for
distant small objects.

### The hidden cost people forget

Uploading 675 GB to a rented machine is not free and not fast. On a 100 Mbit
upstream that is roughly 15 hours of pure transfer before a single epoch runs.
Check whether the provider can mount the public datasets directly, or whether
you are expected to ship them — that single detail often decides the question.

### Recommended sequence

1. **Now, locally**: run the system end to end with COCO-pretrained YOLO11.
   Get cameras, agents, incidents and the dashboard working. You do not need
   any custom training to have a working platform.
2. **Locally, overnight**: fine-tune the detector on a curated subset. Measure
   against a held-out validation split.
3. **Locally**: Re-ID on Market-1501, anomaly heads on UCF-Crime features.
4. **Only if step 2 is not good enough**: rent, and spend the money on 1280px
   rather than on a bigger backbone.

---

## Detector

```bash
python training/train_detector.py --check      # pre-flight: CUDA, data, disk
python training/train_detector.py --epochs 100
python training/train_detector.py --epochs 100 --imgsz 1280 --batch 8   # if renting
```

Pre-flight checks your torch/CUDA build, counts images in each split, warns on
a missing validation split and verifies free disk before anything starts.

The augmentation defaults are set for surveillance rather than web photos:
modest rotation, no vertical flip, wider scale jitter, and HSV ranges that
tolerate poor lighting.

Point the platform at the result:

```bash
# backend/.env
SENTINEL_YOLO_WEIGHTS=sentinel_best.pt      # copied into storage/models/
```

## Behaviour classifiers

```bash
python training/train_behavior.py --all --report
```

Trains a gradient-boosted classifier per behaviour on trajectory features
derived from your generated scenarios. Cross-validated, because a few hundred
scenes is far too few to trust a single split. Feature importances are printed
so you can see *which* signal carries the decision.

**Read the caveat in the report.** These train on simulated trajectories with
exact ground truth, so near-perfect scores mean "the features separate the
scripted behaviours cleanly", not "this works on real CCTV".

## Evaluating the rule-based agents

```bash
python tools/evaluate_agents.py                 # precision / recall / FP rate
python tools/evaluate_agents.py --verbose       # per-scene
python tools/evaluate_agents.py --tune loitering_seconds 60,90,120,150
```

This replays generated scenarios through the real agents and scores them
against exact ground truth — including a false-positive rate measured on
matched negative scenarios (someone waiting normally, two friends walking
together, a bag whose owner stays with it).

The `--tune` sweep is how the shipped defaults were chosen. It is the right way
to set a threshold for *your* site: generate scenarios that match your
environment, sweep, and read the operating point off the curve.

### What the current scores mean

At the time of writing, all five categories score 1.00 precision and 1.00
recall on generated data. **That is not a claim about real-world accuracy.**
It means the agents correctly implement the behaviour definitions on clean
trajectories with perfect detection. Real footage brings detector noise, missed
detections, occlusion, identity switches and human behaviour that does not fit
a script — expect materially lower numbers, and measure them on annotated real
footage before relying on any of this operationally.

What the harness is genuinely good for is catching regressions and logic
errors. Building it immediately surfaced four real bugs that were invisible in
normal operation:

- `measured_fps` reported processing throughput rather than the delivered frame
  rate, which inflated every speed calculation and manufactured false "running"
  detections;
- any passer-by reset the unattended-object clock, so in a crowded concourse —
  the primary deployment — a bag would essentially never have been flagged;
- object ownership was granted to whoever happened to be nearest on the first
  frame, so the "last associated subject" trajectory could name a stranger;
- the following detector could not distinguish *beside* from *behind*, giving a
  100% false-positive rate on pairs of companions.

Each is covered by a test in `backend/tests/test_regressions.py`.

## Public dataset preparation

The public datasets are not downloaded or converted for you. See
`docs/DATASETS.md` for what each one is for and the format Sentinel expects.
