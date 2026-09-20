# Datasets

Sentinel is a system of specialised models, so it needs several specialised
datasets rather than one big one. They split into two groups: **public sets you
download** and **custom sets you generate here**.

## Generate the custom sets

```bash
python tools/make_datasets.py structure          # folder tree + camera/zone metadata
python tools/make_datasets.py behaviour --per-category 40 --render
python tools/make_datasets.py occlusion --source datasets/source_faces
python tools/make_datasets.py reasoning
python tools/make_datasets.py validate           # gate a training run on this
```

Or all at once: `python tools/make_datasets.py all --per-category 40`.

### What gets generated, and what it is good for

| Category | How it is produced | Use it for | Do **not** use it for |
|---|---|---|---|
| Loitering, following, counter-flow, restricted zone, abandoned object | Scripted motion with exact ground truth | Training/evaluating the behaviour, relationship and object agents on trajectory features; threshold tuning; pipeline testing | Training a detector — renders will not generalise to real CCTV. A trained detector correctly finds **nothing** in them, which is why the synthetic camera pins the motion backend |
| Occlusion (mask, helmet, hood, sunglasses, partial) | Augmented from **your own photographs** | Face-occlusion classification, robustness testing | Building a permanent identity database |
| Appearance variation | Same augmentation pipeline | Re-ID robustness across hairstyle, glasses, lighting, blur, angle | — |
| Incident reasoning + Q&A | Derived from incidents **this deployment recorded** | Fine-tuning narration and the assistant; guardrail regression tests | Anything, until you have collected a few hundred real incidents |

`manifest.json` in the dataset root records the provenance of each section.
Read it before training — simulated trajectories, augmented imagery and real
footage do not carry the same weight, and the manifest says which is which.

### Matched negatives

Every behaviour category ships an equal number of **negative** scenarios: a
person waiting normally then leaving, a queue that advances, two friends
walking abreast, someone crossing another's path once, a bag whose owner stays
with it, a boundary walked along but not crossed.

This is not padding. A detector trained or tuned only on positives learns to
say "yes", and the validator fails the dataset if a category has no negatives.

### Face and appearance sets

These need your own photographs. Lay them out one folder per identity:

```
datasets/source_faces/
  subject_a/  photo1.jpg  photo2.jpg
  subject_b/  photo1.jpg
```

Each reference is expanded into ~20 variants (mask, sunglasses, cap, hood,
helmet, partial occlusion, lighting, colour cast, motion blur, low resolution,
sensor noise, three viewpoints, hair and ageing changes) plus composed pairs.
Every output records its source identity and exact parameters.

Use at least two identities. Robustness numbers from a single identity tell you
whether the model memorised one face, not whether it distinguishes people.

## Public datasets to download

| # | Dataset | Module | Size | What it gives you |
|---|---|---|---|---|
| 1 | Open Images V7 | Detection | ~561 GB | 600 classes, 1.74M box-annotated images |
| 2 | XD-Violence | Violence | ~86 GB | 4,754 untrimmed videos, audio + video |
| 3 | UCF-Crime | Anomaly | ~13 GB | 1,900 long surveillance videos, 13 anomaly categories |
| 4 | MOT17 | Tracking | ~5.5 GB | Pedestrian tracks with ground truth |
| 5 | MOT20 | Dense tracking | ~5 GB | Very crowded scenes |
| 6 | ShanghaiTech Campus | Campus anomaly | ~4.3 GB | 330 normal + 107 test videos, 13 scenes |
| 7 | UCF-QNRF | Crowd | ~1–3 GB | 1,535 high-resolution crowd images, 1.25M annotations |
| 8 | Market-1501 | Re-ID | ~0.15 GB | 1,501 identities across multiple cameras |

**Before you download all of it, read `docs/TRAINING.md`.** MOT17 and MOT20
need no training at all (ByteTrack is an association algorithm — use them to
*measure* tracking), and a curated subset of Open Images gets you most of the
detector benefit for a fraction of the compute.

## Formats Sentinel expects

### Detection — YOLO

```
datasets/detection/
  images/{train,val,test}/frame_0001.jpg
  labels/{train,val,test}/frame_0001.txt
  data.yaml
```

One line per object, coordinates normalised 0–1:

```
class_id x_center y_center width height
0 0.512 0.483 0.231 0.641
```

`data.yaml` is written by `make_datasets.py structure` and pins the eleven
classes so the dataset and the runtime policy cannot drift apart.

### Tracking — MOT style

```
frame_id,track_id,x,y,w,h,confidence,class_id
```

`x,y` is the top-left corner in pixels. The validator checks for duplicate
`(frame, track)` rows, non-monotonic frames and non-positive box sizes.

### Behaviour, following, unattended objects

Exactly the shapes the master write-up specifies:

```
# behavior.csv
video_id,track_id,start_frame,end_frame,behavior,label,confidence

# following.csv
scene_id,subject_a,subject_b,start_frame,end_frame,relationship

# unattended_objects.csv
scene_id,object_id,owner_track_id,first_seen,separation_time,stationary_duration,status
```

### Incidents

```json
{
  "incident_id": "INC_001",
  "camera_id": "CAM_07",
  "timestamp": "2026-01-15T18:42:15+00:00",
  "event": "unattended_object",
  "track_ids": ["T_104"],
  "zone_id": "Gate_4",
  "risk_score": 78.0,
  "confidence": 0.91
}
```

## Validation

```bash
python tools/make_datasets.py validate
```

Exits non-zero on any error, so you can gate a training job on it. It checks:

- required folders and `data.yaml` exist;
- YOLO labels have five fields, class ids in range, coordinates normalised, no
  zero-area boxes, and every label has a matching image;
- tracking CSVs have monotonic frames, no duplicate rows, positive box sizes;
- every behaviour category has matched negatives, and warns on class imbalance
  beyond 80/20;
- occlusion samples referenced in annotations actually exist on disk, and warns
  when only one source identity is present;
- **guardrails**: no file anywhere in the dataset may label a sample with
  contents or an identity claim (`explosive`, `bomb`, `weapon_contents`,
  `criminal`, `stalker`, `terrorist`). This is an error, not a warning — the
  platform's scope limit is enforced in the data, not just in prose.

## A note on scope

The write-up is explicit and the code enforces it: Sentinel detects a
*potentially unattended object* and traces the last associated subject's
trajectory. It does not infer contents, and there is no code path that can.
Contents identification is a physical-security and sensor task with a human in
charge. Keep that boundary in your data too — the validator will fail the
dataset if you do not.
