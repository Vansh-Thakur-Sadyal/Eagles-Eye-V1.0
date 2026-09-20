# Training results (local RTX 5070 Ti)

Every model below was trained on this machine's RTX 5070 Ti (16 GB, torch
2.11 + cu128). A model only reaches `storage/models/` - where the live agents
load it from - if it beats a measured baseline on data it never trained on.
Otherwise it stays in `training/runs/` and the platform keeps its previous
behaviour. Machine-readable versions of every number are in `training/runs/`.

| Model | Data | Result on held-out data | Baseline | Live? |
|---|---|---|---|---|
| Re-ID (OSNet x1.0) | Market-1501 | rank-1 **92.8%** | 14.7% (ImageNet weights) | yes |
| Anomaly head (MIL, R(2+1)D) | UCF-Crime | video AUC **0.905**, AP 0.931 | 0.5 (chance) | yes |
| Crowd density (CSRNet) | UCF-QNRF | MAE **119** people/image | 694 (counting YOLO boxes) | yes |
| Violence head (MIL, R(2+1)D) | XD-Violence | video AUC 0.862, false alarms 15.4% | gate: FPR <= 15% | **no** |
| Tracking (YOLO11m + ByteTracker) | MOT17 / MOT20 | MOTA 0.41 / 0.21 | evaluation only | - |
| Detector fine-tune (YOLO11m) | Open Images V7 subset | +6.2 pts mAP50-95 on Open Images, **-2.9 pts on MOT17 people** | stock YOLO11m | **no** |
| LocateAnything-3B fusion | Open Images val (bags) | recall +11.4 pts, precision -7.7 pts | YOLO alone | **no** (off by default) |

## Re-identification - `training/train_reid.py`

- OSNet x1.0, triplet + softmax, 80 epochs, 50 min.
- Market-1501 test (3,368 queries vs 15,913 gallery): rank-1 92.8%.
  ImageNet-only weights score 14.7%.
- **Threshold check** (`training/runs/reid/threshold_calibration.json`), 300
  identities seen on two different cameras:

  | cosine threshold | same person matched | different people matched |
  |---|---|---|
  | 0.60 (watchlist body match) | 95.0% | 0.67% |
  | 0.62 (cross-camera identity) | 94.7% | 0.33% |

  With the ImageNet weights, same-person and different-person similarities
  were 0.67 vs 0.62, so those thresholds meant nothing. Now they are 0.76 vs 0.38.
- Bug found while integrating: torchreid's own checkpoint loader fails under
  torch >= 2.6 (`weights_only=True`), and the embedder was silently falling
  back to ImageNet weights. `BodyEmbedder._load_reid_checkpoint` fixes this,
  and `test_promoted_reid_checkpoint_actually_loads` prevents it recurring.

## Anomaly - `training/train_anomaly.py --task anomaly`

- Weakly-supervised multiple-instance learning (Sultani et al.) on frozen
  R(2+1)D-18 Kinetics features, 32 segments x 16 frames per video.
- Training data: 950 anomaly videos and 663 of the 800 training-normal videos.
  The two `Training-Normal-Videos-Part-*.zip` downloads are truncated (no
  central directory); `tools/recover_partial_zip.py` recovered every complete,
  CRC-verified entry before the cut.
- Test: UCF-Crime's official 140 test anomalies + 106 held-out normals.
  Video AUC 0.905, AP 0.931.
- The alarm threshold (0.558) is calibrated on validation normals. On the
  test split it catches 59% of anomalous videos at a 5.7% false-alarm rate.
- **Cross-dataset: ShanghaiTech frame-level AUC 0.54 - near chance.** The
  head learned what is unusual in UCF-Crime's street/CCTV crime footage, and
  that does not carry over to campus anomalies (bikes, skateboards, running on
  walkways). Expect it to need site-specific training before trusting it in a
  new kind of scene. Treat its alerts as "worth a look", never as proof.
- Not reported: official frame-level UCF-Crime AUC (the temporal annotation
  file was not in the download).

## Crowd counting - `training/train_crowd.py`

- CSRNet (VGG16 frontend + dilated backend), 150 epochs, 45 min.
- UCF-QNRF official test split (334 images, avg ~800 people each):
  MAE 119 / RMSE 210, against MAE 694 / RMSE 1022 for counting YOLO person
  boxes on the same images.
- The crowd agent switches to the density count only when it clearly exceeds
  the detector count (x1.3 and >= 10 people): the signature of a crowd too
  dense to box.
- Served at the training resolution (1536 px). At 1024 px MAE rose to 127 and
  counts were biased low by 42 people per image.

## Violence - `training/train_anomaly.py --task violence`

- XD-Violence: 3,950 training videos (4 corrupt in the source archive,
  quarantined in `C:\sentinel_data\xd_violence_quarantine`). Held-out 15%
  split by source film, so clips of one film never land on both sides.
- Test video AUC 0.862, AP 0.866.
- **Not promoted.** At the threshold calibrated on validation normals, test
  false alarms were 15.4% against the 15% limit. The threshold was not
  re-tuned on the test split; that would make the number meaningless. The
  jump from ~5% on validation to 15% on unseen films is the real finding.
- To finish this properly: download XD-Violence's official test set (800
  videos with frame labels), then re-run `train_anomaly.py --task violence`.

## Tracking - `training/evaluate_mot.py`

Stock YOLO11m (conf 0.1, 1280 px) + the platform's ByteTracker, on the
MOTChallenge train sequences (the only ones with public ground truth).
No training on MOT data.

| | MOTA | IDF1 | ID switches | precision | recall |
|---|---|---|---|---|---|
| MOT17 (7 seq) | 0.409 | 0.540 | 659 | 0.79 | 0.56 |
| MOT20 (4 seq) | 0.207 | 0.322 | 723 | 0.88 | 0.24 |

MOT20 is extremely dense and the COCO detector misses three quarters of the
people. For counting, the density model covers this case. For tracking in
packed crowds, the next step would be a person detector fine-tuned on
MOT17/MOT20 (train on half the sequences, evaluate on the rest).
Detection ran at 28-38 fps; the `mean_fps` field in the JSON measures the
tracker alone.

## Detector fine-tune - `training/train_detector.py` + `compare_detectors.py`

39,011 Open Images V7 images curated for the classes Sentinel watches (person,
bags, luggage, vehicles, bottle, laptop), mapped to COCO ids so stock and
fine-tuned models are scored on identical data. The fine-tune is promoted only
if it gains >= 0.01 mAP50-95 on one validation set (Open Images val or MOT17
people) without losing > 0.005 on the other.

The first attempt (ultralytics `optimizer=auto` -> SGD, lr 0.01) dropped val
mAP50-95 from 0.34 to 0.21 in three epochs. That run is kept in
`training/runs/detect/sentinel_failed_sgd_lr0.01`. It was re-run with AdamW at
lr 5e-4, 40 epochs, ~6 h, finishing at val mAP50-95 0.485.

**Not promoted** (`training/runs/detect/comparison.json`):

| | Open Images val | MOT17 people |
|---|---|---|
| stock YOLO11m | mAP50-95 0.424 | **0.527** |
| fine-tuned | **0.486** (+0.062) | 0.498 (-0.029) |

It learned Open Images - web photographs of bags, vehicles and people - and
got *worse* at the thing this platform exists for: people in surveillance
footage. The gate refused it, so stock YOLO11m stays live. This is the
comparison earning its keep: on the fine-tuning set alone it looks like a
clear win.

Next step for a detector that is genuinely better here: train on
surveillance-domain people as well, i.e. mix MOT17/MOT20 person crops into the
Open Images set, holding out whole sequences for evaluation so the comparison
stays honest.

## LocateAnything-3B + YOLO fusion - `tools/eval_locateanything.py`

`nvidia/LocateAnything-3B` is downloaded and integrated
(`LocateAnythingDetector`, `EnsembleDetector` in `backend/app/vision/detector.py`).
**Licence: NVIDIA License, non-commercial (research / evaluation) use only.**
It stays disabled by default (`SENTINEL_LOCATEANYTHING_ENABLED=false`).

What was verified:
- The integration follows the model card: `AutoModel` with custom
  `generate()`, the card's prompt templates, and output parsed as
  `<ref>label</ref><box><x1><y1><x2><y2></box>` in 0-1000 coordinates. On a
  test image the boxes matched ground truth at IoU 0.82 (person), 0.78
  (backpack) and 0.96 (car).
- **Hybrid decoding looped.** In `hybrid` mode with deterministic decoding,
  the model emitted one real box and then ~170 tiny boxes sliding along a
  row until the token limit. `slow` (autoregressive) mode gave correct output
  on the same image, so `slow` is now the default. The parser also drops such
  chains if they ever appear (`test_locateanything_degenerate_box_chain_is_dropped`).
- How it fuses: YOLO runs on every frame. LocateAnything runs in a background
  thread on keyframes and never blocks the camera loop. Its boxes for
  **static** objects (bags, luggage, smoke/fire, open-vocabulary objects)
  persist until the next keyframe and are added only where YOLO has no
  overlapping box. People and vehicles are left to YOLO, because a
  one-second-old box for something that moves becomes a ghost track.

**Measured on the RTX 5070 Ti**, 92 Open Images val images containing bags,
slow mode (`training/runs/locateanything_eval.json`):

| | bag precision | bag recall |
|---|---|---|
| YOLO alone (conf 0.35) | **0.550** | 0.451 |
| LocateAnything alone | 0.549 | 0.549 |
| YOLO + LocateAnything (live fusion) | 0.473 | **0.565** |

Per class, the fusion finds noticeably more: handbags 0.45 -> 0.70 recall,
backpacks 0.53 -> 0.63, suitcases 0.36 -> 0.41.

- Latency: **3.2 s mean per call** (p50 2.6 s, p95 6.5 s), **8.3 GB VRAM**.
  Far too slow for every frame, which is why it runs on keyframes in a
  background thread.
- **The pre-declared rule did not recommend it**: recall on bags rose 11.4
  points, but precision fell 7.7 points against the 5-point limit.
- **Enabled anyway on this deployment**, by the operator's decision, for
  educational use - which is what the NVIDIA licence permits. It is worth it
  when a missed bag matters more than a dismissed false one, and it is the
  only way to ask for things YOLO's 80 classes cannot express ("a person
  carrying a black backpack"). `SENTINEL_DETECTOR=ensemble`,
  `SENTINEL_LOCATEANYTHING_ENABLED=true`.
  **Any commercial deployment must turn both off again.**

Live cost of running the ensemble, measured on this GPU:

| | value |
|---|---|
| Steady-state detection | 7.1 ms/frame (141 fps capacity) - same as YOLO alone |
| Extra VRAM | 7.5 GB (of 16 GB) |
| Extra start-up | ~9 s to load the 3B model |
| Keyframe call | ~2.2 s, once per 120 frames (~10 s of video) |
| Effect of a keyframe | one ~1.7 s stall in the camera loop, because YOLO and the VLM share one GPU |

The stall is why `SENTINEL_LOCATEANYTHING_EVERY_N_FRAMES` defaults to 120 and
why `max_new_tokens` is capped at 384 (~50 objects): 1024 tokens only made the
worst case longer. If the hitch matters for a given camera, either raise the
interval or give the VLM its own GPU - an edge node with a second card, since
the node runs the same detector stack (docs/EDGE_GPU.md).

## Live pipeline speed (RTX 5070 Ti)

Measured end to end on real MOT17 footage at 1920x1080 - detection, Re-ID
embedding, tracking and all agents, one camera, per frame:

| scene | detect | Re-ID | track | agents | total | sustained |
|---|---|---|---|---|---|---|
| ~11 people (MOT17-09), ensemble detector | 9.2 ms | 14.4 ms | 0.7 ms | 2.6 ms | 28.7 ms | **34.8 fps** |
| ~25 people (MOT17-04), YOLO only | 8.3 ms | 13.5 ms | 1.2 ms | 2.3 ms | 26.3 ms | **38.1 fps** |
| ~9 people, CPU only (GPU toggle off) | - | - | - | - | 88.8 ms | 11.3 fps |

Cameras run at their configured fps (12 by default), so ~35 fps of capacity is
headroom: it is what lets one GPU carry several cameras at once.

**Bottleneck found and fixed while measuring this.** Re-ID embedded people one
at a time, ~6 ms each: 147 ms per frame in a crowd of 25, against 9 ms for
detection. `BodyEmbedder.embed_batch` now runs one batched forward pass for
every person in a frame, and the pipeline calls it once instead of looping:

| | before | after |
|---|---|---|
| Re-ID, ~25 people | 147 ms | **13.5 ms** |
| Whole frame | 168 ms (5.9 fps) | **26 ms (38 fps)** |

The p95 figures (~0.5 s) are the periodic heavy passes: the crowd density
model every 2 s, the video-event clip every 4 s, and - with the ensemble on -
the LocateAnything keyframe every 10 s.

## Bug found during validation

- **Watchlist matching never fired on live cameras.** `watchlist.py` did
  `track.embedding or self.body.embed(crop)`. The live tracker attaches a
  numpy embedding to every person, and `array or ...` raises, so once anyone
  was enrolled the agent failed on every live track. Unit tests missed it
  because their tracks carried no embedding. Fixed, with a regression test
  that drives `process()` with a real embedding
  (`test_watchlist_runs_on_live_tracks_that_carry_an_embedding`).

## Lessons from running this on Windows

- Every DataLoader worker commits several GB just by importing torch. Keeping
  query and gallery workers persistent (25 processes) exhausted the 105 GB
  commit limit and stalled the machine in paging. Only the train loader's
  workers stay persistent now.
- Running training, feature extraction and tests at the same time pinned all
  32 CPU threads and ran the detector's workers out of memory. Run GPU jobs
  one at a time (`training/run_queue.py` does).
- `motmetrics` 1.4 calls `np.asfarray`, which NumPy 2 removed; it is shimmed
  in `evaluate_mot.py`.
- A malformed video can crash OpenCV's decoder inside a worker. The feature
  extractor now rebuilds its pool and isolates the offending video.
