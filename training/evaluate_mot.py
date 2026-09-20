#!/usr/bin/env python
"""Evaluate Sentinel's tracker on MOT17 / MOT20 with MOTChallenge metrics.

ByteTrack is an association algorithm with no learned weights, so these
datasets are used the way they are meant to be: to MEASURE tracking quality,
not to train. Detections come from the platform's own detector (stock YOLO11m,
or a fine-tuned checkpoint), fed through the exact ByteTracker the live
pipeline uses, and scored against ground truth with py-motmetrics.

    python training/evaluate_mot.py --dataset mot17
    python training/evaluate_mot.py --dataset mot20 --weights storage/models/sentinel_best.pt

Reported: MOTA, IDF1, ID switches, precision, recall per sequence and overall.
"""
from __future__ import annotations

import argparse
import configparser
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# motmetrics 1.4 still calls np.asfarray, which NumPy 2.0 removed.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
RAW = ROOT / "datasets" / "raw"


def sequences(dataset: str):
    base = RAW / dataset
    found = []
    for seq in sorted(base.rglob("seqinfo.ini")):
        folder = seq.parent
        if "train" not in str(folder).lower():
            continue                              # only train seqs have public GT
        if dataset == "mot17" and not folder.name.endswith("-FRCNN"):
            continue                              # the 3 variants share one GT
        if (folder / "gt" / "gt.txt").exists():
            found.append(folder)
    return found


def load_gt(folder: Path):
    """frame -> (ids, boxes xywh) for pedestrians flagged for evaluation."""
    data = np.loadtxt(folder / "gt" / "gt.txt", delimiter=",")
    # cols: frame, id, x, y, w, h, considered, class, visibility
    keep = (data[:, 6] == 1) & (data[:, 7] == 1)
    data = data[keep]
    out = {}
    for frame in np.unique(data[:, 0]).astype(int):
        rows = data[data[:, 0] == frame]
        out[frame] = (rows[:, 1].astype(int), rows[:, 2:6])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mot17", "mot20"], default="mot17")
    parser.add_argument("--weights", default="yolo11m.pt")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="low on purpose: ByteTrack's second stage uses weak boxes")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--limit-frames", type=int, default=None)
    args = parser.parse_args()

    import cv2
    import motmetrics as mm
    from ultralytics import YOLO

    from app.vision.tracker import ByteTracker
    from app.vision.types import Detection

    seqs = sequences(args.dataset)
    if not seqs:
        print(f"no {args.dataset} training sequences with ground truth under {RAW / args.dataset}")
        return 1

    model = YOLO(args.weights)
    person_ids = [i for i, n in model.names.items() if n == "person"]
    accumulators, names, timings = [], [], []

    for folder in seqs:
        info = configparser.ConfigParser()
        info.read(folder / "seqinfo.ini")
        fps = float(info["Sequence"].get("frameRate", 30))
        length = int(info["Sequence"]["seqLength"])
        if args.limit_frames:
            length = min(length, args.limit_frames)
        img_dir = folder / info["Sequence"].get("imDir", "img1")
        gt = load_gt(folder)

        tracker = ByteTracker(high_thresh=0.5, low_thresh=0.1, match_thresh=0.8,
                              max_age=int(fps), min_hits=2)
        acc = mm.MOTAccumulator(auto_id=True)
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        started = time.perf_counter()

        for frame in range(1, length + 1):
            img = cv2.imread(str(img_dir / f"{frame:06d}.jpg"))
            if img is None:
                continue
            res = model.predict(img, conf=args.conf, classes=person_ids, imgsz=args.imgsz,
                                device=0, verbose=False)[0]
            dets = [
                Detection(bbox=tuple(float(v) for v in b), score=float(s),
                          class_id=0, class_name="person")
                for b, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())
            ]
            tracks = tracker.update(dets, timestamp=t0 + timedelta(seconds=frame / fps), fps=fps)

            hyp_ids = [t.track_id for t in tracks]
            hyp_boxes = np.array([[t.bbox[0], t.bbox[1], t.bbox[2] - t.bbox[0],
                                   t.bbox[3] - t.bbox[1]] for t in tracks]).reshape(-1, 4)
            gt_ids, gt_boxes = gt.get(frame, (np.array([], int), np.zeros((0, 4))))
            dist = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=0.5)
            acc.update(gt_ids.tolist(), hyp_ids, dist)

        elapsed = time.perf_counter() - started
        accumulators.append(acc)
        names.append(folder.name)
        timings.append(length / elapsed)
        print(f"  {folder.name}: {length} frames at {length / elapsed:.1f} fps", flush=True)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators, names=names, generate_overall=True,
        metrics=["mota", "idf1", "num_switches", "precision", "recall",
                 "mostly_tracked", "mostly_lost", "num_frames"],
    )
    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names))

    overall = summary.loc["OVERALL"]
    result = {
        "dataset": args.dataset,
        "detector": args.weights,
        "tracker": "Sentinel ByteTracker (backend/app/vision/tracker.py)",
        "sequences": names,
        "MOTA": round(float(overall["mota"]), 4),
        "IDF1": round(float(overall["idf1"]), 4),
        "id_switches": int(overall["num_switches"]),
        "precision": round(float(overall["precision"]), 4),
        "recall": round(float(overall["recall"]), 4),
        "mostly_tracked": int(overall["mostly_tracked"]),
        "mostly_lost": int(overall["mostly_lost"]),
        "mean_fps": round(float(np.mean(timings)), 1),
        "note": "MOTChallenge train sequences (the only ones with public ground truth); "
                "no training was done on them.",
    }
    out = ROOT / "training" / "runs" / "tracking"
    out.mkdir(parents=True, exist_ok=True)
    tag = Path(args.weights).stem
    (out / f"{args.dataset}_{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
