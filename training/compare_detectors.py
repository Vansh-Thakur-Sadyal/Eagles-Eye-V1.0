#!/usr/bin/env python
"""Decide whether the fine-tuned detector is actually better than stock YOLO11m.

Both models are validated on the SAME data with the SAME metric:
  1. the held-out slice of the curated Open Images set (in-domain for training)
  2. MOT17 frames with person boxes (street / surveillance footage the
     fine-tune never saw) - the closer proxy for what Sentinel's cameras see

The fine-tune is promoted into storage/models/ and the platform config only if
it does not lose on either set and wins on at least one. Otherwise stock
YOLO11m stays in service and the reason is written down.

    python training/compare_detectors.py --candidate training/runs/detect/sentinel/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OI_DATA = Path("V:/sentinel_cache/oi_yolo/data.yaml")
MOT_YOLO = Path("V:/sentinel_cache/mot17_person")
MODELS = ROOT / "storage" / "models"


def build_mot17_person_set(every: int = 10) -> Path:
    """MOT17 ground truth -> YOLO labels (COCO person = 0), every Nth frame."""
    import configparser

    import cv2

    yaml_path = MOT_YOLO / "data.yaml"
    if yaml_path.exists():
        return yaml_path
    (MOT_YOLO / "images" / "val").mkdir(parents=True, exist_ok=True)
    (MOT_YOLO / "labels" / "val").mkdir(parents=True, exist_ok=True)

    base = ROOT / "datasets" / "raw" / "mot17"
    for ini in sorted(base.rglob("seqinfo.ini")):
        seq = ini.parent
        if "train" not in str(seq).lower() or not seq.name.endswith("-FRCNN"):
            continue
        # MOT17Labels.zip unpacks a second, image-less copy of each sequence
        # (gt only); use the copy that actually has frames.
        if not (seq / "img1").is_dir():
            continue
        info = configparser.ConfigParser()
        info.read(ini)
        w, h = int(info["Sequence"]["imWidth"]), int(info["Sequence"]["imHeight"])
        gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=",")
        # pedestrians flagged for evaluation, at least a quarter visible
        gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1) & (gt[:, 8] >= 0.25)]
        for frame in range(1, int(info["Sequence"]["seqLength"]) + 1, every):
            rows = gt[gt[:, 0] == frame]
            src = seq / "img1" / f"{frame:06d}.jpg"
            name = f"{seq.name}_{frame:06d}"
            shutil.copy2(src, MOT_YOLO / "images" / "val" / f"{name}.jpg")
            lines = []
            for _f, _id, x, y, bw, bh, *_ in rows:
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(w, x + bw), min(h, y + bh)
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (MOT_YOLO / "labels" / "val" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(_coco_names()))
    yaml_path.write_text(f"path: {MOT_YOLO.as_posix()}\ntrain: images/val\nval: images/val\n"
                         f"nc: 80\nnames:\n{names}\n")
    return yaml_path


def _coco_names():
    sys.path.insert(0, str(ROOT / "training"))
    from prepare_open_images import COCO_NAMES

    return COCO_NAMES


SENTINEL_CLASSES = [0, 1, 2, 3, 5, 7, 24, 26, 28, 39, 63]


def validate(weights: str, data: Path, classes, imgsz: int) -> dict:
    from ultralytics import YOLO

    metrics = YOLO(weights).val(data=str(data), imgsz=imgsz, batch=16, device=0,
                                classes=classes, plots=False, verbose=False)
    return {"mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default="yolo11m.pt")
    args = parser.parse_args()

    mot_yaml = build_mot17_person_set()
    report = {}
    for label, weights in (("baseline", args.baseline), ("candidate", args.candidate)):
        report[label] = {
            "open_images_val": validate(weights, OI_DATA, SENTINEL_CLASSES, 640),
            "mot17_person": validate(weights, mot_yaml, [0], 1280),
        }
        print(label, json.dumps(report[label]), flush=True)

    b, c = report["baseline"], report["candidate"]
    deltas = {
        "open_images_val": round(c["open_images_val"]["mAP50_95"] - b["open_images_val"]["mAP50_95"], 4),
        "mot17_person": round(c["mot17_person"]["mAP50_95"] - b["mot17_person"]["mAP50_95"], 4),
    }
    # Allow a hair of noise, but no real regression on either set.
    no_loss = all(d >= -0.005 for d in deltas.values())
    some_gain = any(d >= 0.01 for d in deltas.values())
    promote = no_loss and some_gain

    report["delta_mAP50_95"] = deltas
    report["promoted"] = promote
    report["rule"] = "promote only if no set regresses (>0.5 pt) and one improves by >=1 pt"
    if promote:
        MODELS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate, MODELS / "sentinel_detector.pt")
        report["deployed_as"] = str(MODELS / "sentinel_detector.pt")
    else:
        report["why_not_promoted"] = ("regressed on " + ", ".join(k for k, d in deltas.items() if d < -0.005)
                                      if not no_loss else "gain too small to justify a swap")

    out = ROOT / "training" / "runs" / "detect" / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
