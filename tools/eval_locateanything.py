#!/usr/bin/env python
"""Does fusing nvidia/LocateAnything-3B with YOLO actually help? Measure it.

    python tools/eval_locateanything.py --images 200 [--mode slow]

On Open Images V7 validation images (the curated Sentinel set, labels in COCO
ids) this compares, per class, precision and recall at IoU 0.5 for:

  yolo          the live detector at its operating confidence (SENTINEL_YOLO_CONF)
  locate        LocateAnything-3B alone, same categories
  fused_live    YOLO + LocateAnything boxes for static classes where YOLO has
                no overlapping box - exactly what EnsembleDetector does live
  fused_all     the same, but for every class (what a naive fusion would do)

and LocateAnything latency on this GPU. LocateAnything gives no confidence
scores, so precision/recall at a fixed operating point is reported rather
than mAP. Recommendation rule, written down before the run: enable the
ensemble only if fused_live raises recall on bags/luggage by >= 0.05 while
losing <= 0.05 precision on those classes.

Writes training/runs/locateanything_eval.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

VAL = Path("V:/sentinel_cache/oi_yolo")
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
        24: "backpack", 26: "handbag", 28: "suitcase", 39: "bottle", 63: "laptop"}
CATEGORIES = ["person", "backpack", "handbag", "suitcase", "bicycle", "car", "motorcycle",
              "bus", "truck", "bottle", "laptop"]
BAGS = {"backpack", "handbag", "suitcase"}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, counts) -> None:
    """Greedy per-class matching at IoU 0.5; preds/gts = [(class, box, score)]."""
    for cls in {c for c, *_ in preds} | {c for c, *_ in gts}:
        p = sorted([x for x in preds if x[0] == cls], key=lambda x: -x[2])
        g = [x for x in gts if x[0] == cls]
        used = set()
        for _, box, _s in p:
            best, best_j = 0.0, None
            for j, (_, gbox, _) in enumerate(g):
                if j not in used:
                    v = iou(box, gbox)
                    if v > best:
                        best, best_j = v, j
            if best >= 0.5:
                used.add(best_j)
                counts[cls]["tp"] += 1
            else:
                counts[cls]["fp"] += 1
        counts[cls]["fn"] += len(g) - len(used)


def summarise(counts) -> dict:
    out = {}
    for cls, c in sorted(counts.items()):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        out[cls] = {"precision": round(tp / (tp + fp), 4) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                    "tp": tp, "fp": fp, "fn": fn}
    for name, group in (("bags", BAGS), ("all", set(counts))):
        tp = sum(counts[c]["tp"] for c in group if c in counts)
        fp = sum(counts[c]["fp"] for c in group if c in counts)
        fn = sum(counts[c]["fn"] for c in group if c in counts)
        out[f"_{name}"] = {"precision": round(tp / (tp + fp), 4) if tp + fp else None,
                           "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                           "tp": tp, "fp": fp, "fn": fn}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--mode", default="slow", choices=["fast", "slow", "hybrid"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import os

    os.environ.setdefault("SENTINEL_LOCATEANYTHING_MODEL",
                          "V:/sentinel_cache/hf_models/LocateAnything-3B")
    os.environ["SENTINEL_LOCATEANYTHING_MODE"] = args.mode

    import cv2
    import torch

    from app.config import get_settings
    from app.vision.detector import EnsembleDetector, LocateAnythingDetector, YoloDetector

    # images that contain at least one bag, so the bag comparison has support
    labels = sorted((VAL / "labels" / "val").glob("*.txt"))
    pool = []
    for lab in labels:
        rows = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        classes = {int(r[0]) for r in rows}
        if classes & {24, 26, 28} and classes <= set(COCO):
            pool.append(lab)
    random.Random(args.seed).shuffle(pool)
    chosen = pool[: args.images]
    print(f"{len(chosen)} images with bags (of {len(pool)} eligible)", flush=True)

    yolo = YoloDetector(classes=CATEGORIES)
    locate = LocateAnythingDetector(enabled=True)
    if not locate.ready:
        print("LocateAnything failed to load:", locate.load_error)
        return 1
    print(f"YOLO {yolo.weights_source}; LocateAnything on {next(locate._model.parameters()).device}, "
          f"mode {args.mode}", flush=True)

    counts = {k: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
              for k in ("yolo", "locate", "fused_live", "fused_all")}
    latencies = []
    started = time.time()
    for i, lab in enumerate(chosen, 1):
        img = cv2.imread(str(lab).replace("labels", "images").replace(".txt", ".jpg"))
        h, w = img.shape[:2]
        gts = []
        for r in lab.read_text().splitlines():
            if not r.strip():
                continue
            c, cx, cy, bw, bh = r.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            gts.append((COCO[int(c)], (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 1.0))

        y = [(d.class_name, d.bbox, d.score) for d in yolo.detect(img)]
        t = time.perf_counter()
        la_dets = locate.locate(img, CATEGORIES)
        latencies.append((time.perf_counter() - t) * 1000)
        la = [(d.class_name, d.bbox, 0.5) for d in la_dets if d.class_name in set(CATEGORIES)]

        def fuse(keep_cls):
            extra = [p for p in la if keep_cls(p[0])
                     and all(iou(p[1], q[1]) < 0.5 for q in y)]
            return y + extra

        match(y, gts, counts["yolo"])
        match(la, gts, counts["locate"])
        match(fuse(lambda c: c not in EnsembleDetector.MOVING_CLASSES), gts, counts["fused_live"])
        match(fuse(lambda c: True), gts, counts["fused_all"])
        if i % 20 == 0:
            print(f"  {i}/{len(chosen)}  LA {np.mean(latencies):.0f} ms/img", flush=True)

    result = {
        "images": len(chosen),
        "yolo_weights": yolo.weights_source,
        "yolo_conf": get_settings().yolo_conf,
        "locateanything_mode": args.mode,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "latency_ms": {"mean": round(float(np.mean(latencies)), 1),
                       "p50": round(float(np.percentile(latencies, 50)), 1),
                       "p95": round(float(np.percentile(latencies, 95)), 1)},
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20)
        if torch.cuda.is_available() else None,
        "minutes": round((time.time() - started) / 60, 1),
    }
    for k, c in counts.items():
        result[k] = summarise(c)

    yb, fb = result["yolo"]["_bags"], result["fused_live"]["_bags"]
    gain = (fb["recall"] or 0) - (yb["recall"] or 0)
    loss = (yb["precision"] or 0) - (fb["precision"] or 0)
    result["decision"] = {
        "rule": "enable ensemble if fused_live bag recall +>= 0.05 and bag precision -<= 0.05",
        "bag_recall_gain": round(gain, 4), "bag_precision_loss": round(loss, 4),
        "recommend_ensemble": bool(gain >= 0.05 and loss <= 0.05),
    }
    out = ROOT / "training" / "runs" / "locateanything_eval.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("latency_ms", "peak_vram_mb", "decision")}, indent=2))
    for k in ("yolo", "locate", "fused_live", "fused_all"):
        print(f"{k:11s} bags P {result[k]['_bags']['precision']}  R {result[k]['_bags']['recall']}   "
              f"all P {result[k]['_all']['precision']}  R {result[k]['_all']['recall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
