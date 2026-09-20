#!/usr/bin/env python
"""Weakly-supervised anomaly and violence heads (multiple-instance learning).

Only video-level labels exist for most of this data ("this video contains an
anomaly somewhere"), so each video is a *bag* of 32 segment features and the
model learns to score segments such that the highest-scoring segment of an
anomalous bag outranks the highest-scoring segment of a normal bag
(Sultani, Chen & Shah, CVPR 2018 - the method UCF-Crime was published with).

    python training/train_anomaly.py --task anomaly     # UCF-Crime
    python training/train_anomaly.py --task violence    # XD-Violence

Evaluation, stated plainly:
  anomaly   video-level AUC on UCF-Crime's official test anomalies (every
            anomaly video not in the recovered Anomaly_Train.txt) plus a
            held-out slice of normal videos; frame-level AUC on ShanghaiTech
            test (cross-dataset, never trained on) when its features exist.
  violence  video-level AUC and AP on a held-out 15% of XD-Violence, split by
            source film so clips of one movie never straddle train and test.

The official frame-level UCF-Crime and XD-Violence test annotations were not in
the downloaded archives, so those headline benchmark numbers are NOT reported
here - these are honest substitutes, and are labelled as such in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURES = Path("V:/sentinel_cache/features")
META = ROOT / "datasets" / "raw" / "ucf_crime_meta"
OUT = ROOT / "storage" / "models" / "anomaly"
MIN_TEST_AUC = 0.70     # clearly above chance (0.5) on the held-out test split
MAX_TEST_FPR = 0.15     # calibrated threshold must keep test false alarms bounded


# ------------------------------------------------------------------ model
def build_head(dim: int = 512):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(0.6),
        nn.Linear(512, 32), nn.ReLU(), nn.Dropout(0.6),
        nn.Linear(32, 1), nn.Sigmoid(),
    )


def mil_loss(anom_scores, norm_scores, lambda_smooth=8e-5, lambda_sparse=8e-5):
    """Ranking hinge on bag maxima + temporal smoothness + sparsity."""
    import torch

    rank = torch.relu(1.0 - anom_scores.max(dim=1).values + norm_scores.max(dim=1).values).mean()
    smooth = ((anom_scores[:, 1:] - anom_scores[:, :-1]) ** 2).sum(dim=1).mean()
    sparse = anom_scores.sum(dim=1).mean()
    return rank + lambda_smooth * smooth + lambda_sparse * sparse


# ------------------------------------------------------------------- data
def _load(folder: Path, ids: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for vid in ids:
        p = folder / f"{vid}.npy"
        if p.exists():
            f = np.load(p)
            if f.ndim == 2 and f.shape[0] == 32:
                out[vid] = f
    return out


def _hash_fraction(key: str) -> float:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_ucf() -> Tuple[List[str], List[str], List[str], List[str]]:
    """(train_anom, train_norm, test_anom, test_norm) video ids."""
    folder = FEATURES / "ucf_crime"
    available = {p.stem for p in folder.glob("*.npy")}
    train_list = (META / "Anomaly_Train.txt").read_text().split()
    train_ids = {Path(line).stem for line in train_list}

    def is_normal(vid: str) -> bool:
        return vid.lower().startswith("normal")

    anomalies = sorted(v for v in available if not is_normal(v))
    normals = sorted(v for v in available if is_normal(v))
    train_anom = [v for v in anomalies if v in train_ids]
    test_anom = [v for v in anomalies if v not in train_ids]          # official test set
    # No official normal test videos were downloaded: hold out 15% of normals.
    test_norm = [v for v in normals if _hash_fraction(v) < 0.15]
    train_norm = [v for v in normals if v not in set(test_norm)]
    return train_anom, train_norm, test_anom, test_norm


def split_xd() -> Tuple[List[str], List[str], List[str], List[str]]:
    folder = FEATURES / "xd_violence"
    available = sorted(p.stem for p in folder.glob("*.npy"))

    def film(vid: str) -> str:          # "Movie.Name.2001__#00-01-45_..._label_A"
        return vid.split("__")[0]

    def violent(vid: str) -> bool:
        m = re.search(r"label_([A-Z0-9\-]+)", vid)
        return bool(m) and not m.group(1).startswith("A")

    train_a, train_n, test_a, test_n = [], [], [], []
    for vid in available:
        held_out = _hash_fraction(film(vid)) < 0.15
        (test_a if violent(vid) else test_n).append(vid) if held_out else \
            (train_a if violent(vid) else train_n).append(vid)
    return train_a, train_n, test_a, test_n


# ---------------------------------------------------------------- metrics
def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, y_score))


def shanghaitech_frame_auc(model, device) -> Optional[dict]:
    """Frame-level AUC on ShanghaiTech test, from dense 16-frame clip scores."""
    import torch

    feat_dir = FEATURES / "shanghaitech_test_dense"
    raw = ROOT / "datasets" / "raw" / "shanghaitech"
    mask_dir = next((p for p in raw.rglob("test_frame_mask") if p.is_dir()), None)
    if not feat_dir.is_dir() or mask_dir is None:
        return None
    scores, labels = [], []
    for mask_path in sorted(mask_dir.glob("*.npy")):
        feat_path = feat_dir / f"{mask_path.stem}.npy"
        if not feat_path.exists():
            continue
        gt = np.load(mask_path).astype(int)
        feats = np.load(feat_path)
        with torch.inference_mode():
            clip_scores = model(torch.from_numpy(feats).float().to(device)).cpu().numpy().ravel()
        per_frame = np.repeat(clip_scores, 16)
        if len(per_frame) < len(gt):
            per_frame = np.concatenate([per_frame, np.full(len(gt) - len(per_frame), per_frame[-1])])
        scores.append(per_frame[: len(gt)])
        labels.append(gt)
    if not scores:
        return None
    y, s = np.concatenate(labels), np.concatenate(scores)
    return {"frame_auc": round(auc(y, s), 4), "videos": len(scores), "frames": int(len(y))}


# ------------------------------------------------------------------- train
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["anomaly", "violence"], required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=30, help="bags of each class per step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.task == "anomaly":
        train_a, train_n, test_a, test_n = split_ucf()
        folder = FEATURES / "ucf_crime"
        eval_note = ("video-level AUC on the official UCF-Crime test anomalies + 15% held-out "
                     "normals (official frame-level annotations not downloaded)")
    else:
        train_a, train_n, test_a, test_n = split_xd()
        folder = FEATURES / "xd_violence"
        eval_note = ("video-level AUC/AP on a held-out 15% of XD-Violence, split by source film "
                     "(official test set not downloaded)")

    feats = _load(folder, train_a + train_n + test_a + test_n)
    train_a = [v for v in train_a if v in feats]
    train_n = [v for v in train_n if v in feats]
    test_a = [v for v in test_a if v in feats]
    test_n = [v for v in test_n if v in feats]
    print(f"{args.task}: train {len(train_a)} positive / {len(train_n)} negative, "
          f"test {len(test_a)} / {len(test_n)}")
    if min(len(train_a), len(train_n), len(test_a), len(test_n)) == 0:
        print("not enough extracted features yet")
        return 1

    # Model selection on a validation slice carved out of TRAINING data. The
    # test split is touched exactly once, at the end.
    val_a = [v for v in train_a if _hash_fraction("val" + v) < 0.12]
    val_n = [v for v in train_n if _hash_fraction("val" + v) < 0.12]
    train_a = [v for v in train_a if v not in set(val_a)]
    train_n = [v for v in train_n if v not in set(val_n)]
    print(f"  model selection on {len(val_a)} / {len(val_n)} validation bags")

    model = build_head().to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=1e-3)

    def evaluate(pos=None, neg=None) -> dict:
        pos = test_a if pos is None else pos
        neg = test_n if neg is None else neg
        model.eval()
        ids = pos + neg
        y = np.array([1] * len(pos) + [0] * len(neg))
        with torch.inference_mode():
            x = torch.from_numpy(np.stack([feats[v] for v in ids])).float().to(device)
            s = model(x).squeeze(-1).max(dim=1).values.cpu().numpy()
        model.train()
        return {"video_auc": round(auc(y, s), 4), "video_ap": round(ap(y, s), 4)}

    best, best_state, history = -1.0, None, []
    steps_per_epoch = max(1, len(train_a) // args.batch)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        for _ in range(steps_per_epoch):
            a = rng.choice(train_a, args.batch, replace=len(train_a) < args.batch)
            n = rng.choice(train_n, args.batch, replace=len(train_n) < args.batch)
            xa = torch.from_numpy(np.stack([feats[v] for v in a])).float().to(device)
            xn = torch.from_numpy(np.stack([feats[v] for v in n])).float().to(device)
            loss = mil_loss(model(xa).squeeze(-1), model(xn).squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = evaluate(val_a, val_n)
        history.append({"epoch": epoch, "loss": round(float(loss), 4),
                        "val_auc": metrics["video_auc"]})
        if metrics["video_auc"] > best:
            best = metrics["video_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss {float(loss):.4f}  "
                  f"val AUC {metrics['video_auc']:.4f}", flush=True)

    model.load_state_dict(best_state)
    final = evaluate()

    # Alarm threshold: calibrated on VALIDATION normals so that ~5% of normal
    # videos would raise an alarm, then that operating point is reported on the
    # untouched test split. This is the number the live agent uses.
    def max_scores(ids):
        model.eval()
        with torch.inference_mode():
            x = torch.from_numpy(np.stack([feats[v] for v in ids])).float().to(device)
            return model(x).squeeze(-1).max(dim=1).values.cpu().numpy()

    threshold = float(np.quantile(max_scores(val_n), 0.95))
    test_pos, test_neg = max_scores(test_a), max_scores(test_n)
    operating_point = {
        "threshold": round(threshold, 4),
        "calibrated_on": "95th percentile of validation-normal video scores",
        "test_true_positive_rate": round(float((test_pos >= threshold).mean()), 4),
        "test_false_positive_rate": round(float((test_neg >= threshold).mean()), 4),
    }
    result = {
        "task": args.task,
        "backbone": "r2plus1d_18 (Kinetics-400), 32 segments x 16 frames",
        "evaluation": eval_note,
        **final,
        "threshold": operating_point["threshold"],
        "operating_point": operating_point,
        "train_positive": len(train_a), "train_negative": len(train_n),
        "test_positive": len(test_a), "test_negative": len(test_n),
        "epochs": args.epochs,
        "train_seconds": round(time.time() - started, 1),
        "model_selection": "best epoch on a validation slice of the training data; "
                           "the test split was evaluated once, after selection",
    }
    if args.task == "anomaly":
        cross = shanghaitech_frame_auc(model, device)
        result["shanghaitech_cross_dataset"] = cross or "features not extracted"

    # Promotion gate: the live agent raises alarms from this head, so it only
    # goes live if it is clearly better than chance on the untouched test split
    # AND its calibrated threshold keeps false alarms on test normals bounded.
    promote = (final["video_auc"] >= MIN_TEST_AUC
               and operating_point["test_false_positive_rate"] <= MAX_TEST_FPR)
    result["promotion"] = {"promoted": promote, "min_test_auc": MIN_TEST_AUC,
                           "max_test_fpr": MAX_TEST_FPR}

    runs = ROOT / "training" / "runs" / "anomaly"
    runs.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": result}, runs / f"{args.task}_head.pt")
    report = json.dumps({**result, "history": history}, indent=2)
    (runs / f"{args.task}_metrics.json").write_text(report, encoding="utf-8")
    if promote:
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "meta": result}, OUT / f"{args.task}_head.pt")
        (OUT / f"{args.task}_metrics.json").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("PROMOTED to storage/models/anomaly" if promote else
          "NOT promoted - kept in training/runs/anomaly only", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
