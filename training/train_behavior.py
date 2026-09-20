#!/usr/bin/env python
"""Train a behaviour classifier on trajectory features.

    python training/train_behavior.py --category following
    python training/train_behavior.py --all --report

This is the model that learns *from your generated scenarios*, and it is the
one place where simulated data is genuinely the right tool: the agents operate
on trajectories, so a classifier trained on trajectory features learns the same
representation the production system uses.

It trains a small gradient-boosted classifier over hand-derived features rather
than a deep net, because with a few hundred scenes that is what actually works,
it trains in seconds, and - importantly for this platform - the feature
importances are readable, so an operator can be told *why* a behaviour scored.

The output is a model plus the operating point, which you can compare against
the rule-based agent using tools/evaluate_agents.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUT = ROOT / "storage" / "models"


# ------------------------------------------------------------------ features
def _load_tracks(csv_path: Path) -> Dict[str, List[Tuple[int, float, float]]]:
    tracks: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["class_id"]) != 0:
                continue
            frame = int(float(row["frame_id"]))
            cx = float(row["x"]) + float(row["w"]) / 2
            cy = float(row["y"]) + float(row["h"])
            tracks[row["track_id"]].append((frame, cx, cy))
    for key in tracks:
        tracks[key].sort()
    return tracks


def _path_features(points: Sequence[Tuple[int, float, float]], fps: float) -> Dict[str, float]:
    """Motion descriptors for one subject."""
    if len(points) < 4:
        return {}
    xy = np.array([(p[1], p[2]) for p in points], dtype=np.float64)
    deltas = np.diff(xy, axis=0)
    step = np.linalg.norm(deltas, axis=1)
    speeds = step * fps

    distance = float(step.sum())
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    centroid = xy.mean(axis=0)
    radius = np.linalg.norm(xy - centroid, axis=1)

    headings = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0]))
    heading_change = np.abs(np.diff(headings))
    heading_change = np.minimum(heading_change, 360 - heading_change)

    duration = len(points) / fps
    return {
        "duration_s": duration,
        "path_length": distance,
        "displacement": displacement,
        "straightness": displacement / distance if distance > 1e-6 else 0.0,
        "mean_speed": float(speeds.mean()),
        "max_speed": float(speeds.max()),
        "speed_std": float(speeds.std()),
        "stationary_fraction": float((speeds < 12).mean()),
        "mean_radius": float(radius.mean()),
        "max_radius": float(radius.max()),
        "radius_std": float(radius.std()),
        "confinement": float(radius.mean() / (distance + 1e-6)),
        "mean_turn": float(heading_change.mean()) if len(heading_change) else 0.0,
        "sharp_turns": float((heading_change > 45).sum()),
        "heading_std": float(headings.std()) if len(headings) > 1 else 0.0,
    }


def _pair_features(a, b, fps: float) -> Dict[str, float]:
    """Descriptors for a pair - what separates following from co-travel."""
    common = sorted(set(f for f, _, _ in a) & set(f for f, _, _ in b))
    if len(common) < 10:
        return {}
    pa = {f: (x, y) for f, x, y in a}
    pb = {f: (x, y) for f, x, y in b}
    A = np.array([pa[f] for f in common])
    B = np.array([pb[f] for f in common])

    separation = np.linalg.norm(A - B, axis=1)

    # Along/cross decomposition against the leader's direction of travel -
    # the signal that distinguishes "behind" from "beside".
    window = 5
    along, cross = [], []
    for i in range(window, len(common)):
        forward = A[i] - A[i - window]
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            continue
        forward = forward / norm
        offset = B[i] - A[i]
        offset_norm = np.linalg.norm(offset)
        if offset_norm < 1e-6:
            continue
        unit = offset / offset_norm
        along.append(float(-np.dot(unit, forward)))
        # 2-D cross product by hand (np.cross on 2-vectors is deprecated)
        cross.append(float(abs(unit[0] * forward[1] - unit[1] * forward[0])))

    # Does B occupy where A was? Compare best lagged distance to zero-lag.
    zero_lag = float(separation.mean())
    best_lagged = zero_lag
    for lag in range(3, min(30, len(common) // 2)):
        d = float(np.linalg.norm(A[: len(A) - lag] - B[lag:], axis=1).mean())
        best_lagged = min(best_lagged, d)

    return {
        "mean_separation": zero_lag,
        "separation_std": float(separation.std()),
        "separation_cv": float(separation.std() / zero_lag) if zero_lag > 1e-6 else 1.0,
        "min_separation": float(separation.min()),
        "max_separation": float(separation.max()),
        "mean_along": float(np.mean(along)) if along else 0.0,
        "mean_cross": float(np.mean(cross)) if cross else 1.0,
        "behind_fraction": float(np.mean(np.array(along) > 0.5)) if along else 0.0,
        "lag_gain": float(1.0 - best_lagged / zero_lag) if zero_lag > 1e-6 else 0.0,
        "co_duration_s": len(common) / fps,
    }


def build_dataset(category: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    index_path = DATASETS / "behavior" / category / "annotations" / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"no generated scenarios for '{category}'. "
            f"Run: python tools/make_datasets.py behaviour"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))

    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    scene_ids: List[str] = []

    for record in index:
        tracks = _load_tracks(DATASETS / record["tracking_csv"])
        fps = float(record.get("fps", 12))

        if category == "following":
            a = tracks.get(record.get("subject_a", ""))
            b = tracks.get(record.get("subject_b", ""))
            if not a or not b:
                continue
            features = _pair_features(a, b, fps)
            features.update({f"a_{k}": v for k, v in _path_features(a, fps).items()})
            features.update({f"b_{k}": v for k, v in _path_features(b, fps).items()})
        else:
            # Scenario categories name their subject differently: an abandoned
            # -object scene is labelled on the owner, not a "subject_id".
            subject_key = (
                record.get("subject_id")
                or record.get("owner_track_id")
                or ""
            )
            subject = tracks.get(subject_key)
            if not subject:
                continue
            features = _path_features(subject, fps)
            # Context: how the subject compares to everyone else in the scene.
            others = [t for k, t in tracks.items() if k != subject_key]
            if others:
                other_headings = []
                for other in others:
                    pts = np.array([(p[1], p[2]) for p in other])
                    if len(pts) > 6:
                        d = pts[-1] - pts[0]
                        if np.linalg.norm(d) > 1e-6:
                            other_headings.append(math.degrees(math.atan2(d[1], d[0])))
                if other_headings:
                    sx = np.mean(np.cos(np.radians(other_headings)))
                    sy = np.mean(np.sin(np.radians(other_headings)))
                    crowd_heading = math.degrees(math.atan2(sy, sx))
                    pts = np.array([(p[1], p[2]) for p in subject])
                    d = pts[-1] - pts[0]
                    subject_heading = math.degrees(math.atan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0
                    deviation = abs((subject_heading - crowd_heading + 180) % 360 - 180)
                    features["crowd_deviation_deg"] = float(deviation)
                    features["crowd_size"] = float(len(others))

        if not features:
            continue
        rows.append(features)
        labels.append(1 if record["positive"] else 0)
        scene_ids.append(record["scene_id"])

    if not rows:
        raise RuntimeError(f"no usable scenes for '{category}'")

    names = sorted({k for row in rows for k in row})
    X = np.array([[row.get(n, 0.0) for n in names] for row in rows], dtype=np.float64)
    y = np.array(labels, dtype=np.int64)
    return X, y, names, scene_ids


# ------------------------------------------------------------------ training
def train(category: str, *, seed: int = 20260919, folds: int = 5) -> Dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import StratifiedKFold

    X, y, names, scene_ids = build_dataset(category)
    positives = int(y.sum())
    print(f"\n{category}: {len(y)} scenes ({positives} positive, {len(y) - positives} negative), "
          f"{len(names)} features")

    if positives < 3 or (len(y) - positives) < 3:
        return {"category": category, "error": "not enough scenes in one class to train"}

    # Cross-validated, because a few hundred scenes is far too few to trust a
    # single split.
    n_splits = min(folds, positives, len(y) - positives)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: Dict[str, List[float]] = defaultdict(list)
    oof = np.zeros(len(y))

    for train_idx, test_idx in splitter.split(X, y):
        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=seed,
        )
        model.fit(X[train_idx], y[train_idx])
        probability = model.predict_proba(X[test_idx])[:, 1]
        oof[test_idx] = probability
        prediction = (probability >= 0.5).astype(int)
        scores["accuracy"].append(accuracy_score(y[test_idx], prediction))
        scores["precision"].append(precision_score(y[test_idx], prediction, zero_division=0))
        scores["recall"].append(recall_score(y[test_idx], prediction, zero_division=0))
        scores["f1"].append(f1_score(y[test_idx], prediction, zero_division=0))
        if len(set(y[test_idx])) > 1:
            scores["roc_auc"].append(roc_auc_score(y[test_idx], probability))

    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}
    print(f"  {n_splits}-fold cross-validation:")
    for key, (mean, std) in summary.items():
        print(f"    {key:<12} {mean:.3f} +/- {std:.3f}")

    final = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.9, random_state=seed,
    ).fit(X, y)

    importances = sorted(zip(names, final.feature_importances_), key=lambda kv: -kv[1])
    print("  most informative features:")
    for name, weight in importances[:6]:
        print(f"    {name:<24} {weight:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    model_path = OUT / f"behavior_{category}.pkl"
    with model_path.open("wb") as fh:
        pickle.dump({"model": final, "feature_names": names, "category": category}, fh)
    print(f"  saved -> {model_path}")

    return {
        "category": category,
        "scenes": len(y),
        "positives": positives,
        "features": len(names),
        "cross_validation": {k: {"mean": m, "std": s} for k, (m, s) in summary.items()},
        "top_features": [{"name": n, "importance": float(w)} for n, w in importances[:10]],
        "model_path": str(model_path),
        "caveat": (
            "Trained on simulated trajectories with exact ground truth. These scores "
            "measure whether the features separate the scripted behaviours, not "
            "real-world accuracy. Validate against annotated real footage before "
            "trusting them operationally."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train behaviour classifiers")
    parser.add_argument("--category", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--report", action="store_true", help="write a JSON report")
    args = parser.parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("scikit-learn is required:  pip install scikit-learn")
        return 1

    categories = (
        ["loitering", "following", "counter_flow", "restricted_zone", "abandoned_object"]
        if args.all or not args.category
        else [args.category]
    )

    results = []
    for category in categories:
        try:
            results.append(train(category))
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"\n{category}: skipped - {exc}")

    if args.report and results:
        path = OUT / "behavior_training_report.json"
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nreport -> {path}")

    print("\nCompare against the rule-based agents with:")
    print("  python tools/evaluate_agents.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
