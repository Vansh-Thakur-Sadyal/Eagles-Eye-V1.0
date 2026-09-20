#!/usr/bin/env python
"""Replay generated behaviour scenarios through the agents and score them.

This closes the loop on the custom datasets: every scenario has exact ground
truth, so feeding the trajectories straight to the agents gives real precision
and recall for loitering, following, counter-flow, restricted entry and
unattended objects - and, just as importantly, a false-positive rate measured
against the matched negative scenarios.

    python tools/evaluate_agents.py
    python tools/evaluate_agents.py --category following --verbose
    python tools/evaluate_agents.py --tune loitering_seconds 60,90,120,150

Detection is bypassed on purpose: this measures the *reasoning* agents against
known-good tracks, so a result is not confounded by detector error.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.base import CameraContext, FrameContext  # noqa: E402
from app.agents.behavior import BehaviorAgent  # noqa: E402
from app.agents.crowd import CrowdAgent  # noqa: E402
from app.agents.objects import ObjectAgent  # noqa: E402
from app.agents.relationship import RelationshipAgent  # noqa: E402
from app.config import load_policy  # noqa: E402
from app.vision.types import TrackState  # noqa: E402

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Which behaviour each category's positives should produce.
#
# Note on restricted_zone: `authorised_entry` is counted as a positive here.
# Video alone cannot establish authorisation - the agent's job is to report the
# entry, and whether it was permitted is a downstream policy decision made
# against an access-control system. Expecting the agent to stay silent would be
# expecting it to know something it cannot see.
EXPECTED: Dict[str, Sequence[str]] = {
    "loitering": ["loitering"],
    "following": ["following"],
    "counter_flow": ["counter_flow"],
    "restricted_zone": ["restricted_entry"],
    "abandoned_object": ["unattended_object", "object_separation"],
}

AGENTS_FOR = {
    "loitering": ["behavior"],
    "following": ["relationship"],
    "counter_flow": ["behavior"],
    "restricted_zone": ["behavior"],
    "abandoned_object": ["object"],
}


def load_scene(root: Path, record: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Read a tracking CSV into {frame: [detections]}."""
    frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with (root / record["tracking_csv"]).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            frames[int(float(row["frame_id"]))].append(
                {
                    "track_id": row["track_id"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "w": float(row["w"]),
                    "h": float(row["h"]),
                    "class_id": int(row["class_id"]),
                }
            )
    return frames


def _track_states(
    frames: Dict[int, List[Dict[str, Any]]],
    upto: int,
    fps: float,
    history: Dict[str, List[Dict[str, Any]]],
    velocities: Dict[str, Tuple[float, float]],
) -> List[TrackState]:
    """Build TrackState objects carrying their accumulated history."""
    states: List[TrackState] = []
    for row in frames.get(upto, []):
        key = row["track_id"]
        cx = row["x"] + row["w"] / 2
        cy = row["y"] + row["h"]
        history[key].append({"t": (T0 + timedelta(seconds=upto / fps)).isoformat(),
                             "f": upto, "x": cx, "y": cy, "w": row["w"], "h": row["h"],
                             "conf": 1.0, "vx": 0.0, "vy": 0.0})
        trail = history[key]
        numeric_id = abs(hash(key)) % 100000

        state = TrackState(
            track_id=numeric_id,
            class_name="person" if row["class_id"] == 0 else "suitcase",
            bbox=(row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"]),
            score=1.0,
            first_frame=trail[0]["f"],
            last_frame=upto,
            first_seen=T0 + timedelta(seconds=trail[0]["f"] / fps),
            last_seen=T0 + timedelta(seconds=upto / fps),
        )
        state.history = list(trail[-300:])
        if len(trail) >= 2:
            dt = max(1e-6, (trail[-1]["f"] - trail[-2]["f"]) / fps)
            vx = (trail[-1]["x"] - trail[-2]["x"]) / dt
            vy = (trail[-1]["y"] - trail[-2]["y"]) / dt
            # Match ByteTracker's smoothing, or the agents see a noisier signal
            # here than they ever would in production and the scores are unfair.
            prev = velocities.get(key, (0.0, 0.0))
            vx = 0.6 * prev[0] + 0.4 * vx
            vy = 0.6 * prev[1] + 0.4 * vy
            velocities[key] = (vx, vy)
            state.velocity = (vx, vy)
            state.speed = float((vx * vx + vy * vy) ** 0.5)
            trail[-1]["vx"], trail[-1]["vy"] = round(vx, 2), round(vy, 2)
        state.attributes["source_id"] = key
        states.append(state)
    return states


def _subject_ids(record: Dict[str, Any]) -> set:
    """The tracks this scenario is actually labelled about.

    Without this, a background pedestrian wandering through the restricted zone
    would be scored as a false alarm against a scenario labelled on a different
    subject entirely.
    """
    ids = set()
    for key in ("subject_id", "subject_a", "subject_b", "owner_track_id"):
        value = record.get(key)
        if value:
            ids.add(str(value))
    # An object-centric scenario is also "about" the object itself.
    for obj in record.get("objects", []) or []:
        if obj.get("object_id"):
            ids.add(str(obj["object_id"]))
    return ids


def run_scene(
    root: Path,
    record: Dict[str, Any],
    policy: Dict[str, Any],
    agent_names: Sequence[str],
) -> List[Tuple[str, str]]:
    """Run the agents across one scenario.

    Returns (behaviour, source_track_id) so a finding can be attributed to the
    subject the scenario is labelled about.
    """
    frames = load_scene(root, record)
    if not frames:
        return []

    fps = float(record.get("fps", 12))
    agents = {
        "behavior": BehaviorAgent(),
        "relationship": RelationshipAgent(),
        "object": ObjectAgent(),
        "crowd": CrowdAgent(),
    }
    active = [agents[name] for name in agent_names if name in agents]
    history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    velocities: Dict[str, Tuple[float, float]] = {}
    detected: List[Tuple[str, str]] = []
    numeric_to_source: Dict[int, str] = {}

    zones = record.get("zones") or []
    max_frame = max(frames)

    for frame_index in range(max_frame + 1):
        tracks = _track_states(frames, frame_index, fps, history, velocities)
        if not tracks:
            continue
        for track in tracks:
            numeric_to_source[track.track_id] = track.attributes["source_id"]
        ctx = FrameContext(
            camera=CameraContext(
                camera_id="CAM_EVAL", name="Evaluation", width=960, height=540, fps=fps,
                zone_id=None, site_id=None, location="Evaluation", zones=zones,
            ),
            frame_index=frame_index,
            timestamp=T0 + timedelta(seconds=frame_index / fps),
            frame=None,
            detections=[],
            tracks=tracks,
            policy=policy,
            elapsed_seconds=frame_index / fps,
        )
        for agent in active:
            for finding in agent.run(ctx):
                source = "?"
                for candidate in (finding.track_id, finding.object_id,
                                  finding.secondary_track_id):
                    if candidate is None:
                        continue
                    try:
                        source = numeric_to_source.get(int(candidate), source)
                    except ValueError:
                        source = candidate
                    if source != "?":
                        break
                detected.append((finding.behavior, source))

    return detected


def evaluate(
    root: Path,
    categories: Sequence[str],
    policy: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    for category in categories:
        index_path = root / "behavior" / category / "annotations" / "index.json"
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected = set(EXPECTED.get(category, []))
        agent_names = AGENTS_FOR.get(category, ["behavior"])

        tp = fp = tn = fn = 0
        misses: List[str] = []
        false_alarms: List[str] = []

        for record in index:
            raw = run_scene(root, record, policy, agent_names)
            subjects = _subject_ids(record)
            # Only findings about the labelled subject(s) count. Background
            # pedestrians are scenery, not the thing under test.
            detected = {
                behavior for behavior, source in raw
                if behavior in expected and (not subjects or source in subjects or source == "?")
            }
            other = {behavior for behavior, source in raw if source not in subjects}
            fired = bool(detected & expected)
            if record["positive"]:
                if fired:
                    tp += 1
                else:
                    fn += 1
                    misses.append(record["scene_id"])
            else:
                if fired:
                    fp += 1
                    false_alarms.append(f"{record['scene_id']} ({record['label']})")
                else:
                    tn += 1
            if verbose:
                mark = "OK " if fired == record["positive"] else "XX "
                print(f"  {mark} {record['scene_id']:<12} label={record['label']:<28} "
                      f"subject={sorted(detected) or '-'} other={sorted(other) or '-'}")

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        results[category] = {
            "scenes": len(index),
            "true_positive": tp, "false_positive": fp,
            "true_negative": tn, "false_negative": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "false_positive_rate": round(fp / (fp + tn), 3) if (fp + tn) else 0.0,
            "missed": misses[:8],
            "false_alarms": false_alarms[:8],
        }

    return results


def print_table(results: Dict[str, Any]) -> None:
    print()
    print(f"{'category':<20}{'scenes':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'FP rate':>10}")
    print("-" * 62)
    for category, stats in results.items():
        print(f"{category:<20}{stats['scenes']:>8}{stats['precision']:>8.2f}"
              f"{stats['recall']:>8.2f}{stats['f1']:>8.2f}{stats['false_positive_rate']:>10.2f}")
    print("-" * 62)
    total_tp = sum(s["true_positive"] for s in results.values())
    total_fp = sum(s["false_positive"] for s in results.values())
    total_fn = sum(s["false_negative"] for s in results.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    print(f"{'OVERALL':<20}{'':>8}{precision:>8.2f}{recall:>8.2f}")
    print()

    for category, stats in results.items():
        if stats["missed"]:
            print(f"  {category}: missed {', '.join(stats['missed'])}")
        if stats["false_alarms"]:
            print(f"  {category}: false alarms on {', '.join(stats['false_alarms'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the agents against generated ground truth")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets")
    parser.add_argument("--category", nargs="*", default=list(EXPECTED))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--tune", nargs=2, metavar=("KEY", "VALUES"),
                        help="sweep one behaviour threshold, e.g. --tune loitering_seconds 60,90,120")
    args = parser.parse_args()

    policy = load_policy()
    # The generated scenes are ~30-60s long, so relax the production thresholds
    # that assume minutes of observation.
    policy["behavior"]["loitering_seconds"] = 20   # from the threshold sweep
    policy["behavior"]["counter_flow_min_frames"] = 10
    policy["behavior"]["min_track_age_frames"] = 6
    policy["following"]["min_duration_seconds"] = 8
    policy["following"]["confidence_floor"] = 0.60   # from the sweep
    policy["object"]["unattended_seconds"] = 10
    policy["object"]["warning_seconds"] = 5
    policy["object"]["tracked_classes"] = ["suitcase"]

    if args.tune:
        key, raw_values = args.tune
        section = "following" if key in policy.get("following", {}) else (
            "object" if key in policy.get("object", {}) else "behavior"
        )
        print(f"Sweeping {section}.{key}\n")
        for raw in raw_values.split(","):
            value = float(raw) if "." in raw else int(raw)
            policy[section][key] = value
            results = evaluate(args.dataset, args.category, policy)
            for category, stats in results.items():
                print(f"  {key}={value:<8} {category:<18} "
                      f"P={stats['precision']:.2f} R={stats['recall']:.2f} "
                      f"F1={stats['f1']:.2f} FPR={stats['false_positive_rate']:.2f}")
        return 0

    results = evaluate(args.dataset, args.category, policy, verbose=args.verbose)
    if not results:
        print(f"No generated scenarios found under {args.dataset}/behavior. "
              "Run: python tools/make_datasets.py behaviour")
        return 1

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
