"""Scripted behaviour scenarios with exact ground truth.

Generates the five custom behaviour categories the write-up calls for, as
trajectories (the representation the agents actually consume) and optionally as
rendered video (for end-to-end pipeline testing).

Each scenario is built from a small set of motion primitives so the ground truth
is exact by construction rather than annotated after the fact.  Crucially every
category ships with matched NEGATIVE scenarios - a person waiting normally, two
friends walking together, a bag whose owner stays beside it - because a model
trained only on positives learns to say "yes" to everything.
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

FRAME_W, FRAME_H = 960, 540
FPS = 12


# --------------------------------------------------------------- primitives
@dataclass
class Walker:
    """One simulated subject."""

    subject_id: str
    role: str
    path: List[Point] = field(default_factory=list)
    height: float = 110.0
    present_from: int = 0

    def at(self, frame: int) -> Optional[Point]:
        idx = frame - self.present_from
        if idx < 0 or idx >= len(self.path):
            return None
        return self.path[idx]


def _jitter(rng: random.Random, amount: float = 1.2) -> float:
    return rng.uniform(-amount, amount)


def straight(start: Point, heading_deg: float, speed: float, frames: int,
             rng: random.Random) -> List[Point]:
    rad = math.radians(heading_deg)
    dx, dy = math.cos(rad) * speed, math.sin(rad) * speed
    out: List[Point] = []
    x, y = start
    for _ in range(frames):
        x += dx + _jitter(rng, 0.8)
        y += dy + _jitter(rng, 0.5)
        out.append((x, y))
    return out


def waypoints(points: Sequence[Point], frames_per_leg: int, rng: random.Random) -> List[Point]:
    """Walk through a sequence of waypoints at a steady pace."""
    out: List[Point] = []
    for i in range(len(points) - 1):
        (x0, y0), (x1, y1) = points[i], points[i + 1]
        for f in range(frames_per_leg):
            t = f / max(1, frames_per_leg - 1)
            out.append((x0 + (x1 - x0) * t + _jitter(rng), y0 + (y1 - y0) * t + _jitter(rng)))
    return out


def dwell(centre: Point, frames: int, radius: float, rng: random.Random) -> List[Point]:
    """Confined movement - the signature of loitering, or of ordinary waiting."""
    out: List[Point] = []
    for f in range(frames):
        angle = rng.uniform(0, math.tau)
        r = rng.uniform(0, radius)
        out.append((centre[0] + math.cos(angle) * r, centre[1] + math.sin(angle) * r))
    return out


def follow(leader: Sequence[Point], lag_frames: int, offset: Point,
           rng: random.Random) -> List[Point]:
    """Trail a leader by a time lag - mirrors their turns after a delay."""
    out: List[Point] = []
    for f in range(len(leader)):
        src = leader[max(0, f - lag_frames)]
        out.append((src[0] + offset[0] + _jitter(rng), src[1] + offset[1] + _jitter(rng)))
    return out


def abreast(leader: Sequence[Point], separation: float, rng: random.Random) -> List[Point]:
    """Walk alongside with no lag - two companions, the key negative case.

    The offset must be perpendicular to the direction of travel at each step. A
    fixed offset would silently become a *trailing* offset whenever the route
    turns, which would make this "negative" case an unlabelled positive.
    """
    out: List[Point] = []
    for i, p in enumerate(leader):
        nxt = leader[min(i + 1, len(leader) - 1)]
        dx, dy = nxt[0] - p[0], nxt[1] - p[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            px, py = 0.0, 1.0
        else:
            px, py = -dy / norm, dx / norm      # unit normal
        out.append((p[0] + px * separation + _jitter(rng), p[1] + py * separation + _jitter(rng)))
    return out


# ---------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    scene_id: str
    category: str
    label: str                        # the ground-truth class
    positive: bool
    walkers: List[Walker]
    frames: int
    annotations: Dict[str, Any] = field(default_factory=dict)
    zones: List[Dict[str, Any]] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)


RESTRICTED_ZONE = {
    "id": "ZONE_RESTRICTED",
    "name": "Restricted Apron",
    "zone_type": "restricted",
    "polygon": [[0.70, 0.52], [0.97, 0.52], [0.97, 0.94], [0.70, 0.94]],
}


def _crowd(rng: random.Random, n: int, frames: int, heading: float = 0.0) -> List[Walker]:
    walkers = []
    for i in range(n):
        start = (rng.uniform(-150, 200), rng.uniform(FRAME_H * 0.36, FRAME_H * 0.86))
        walkers.append(
            Walker(
                subject_id=f"T_BG{i:02d}",
                role="background",
                path=straight(start, heading, rng.uniform(2.2, 3.4), frames, rng),
                height=rng.uniform(96, 126),
            )
        )
    return walkers


# --- loitering --------------------------------------------------------------
def loitering_scenarios(rng: random.Random, count: int) -> List[Scenario]:
    out: List[Scenario] = []
    for i in range(count):
        frames = rng.randint(420, 600)
        positive = i % 2 == 0
        centre = (rng.uniform(220, 700), rng.uniform(250, 420))

        if positive:
            # Long confined dwell - the target behaviour.
            path = dwell(centre, frames, radius=rng.uniform(18, 40), rng=rng)
            label = "loitering"
            note = "confined dwell well past the threshold"
        else:
            # Normal waiting then leaving, or browsing along a line.
            style = rng.choice(["waiting_then_leaves", "browsing", "queue"])
            if style == "waiting_then_leaves":
                hold = int(frames * 0.35)
                path = dwell(centre, hold, radius=25, rng=rng) + straight(
                    centre, rng.uniform(-30, 30), 3.0, frames - hold, rng
                )
            elif style == "browsing":
                path = waypoints(
                    [(centre[0] - 180, centre[1]), (centre[0], centre[1] - 40),
                     (centre[0] + 180, centre[1] + 20)],
                    frames // 2, rng,
                )
            else:
                # A queue advances. It must travel further than the loitering
                # radius, or it is loitering and the label would be wrong.
                path = waypoints(
                    [(centre[0], centre[1]), (centre[0] - 260, centre[1] - 40)], frames, rng
                )
            label = f"normal_{style}"
            note = "benign control case"

        walkers = [Walker(f"T_S{i:03d}", "subject", path)] + _crowd(rng, 4, frames)
        out.append(
            Scenario(
                scene_id=f"LOIT_{i:04d}",
                category="loitering",
                label=label,
                positive=positive,
                walkers=walkers,
                frames=frames,
                annotations={
                    "subject_id": f"T_S{i:03d}",
                    "dwell_centre": [round(c, 1) for c in centre],
                    "note": note,
                },
            )
        )
    return out


# --- following --------------------------------------------------------------
def following_scenarios(rng: random.Random, count: int) -> List[Scenario]:
    out: List[Scenario] = []
    for i in range(count):
        frames = rng.randint(420, 620)
        positive = i % 2 == 0
        legs = frames // 4
        route = [
            (rng.uniform(60, 160), rng.uniform(200, 360)),
            (rng.uniform(380, 520), rng.uniform(180, 300)),
            (rng.uniform(560, 700), rng.uniform(330, 460)),
            (rng.uniform(760, 900), rng.uniform(250, 400)),
            (rng.uniform(860, 940), rng.uniform(180, 300)),
        ]
        leader_path = waypoints(route, legs, rng)

        if positive:
            lag = rng.randint(10, 22)
            second = follow(leader_path, lag, (-rng.uniform(55, 110), rng.uniform(15, 55)), rng)
            label, note = "following", f"trails with a {lag}-frame lag and mirrors turns"
        else:
            style = rng.choice(["companions", "crossing", "independent"])
            if style == "companions":
                second = abreast(leader_path, rng.uniform(45, 75), rng)
                note = "walking side by side, no lag"
            elif style == "crossing":
                second = straight((rng.uniform(400, 600), 60), 90, 3.0, len(leader_path), rng)
                note = "crossing the leader's path once"
            else:
                second = waypoints(
                    [(900, 420), (600, 200), (300, 380), (80, 220)], legs, rng
                )
                note = "independent route"
            label = f"normal_{style}"
            lag = 0

        walkers = [
            Walker(f"T_L{i:03d}", "leader", leader_path),
            Walker(f"T_F{i:03d}", "second", second),
        ] + _crowd(rng, 3, frames)

        out.append(
            Scenario(
                scene_id=f"FOLW_{i:04d}",
                category="following",
                label=label,
                positive=positive,
                walkers=walkers,
                frames=min(frames, len(leader_path)),
                annotations={
                    "subject_a": f"T_L{i:03d}",
                    "subject_b": f"T_F{i:03d}",
                    "lag_frames": lag,
                    "relationship": "following" if positive else "normal",
                    "note": note,
                },
            )
        )
    return out


# --- counter-flow -----------------------------------------------------------
def counterflow_scenarios(rng: random.Random, count: int) -> List[Scenario]:
    out: List[Scenario] = []
    for i in range(count):
        frames = rng.randint(300, 420)
        positive = i % 2 == 0
        crowd_heading = rng.choice([0.0, 180.0])
        crowd = _crowd(rng, rng.randint(6, 10), frames, heading=crowd_heading)

        if positive:
            start = (FRAME_W - 60.0, rng.uniform(260, 420)) if crowd_heading == 0 else (60.0, rng.uniform(260, 420))
            subject_path = straight(start, crowd_heading + 180, rng.uniform(2.4, 3.2), frames, rng)
            label, note = "counter_flow", "sustained movement against the prevailing flow"
        else:
            start = (rng.uniform(-80, 120), rng.uniform(260, 420))
            subject_path = straight(start, crowd_heading + rng.uniform(-25, 25),
                                    rng.uniform(2.4, 3.2), frames, rng)
            label, note = "normal_with_flow", "moving with the crowd"

        walkers = [Walker(f"T_S{i:03d}", "subject", subject_path)] + crowd
        out.append(
            Scenario(
                scene_id=f"CFLW_{i:04d}",
                category="counter_flow",
                label=label,
                positive=positive,
                walkers=walkers,
                frames=frames,
                annotations={
                    "subject_id": f"T_S{i:03d}",
                    "crowd_heading_deg": crowd_heading,
                    "crowd_size": len(crowd),
                    "note": note,
                },
            )
        )
    return out


# --- restricted zone --------------------------------------------------------
def restricted_scenarios(rng: random.Random, count: int) -> List[Scenario]:
    out: List[Scenario] = []
    zx1, zy1 = RESTRICTED_ZONE["polygon"][0]
    zx2, zy2 = RESTRICTED_ZONE["polygon"][2]
    inside = ((zx1 + zx2) / 2 * FRAME_W, (zy1 + zy2) / 2 * FRAME_H)

    for i in range(count):
        frames = rng.randint(260, 380)
        style = ["entry", "tailgating", "boundary_pass", "authorised_entry"][i % 4]
        # An authorised entry is still a zone entry: video cannot verify
        # authorisation, so the agent is expected to report it and the
        # access decision is made downstream. Only the boundary pass, where
        # nobody crosses the line, is a true negative.
        positive = style != "boundary_pass"

        if style in ("entry", "tailgating", "authorised_entry"):
            path = waypoints([(120, 480), (420, 470), inside], frames // 2, rng)
        else:
            # walks along the boundary without crossing it
            path = waypoints(
                [(120, 480), (zx1 * FRAME_W - 30, zy1 * FRAME_H + 40),
                 (zx1 * FRAME_W - 25, zy2 * FRAME_H - 30)],
                frames // 2, rng,
            )

        walkers = [Walker(f"T_S{i:03d}", "subject", path)]
        if style == "tailgating":
            walkers.append(Walker(f"T_T{i:03d}", "tailgater", follow(path, 8, (-45.0, 12.0), rng)))
        walkers += _crowd(rng, 3, frames)

        out.append(
            Scenario(
                scene_id=f"RSTR_{i:04d}",
                category="restricted_zone",
                label=style,
                positive=positive,
                walkers=walkers,
                frames=min(frames, len(path)),
                zones=[RESTRICTED_ZONE],
                annotations={
                    "subject_id": f"T_S{i:03d}",
                    "zone_id": RESTRICTED_ZONE["id"],
                    "authorization_status": "authorised" if style == "authorised_entry" else "unverified",
                    "event_type": style,
                },
            )
        )
    return out


# --- abandoned object -------------------------------------------------------
def abandoned_scenarios(rng: random.Random, count: int) -> List[Scenario]:
    out: List[Scenario] = []
    for i in range(count):
        frames = rng.randint(480, 700)
        positive = i % 2 == 0
        drop_at = int(frames * rng.uniform(0.25, 0.4))
        spot = (rng.uniform(300, 640), rng.uniform(300, 430))

        # The owner arrives, then STAYS WITH the object for a while before
        # anything happens. Without that attended phase the scenario is
        # unrealistic and gives an association agent nothing to bind to - the
        # nearest passer-by would be credited as the owner.
        settle = rng.randint(36, 72)           # 3-6 seconds at 12 fps
        approach = waypoints([(80, 460), spot], drop_at, rng)
        settled = dwell(spot, settle, radius=14, rng=rng)
        leaves_at = drop_at + settle

        if positive:
            # owner leaves and does not come back
            departure = straight(spot, rng.uniform(-40, 40), 3.2,
                                 max(1, frames - leaves_at), rng)
            owner_path = approach + settled + departure
            label, note = "unattended", "owner settles with the bag, then departs for good"
            separation_frame = leaves_at
        else:
            style = rng.choice(["stays", "returns"])
            if style == "stays":
                owner_path = approach + settled + dwell(spot, max(1, frames - leaves_at), 26, rng)
                separation_frame = None
                note = "owner remains with the bag throughout"
            else:
                # Brief absence: short enough that a correctly-configured
                # threshold should NOT escalate, which is what makes this a
                # usable negative rather than a mislabelled positive.
                away = max(6, int((frames - leaves_at) * 0.12))
                owner_path = (
                    approach
                    + settled
                    + straight(spot, 20, 3.0, away, rng)
                    + waypoints([(spot[0] + away * 3, spot[1] + away), spot],
                                max(1, frames - leaves_at - away), rng)
                )
                separation_frame = leaves_at
                note = "owner steps away briefly then reclaims the bag"
            label = f"normal_{style}"

        walkers = [Walker(f"T_O{i:03d}", "owner", owner_path)] + _crowd(rng, 4, frames)
        out.append(
            Scenario(
                scene_id=f"ABND_{i:04d}",
                category="abandoned_object",
                label=label,
                positive=positive,
                walkers=walkers,
                frames=min(frames, len(owner_path)),
                objects=[
                    {
                        "object_id": f"O_{i:03d}",
                        "class_name": "suitcase",
                        "position": [round(spot[0], 1), round(spot[1], 1)],
                        "present_from": drop_at,
                        "width": 46,
                        "height": 38,
                    }
                ],
                annotations={
                    "object_id": f"O_{i:03d}",
                    "owner_track_id": f"T_O{i:03d}",
                    "first_seen": drop_at,
                    "attended_until": leaves_at,
                    "separation_frame": separation_frame,
                    "stationary_duration_frames": frames - drop_at,
                    "status": "unattended" if positive else "attended",
                    "note": note,
                    # Enforced by design: contents are never labelled.
                    "contents": None,
                },
            )
        )
    return out


GENERATORS = {
    "loitering": loitering_scenarios,
    "following": following_scenarios,
    "counter_flow": counterflow_scenarios,
    "restricted_zone": restricted_scenarios,
    "abandoned_object": abandoned_scenarios,
}


# ----------------------------------------------------------------- writing
def _tracking_rows(scenario: Scenario) -> List[List[Any]]:
    """MOT-style rows: frame_id, track_id, x, y, w, h, confidence, class_id."""
    rows: List[List[Any]] = []
    for walker in scenario.walkers:
        for frame in range(scenario.frames):
            point = walker.at(frame)
            if point is None:
                continue
            w = walker.height * 0.36
            rows.append([
                frame, walker.subject_id,
                round(point[0] - w / 2, 2), round(point[1] - walker.height, 2),
                round(w, 2), round(walker.height, 2),
                1.0, 0,
            ])
    for obj in scenario.objects:
        for frame in range(obj["present_from"], scenario.frames):
            rows.append([
                frame, obj["object_id"],
                round(obj["position"][0], 2), round(obj["position"][1], 2),
                obj["width"], obj["height"], 1.0, 28,
            ])
    rows.sort(key=lambda r: (r[0], str(r[1])))
    return rows


def render_video(scenario: Scenario, path: Path) -> bool:
    """Render the scenario to MP4 for end-to-end pipeline testing."""
    try:
        import cv2
    except Exception:
        return False

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        return False

    for frame_index in range(scenario.frames):
        frame = np.full((FRAME_H, FRAME_W, 3), 205, dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (FRAME_W, int(FRAME_H * 0.30)), (178, 172, 165), -1)
        for i in range(1, 6):
            y = int(FRAME_H * 0.30 + (FRAME_H * 0.70) * (i / 6) ** 1.4)
            cv2.line(frame, (0, y), (FRAME_W, y), (192, 188, 182), 1)

        for zone in scenario.zones:
            pts = np.array(
                [[int(x * FRAME_W), int(y * FRAME_H)] for x, y in zone["polygon"]], dtype=np.int32
            )
            cv2.polylines(frame, [pts], True, (60, 60, 200), 2)

        for obj in scenario.objects:
            if frame_index < obj["present_from"]:
                continue
            x, y = int(obj["position"][0]), int(obj["position"][1])
            cv2.rectangle(frame, (x, y), (x + obj["width"], y + obj["height"]), (60, 60, 120), -1)
            cv2.rectangle(frame, (x, y), (x + obj["width"], y + obj["height"]), (40, 40, 80), 2)

        for walker in scenario.walkers:
            point = walker.at(frame_index)
            if point is None:
                continue
            w = walker.height * 0.36
            x1, y1 = int(point[0] - w / 2), int(point[1] - walker.height)
            x2, y2 = int(point[0] + w / 2), int(point[1])
            cv2.rectangle(frame, (x1, int(y1 + walker.height * 0.26)), (x2, y2), (86, 92, 110), -1)
            head_r = int(w * 0.34)
            cv2.circle(frame, (int(point[0]), int(y1 + head_r)), head_r, (140, 132, 122), -1)

        writer.write(frame)

    writer.release()
    return True


def build(
    out_dir: Path,
    *,
    categories: Sequence[str],
    per_category: int,
    seed: int = 20260919,
    render: bool = False,
) -> Dict[str, Any]:
    """Generate every requested category and write it to disk."""
    rng = random.Random(seed)
    summary: Dict[str, Any] = {"categories": {}, "seed": seed, "fps": FPS,
                               "resolution": [FRAME_W, FRAME_H]}

    for category in categories:
        generator = GENERATORS.get(category)
        if generator is None:
            continue
        scenarios = generator(rng, per_category)
        base = out_dir / "behavior" / category
        (base / "tracking").mkdir(parents=True, exist_ok=True)
        (base / "annotations").mkdir(parents=True, exist_ok=True)
        if render:
            (base / "video").mkdir(parents=True, exist_ok=True)

        index: List[Dict[str, Any]] = []
        for scenario in scenarios:
            # MOT-style tracking ground truth
            gt_path = base / "tracking" / f"{scenario.scene_id}.csv"
            with gt_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    ["frame_id", "track_id", "x", "y", "w", "h", "confidence", "class_id"]
                )
                writer.writerows(_tracking_rows(scenario))

            record = {
                "scene_id": scenario.scene_id,
                "category": scenario.category,
                "label": scenario.label,
                "positive": scenario.positive,
                "frames": scenario.frames,
                "fps": FPS,
                "subjects": [w.subject_id for w in scenario.walkers],
                "zones": scenario.zones,
                "objects": scenario.objects,
                "tracking_csv": str(gt_path.relative_to(out_dir)).replace("\\", "/"),
                **scenario.annotations,
            }
            if render:
                video_path = base / "video" / f"{scenario.scene_id}.mp4"
                if render_video(scenario, video_path):
                    record["video"] = str(video_path.relative_to(out_dir)).replace("\\", "/")
            index.append(record)

        (base / "annotations" / "index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )
        _write_category_csv(base / "annotations", category, index)

        positives = sum(1 for r in index if r["positive"])
        summary["categories"][category] = {
            "scenes": len(index),
            "positive": positives,
            "negative": len(index) - positives,
            "path": str(base.relative_to(out_dir)).replace("\\", "/"),
        }

    return summary


def _write_category_csv(folder: Path, category: str, index: List[Dict[str, Any]]) -> None:
    """Emit the exact CSV shapes the write-up specifies (S42-S44)."""
    if category == "following":
        path = folder / "following.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["scene_id", "subject_a", "subject_b", "start_frame",
                             "end_frame", "relationship"])
            for row in index:
                writer.writerow([row["scene_id"], row["subject_a"], row["subject_b"],
                                 0, row["frames"], row["relationship"]])

    elif category == "abandoned_object":
        path = folder / "unattended_objects.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["scene_id", "object_id", "owner_track_id", "first_seen",
                             "separation_time", "stationary_duration", "status"])
            for row in index:
                writer.writerow([
                    row["scene_id"], row["object_id"], row["owner_track_id"],
                    row["first_seen"], row["separation_frame"] if row["separation_frame"] is not None else "",
                    row["stationary_duration_frames"], row["status"],
                ])

    elif category == "restricted_zone":
        path = folder / "restricted_zone.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["scene_id", "zone_id", "track_id", "start_frame",
                             "event_type", "authorization_status"])
            for row in index:
                writer.writerow([row["scene_id"], row["zone_id"], row["subject_id"],
                                 0, row["event_type"], row["authorization_status"]])

    else:
        # S42 behavioural annotation format
        path = folder / "behavior.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["video_id", "track_id", "start_frame", "end_frame",
                             "behavior", "label", "confidence"])
            for row in index:
                writer.writerow([
                    row["scene_id"], row.get("subject_id", ""), 0, row["frames"],
                    row["label"], 1 if row["positive"] else 0, 1.0,
                ])
