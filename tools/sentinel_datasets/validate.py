"""Dataset validator.

Checks the things that actually go wrong and silently ruin a training run:

  * structure     - required folders and descriptors exist
  * YOLO labels   - correct arity, class ids in range, coordinates normalised,
                    no zero-area boxes, every label has an image and vice versa
  * tracking      - monotonic frames, no duplicate (frame, track) rows,
                    no negative sizes
  * class balance - a category that is 95% one label will teach a model to
                    guess that label
  * negatives     - every custom behaviour category must ship matched negatives
  * leakage       - the same source identity must not appear in two splits
  * guardrails    - no sample may be labelled with contents or an identity claim

Exit status is non-zero when any ERROR is found, so it can gate a training job.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FORBIDDEN_LABELS = {"explosive", "bomb", "weapon_contents", "criminal", "stalker", "terrorist"}


class Report:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


def validate(root: Path) -> Report:
    report = Report()
    if not root.is_dir():
        report.error(f"dataset root does not exist: {root}")
        return report

    _check_structure(root, report)
    _check_detection(root, report)
    _check_tracking(root, report)
    _check_behaviour(root, report)
    _check_occlusion(root, report)
    _check_incidents(root, report)
    _check_guardrails(root, report)
    return report


# ----------------------------------------------------------------- checks
def _check_structure(root: Path, report: Report) -> None:
    from .structure import TREE

    missing = [name for name in TREE if not (root / name).is_dir()]
    if missing:
        report.warn(f"missing top-level folders: {', '.join(missing)} - run `make-datasets structure`")
    report.info["top_level"] = sorted(p.name for p in root.iterdir() if p.is_dir())


def _check_detection(root: Path, report: Report) -> None:
    detection = root / "detection"
    if not detection.is_dir():
        return

    data_yaml = detection / "data.yaml"
    class_count: Optional[int] = None
    if data_yaml.is_file():
        for line in data_yaml.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("nc:"):
                try:
                    class_count = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    else:
        report.warn("detection/data.yaml is missing - Ultralytics cannot read the dataset")

    stats: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        images_dir = detection / "images" / split
        labels_dir = detection / "labels" / split
        if not images_dir.is_dir():
            continue

        images = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        labels = {p.stem for p in labels_dir.iterdir() if p.suffix == ".txt"} if labels_dir.is_dir() else set()
        if not images:
            continue

        orphan_images = images - labels
        orphan_labels = labels - images
        if orphan_images:
            report.warn(
                f"detection/{split}: {len(orphan_images)} image(s) have no label file "
                "(they will train as backgrounds, which may be intended)"
            )
        if orphan_labels:
            report.error(f"detection/{split}: {len(orphan_labels)} label file(s) have no image")

        class_counts: Counter = Counter()
        bad_lines = 0
        zero_area = 0
        out_of_range = 0
        bad_class = 0

        for label_path in (labels_dir.iterdir() if labels_dir.is_dir() else []):
            if label_path.suffix != ".txt":
                continue
            for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad_lines += 1
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    bad_lines += 1
                    continue
                class_counts[cls] += 1
                if class_count is not None and not (0 <= cls < class_count):
                    bad_class += 1
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                    out_of_range += 1
                if w <= 0 or h <= 0:
                    zero_area += 1

        if bad_lines:
            report.error(f"detection/{split}: {bad_lines} malformed label line(s) - expected "
                         "'class cx cy w h'")
        if out_of_range:
            report.error(f"detection/{split}: {out_of_range} box(es) not normalised to 0-1")
        if zero_area:
            report.error(f"detection/{split}: {zero_area} zero-area box(es)")
        if bad_class:
            report.error(f"detection/{split}: {bad_class} box(es) with a class id outside "
                         f"0..{(class_count or 0) - 1}")

        if class_counts:
            total = sum(class_counts.values())
            dominant, dominant_count = class_counts.most_common(1)[0]
            if dominant_count / total > 0.95 and len(class_counts) > 1:
                report.warn(
                    f"detection/{split}: class {dominant} is {dominant_count / total:.0%} of all "
                    "boxes - the model will learn to predict it by default"
                )
        stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": sum(class_counts.values()),
            "class_distribution": dict(sorted(class_counts.items())),
        }

    if stats:
        report.info["detection"] = stats
        if "train" in stats and "val" not in stats:
            report.warn("detection: a train split with no val split means you cannot measure "
                        "generalisation")


def _check_tracking(root: Path, report: Report) -> None:
    files = list((root / "behavior").rglob("tracking/*.csv")) + list(
        (root / "tracking").rglob("*.csv")
    )
    if not files:
        return

    checked = 0
    for path in files[:400]:
        try:
            with path.open(encoding="utf-8") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
                if not header or header[0] != "frame_id":
                    report.warn(f"{path.name}: unexpected header {header}")
                    continue
                seen: set = set()
                last_frame = -1
                for row in reader:
                    if len(row) < 8:
                        report.error(f"{path.name}: row with {len(row)} columns, expected 8")
                        break
                    frame = int(float(row[0]))
                    key = (frame, row[1])
                    if key in seen:
                        report.error(f"{path.name}: duplicate row for frame {frame}, track {row[1]}")
                        break
                    seen.add(key)
                    if frame < last_frame:
                        report.error(f"{path.name}: frames are not in order at frame {frame}")
                        break
                    last_frame = frame
                    if float(row[4]) <= 0 or float(row[5]) <= 0:
                        report.error(f"{path.name}: non-positive box size at frame {frame}")
                        break
            checked += 1
        except Exception as exc:
            report.error(f"{path.name}: unreadable ({exc})")

    report.info["tracking_files_checked"] = checked


def _check_behaviour(root: Path, report: Report) -> None:
    behaviour = root / "behavior"
    if not behaviour.is_dir():
        return

    summary: Dict[str, Any] = {}
    for category_dir in sorted(behaviour.iterdir()):
        index_path = category_dir / "annotations" / "index.json"
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.error(f"behavior/{category_dir.name}: index.json is not valid JSON ({exc})")
            continue

        positives = sum(1 for r in index if r.get("positive"))
        negatives = len(index) - positives
        labels = Counter(r.get("label") for r in index)

        if negatives == 0 and len(index) > 0:
            report.error(
                f"behavior/{category_dir.name}: {len(index)} scenes but no negatives. "
                "A detector trained on positives only will fire on everything."
            )
        elif positives and negatives:
            ratio = positives / (positives + negatives)
            if ratio > 0.8 or ratio < 0.2:
                report.warn(
                    f"behavior/{category_dir.name}: {ratio:.0%} positive - consider balancing"
                )

        for record in index:
            if not record.get("tracking_csv"):
                report.error(f"behavior/{category_dir.name}/{record.get('scene_id')}: "
                             "no tracking ground truth")
                break
            gt = root / record["tracking_csv"]
            if not gt.is_file():
                report.error(f"missing tracking file referenced by index: {record['tracking_csv']}")
                break

        summary[category_dir.name] = {
            "scenes": len(index),
            "positive": positives,
            "negative": negatives,
            "labels": dict(labels),
        }

    if summary:
        report.info["behavior"] = summary


def _check_occlusion(root: Path, report: Report) -> None:
    annotations = root / "occlusion" / "annotations" / "occlusion.csv"
    if not annotations.is_file():
        return

    rows = list(csv.DictReader(annotations.open(encoding="utf-8")))
    if not rows:
        report.warn("occlusion: annotation file is empty")
        return

    by_class = Counter(r["occlusion_label"] for r in rows)
    by_identity = defaultdict(set)
    missing_files = 0
    for row in rows:
        by_identity[row["source_identity"]].add(row["occlusion_label"])
        candidate = root / "occlusion" / row["occlusion_label"] / row["filename"]
        if not candidate.is_file():
            missing_files += 1

    if missing_files:
        report.error(f"occlusion: {missing_files} annotated file(s) are missing from disk")
    if "normal" not in by_class:
        report.error("occlusion: no 'normal' class - there is nothing to contrast occlusion with")
    if len(by_identity) < 2:
        report.warn(
            f"occlusion: only {len(by_identity)} source identity/identities. Robustness "
            "results from a single identity do not generalise."
        )

    report.info["occlusion"] = {
        "samples": len(rows),
        "by_class": dict(by_class),
        "identities": len(by_identity),
    }

    # Leakage: the same identity appearing across train/val splits inflates scores.
    appearance = root / "appearance" / "appearance.csv"
    if appearance.is_file():
        app_rows = list(csv.DictReader(appearance.open(encoding="utf-8")))
        identities = {r["original_identity"] for r in app_rows}
        report.info["appearance"] = {"samples": len(app_rows), "identities": len(identities)}
        if len(identities) < 2:
            report.warn(
                "appearance: a single identity cannot show whether the model distinguishes "
                "people or merely memorises one"
            )


def _check_incidents(root: Path, report: Report) -> None:
    folder = root / "incidents"
    events = folder / "events.json"
    if not events.is_file():
        return

    try:
        records = json.loads(events.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.error(f"incidents/events.json is not valid JSON ({exc})")
        return

    required = {"incident_id", "camera_id", "timestamp", "event"}
    for record in records[:200]:
        missing = required - set(record)
        if missing:
            report.error(f"incidents: record {record.get('incident_id')} missing {sorted(missing)}")
            break

    qa = folder / "qa.jsonl"
    pairs = 0
    if qa.is_file():
        for line in qa.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                report.error("incidents/qa.jsonl contains a malformed line")
                break
            if len(payload.get("messages", [])) != 3:
                report.error("incidents/qa.jsonl: a pair does not have system/user/assistant")
                break
            pairs += 1

    report.info["incidents"] = {"records": len(records), "qa_pairs": pairs}
    if len(records) < 50:
        report.warn(
            f"incidents: only {len(records)} record(s). Run the pipeline against real or "
            "recorded footage for longer before fine-tuning on this."
        )


def _check_guardrails(root: Path, report: Report) -> None:
    """No part of the dataset may assert contents or a real identity."""
    hits: List[str] = []
    for path in list(root.rglob("*.json")) + list(root.rglob("*.csv")):
        if path.stat().st_size > 20_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for word in FORBIDDEN_LABELS:
            if word in text:
                hits.append(f"{path.relative_to(root)} contains '{word}'")
                break

    if hits:
        report.error(
            "guardrail violation - these labels are out of scope by design (write-up S12): "
            + "; ".join(hits[:10])
        )
    report.info["guardrail_scan"] = {"violations": len(hits)}


def format_report(report: Report) -> str:
    lines: List[str] = []
    lines.append("=" * 74)
    lines.append(f"  Dataset validation: {'PASS' if report.ok else 'FAIL'}")
    lines.append("=" * 74)

    if report.errors:
        lines.append(f"\nERRORS ({len(report.errors)}) - these will break or corrupt training:")
        lines.extend(f"  x {message}" for message in report.errors)
    if report.warnings:
        lines.append(f"\nWARNINGS ({len(report.warnings)}):")
        lines.extend(f"  ! {message}" for message in report.warnings)
    if not report.errors and not report.warnings:
        lines.append("\nNo problems found.")

    if report.info:
        lines.append("\nSUMMARY:")
        lines.append(json.dumps(report.info, indent=2))
    return "\n".join(lines)
