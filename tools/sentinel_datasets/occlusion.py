"""Face-occlusion and appearance-variation datasets (write-up S37, S38).

These are built by augmenting *real photographs you supply*, which is what the
write-up prescribes: vary an authorised reference to learn robust features,
rather than fabricating identities and storing them as surveillance records.

Every output carries its source identity and the exact augmentation parameters,
so the dataset is auditable and any sample can be traced back to the reference
it came from.

Input layout (one folder per identity):

    source/
      identity_a/  photo1.jpg  photo2.jpg
      identity_b/  photo1.jpg

Output follows the write-up's folder structure:

    occlusion/{normal,mask,helmet,hood,sunglasses,partial}/
    appearance/{identity}/...
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Which augmentation lands in which occlusion class (write-up S37 labels).
OCCLUSION_CLASS = {
    "mask": "mask",
    "helmet": "helmet",
    "hood": "hood",
    "cap": "hood",
    "sunglasses": "sunglasses",
    "occluded_left": "partial",
    "occluded_right": "partial",
    "occluded_bottom": "partial",
}

APPEARANCE_KINDS = [
    "hair_change", "hair_light", "facial_hair", "aging",
    "low_light", "bright_light", "warm_cast", "cool_cast",
    "motion_blur", "low_resolution", "sensor_noise",
    "viewpoint_left", "viewpoint_right", "viewpoint_high",
]


def discover_identities(source: Path) -> Dict[str, List[Path]]:
    """Map identity folder -> its reference images."""
    identities: Dict[str, List[Path]] = {}
    if not source.is_dir():
        return identities
    for child in sorted(source.iterdir()):
        if not child.is_dir():
            continue
        images = [p for p in sorted(child.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
        if images:
            identities[child.name] = images
    return identities


def build(
    source: Path,
    out_dir: Path,
    *,
    compose: int = 3,
    include_appearance: bool = True,
) -> Dict[str, Any]:
    """Generate occlusion and appearance-variation sets from real references."""
    import sys

    backend = Path(__file__).resolve().parents[2] / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))

    import cv2
    from app.vision.augment import AUGMENTATIONS, augment
    from app.vision.reid import get_face_embedder

    identities = discover_identities(source)
    if not identities:
        return {
            "error": f"no identity folders with images found under {source}",
            "expected_layout": "source/<identity>/<photo>.jpg",
            "identities": 0,
        }

    face = get_face_embedder()
    occlusion_root = out_dir / "occlusion"
    appearance_root = out_dir / "appearance"
    for label in set(OCCLUSION_CLASS.values()) | {"normal"}:
        (occlusion_root / label).mkdir(parents=True, exist_ok=True)

    occlusion_rows: List[List[Any]] = []
    appearance_rows: List[List[Any]] = []
    counts: Dict[str, int] = {}
    skipped: List[str] = []

    for identity, images in identities.items():
        if include_appearance:
            (appearance_root / identity).mkdir(parents=True, exist_ok=True)

        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                skipped.append(f"{image_path} (unreadable)")
                continue

            _emb, face_bbox, score = face.embed_largest(image)
            if face_bbox is None:
                skipped.append(f"{image_path} (no face detected - face crops will be approximate)")

            # the unmodified reference is the 'normal' class
            normal_name = f"{identity}__{image_path.stem}__normal.jpg"
            cv2.imwrite(str(occlusion_root / "normal" / normal_name), image)
            occlusion_rows.append([
                normal_name, identity, "normal", "none", "{}",
                round(float(score), 3), "face_visible",
            ])
            counts["normal"] = counts.get("normal", 0) + 1

            kinds = list(OCCLUSION_CLASS) + (APPEARANCE_KINDS if include_appearance else [])
            samples = augment(
                image, kinds,
                face_bbox=tuple(face_bbox) if face_bbox else None,
                source_identity=identity,
                compose=compose,
            )

            for sample in samples:
                primary = sample.augmentation.split("+")[0]
                occlusion_class = OCCLUSION_CLASS.get(primary)
                filename = f"{identity}__{image_path.stem}__{sample.augmentation.replace('+', '_')}.jpg"

                if occlusion_class:
                    cv2.imwrite(str(occlusion_root / occlusion_class / filename), sample.image)
                    _emb2, bbox2, score2 = face.embed_largest(sample.image)
                    occlusion_rows.append([
                        filename, identity, occlusion_class, sample.augmentation,
                        json.dumps(sample.params), round(float(score2), 3),
                        "face_visible" if bbox2 is not None else "face_not_detected",
                    ])
                    counts[occlusion_class] = counts.get(occlusion_class, 0) + 1

                if include_appearance and primary in APPEARANCE_KINDS:
                    cv2.imwrite(str(appearance_root / identity / filename), sample.image)
                    appearance_rows.append([
                        filename, identity, sample.augmentation,
                        json.dumps(sample.params), identity,
                    ])
                    counts["appearance"] = counts.get("appearance", 0) + 1

    (occlusion_root / "annotations").mkdir(parents=True, exist_ok=True)
    with (occlusion_root / "annotations" / "occlusion.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "source_identity", "occlusion_label", "augmentation",
                         "augmentation_params", "face_detection_score", "face_state"])
        writer.writerows(occlusion_rows)

    if include_appearance and appearance_rows:
        with (appearance_root / "appearance.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["filename", "original_identity", "augmentation_type",
                             "augmentation_parameters", "ground_truth_identity"])
            writer.writerows(appearance_rows)

    return {
        "identities": len(identities),
        "source_images": sum(len(v) for v in identities.values()),
        "generated": counts,
        "total_generated": sum(counts.values()),
        "face_backend": face.info(),
        "skipped": skipped[:20],
        "skipped_count": len(skipped),
        "note": (
            "Derived from your own reference photographs by parameterised augmentation. "
            "Every sample records its source identity, so the set is auditable and is "
            "suitable for robustness training and testing - not for building a "
            "permanent identity database."
        ),
    }
