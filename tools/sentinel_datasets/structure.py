"""Dataset folder structure and camera/zone metadata (write-up S46, S47)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Exactly the tree the master write-up specifies in section 46.
TREE: Dict[str, List[str]] = {
    "detection": ["images", "labels"],
    "tracking": ["sequences", "gt", "metadata"],
    "reid": ["train", "query", "gallery"],
    "anomaly": ["normal", "abnormal", "annotations"],
    "behavior": ["loitering", "running", "counter_flow", "following", "restricted_zone",
                 "abandoned_object", "normal"],
    "objects": ["unattended", "normal", "annotations"],
    "occlusion": ["mask", "helmet", "hood", "sunglasses", "partial", "normal", "annotations"],
    "appearance": [],
    "crowd": ["dense", "normal", "annotations"],
    "incidents": [],
    "metadata": [],
}

# Classes the detector is trained on. Kept here so data.yaml and the runtime
# policy cannot drift apart.
DETECTION_CLASSES = [
    "person", "backpack", "handbag", "suitcase", "bicycle", "motorcycle",
    "car", "bus", "truck", "bottle", "laptop",
]

README = """# Sentinel dataset

Layout follows section 46 of the master write-up.

    detection/   YOLO-format images + labels (class x_center y_center w h, normalised)
    tracking/    MOT-style sequences: frame_id,track_id,x,y,w,h,confidence,class_id
    reid/        train / query / gallery crops for person re-identification
    anomaly/     normal vs abnormal clips
    behavior/    the custom behaviour categories, with matched negatives
    objects/     unattended vs attended object events
    occlusion/   face-occlusion classes, augmented from real references
    appearance/  appearance-variation sets, per source identity
    crowd/       density and counting data
    incidents/   structured incident records and conversational Q&A
    metadata/    cameras.json, zones.json, locations.json

## Provenance

`manifest.json` in this folder records how each part was produced. Read it
before training anything: simulated trajectory data and augmented imagery have
different standing from real footage, and the manifest says which is which.

## What you still need to download

The public datasets are not generated here. See docs/DATASETS.md for the list
and what each one is for.
"""


def build(out_dir: Path, *, database_url: Optional[str] = None) -> Dict[str, Any]:
    """Create the folder tree, data.yaml, metadata files and a README."""
    out_dir.mkdir(parents=True, exist_ok=True)

    created: List[str] = []
    for top, subs in TREE.items():
        (out_dir / top).mkdir(parents=True, exist_ok=True)
        created.append(top)
        for sub in subs:
            (out_dir / top / sub).mkdir(parents=True, exist_ok=True)

    # YOLO dataset descriptor
    data_yaml = out_dir / "detection" / "data.yaml"
    data_yaml.write_text(
        "# Sentinel detection dataset\n"
        f"path: {out_dir.resolve().as_posix()}/detection\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(DETECTION_CLASSES)}\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(DETECTION_CLASSES)),
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        (out_dir / "detection" / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "detection" / "labels" / split).mkdir(parents=True, exist_ok=True)

    (out_dir / "README.md").write_text(README, encoding="utf-8")

    metadata = _export_metadata(out_dir / "metadata", database_url)

    return {
        "root": str(out_dir.resolve()),
        "folders_created": len(created),
        "detection_classes": DETECTION_CLASSES,
        "metadata": metadata,
    }


def _export_metadata(folder: Path, database_url: Optional[str]) -> Dict[str, Any]:
    """Write cameras.json / zones.json / locations.json from the live database.

    Falls back to documented empty templates when the database is unavailable,
    so the structure is still correct on a fresh checkout.
    """
    folder.mkdir(parents=True, exist_ok=True)
    cameras: List[Dict[str, Any]] = []
    zones: List[Dict[str, Any]] = []
    sites: List[Dict[str, Any]] = []
    source = "database"

    try:
        import os
        import sys

        backend = Path(__file__).resolve().parents[2] / "backend"
        if str(backend) not in sys.path:
            sys.path.insert(0, str(backend))
        if database_url:
            os.environ["SENTINEL_DATABASE_URL"] = database_url

        from sqlalchemy import select

        from app.db import session_scope
        from app.models import Camera, Site, Zone

        with session_scope() as db:
            for camera in db.scalars(select(Camera)).all():
                cameras.append({
                    "camera_id": camera.id,
                    "name": camera.name,
                    "location": camera.location,
                    "site_id": camera.site_id,
                    "zone": camera.zone_id,
                    "latitude": camera.latitude,
                    "longitude": camera.longitude,
                    "floor": camera.floor,
                    "orientation": camera.orientation_deg,
                    "field_of_view": camera.field_of_view_deg,
                    "range_m": camera.range_m,
                    "resolution": [camera.width, camera.height],
                    "fps": camera.fps,
                    "homography": camera.homography,
                })
            for zone in db.scalars(select(Zone)).all():
                zones.append({
                    "zone_id": zone.id,
                    "name": zone.name,
                    "zone_type": zone.zone_type,
                    "site_id": zone.site_id,
                    "floor": zone.floor,
                    "polygon": zone.polygon,
                    "authorized_roles": zone.authorized_roles,
                    "risk_weight": zone.risk_weight,
                })
            for site in db.scalars(select(Site)).all():
                sites.append({
                    "site_id": site.id,
                    "name": site.name,
                    "site_type": site.site_type,
                    "latitude": site.latitude,
                    "longitude": site.longitude,
                    "timezone": site.timezone,
                    "floors": site.floors,
                })
    except Exception as exc:
        source = f"template (database unavailable: {type(exc).__name__})"
        cameras = [{
            "camera_id": "CAM_042", "name": "Gate 4 North", "location": "Terminal A",
            "site_id": None, "zone": "Gate_4", "latitude": None, "longitude": None,
            "floor": 1, "orientation": 135, "field_of_view": 82, "range_m": 25,
            "resolution": [1280, 720], "fps": 12, "homography": None,
        }]

    (folder / "cameras.json").write_text(json.dumps(cameras, indent=2), encoding="utf-8")
    (folder / "zones.json").write_text(json.dumps(zones, indent=2), encoding="utf-8")
    (folder / "locations.json").write_text(json.dumps(sites, indent=2), encoding="utf-8")

    return {"source": source, "cameras": len(cameras), "zones": len(zones), "sites": len(sites)}
