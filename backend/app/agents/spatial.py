"""Spatial Intelligence agent (spec S17-S20) - the Digital Twin / AR bridge.

Turns image-space observations into world-space objects the 3D twin, the map,
the AR overlay and the VR command centre can all consume from one payload.

Two projections are supported, chosen per camera:
  * homography - when the camera has been calibrated against ground points,
    giving metric positions on the floor plane;
  * bearing estimate - otherwise, the subject's horizontal position in frame
    is mapped across the camera's field of view to a bearing, and apparent
    height gives a coarse range.  Marked ``estimated`` so the twin can render
    it with the uncertainty it deserves.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..vision.geometry import camera_footprint, destination_point, project_point
from .base import Agent, FrameContext, Finding

# Rough mean standing height, used to turn apparent pixel height into range.
ASSUMED_PERSON_HEIGHT_M = 1.7


class SpatialAgent(Agent):
    name = "spatial"
    spec_id = 12
    description = "Projects observations into world coordinates for twin, AR and VR"

    def __init__(self) -> None:
        super().__init__()
        self._latest: Dict[str, List[Dict[str, Any]]] = {}

    def process(self, ctx: FrameContext) -> List[Finding]:
        placed: List[Dict[str, Any]] = []
        for track in ctx.tracks:
            position = self.locate(ctx, track)
            if position is None:
                continue
            track.attributes["world"] = position
            if position.get("latitude") is not None:
                track.world_path = getattr(track, "world_path", [])
            placed.append(
                {
                    "track_id": track.track_id,
                    "global_id": track.global_id,
                    "class_name": track.class_name,
                    "camera_id": ctx.camera.camera_id,
                    "zone_id": track.zone_id or ctx.camera.zone_id,
                    "floor": ctx.camera.floor,
                    **position,
                }
            )
        self._latest[ctx.camera.camera_id] = placed
        return []

    # -------------------------------------------------------------- locate
    def locate(self, ctx: FrameContext, track) -> Optional[Dict[str, Any]]:
        cam = ctx.camera
        fx, fy = track.foot_point

        if cam.homography:
            world = project_point(cam.homography, (fx, fy))
            if world is not None:
                return {
                    "method": "homography",
                    "estimated": False,
                    "x_m": round(world[0], 2),
                    "y_m": round(world[1], 2),
                    "latitude": None,
                    "longitude": None,
                    "range_m": None,
                    "bearing_deg": None,
                }

        if cam.latitude is None or cam.longitude is None:
            return {
                "method": "image_only",
                "estimated": True,
                "image_x": round(fx, 1),
                "image_y": round(fy, 1),
                "x_m": None,
                "y_m": None,
                "latitude": None,
                "longitude": None,
                "range_m": None,
                "bearing_deg": None,
            }

        # Bearing from horizontal position across the field of view.
        half_fov = cam.field_of_view_deg / 2.0 if cam.field_of_view_deg else 41.0
        offset = (fx / max(1, cam.width)) - 0.5              # -0.5 .. +0.5
        bearing = (cam.orientation_deg + offset * 2 * half_fov) % 360.0

        # Range from apparent height; a taller box means a closer subject.
        box_height = max(1.0, track.bbox[3] - track.bbox[1])
        vertical_fov = half_fov * 2 * (cam.height / max(1, cam.width))
        angular_height = math.radians(vertical_fov * (box_height / max(1, cam.height)))
        range_m = (
            ASSUMED_PERSON_HEIGHT_M / max(1e-4, math.tan(angular_height))
            if track.class_name == "person"
            else None
        )
        if range_m is not None:
            range_m = float(min(range_m, cam.__dict__.get("range_m", 60.0) or 60.0))

        lat, lon = destination_point(
            cam.latitude, cam.longitude, bearing, range_m if range_m else 8.0
        )
        return {
            "method": "bearing_estimate",
            "estimated": True,
            "image_x": round(fx, 1),
            "image_y": round(fy, 1),
            "x_m": None,
            "y_m": None,
            "bearing_deg": round(bearing, 1),
            "range_m": round(range_m, 1) if range_m else None,
            "latitude": round(lat, 7),
            "longitude": round(lon, 7),
            "confidence": 0.45,
            "note": "Estimated from camera geometry; calibrate a homography for metric accuracy.",
        }

    # -------------------------------------------------------------- readers
    def latest(self, camera_id: str) -> List[Dict[str, Any]]:
        return self._latest.get(camera_id, [])

    def scene(self) -> Dict[str, List[Dict[str, Any]]]:
        return dict(self._latest)

    @staticmethod
    def camera_coverage(camera: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """GeoJSON-ish wedge for the map and twin coverage layer."""
        lat, lon = camera.get("latitude"), camera.get("longitude")
        if lat is None or lon is None:
            return None
        return {
            "camera_id": camera.get("id"),
            "type": "Polygon",
            "coordinates": [
                camera_footprint(
                    float(lat),
                    float(lon),
                    float(camera.get("orientation_deg", 0.0)),
                    float(camera.get("field_of_view_deg", 82.0)),
                    float(camera.get("range_m", 25.0)),
                )
            ],
            "floor": camera.get("floor", 0),
        }

    @staticmethod
    def ar_payload(incident: Dict[str, Any], camera: Dict[str, Any]) -> Dict[str, Any]:
        """What a field device renders for one incident (spec S18)."""
        return {
            "incident_id": incident.get("id"),
            "title": incident.get("title"),
            "severity": incident.get("severity"),
            "risk_score": incident.get("risk_score"),
            "anchor": {
                "latitude": camera.get("latitude"),
                "longitude": camera.get("longitude"),
                "floor": camera.get("floor", 0),
                "bearing_deg": camera.get("orientation_deg"),
            },
            "label": incident.get("title"),
            "detail": incident.get("summary"),
            "zone": incident.get("zone_id"),
            "camera_name": camera.get("name"),
            "overlays": {
                "incident_marker": True,
                "camera_coverage": True,
                "restricted_zones": True,
                "safe_route": incident.get("severity") in ("high", "critical"),
            },
        }
