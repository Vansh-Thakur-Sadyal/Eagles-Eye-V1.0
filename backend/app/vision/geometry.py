"""Spatial helpers: zone polygons, homography, trajectory similarity.

These back the restricted-zone agent, the digital twin projection and the
following-pattern correlation maths.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


# ------------------------------------------------------------------ polygons
def point_in_polygon(point: Point, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray casting. Polygon is [[x, y], ...] in the same space as `point`."""
    if not polygon or len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if (yi > y) != (yj > y):
            denom = (yj - yi) or 1e-12
            if x < (xj - xi) * (y - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


def polygon_centroid(polygon: Sequence[Sequence[float]]) -> Optional[Point]:
    if not polygon:
        return None
    arr = np.asarray(polygon, dtype=np.float64)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def normalised_polygon_to_pixels(
    polygon: Sequence[Sequence[float]], width: int, height: int
) -> List[List[float]]:
    """Zones are stored normalised 0-1 so they survive a resolution change."""
    out = []
    for p in polygon:
        x, y = float(p[0]), float(p[1])
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            out.append([x * width, y * height])
        else:
            out.append([x, y])
    return out


def segment_intersects(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """Used for tripwire / line-crossing rules."""

    def orient(p: Point, q: Point, r: Point) -> float:
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    def on_seg(p: Point, q: Point, r: Point) -> bool:
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1, o2, o3, o4 = orient(a1, a2, b1), orient(a1, a2, b2), orient(b1, b2, a1), orient(b1, b2, a2)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    if abs(o1) < 1e-9 and on_seg(a1, b1, a2):
        return True
    if abs(o2) < 1e-9 and on_seg(a1, b2, a2):
        return True
    if abs(o3) < 1e-9 and on_seg(b1, a1, b2):
        return True
    if abs(o4) < 1e-9 and on_seg(b1, a2, b2):
        return True
    return False


def crossing_direction(prev: Point, curr: Point, line: Sequence[Sequence[float]]) -> Optional[str]:
    """Which way a track crossed a line: 'a_to_b', 'b_to_a', or None."""
    if len(line) < 2:
        return None
    p1, p2 = (line[0][0], line[0][1]), (line[1][0], line[1][1])
    if not segment_intersects(prev, curr, p1, p2):
        return None
    side_before = _side(p1, p2, prev)
    side_after = _side(p1, p2, curr)
    if side_before == side_after:
        return None
    return "a_to_b" if side_after > 0 else "b_to_a"


def _side(p1: Point, p2: Point, p: Point) -> float:
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


# --------------------------------------------------------------- homography
def compute_homography(image_points: Sequence[Point], world_points: Sequence[Point]):
    """Fit image -> ground-plane mapping from >= 4 correspondences."""
    if len(image_points) < 4 or len(world_points) < 4:
        return None
    try:
        import cv2

        src = np.asarray(image_points, dtype=np.float32)
        dst = np.asarray(world_points, dtype=np.float32)
        h, _mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        return h.tolist() if h is not None else None
    except Exception:
        return None


def project_point(homography: Optional[Sequence[Sequence[float]]], point: Point) -> Optional[Point]:
    """Image pixel -> world/ground coordinate."""
    if not homography:
        return None
    h = np.asarray(homography, dtype=np.float64)
    if h.shape != (3, 3):
        return None
    vec = np.array([point[0], point[1], 1.0], dtype=np.float64)
    out = h @ vec
    if abs(out[2]) < 1e-9:
        return None
    return float(out[0] / out[2]), float(out[1] / out[2])


def camera_footprint(
    lat: float, lon: float, orientation_deg: float, fov_deg: float, range_m: float, steps: int = 12
) -> List[List[float]]:
    """Approximate the visible ground wedge, for the map and digital twin."""
    pts: List[List[float]] = [[lon, lat]]
    half = fov_deg / 2.0
    for i in range(steps + 1):
        bearing = orientation_deg - half + (fov_deg * i / steps)
        pts.append(list(reversed(destination_point(lat, lon, bearing, range_m))))
    pts.append([lon, lat])
    return pts


def destination_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """Great-circle offset. Returns (lat, lon)."""
    r = 6371000.0
    br = math.radians(bearing_deg)
    lat1, lon1 = math.radians(lat), math.radians(lon)
    ang = distance_m / r
    lat2 = math.asin(math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ------------------------------------------------------------- trajectories
def resample_path(points: Sequence[Point], n: int = 32) -> np.ndarray:
    """Arc-length resampling so two paths of different lengths are comparable."""
    arr = np.asarray(points, dtype=np.float64)
    if len(arr) < 2:
        return np.repeat(arr if len(arr) else np.zeros((1, 2)), n, axis=0)[:n]
    deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total = cumulative[-1]
    if total <= 0:
        return np.repeat(arr[:1], n, axis=0)
    targets = np.linspace(0, total, n)
    xs = np.interp(targets, cumulative, arr[:, 0])
    ys = np.interp(targets, cumulative, arr[:, 1])
    return np.stack([xs, ys], axis=1)


def trajectory_similarity(a: Sequence[Point], b: Sequence[Point], *, scale: float = 200.0) -> float:
    """Shape similarity in 0-1 after removing translation.

    Translation is removed on purpose: two people walking the same route a few
    metres apart should score high, which is exactly the following signal.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ra, rb = resample_path(a), resample_path(b)
    ra = ra - ra.mean(axis=0)
    rb = rb - rb.mean(axis=0)
    dist = float(np.linalg.norm(ra - rb, axis=1).mean())
    return float(math.exp(-dist / max(1e-6, scale)))


def direction_changes(points: Sequence[Point], *, min_angle_deg: float = 35.0,
                      window: int = 4) -> List[int]:
    """Indices where the path turns by more than `min_angle_deg`."""
    arr = np.asarray(points, dtype=np.float64)
    if len(arr) < window * 2 + 1:
        return []
    out: List[int] = []
    for i in range(window, len(arr) - window):
        v1 = arr[i] - arr[i - window]
        v2 = arr[i + window] - arr[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        if math.degrees(math.acos(cos)) >= min_angle_deg:
            out.append(i)
    return out


def path_length(points: Sequence[Point]) -> float:
    arr = np.asarray(points, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def angular_difference(a_deg: float, b_deg: float) -> float:
    """Smallest absolute angle between two headings, 0-180."""
    d = abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)
    return float(d)


def dominant_heading(headings: Sequence[float]) -> Optional[float]:
    """Circular mean - a plain average would break across the 0/360 seam."""
    vals = [h for h in headings if h is not None]
    if not vals:
        return None
    rad = np.radians(vals)
    return float((math.degrees(math.atan2(np.sin(rad).mean(), np.cos(rad).mean())) + 360.0) % 360.0)


def zone_for_point(point: Point, zones: List[Dict[str, Any]], width: int, height: int) -> Optional[str]:
    """First zone containing the point; zones are stored normalised."""
    for z in zones:
        poly = z.get("polygon") or []
        if not poly:
            continue
        pixels = normalised_polygon_to_pixels(poly, width, height)
        if point_in_polygon(point, pixels):
            return z.get("id")
    return None
