"""Agent 8 - Crowd Intelligence (spec S11).

Continuously samples count, density, velocity, flow direction, compression,
counter-flow ratio and surge, then bands the result into a risk level so
intervention can happen before a crowd becomes dangerous.

Density is expressed as occupied-cell fraction on a coarse grid rather than
raw head count, because a count alone says nothing about whether people are
spread across a concourse or compressed at one gate.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..vision.crowd_density import get_density_counter
from ..vision.geometry import angular_difference, dominant_heading
from .base import Agent, FrameContext, Finding


@dataclass
class _CrowdWindow:
    samples: Deque[Tuple[float, int, float, Optional[float]]] = field(
        default_factory=lambda: deque(maxlen=600)
    )  # (t, count, density, heading)
    last_report: Dict[str, float] = field(default_factory=dict)
    last_sample_at: float = -1e9


class CrowdAgent(Agent):
    name = "crowd"
    spec_id = 8
    description = "Density, flow, compression, surge and crowd risk"

    SAMPLE_INTERVAL = 1.0      # seconds between telemetry samples
    DENSITY_INTERVAL = 2.0     # seconds between density-model passes (it is heavier)
    # Switch to the density estimate only when it clearly exceeds the detector
    # count: that is the signature of a crowd too dense to separate into boxes.
    DENSITY_TAKEOVER_RATIO = 1.3
    DENSITY_TAKEOVER_MIN = 10
    REPORT_COOLDOWN = 60.0

    def __init__(self) -> None:
        super().__init__()
        self._windows: Dict[str, _CrowdWindow] = defaultdict(_CrowdWindow)
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._density_estimates: Dict[str, Any] = {}
        self._density_at: Dict[str, float] = {}
        self.density = get_density_counter()

    def reset(self, camera_id: Optional[str] = None) -> None:
        if camera_id:
            self._windows.pop(camera_id, None)
            self._latest.pop(camera_id, None)
            self._density_estimates.pop(camera_id, None)
            self._density_at.pop(camera_id, None)
        else:
            self._windows.clear()
            self._latest.clear()
            self._density_estimates.clear()
            self._density_at.clear()

    def process(self, ctx: FrameContext) -> List[Finding]:
        cfg = ctx.policy.get("crowd", {})
        bands = cfg.get("density_bands", {"low": 0.15, "medium": 0.35, "high": 0.6, "critical": 0.8})
        cell = int(cfg.get("cell_size_px", 64))
        window_seconds = float(cfg.get("flow_window_seconds", 30))
        surge_threshold = float(cfg.get("surge_percent_threshold", 40))
        reversal_ratio = float(cfg.get("reversal_ratio", 0.6))

        now = ctx.elapsed_seconds
        win = self._windows[ctx.camera.camera_id]
        persons = ctx.persons()

        if self.density.available and ctx.frame is not None and                 now - self._density_at.get(ctx.camera.camera_id, -1e9) >= self.DENSITY_INTERVAL:
            self._density_at[ctx.camera.camera_id] = now
            self._density_estimates[ctx.camera.camera_id] = self.density.estimate(ctx.frame)

        metrics = self._measure(ctx, persons, cell, bands)
        self._latest[ctx.camera.camera_id] = metrics

        if (now - win.last_sample_at) < self.SAMPLE_INTERVAL:
            return []
        win.last_sample_at = now
        win.samples.append((now, metrics["count"], metrics["density"], metrics["flow_direction_deg"]))

        findings: List[Finding] = []
        history = [s for s in win.samples if now - s[0] <= window_seconds]

        # ---- surge -------------------------------------------------------
        delta_percent = 0.0
        if len(history) >= 4:
            baseline = float(np.mean([s[1] for s in history[: max(1, len(history) // 3)]]))
            current = float(np.mean([s[1] for s in history[-max(1, len(history) // 3):]]))
            if baseline >= 3:
                delta_percent = (current - baseline) / baseline * 100.0
        metrics["delta_percent"] = round(delta_percent, 1)

        if delta_percent >= surge_threshold and self._cooldown(win, "surge", now):
            win.last_report["surge"] = now
            confidence = float(np.clip(0.5 + delta_percent / 200.0, 0, 0.92))
            findings.append(
                Finding(
                    behavior="crowd_surge",
                    confidence=round(confidence, 3),
                    severity="high" if delta_percent >= surge_threshold * 2 else "medium",
                    zone_id=ctx.camera.zone_id,
                    explanation=(
                        f"Occupancy rose {delta_percent:.0f}% over {int(window_seconds)}s "
                        f"at {ctx.camera.name} (now {metrics['count']} subjects, "
                        f"density band {metrics['density_band'].upper()})."
                    ),
                    evidence=dict(metrics),
                    dedupe_key=f"surge:{ctx.camera.camera_id}",
                )
            )

        # ---- reversal ----------------------------------------------------
        if len(history) >= 8:
            headings = [s[3] for s in history if s[3] is not None]
            if len(headings) >= 8:
                early = dominant_heading(headings[: len(headings) // 2])
                late = dominant_heading(headings[len(headings) // 2 :])
                if early is not None and late is not None:
                    swing = angular_difference(early, late)
                    if swing >= 130 and self._cooldown(win, "reversal", now):
                        win.last_report["reversal"] = now
                        findings.append(
                            Finding(
                                behavior="crowd_reversal",
                                confidence=round(float(np.clip(swing / 180.0, 0, 0.95)), 3),
                                severity="high",
                                zone_id=ctx.camera.zone_id,
                                explanation=(
                                    f"Prevailing crowd direction swung {int(swing)}deg "
                                    f"(from {int(early)}deg to {int(late)}deg) within "
                                    f"{int(window_seconds)}s - consistent with a crowd turning back."
                                ),
                                evidence={**metrics, "swing_deg": round(swing, 1)},
                                dedupe_key=f"reversal:{ctx.camera.camera_id}",
                            )
                        )

        # ---- counter-flow share -------------------------------------------
        if metrics["counter_flow_ratio"] >= reversal_ratio and metrics["count"] >= 5 \
                and self._cooldown(win, "counterflow", now):
            win.last_report["counterflow"] = now
            findings.append(
                Finding(
                    behavior="crowd_flow_anomaly",
                    confidence=round(float(np.clip(metrics["counter_flow_ratio"], 0, 0.9)), 3),
                    severity="medium",
                    zone_id=ctx.camera.zone_id,
                    explanation=(
                        f"{metrics['counter_flow_ratio'] * 100:.0f}% of moving subjects oppose "
                        f"the prevailing flow - the crowd is no longer moving coherently."
                    ),
                    evidence=dict(metrics),
                    dedupe_key=f"flow:{ctx.camera.camera_id}",
                )
            )

        # ---- sustained critical density -----------------------------------
        if metrics["density_band"] == "critical" and self._cooldown(win, "density", now):
            win.last_report["density"] = now
            findings.append(
                Finding(
                    behavior="crowd_density_critical",
                    confidence=0.85,
                    severity="critical",
                    zone_id=ctx.camera.zone_id,
                    explanation=(
                        f"Crowd density reached CRITICAL at {ctx.camera.name}: "
                        f"{metrics['count']} subjects, compression {metrics['compression']:.2f}."
                    ),
                    evidence=dict(metrics),
                    dedupe_key=f"density:{ctx.camera.camera_id}",
                )
            )

        return findings

    # ------------------------------------------------------------- measuring
    def _measure(self, ctx: FrameContext, persons, cell: int, bands: Dict[str, float]) -> Dict[str, Any]:
        w, h = max(1, ctx.camera.width), max(1, ctx.camera.height)
        cols = max(1, w // cell)
        rows = max(1, h // cell)
        grid = np.zeros((rows, cols), dtype=np.float32)

        for p in persons:
            x, y = p.foot_point
            c = int(np.clip(x / w * cols, 0, cols - 1))
            r = int(np.clip(y / h * rows, 0, rows - 1))
            grid[r, c] += 1

        detected = len(persons)
        count, method = detected, "detection"
        estimate = self._density_estimates.get(ctx.camera.camera_id)
        if estimate is not None:
            est_count, density_map = estimate
            if est_count >= self.DENSITY_TAKEOVER_MIN and                     est_count > detected * self.DENSITY_TAKEOVER_RATIO:
                # Dense crowd: detections are undercounting. Use the density
                # map for the count AND for the occupancy grid.
                count, method = int(round(est_count)), "density"
                grid = self._pool(density_map, rows, cols)

        occupied = float((grid > 0.5).sum()) if method == "density" else float((grid > 0).sum())
        total_cells = float(rows * cols)
        density = occupied / total_cells if total_cells else 0.0
        compression = float(grid.max() / max(1.0, grid[grid > 0].mean())) if occupied else 0.0
        compression = compression if np.isfinite(compression) else 0.0

        speeds = [p.speed for p in persons]
        headings = [p.heading_deg() for p in persons if p.speed > 12]
        headings = [x for x in headings if x is not None]
        flow = dominant_heading(headings)

        counter = 0
        if flow is not None and headings:
            counter = sum(1 for x in headings if angular_difference(x, flow) > 120)
        counter_ratio = counter / len(headings) if headings else 0.0

        variance = 0.0
        if len(headings) >= 2 and flow is not None:
            variance = float(np.mean([angular_difference(x, flow) for x in headings]) / 180.0)

        band = "low"
        for label in ("medium", "high", "critical"):
            if density >= float(bands.get(label, 1.0)):
                band = label

        risk = band
        if band in ("medium", "high") and (counter_ratio > 0.5 or compression > 3.0):
            risk = "high" if band == "medium" else "critical"

        return {
            "count": count,
            "count_method": method,
            "count_detected": detected,
            "count_density": (int(round(estimate[0])) if estimate is not None else None),
            "density": round(density, 4),
            "density_band": band,
            "risk": risk,
            "mean_speed": round(float(np.mean(speeds)), 2) if speeds else 0.0,
            "flow_direction_deg": round(flow, 1) if flow is not None else None,
            "flow_variance": round(variance, 3),
            "counter_flow_ratio": round(counter_ratio, 3),
            "compression": round(compression, 3),
            "delta_percent": 0.0,
            "grid_rows": rows,
            "grid_cols": cols,
            "heatmap": grid.tolist(),
        }

    @staticmethod
    def _pool(density_map: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """Sum a density map into the agent's coarse grid, preserving the count."""
        h, w = density_map.shape
        out = np.zeros((rows, cols), dtype=np.float32)
        ys = np.linspace(0, h, rows + 1).astype(int)
        xs = np.linspace(0, w, cols + 1).astype(int)
        for r in range(rows):
            for c in range(cols):
                out[r, c] = density_map[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].sum()
        return out

    def _cooldown(self, win: _CrowdWindow, key: str, now: float) -> bool:
        last = win.last_report.get(key)
        return last is None or (now - last) >= self.REPORT_COOLDOWN

    # -------------------------------------------------------------- readers
    def latest(self, camera_id: str) -> Optional[Dict[str, Any]]:
        return self._latest.get(camera_id)

    def all_latest(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._latest)
