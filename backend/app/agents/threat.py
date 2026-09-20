"""Agent 10 - Threat Assessment (spec S14).

Combines the signals the other agents produced into one explainable score.

Two properties the spec insists on, both implemented literally:
  * weights are configuration, not constants baked into code - they live in
    policy.json and are editable from the Settings screen;
  * the score is never returned alone.  Every assessment carries the per-factor
    contribution table and a sentence saying *why* the number is what it is.

Scores saturate rather than sum linearly, so five weak signals cannot
manufacture a critical incident, and recency decay means a stale signal stops
propping up a live score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .base import Finding

# Signals that describe observability rather than risk must never add weight.
ZERO_WEIGHT_BEHAVIORS = {"face_unavailable", "appearance_change", "cross_camera_association"}


@dataclass
class ThreatAssessment:
    score: float
    band: str
    factors: List[Dict[str, Any]]
    explanation: str
    confidence: float
    dominant_factor: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "band": self.band,
            "factors": self.factors,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 3),
            "dominant_factor": self.dominant_factor,
        }


class ThreatAssessor:
    """Not a per-frame Agent: called by the orchestrator once findings exist."""

    name = "threat"
    spec_id = 10
    description = "Explainable multi-signal risk scoring"

    HALF_LIFE_SECONDS = 180.0      # a signal contributes half as much after 3 min

    def assess(
        self,
        findings: List[Finding],
        policy: Dict[str, Any],
        *,
        now: Optional[datetime] = None,
        zone_risk_weight: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> ThreatAssessment:
        weights = policy.get("threat_weights", {})
        bands = policy.get("threat_bands", {"low": 25, "medium": 50, "high": 75, "critical": 90})
        now = now or datetime.now(timezone.utc)
        context = context or {}

        factors: List[Dict[str, Any]] = []
        raw_total = 0.0
        confidences: List[float] = []

        # Only the strongest instance of each behaviour counts, so a repeated
        # signal cannot be double-counted into a higher band.
        strongest: Dict[str, Finding] = {}
        for f in findings:
            current = strongest.get(f.behavior)
            if current is None or f.confidence > current.confidence:
                strongest[f.behavior] = f

        for behavior, finding in strongest.items():
            base_weight = float(weights.get(behavior, 0))
            if behavior in ZERO_WEIGHT_BEHAVIORS:
                base_weight = 0.0

            decay = self._recency(finding, now)
            contribution = base_weight * finding.confidence * decay
            raw_total += contribution
            if base_weight > 0:
                confidences.append(finding.confidence)

            factors.append(
                {
                    "behavior": behavior,
                    "weight": base_weight,
                    "confidence": round(finding.confidence, 3),
                    "recency_factor": round(decay, 3),
                    "contribution": round(contribution, 2),
                    "severity": finding.severity,
                    "explanation": finding.explanation,
                    "track_id": finding.track_id,
                    "zone_id": finding.zone_id,
                    "counted": base_weight > 0,
                    "note": (
                        "Observability signal - carries no risk weight by design."
                        if behavior in ZERO_WEIGHT_BEHAVIORS else None
                    ),
                }
            )

        # Zone sensitivity scales the whole assessment.
        raw_total *= max(0.1, zone_risk_weight)

        # Saturating curve: approaches 100 but never fabricates certainty.
        score = 100.0 * (1.0 - math.exp(-raw_total / 55.0))
        score = float(min(99.0, max(0.0, score)))

        band = "low"
        for label in ("medium", "high", "critical"):
            if score >= float(bands.get(label, 100)):
                band = label
        if score < float(bands.get("low", 25)):
            band = "info" if score < 8 else "low"

        factors.sort(key=lambda f: -f["contribution"])
        dominant = factors[0]["behavior"] if factors and factors[0]["contribution"] > 0 else None
        confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return ThreatAssessment(
            score=score,
            band=band,
            factors=factors,
            explanation=self._explain(score, band, factors, context),
            confidence=confidence,
            dominant_factor=dominant,
        )

    def _recency(self, finding: Finding, now: datetime) -> float:
        ts = finding.ended_at or finding.started_at
        if ts is None:
            return 1.0
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = max(0.0, (now - ts).total_seconds())
        return float(0.5 ** (age / self.HALF_LIFE_SECONDS))

    @staticmethod
    def _explain(score: float, band: str, factors: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        counted = [f for f in factors if f["counted"] and f["contribution"] > 0.5]
        if not counted:
            return (
                "No weighted risk signals are currently active. "
                "Any observations present are descriptive only."
            )

        names = {
            "unattended_object": "an object left without an associated subject",
            "object_separation": "an object separating from its associated subject",
            "restricted_zone_entry": "entry into a restricted zone",
            "restricted_entry": "entry into a restricted zone",
            "abnormal_trajectory": "movement that is not goal-directed",
            "crowd_anomaly": "an anomaly in crowd movement",
            "crowd_surge": "a rapid rise in occupancy",
            "crowd_reversal": "the crowd reversing direction",
            "crowd_density_critical": "critical crowd density",
            "crowd_flow_anomaly": "incoherent crowd flow",
            "following_pattern": "a possible persistent following pattern",
            "following": "a possible persistent following pattern",
            "counter_flow": "movement against the prevailing flow",
            "loitering": "prolonged confined presence",
            "running": "running",
            "sudden_movement": "a sudden acceleration",
            "violence_detected": "possible violent activity",
            "anomalous_activity": "anomalous scene activity",
            "fire_smoke": "possible smoke or fire",
            "watchlist_match": "a possible sighting of an enrolled subject",
        }

        described = [names.get(f["behavior"], f["behavior"].replace("_", " ")) for f in counted[:4]]
        if len(described) == 1:
            reason = described[0]
        else:
            reason = ", ".join(described[:-1]) + f" and {described[-1]}"

        lead = f"Risk assessed at {score:.0f}% ({band.upper()}) because of {reason}."
        top = counted[0]
        detail = (
            f" The dominant contributor is '{top['behavior']}' "
            f"(weight {top['weight']:.0f} x confidence {top['confidence']:.2f} "
            f"x recency {top['recency_factor']:.2f} = {top['contribution']:.1f})."
        )
        zone = context.get("zone_name")
        where = f" Location: {zone}." if zone else ""
        return lead + detail + where
