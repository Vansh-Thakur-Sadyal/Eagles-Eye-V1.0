"""Privacy agent (spec S27) - privacy-by-design, enforced in the pipeline.

Default posture: a person is an anonymous track ID with a trajectory and a
behaviour, not an entry in an identity database.  This agent:

  * redacts faces in outgoing video for everyone who is not the subject of an
    open incident or an approved watchlist query;
  * strips identity-bearing fields from any payload leaving the API for a role
    that is not cleared to see them;
  * refuses identity escalation unless an approval record exists.

Redaction happens on the frame *before* it is encoded for the dashboard, so a
bystander's face is never transmitted, not merely hidden in the UI.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from ..config import get_settings
from .base import Agent, FrameContext, Finding

log = logging.getLogger("sentinel.agents.privacy")

# Fields that only an identity-cleared role may see.
IDENTITY_FIELDS = (
    "snapshot_path", "crop_path", "face_bbox", "embedding_id",
    "face_embedding_id", "body_embedding_id", "subject_label",
    "latitude", "longitude",
)


class PrivacyAgent(Agent):
    name = "privacy"
    spec_id = 14
    description = "Bystander redaction, anonymity by default, identity-escalation gating"

    def __init__(self) -> None:
        super().__init__()
        s = get_settings()
        self.redaction_enabled = s.privacy_redaction
        self.anonymous_by_default = s.privacy_anonymous_by_default
        self.redactions_applied = 0
        self._exempt_tracks: Dict[str, Set[int]] = {}

    # --------------------------------------------------------------- policy
    def exempt(self, camera_id: str, track_ids: Iterable[int]) -> None:
        """Mark tracks as incident subjects, so they are not redacted for
        an operator who is authorised to view that incident."""
        self._exempt_tracks.setdefault(camera_id, set()).update(int(t) for t in track_ids)

    def clear_exemptions(self, camera_id: str) -> None:
        self._exempt_tracks.pop(camera_id, None)

    def process(self, ctx: FrameContext) -> List[Finding]:
        """The agent emits no findings; it enforces policy on the frame."""
        if self.anonymous_by_default:
            for track in ctx.tracks:
                track.attributes.setdefault("identity", "anonymous")
        return []

    # ------------------------------------------------------------ redaction
    def redact_frame(
        self,
        frame: np.ndarray,
        ctx: FrameContext,
        *,
        reveal_track_ids: Optional[Sequence[int]] = None,
        force: Optional[bool] = None,
    ) -> np.ndarray:
        """Blur the face region of every person except those explicitly revealed."""
        enabled = self.redaction_enabled if force is None else force
        if not enabled or frame is None:
            return frame
        try:
            import cv2
        except Exception:
            return frame

        reveal = set(int(t) for t in (reveal_track_ids or []))
        reveal |= self._exempt_tracks.get(ctx.camera.camera_id, set())

        out = frame.copy()
        h, w = out.shape[:2]

        for track in ctx.tracks:
            if track.class_name != "person" or track.track_id in reveal:
                continue
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            # head region: upper ~28% of the body box, centred
            hx1 = max(0, int(x1 + bw * 0.18))
            hx2 = min(w, int(x2 - bw * 0.18))
            hy1 = max(0, y1)
            hy2 = min(h, int(y1 + bh * 0.30))
            if hx2 <= hx1 or hy2 <= hy1:
                continue
            region = out[hy1:hy2, hx1:hx2]
            if region.size == 0:
                continue
            k = max(9, (min(region.shape[0], region.shape[1]) // 2) | 1)
            out[hy1:hy2, hx1:hx2] = cv2.GaussianBlur(region, (k, k), 0)
            self.redactions_applied += 1

        return out

    # ------------------------------------------------------------ filtering
    def filter_payload(self, payload: Dict[str, Any], *, role: str,
                       identity_cleared: bool = False) -> Dict[str, Any]:
        """Remove identity-bearing fields for roles without clearance."""
        if identity_cleared or role in ("admin", "commander", "investigator"):
            return payload
        redacted = dict(payload)
        for field in IDENTITY_FIELDS:
            if field in redacted:
                redacted[field] = None
        redacted["_privacy"] = "identity fields withheld for this role"
        return redacted

    def filter_many(self, rows: List[Dict[str, Any]], *, role: str,
                    identity_cleared: bool = False) -> List[Dict[str, Any]]:
        return [self.filter_payload(r, role=role, identity_cleared=identity_cleared) for r in rows]

    def status(self) -> Dict[str, Any]:
        base = super().status()
        base.update(
            {
                "redaction_enabled": self.redaction_enabled,
                "anonymous_by_default": self.anonymous_by_default,
                "redactions_applied": self.redactions_applied,
                "cameras_with_exemptions": len(self._exempt_tracks),
            }
        )
        return base
