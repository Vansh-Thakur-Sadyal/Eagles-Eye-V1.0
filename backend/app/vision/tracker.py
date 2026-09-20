"""ByteTrack multi-object tracker (spec Agent 2).

A self-contained implementation: constant-velocity Kalman filter in
(cx, cy, aspect, height) space plus ByteTrack's two-stage association, which
recovers occluded subjects by matching low-confidence detections that a
single-threshold tracker would throw away.

Optional appearance gating (DeepSORT-style) is applied when the caller supplies
Re-ID embeddings, which materially helps in the crowded scenes the platform
targets.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import Detection, TrackState


# ---------------------------------------------------------------- utilities
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def linear_assignment(cost: np.ndarray, threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian assignment with a cost cutoff; greedy fallback if SciPy is absent."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= threshold]
    except Exception:
        pairs = []
        used_r, used_c = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in order:
            r, c = int(r), int(c)
            if r in used_r or c in used_c or cost[r, c] > threshold:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
    matched_r = {r for r, _ in pairs}
    matched_c = {c for _, c in pairs}
    un_r = [i for i in range(cost.shape[0]) if i not in matched_r]
    un_c = [j for j in range(cost.shape[1]) if j not in matched_c]
    return pairs, un_r, un_c


class KalmanBox:
    """8-state constant-velocity filter over (cx, cy, aspect, height)."""

    def __init__(self, bbox: Sequence[float]) -> None:
        self.ndim = 4
        dt = 1.0
        self._motion = np.eye(8, dtype=np.float32)
        for i in range(self.ndim):
            self._motion[i, self.ndim + i] = dt
        self._update = np.eye(4, 8, dtype=np.float32)
        self._std_pos = 1.0 / 20
        self._std_vel = 1.0 / 160

        z = self._to_z(bbox)
        self.mean = np.concatenate([z, np.zeros(4, dtype=np.float32)])
        std = np.array(
            [
                2 * self._std_pos * z[3], 2 * self._std_pos * z[3], 1e-2, 2 * self._std_pos * z[3],
                10 * self._std_vel * z[3], 10 * self._std_vel * z[3], 1e-5, 10 * self._std_vel * z[3],
            ],
            dtype=np.float32,
        )
        self.covariance = np.diag(np.square(std))

    @staticmethod
    def _to_z(bbox: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w = max(1e-3, x2 - x1)
        h = max(1e-3, y2 - y1)
        return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=np.float32)

    @staticmethod
    def _to_bbox(z: np.ndarray) -> Tuple[float, float, float, float]:
        cx, cy, a, h = z[:4]
        w = max(1e-3, a * h)
        return float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)

    def predict(self) -> None:
        h = max(1e-3, abs(float(self.mean[3])))
        std = np.array(
            [
                self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
                self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h,
            ],
            dtype=np.float32,
        )
        motion_cov = np.diag(np.square(std))
        self.mean = self._motion @ self.mean
        self.covariance = self._motion @ self.covariance @ self._motion.T + motion_cov

    def update(self, bbox: Sequence[float]) -> None:
        z = self._to_z(bbox)
        h = max(1e-3, abs(float(self.mean[3])))
        std = np.array([self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h], dtype=np.float32)
        innovation_cov = np.diag(np.square(std))

        projected_mean = self._update @ self.mean
        projected_cov = self._update @ self.covariance @ self._update.T + innovation_cov
        try:
            kalman_gain = self.covariance @ self._update.T @ np.linalg.inv(projected_cov)
        except np.linalg.LinAlgError:
            kalman_gain = self.covariance @ self._update.T @ np.linalg.pinv(projected_cov)
        innovation = z - projected_mean
        self.mean = self.mean + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ projected_cov @ kalman_gain.T

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self._to_bbox(self.mean)

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.mean[4]), float(self.mean[5])


class _Candidate:
    __slots__ = ("kf", "state", "time_since_update", "hits", "confirmed")

    def __init__(self, state: TrackState) -> None:
        self.kf = KalmanBox(state.bbox)
        self.state = state
        self.time_since_update = 0
        self.hits = 1
        self.confirmed = False


class ByteTracker:
    """Per-camera tracker instance."""

    def __init__(
        self,
        *,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        second_match_thresh: float = 0.5,
        max_age: int = 30,
        min_hits: int = 3,
        appearance_weight: float = 0.0,
        history_limit: int = 900,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.appearance_weight = appearance_weight
        self.history_limit = history_limit

        self._tracks: Dict[int, _Candidate] = {}
        self._next_id = 1
        self.frame_index = 0

    # ------------------------------------------------------------------ api
    def update(
        self,
        detections: List[Detection],
        *,
        timestamp: datetime,
        fps: float = 12.0,
        embeddings: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[TrackState]:
        self.frame_index += 1
        embeddings = embeddings or [None] * len(detections)

        for cand in self._tracks.values():
            cand.kf.predict()
            cand.time_since_update += 1

        high_idx = [i for i, d in enumerate(detections) if d.score >= self.high_thresh]
        low_idx = [
            i for i, d in enumerate(detections)
            if self.low_thresh <= d.score < self.high_thresh
        ]

        track_ids = list(self._tracks.keys())
        track_boxes = np.array(
            [self._tracks[t].kf.bbox for t in track_ids], dtype=np.float32
        ) if track_ids else np.zeros((0, 4), dtype=np.float32)

        # ---- stage 1: high-confidence detections --------------------------
        matched, un_tracks, un_dets = self._associate(
            track_ids, track_boxes, detections, high_idx, embeddings, self.match_thresh
        )
        for t_pos, d_pos in matched:
            self._apply(track_ids[t_pos], detections[high_idx[d_pos]],
                        embeddings[high_idx[d_pos]], timestamp, fps)

        # ---- stage 2: rescue with low-confidence detections ---------------
        if low_idx and un_tracks:
            rem_ids = [track_ids[i] for i in un_tracks]
            rem_boxes = np.array([self._tracks[t].kf.bbox for t in rem_ids], dtype=np.float32)
            matched2, un_tracks2, _ = self._associate(
                rem_ids, rem_boxes, detections, low_idx, embeddings,
                self.second_match_thresh, use_appearance=False,
            )
            for t_pos, d_pos in matched2:
                self._apply(rem_ids[t_pos], detections[low_idx[d_pos]],
                            embeddings[low_idx[d_pos]], timestamp, fps)
            # map the still-unmatched positions back into `track_ids` space
            un_tracks = [track_ids.index(rem_ids[i]) for i in un_tracks2]

        # ---- spawn new tracks from unmatched high detections --------------
        for d_pos in un_dets:
            det = detections[high_idx[d_pos]]
            self._spawn(det, embeddings[high_idx[d_pos]], timestamp)

        # ---- retire stale tracks ------------------------------------------
        for tid in [t for t, c in self._tracks.items() if c.time_since_update > self.max_age]:
            self._tracks.pop(tid, None)

        return self.active_tracks()

    def _associate(
        self,
        track_ids: List[int],
        track_boxes: np.ndarray,
        detections: List[Detection],
        det_index: List[int],
        embeddings: List[Optional[np.ndarray]],
        threshold: float,
        use_appearance: bool = True,
    ):
        if not track_ids or not det_index:
            return [], list(range(len(track_ids))), list(range(len(det_index)))

        det_boxes = np.array([detections[i].bbox for i in det_index], dtype=np.float32)
        ious = iou_matrix(track_boxes, det_boxes)
        cost = 1.0 - ious

        # class gate: never associate a suitcase track with a person detection
        for ti, tid in enumerate(track_ids):
            t_cls = self._tracks[tid].state.class_name
            for di, i in enumerate(det_index):
                if detections[i].class_name != t_cls:
                    cost[ti, di] = 1.0

        if use_appearance and self.appearance_weight > 0:
            app = np.ones_like(cost)
            for ti, tid in enumerate(track_ids):
                emb_t = self._tracks[tid].state.embedding
                if emb_t is None:
                    continue
                for di, i in enumerate(det_index):
                    emb_d = embeddings[i]
                    if emb_d is None:
                        continue
                    app[ti, di] = 1.0 - float(np.dot(emb_t, emb_d))
            cost = (1 - self.appearance_weight) * cost + self.appearance_weight * app

        # cost is 1-IoU, so `threshold` is ByteTrack's match_thresh directly
        return linear_assignment(cost, threshold)

    def _apply(
        self,
        track_id: int,
        det: Detection,
        embedding: Optional[np.ndarray],
        timestamp: datetime,
        fps: float,
    ) -> None:
        cand = self._tracks[track_id]
        cand.kf.update(det.bbox)
        cand.time_since_update = 0
        cand.hits += 1
        if cand.hits >= self.min_hits:
            cand.confirmed = True

        st = cand.state
        prev_center = st.center
        st.bbox = cand.kf.bbox
        st.score = det.score
        st.last_frame = self.frame_index
        st.last_seen = timestamp
        st.attributes.update(det.attributes)

        cx, cy = st.center
        dt = 1.0 / max(1e-3, fps)
        vx = (cx - prev_center[0]) / dt
        vy = (cy - prev_center[1]) / dt
        # light smoothing keeps heading stable enough for counter-flow logic
        st.velocity = (0.6 * st.velocity[0] + 0.4 * vx, 0.6 * st.velocity[1] + 0.4 * vy)
        st.speed = float(np.hypot(*st.velocity))

        st.history.append(
            {
                "t": timestamp.isoformat(),
                "f": self.frame_index,
                "x": round(cx, 2),
                "y": round(cy, 2),
                "w": round(st.bbox[2] - st.bbox[0], 2),
                "h": round(st.bbox[3] - st.bbox[1], 2),
                "conf": round(det.score, 3),
                "vx": round(st.velocity[0], 2),
                "vy": round(st.velocity[1], 2),
            }
        )
        if len(st.history) > self.history_limit:
            del st.history[: len(st.history) - self.history_limit]

        if embedding is not None:
            st.embedding_history.append(embedding)
            if len(st.embedding_history) > 20:
                st.embedding_history.pop(0)
            stacked = np.stack(st.embedding_history)
            avg = stacked.mean(axis=0)
            norm = np.linalg.norm(avg)
            st.embedding = avg / norm if norm > 0 else avg

    def _spawn(self, det: Detection, embedding: Optional[np.ndarray], timestamp: datetime) -> None:
        tid = self._next_id
        self._next_id += 1
        cx, cy = det.center
        state = TrackState(
            track_id=tid,
            class_name=det.class_name,
            bbox=det.bbox,
            score=det.score,
            first_frame=self.frame_index,
            last_frame=self.frame_index,
            first_seen=timestamp,
            last_seen=timestamp,
            embedding=embedding,
            attributes=dict(det.attributes),
        )
        state.history.append(
            {
                "t": timestamp.isoformat(),
                "f": self.frame_index,
                "x": round(cx, 2),
                "y": round(cy, 2),
                "w": round(det.bbox[2] - det.bbox[0], 2),
                "h": round(det.bbox[3] - det.bbox[1], 2),
                "conf": round(det.score, 3),
                "vx": 0.0,
                "vy": 0.0,
            }
        )
        if embedding is not None:
            state.embedding_history.append(embedding)
        self._tracks[tid] = _Candidate(state)

    # ------------------------------------------------------------- readers
    def active_tracks(self) -> List[TrackState]:
        return [
            c.state
            for c in self._tracks.values()
            if c.time_since_update == 0 and (c.confirmed or c.hits >= self.min_hits)
        ]

    def all_tracks(self) -> List[TrackState]:
        return [c.state for c in self._tracks.values()]

    def get(self, track_id: int) -> Optional[TrackState]:
        cand = self._tracks.get(track_id)
        return cand.state if cand else None

    def lost_tracks(self) -> List[TrackState]:
        return [c.state for c in self._tracks.values() if c.time_since_update > 0]

    def reset(self) -> None:
        self._tracks.clear()
        self.frame_index = 0
