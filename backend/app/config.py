"""Runtime configuration for Eagles Eye.

Every tunable lives here and is sourced from the environment (.env) or from
JSON policy files on disk.  No thresholds, camera lists, zone definitions or
model names are hardcoded anywhere else in the codebase.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (BACKEND_DIR / path).resolve()
    return path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_prefix="SENTINEL_",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    storage_dir: Path = Path("../storage")
    dataset_dir: Path = Path("../datasets")
    config_dir: Path = Path("../config")

    database_url: str = "sqlite:///../storage/sentinel.db"

    jwt_secret: str = "change-me-in-production"
    jwt_ttl_minutes: int = 720
    # Kept as a raw string: pydantic-settings would try to JSON-decode a
    # list-typed field coming from .env.  Read it via `cors_origin_list`.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # compute
    device_mode: str = "auto"          # auto | cuda | cpu
    gpu_index: int = 0
    gpu_memory_fraction: float = 0.85
    use_half_precision: bool = True

    # models
    detector: str = "yolo"
    yolo_weights: str = "yolo11m.pt"
    yolo_conf: float = 0.35
    yolo_iou: float = 0.5
    yolo_imgsz: int = 640
    # NVIDIA License: non-commercial (research / evaluation) use only.
    locateanything_model: str = "nvidia/LocateAnything-3B"
    locateanything_enabled: bool = False
    # One keyframe call costs ~2 s of GPU time, and YOLO shares that GPU, so
    # every call briefly stalls the camera loop. 120 frames = one call per 10 s
    # at 12 fps. Lower it only if the GPU is not also serving live video.
    locateanything_every_n_frames: int = 120
    # "slow" = plain autoregressive decoding. On this stack (Windows, SDPA
    # fallback, no MagiAttention) the parallel "hybrid"/"fast" modes fell into
    # degenerate box loops under deterministic decoding; slow mode was correct.
    locateanything_mode: str = "slow"        # fast | slow | hybrid (model card)
    reid_weights: str = "osnet_x1_0"
    face_model: str = "buffalo_l"
    tracker: str = "bytetrack"

    # pipeline
    target_fps: int = 12
    max_concurrent_cameras: int = 8
    frame_queue_size: int = 4

    # Where camera processing runs (see docs/EDGE_GPU.md):
    #   auto    - on a connected GPU edge node if one is online, else here
    #   server  - always on this machine
    #   edge    - only on edge nodes; refuse to start a camera if none is online
    # A camera overrides it with meta.processing_node = "server" | "auto" |
    # "edge" | <node id> (pin it to one node, e.g. the PC its webcam is on).
    processing_placement: str = "auto"
    edge_node_timeout_seconds: float = 30.0
    snapshot_on_incident: bool = True

    # portal (public sign-up / sign-in). The identity store is the Sentinel_Auth
    # workflow in n8n: scrypt passwords, e-mail + SMS one-time codes, Mongo.
    portal_auth_base_url: str = ""
    portal_enabled: bool = True
    portal_admin_email: str = ""          # gets a copy of every sign-up
    portal_notify_key: str = ""           # shared secret for the n8n notify hook

    # llm
    llm_provider: str = "none"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout: int = 60

    # rag
    vector_backend: str = "chroma"
    vector_dir: Path = Path("../storage/vectors")
    embedding_model: str = "all-MiniLM-L6-v2"

    # automation
    n8n_enabled: bool = False
    n8n_base_url: str = ""
    n8n_webhook_incident: str = ""
    n8n_webhook_unattended: str = ""
    n8n_webhook_daily_report: str = ""
    n8n_webhook_system_health: str = ""

    # privacy
    privacy_redaction: bool = True
    privacy_anonymous_by_default: bool = True
    retention_days: int = 30
    audit_all_reads: bool = True

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @field_validator("storage_dir", "dataset_dir", "config_dir", "vector_dir", mode="after")
    @classmethod
    def _abs(cls, v: Path) -> Path:
        return _resolve(v)

    def sqlalchemy_url(self) -> str:
        """Turn a relative sqlite path into an absolute one."""
        url = self.database_url
        if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
            rel = url[len("sqlite:///"):]
            return f"sqlite:///{_resolve(rel).as_posix()}"
        return url

    def ensure_dirs(self) -> None:
        for d in (self.storage_dir, self.dataset_dir, self.config_dir, self.vector_dir):
            d.mkdir(parents=True, exist_ok=True)
        for sub in ("snapshots", "clips", "evidence", "watchlist", "models", "reports"):
            (self.storage_dir / sub).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


# --------------------------------------------------------------------------
# Policy files — operator-editable JSON, hot-reloaded, never baked into code.
# --------------------------------------------------------------------------

DEFAULT_POLICY: Dict[str, Any] = {
    "threat_weights": {
        "unattended_object": 35,
        "restricted_zone_entry": 20,
        "abnormal_trajectory": 12,
        "crowd_anomaly": 10,
        "following_pattern": 8,
        "counter_flow": 8,
        "loitering": 6,
        "sudden_running": 6,
        "crowd_reversal": 10,
        "violence_detected": 40,
        "anomalous_activity": 15,
        "fire_smoke": 38,
        "watchlist_match": 30,
        "face_unavailable": 0,
    },
    "threat_bands": {"low": 25, "medium": 50, "high": 75, "critical": 90},
    "behavior": {
        "loitering_seconds": 120,
        "loitering_radius_px": 90,
        "running_speed_px_s": 210,
        "counter_flow_angle_deg": 120,
        "counter_flow_min_frames": 15,
        "sudden_movement_accel": 160,
        "min_track_age_frames": 8
    },
    "following": {
        "min_duration_seconds": 25,
        "max_pair_distance_px": 260,
        "min_distance_px": 30,
        "trajectory_similarity": 0.72,
        "direction_change_matches": 2,
        "confidence_floor": 0.60
    },
    "crowd": {
        "density_bands": {"low": 0.15, "medium": 0.35, "high": 0.6, "critical": 0.8},
        "cell_size_px": 64,
        "flow_window_seconds": 30,
        "surge_percent_threshold": 40,
        "reversal_ratio": 0.6
    },
    "object": {
        "separation_distance_px": 180,
        "stationary_tolerance_px": 22,
        "unattended_seconds": 60,
        "warning_seconds": 30,
        "tracked_classes": ["backpack", "handbag", "suitcase", "bottle", "laptop"]
    },
    "reid": {
        "match_threshold": 0.62,
        "gallery_size": 5000,
        "max_time_gap_seconds": 900,
        "embedding_dim": 512
    },
    "watchlist": {
        "face_match_threshold": 0.55,
        "body_match_threshold": 0.60,
        "require_approval": True,
        "alert_on_match": True
    },
    "occlusion": {
        "face_visible_min_area": 0.012,
        "low_confidence_below": 0.45
    },
    "privacy": {
        "redact_bystanders": True,
        "identity_escalation_requires_approval": True,
        "retention_days": 30
    },
}


def policy_path() -> Path:
    return get_settings().config_dir / "policy.json"


def load_policy() -> Dict[str, Any]:
    """Read the operator policy file, seeding it from defaults on first run."""
    p = policy_path()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(DEFAULT_POLICY, indent=2), encoding="utf-8")
        return json.loads(json.dumps(DEFAULT_POLICY))
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return json.loads(json.dumps(DEFAULT_POLICY))
    # deep-merge so new keys added by an upgrade appear without losing edits
    return _merge(json.loads(json.dumps(DEFAULT_POLICY)), data)


def save_policy(data: Dict[str, Any]) -> Dict[str, Any]:
    merged = _merge(load_policy(), data)
    policy_path().write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _merge(base[k], v)
        else:
            base[k] = v
    return base
