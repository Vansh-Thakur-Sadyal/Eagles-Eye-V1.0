"""Compute device management - the runtime GPU on/off switch.

Eagles Eye can execute inference in three places. The dashboard toggle selects
between them at runtime, with no restart and no OS-level permission prompt:

  1. ``server``  - torch models on the machine running this API (CUDA or CPU).
  2. ``worker``  - a registered remote GPU node (e.g. the operator's own
                   workstation) that pulls inference jobs from the API. This
                   is how a VPS-hosted dashboard uses a discrete GPU that is
                   not inside the VPS.
  3. ``client``  - WebGPU in the operator's browser, used for digital-twin and
                   heatmap rendering plus optional ONNX inference. WebGPU asks
                   for the high-performance adapter, which selects the discrete
                   GPU without prompting the user.

Turning the toggle off moves every loaded torch module to CPU immediately and
tells the frontend to drop to a software renderer.
"""
from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..config import get_settings

log = logging.getLogger("sentinel.device")


@dataclass
class GpuInfo:
    index: int
    name: str
    total_memory_mb: int
    compute_capability: str = ""
    driver: str = ""


@dataclass
class WorkerNode:
    """A remote machine offering its discrete GPU to this deployment."""

    node_id: str
    name: str
    device: str
    gpu_name: str
    total_memory_mb: int
    capabilities: List[str] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)
    jobs_completed: int = 0

    @property
    def online(self) -> bool:
        return (time.time() - self.last_heartbeat) < 30.0


class DeviceManager:
    """Singleton owning the active compute device for every torch model."""

    def __init__(self) -> None:
        s = get_settings()
        self._lock = threading.RLock()
        self._requested_mode: str = s.device_mode          # auto | cuda | cpu
        self._gpu_enabled: bool = s.device_mode != "cpu"
        self._gpu_index: int = s.gpu_index
        self._half: bool = s.use_half_precision
        self._torch: Any = None
        self._cuda_available: bool = False
        self._gpus: List[GpuInfo] = []
        self._registry: Dict[str, Any] = {}                # model_key -> module
        self._listeners: List[Callable[[str], None]] = []
        self._workers: Dict[str, WorkerNode] = {}
        self._probe()

    # ------------------------------------------------------------------ probe
    def _probe(self) -> None:
        try:
            import torch

            self._torch = torch
            self._cuda_available = bool(torch.cuda.is_available())
            if self._cuda_available:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    self._gpus.append(
                        GpuInfo(
                            index=i,
                            name=props.name,
                            total_memory_mb=int(props.total_memory / (1024 * 1024)),
                            compute_capability=f"{props.major}.{props.minor}",
                            driver=getattr(torch.version, "cuda", "") or "",
                        )
                    )
        except Exception as exc:
            log.info("torch unavailable, running in CPU/degraded mode: %s", exc)
            self._torch = None
            self._cuda_available = False

        if not self._gpus:
            # Report hardware even without torch, so the dashboard shows truth.
            self._gpus = _nvidia_smi_probe()

    # ----------------------------------------------------------------- state
    @property
    def torch(self) -> Any:
        return self._torch

    @property
    def cuda_available(self) -> bool:
        return self._cuda_available

    @property
    def active_device(self) -> str:
        with self._lock:
            if self._gpu_enabled and self._cuda_available:
                return f"cuda:{self._gpu_index}"
            return "cpu"

    @property
    def gpu_enabled(self) -> bool:
        with self._lock:
            return self._gpu_enabled and self._cuda_available

    @property
    def use_half(self) -> bool:
        return self._half and self.gpu_enabled

    def status(self) -> Dict[str, Any]:
        with self._lock:
            mem: Dict[str, Any] = {}
            if self._cuda_available and self._torch is not None:
                try:
                    free, total = self._torch.cuda.mem_get_info(self._gpu_index)
                    mem = {
                        "free_mb": int(free / 1024 / 1024),
                        "total_mb": int(total / 1024 / 1024),
                        "used_mb": int((total - free) / 1024 / 1024),
                        "allocated_mb": int(
                            self._torch.cuda.memory_allocated(self._gpu_index) / 1024 / 1024
                        ),
                        "utilisation_percent": round((total - free) / total * 100, 1) if total else 0.0,
                    }
                except Exception:
                    mem = {}
            torch_version = getattr(self._torch, "__version__", None) if self._torch else None
            cuda_version = None
            if self._torch is not None:
                cuda_version = getattr(getattr(self._torch, "version", None), "cuda", None)
            return {
                "requested_mode": self._requested_mode,
                "gpu_enabled": self._gpu_enabled,
                "cuda_available": self._cuda_available,
                "active_device": self.active_device,
                "half_precision": self.use_half,
                "gpu_index": self._gpu_index,
                "torch_available": self._torch is not None,
                "torch_version": torch_version,
                "cuda_version": cuda_version,
                "gpus": [g.__dict__ for g in self._gpus],
                "memory": mem,
                "loaded_models": sorted(self._registry.keys()),
                "workers": [{**w.__dict__, "online": w.online} for w in self._workers.values()],
            }

    # --------------------------------------------------------------- toggling
    def set_gpu(self, enabled: bool, gpu_index: Optional[int] = None) -> Dict[str, Any]:
        """Flip the compute target and relocate every registered model."""
        with self._lock:
            if gpu_index is not None:
                self._gpu_index = max(0, int(gpu_index))
            self._gpu_enabled = bool(enabled)
            self._requested_mode = "cuda" if enabled else "cpu"
            target = self.active_device
            moved = self._relocate(target)
        for cb in list(self._listeners):
            try:
                cb(target)
            except Exception:
                log.exception("device listener failed")
        log.info("compute target -> %s (%d models moved)", target, moved)
        return self.status()

    def _relocate(self, target: str) -> int:
        if self._torch is None:
            return 0
        moved = 0
        for key, model in list(self._registry.items()):
            try:
                mover = getattr(model, "to", None)
                if not callable(mover):
                    continue
                model.to(target)
                if target.startswith("cuda") and self._half:
                    with contextlib.suppress(Exception):
                        model.half()
                else:
                    with contextlib.suppress(Exception):
                        model.float()
                moved += 1
            except Exception:
                log.exception("could not move model %s to %s", key, target)
        if target == "cpu" and self._cuda_available:
            with contextlib.suppress(Exception):
                self._torch.cuda.empty_cache()
        return moved

    # -------------------------------------------------------------- registry
    def register_model(self, key: str, model: Any) -> Any:
        """Track a model so the toggle can relocate it later."""
        with self._lock:
            self._registry[key] = model
            target = self.active_device
        try:
            if hasattr(model, "to"):
                model.to(target)
                if target.startswith("cuda") and self._half and hasattr(model, "half"):
                    model.half()
        except Exception:
            log.exception("initial placement of %s failed", key)
        return model

    def unregister_model(self, key: str) -> None:
        with self._lock:
            self._registry.pop(key, None)

    def on_change(self, callback: Callable[[str], None]) -> None:
        self._listeners.append(callback)

    # --------------------------------------------------------- remote workers
    def register_worker(self, payload: Dict[str, Any]) -> WorkerNode:
        node = WorkerNode(
            node_id=str(payload["node_id"]),
            name=str(payload.get("name", payload["node_id"])),
            device=str(payload.get("device", "cuda:0")),
            gpu_name=str(payload.get("gpu_name", "unknown")),
            total_memory_mb=int(payload.get("total_memory_mb", 0)),
            capabilities=list(payload.get("capabilities", [])),
        )
        with self._lock:
            existing = self._workers.get(node.node_id)
            if existing:
                node.jobs_completed = existing.jobs_completed
            self._workers[node.node_id] = node
        return node

    def heartbeat_worker(self, node_id: str, jobs_completed: Optional[int] = None) -> bool:
        with self._lock:
            node = self._workers.get(node_id)
            if not node:
                return False
            node.last_heartbeat = time.time()
            if jobs_completed is not None:
                node.jobs_completed = int(jobs_completed)
            return True

    def drop_worker(self, node_id: str) -> bool:
        with self._lock:
            return self._workers.pop(node_id, None) is not None

    def pick_worker(self, capability: str) -> Optional[WorkerNode]:
        with self._lock:
            candidates = [
                w
                for w in self._workers.values()
                if w.online and (not w.capabilities or capability in w.capabilities)
            ]
        if not candidates:
            return None
        return min(candidates, key=lambda w: w.jobs_completed)

    def autocast(self):
        """Mixed-precision context manager, a no-op on CPU."""
        if self._torch is not None and self.gpu_enabled:
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
        return contextlib.nullcontext()


def _nvidia_smi_probe() -> List[GpuInfo]:
    """Report GPUs even when torch is missing, so the dashboard is honest."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpus: List[GpuInfo] = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append(GpuInfo(int(parts[0]), parts[1], int(float(parts[2])), driver=parts[3]))
        return gpus
    except Exception:
        return []


_manager: Optional[DeviceManager] = None
_manager_lock = threading.Lock()


def get_device_manager() -> DeviceManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = DeviceManager()
    return _manager
