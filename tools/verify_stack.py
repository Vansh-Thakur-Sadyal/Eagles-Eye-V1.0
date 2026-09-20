#!/usr/bin/env python
"""Verify what is actually installed and working, end to end.

    python tools/verify_stack.py
    python tools/verify_stack.py --bench      # also time inference on GPU vs CPU

Reports the real state of every optional component - and, where one is missing,
exactly which command installs it. Exits non-zero if a *required* piece is
broken; optional gaps are reported but do not fail.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class Report:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.failures = 0

    def add(self, name: str, ok: bool, detail: str, *, required: bool = False,
            fix: Optional[str] = None) -> None:
        self.rows.append({"name": name, "ok": ok, "detail": detail,
                          "required": required, "fix": fix})
        if required and not ok:
            self.failures += 1

    def render(self) -> None:
        width = max(len(r["name"]) for r in self.rows) + 2
        for row in self.rows:
            if row["ok"]:
                mark, colour = "OK  ", GREEN
            elif row["required"]:
                mark, colour = "FAIL", RED
            else:
                mark, colour = "----", YELLOW
            print(f"  {colour}{mark}{RESET} {row['name']:<{width}} {row['detail']}")
            if not row["ok"] and row["fix"]:
                print(f"       {DIM}fix: {row['fix']}{RESET}")


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * 74)


def check_core(report: Report) -> None:
    section("Core")
    for module, required, fix in (
        ("fastapi", True, "pip install -r backend/requirements.txt"),
        ("sqlalchemy", True, "pip install -r backend/requirements.txt"),
        ("cv2", True, "pip install opencv-python-headless"),
        ("numpy", True, "pip install numpy"),
        ("scipy", False, "pip install scipy  (Hungarian matching; greedy fallback otherwise)"),
    ):
        try:
            mod = __import__(module)
            report.add(module, True, getattr(mod, "__version__", "installed"), required=required)
        except ImportError as exc:
            report.add(module, False, str(exc), required=required, fix=fix)


def check_torch(report: Report) -> Dict[str, Any]:
    section("Compute")
    info: Dict[str, Any] = {"torch": False, "cuda": False}
    try:
        import torch
    except ImportError:
        report.add("torch", False, "not installed", fix=(
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        ))
        report.add("CUDA", False, "unavailable without torch")
        return info

    info["torch"] = True
    build = torch.version.cuda or "cpu-only build"
    report.add("torch", True, f"{torch.__version__}  (CUDA build: {build})")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["cuda"] = True
        info["gpu"] = props.name
        info["sm"] = f"{props.major}.{props.minor}"
        report.add("CUDA", True,
                   f"{props.name}  {props.total_memory / 1024**3:.1f} GB  SM {props.major}.{props.minor}")

        # A matching-but-unsupported build fails only when a kernel launches.
        try:
            x = torch.randn(64, 64, device="cuda")
            _ = (x @ x).sum().item()
            torch.cuda.synchronize()
            report.add("CUDA kernel launch", True, "matmul succeeded on device")
        except Exception as exc:
            report.add("CUDA kernel launch", False, f"{type(exc).__name__}: {exc}",
                       required=True,
                       fix=("your torch build does not support this GPU's compute capability; "
                            "install the cu128 wheel"))
    else:
        detail = "no CUDA device visible to torch"
        if torch.version.cuda is None:
            detail += " (this is a CPU-only torch build)"
        report.add("CUDA", False, detail, fix=(
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        ))
    return info


def check_models(report: Report) -> None:
    section("Models")

    try:
        import ultralytics

        report.add("ultralytics", True, ultralytics.__version__)
    except ImportError:
        report.add("ultralytics", False, "not installed", fix="pip install ultralytics")

    from app.vision.detector import build_detector

    detector = build_detector()
    info = detector.info()
    real = info["backend"] not in ("fallback-motion", "none")
    report.add("detector", real,
               f"{info['backend']}" + ("" if real else "  (no trained model - blob detection only)"),
               fix=None if real else "pip install ultralytics")

    from app.vision.reid import get_body_embedder, get_face_embedder

    body = get_body_embedder().info()
    # torchreid present is not enough: ImageNet weights were never trained to
    # tell two people apart, so report that distinction rather than a green tick.
    reid_ready = body.get("trained_for_reid", False)
    report.add(
        "re-identification", reid_ready,
        f"{body['backend']}  dim={body['dim']}  weights={body.get('weights', '?')}"
        + ("" if reid_ready else "  (ImageNet weights - not trained for re-ID)"),
        fix=("pip install torchreid" if not body["backend"].startswith("torchreid")
             else "drop a Market-1501 checkpoint in storage/models/reid/"),
    )

    face = get_face_embedder().info()
    report.add("face matching", face["backend"].startswith("insightface"),
               f"{face['backend']}  dim={face['dim']}",
               fix="pip install insightface onnxruntime-gpu")

    # onnxruntime's CUDA provider is fragile: anything that depends on plain
    # `onnxruntime` (insightface and chromadb both do) reinstalls the CPU build
    # over the GPU one, and the provider silently disappears. Check it directly
    # rather than trusting that a previous install still holds.
    try:
        import torch  # noqa: F401  - must be imported first; ORT reuses its DLLs
    except Exception:
        pass
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
        on_gpu = "CUDAExecutionProvider" in providers
        report.add(
            "onnxruntime provider",
            on_gpu or not face["backend"].startswith("insightface"),
            ", ".join(providers),
            fix=("pip uninstall -y onnxruntime && "
                 "pip install --force-reinstall --no-deps onnxruntime-gpu"),
        )
        if face.get("on_gpu") is False and on_gpu:
            report.add("face matching device", False,
                       "provider available but the embedder loaded on CPU - restart the API",
                       fix="restart the process so InsightFace re-initialises")
    except ImportError:
        pass


def check_rag(report: Report) -> None:
    section("Retrieval and language")

    from app.rag.llm import get_llm
    from app.rag.vectorstore import get_store

    store = get_store().info()
    report.add("vector store", store["backend"].startswith("chroma"),
               f"{store['backend']}  {store['documents']} documents",
               fix="pip install chromadb sentence-transformers")

    llm = get_llm()
    if llm.enabled:
        health = llm.health()
        report.add("language model", health["ok"],
                   f"{llm.provider}/{llm.model}  " + ("reachable" if health["ok"]
                                                      else health.get("reason", "unreachable")))
    else:
        report.add("language model", False,
                   "not configured - narration uses deterministic templates",
                   fix="set SENTINEL_LLM_PROVIDER and related values in backend/.env")


def check_pipeline(report: Report) -> None:
    section("Pipeline")
    from datetime import datetime, timezone

    from app.agents.base import CameraContext, FrameContext
    from app.agents.orchestrator import Orchestrator
    from app.pipeline.capture import SourceSpec, build_source
    from app.vision.detector import build_detector
    from app.vision.tracker import ByteTracker

    try:
        source = build_source(SourceSpec(source_type="synthetic", uri="",
                                         width=640, height=360, fps=12))
        assert source.open()
        # The synthetic scene renders schematic figures. A trained detector
        # correctly finds nothing in them, so this check would prove nothing
        # about the pipeline - use the motion backend, exactly as the synthetic
        # camera does in production.
        detector = build_detector("motion")
        tracker = ByteTracker(min_hits=2)
        orch = Orchestrator()

        peak_tracks = 0
        findings = 0
        started = time.perf_counter()
        for i in range(60):
            ok, frame = source.read()
            if not ok:
                break
            now = datetime.now(timezone.utc)
            detections = detector.detect(frame)
            tracks = tracker.update(detections, timestamp=now, fps=12)
            peak_tracks = max(peak_tracks, len(tracks))
            ctx = FrameContext(
                camera=CameraContext(camera_id="VERIFY", name="verify",
                                     width=frame.shape[1], height=frame.shape[0], fps=12),
                frame_index=i, timestamp=now, frame=frame,
                detections=detections, tracks=tracks,
                policy=orch.policy, elapsed_seconds=i / 12,
            )
            result = orch.process_frame(ctx)
            findings += result["findings"]
        elapsed = time.perf_counter() - started
        source.release()

        report.add("end-to-end", peak_tracks > 0 and findings > 0,
                   f"60 frames in {elapsed:.1f}s  ({60 / elapsed:.1f} fps)  "
                   f"peak {peak_tracks} tracks, {findings} findings", required=True)

        errored = [a for a in orch.status()["agents"] if a["last_error"]]
        report.add("agents", not errored,
                   "no errors" if not errored else
                   "; ".join(f"{a['name']}: {a['last_error']}" for a in errored),
                   required=True)
    except Exception as exc:
        report.add("end-to-end", False, f"{type(exc).__name__}: {exc}", required=True)


def benchmark(compute: Dict[str, Any]) -> None:
    section("Inference benchmark")
    if not compute.get("cuda"):
        print("  skipped: no CUDA device")
        return

    import numpy as np

    from app.core.device import get_device_manager
    from app.vision.detector import build_detector

    detector = build_detector()
    if detector.backend == "fallback-motion":
        print("  skipped: no trained detector installed")
        return

    frame = (np.random.rand(720, 1280, 3) * 255).astype("uint8")
    manager = get_device_manager()

    for label, enabled in (("GPU", True), ("CPU", False)):
        manager.set_gpu(enabled)
        detector.detect(frame)                       # warm up
        started = time.perf_counter()
        for _ in range(12):
            detector.detect(frame)
        per_frame = (time.perf_counter() - started) / 12
        print(f"  {label:<4} {per_frame * 1000:7.1f} ms/frame   {1 / per_frame:6.1f} fps")

    manager.set_gpu(True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Sentinel stack")
    parser.add_argument("--bench", action="store_true", help="time GPU vs CPU inference")
    args = parser.parse_args()

    print("=" * 74)
    print("  Sentinel AI - stack verification")
    print("=" * 74)

    report = Report()
    check_core(report)
    compute = check_torch(report)
    check_models(report)
    check_rag(report)
    check_pipeline(report)

    print("\n" + "=" * 74)
    report.render()
    print("=" * 74)

    optional = [r for r in report.rows if not r["ok"] and not r["required"]]
    if report.failures:
        print(f"\n{RED}{report.failures} required component(s) are broken.{RESET}")
    elif optional:
        print(f"\n{YELLOW}Running, with {len(optional)} optional component(s) on fallbacks.{RESET}")
        print("The dashboard reports each fallback honestly in Settings -> Compute & models.")
    else:
        print(f"\n{GREEN}Full stack present.{RESET}")

    if args.bench:
        benchmark(compute)

    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
