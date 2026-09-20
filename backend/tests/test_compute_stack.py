"""Compute-stack consistency.

These guard against a failure mode that is completely silent in normal use:
installing a package that depends on `torch` without an index pin (ultralytics
is the usual culprit) replaces a working CUDA build with the CPU-only wheel.
Nothing errors. `torch.cuda.is_available()` just flips to False and every
inference quietly runs on CPU at a fraction of the speed.

The tests skip cleanly on a machine with no NVIDIA GPU, so they are safe in CI.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from app.core.device import get_device_manager


def _nvidia_smi_gpus():
    """What the driver reports, independent of torch."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
    except Exception:
        return []


HAS_NVIDIA_GPU = bool(_nvidia_smi_gpus())

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def test_device_manager_reports_hardware_even_without_torch():
    """The dashboard must show a GPU that exists, whatever torch thinks."""
    manager = get_device_manager()
    status = manager.status()
    if HAS_NVIDIA_GPU:
        assert status["gpus"], (
            "nvidia-smi sees a GPU but the device manager reports none - "
            "the dashboard would hide hardware that is present"
        )


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_a_cuda_torch_build_is_not_silently_a_cpu_build():
    """A CUDA build on a CUDA machine must actually reach the device.

    If this fails with a GPU present, something reinstalled torch from plain
    PyPI. Repair with:
        pip install --force-reinstall torch torchvision \\
            --index-url https://download.pytorch.org/whl/cu128
    """
    if not HAS_NVIDIA_GPU:
        pytest.skip("no NVIDIA GPU on this host")

    build = torch.version.cuda
    assert build is not None, (
        f"torch {torch.__version__} is a CPU-only build on a machine with "
        f"{_nvidia_smi_gpus()}. Something replaced the CUDA wheel - see "
        "backend/requirements-gpu.txt."
    )
    assert torch.cuda.is_available(), (
        f"torch {torch.__version__} reports a CUDA {build} build but no device "
        "is available. Check the driver version against the CUDA build."
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_kernels_actually_launch_on_this_gpu():
    """A build can match on paper and still have no kernel for this architecture.

    Blackwell (SM 12.0) needs cu128; an older wheel imports fine and then fails
    at the first launch with "no kernel image is available".
    """
    if not (HAS_NVIDIA_GPU and torch.cuda.is_available()):
        pytest.skip("no usable CUDA device")

    props = torch.cuda.get_device_properties(0)
    x = torch.randn(256, 256, device="cuda")
    result = (x @ x).sum().item()
    torch.cuda.synchronize()
    assert result == result, "matmul produced NaN"          # NaN != NaN
    assert props.total_memory > 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_toggling_the_device_moves_a_real_model():
    """The dashboard switch must actually relocate registered modules."""
    if not (HAS_NVIDIA_GPU and torch.cuda.is_available()):
        pytest.skip("no usable CUDA device")

    manager = get_device_manager()
    model = torch.nn.Linear(16, 16)
    manager.register_model("test:linear", model)
    try:
        manager.set_gpu(True)
        assert manager.active_device.startswith("cuda")
        assert next(model.parameters()).device.type == "cuda", (
            "the model did not follow the toggle onto the GPU"
        )

        manager.set_gpu(False)
        assert manager.active_device == "cpu"
        assert next(model.parameters()).device.type == "cpu", (
            "the model did not follow the toggle back to CPU"
        )
    finally:
        manager.unregister_model("test:linear")
        manager.set_gpu(True)


def test_detector_reports_its_real_backend():
    """The UI must never imply a trained model when a fallback is running."""
    from app.vision.detector import build_detector

    detector = build_detector()
    info = detector.info()
    assert info["backend"], "the detector did not report a backend at all"

    if info["backend"] == "fallback-motion":
        assert detector.name == "motion"
    else:
        assert info.get("ready") is True, (
            f"backend '{info['backend']}' is advertised but not ready"
        )
