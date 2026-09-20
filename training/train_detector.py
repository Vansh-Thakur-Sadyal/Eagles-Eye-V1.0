#!/usr/bin/env python
"""Fine-tune the Sentinel detector.

    python training/train_detector.py --check
    python training/train_detector.py --epochs 100 --batch -1
    python training/train_detector.py --resume runs/detect/sentinel/weights/last.pt

Defaults are tuned for a 16 GB card (RTX 5070 Ti): batch is auto-sized, AMP is
on, and the cache stays on disk because a large detection set will not fit in
16 GB of RAM.

Reality check before you start: you do not need to train on all 561 GB of Open
Images. Start from COCO-pretrained weights and fine-tune on a curated subset of
the eleven classes Sentinel actually uses. That reaches a usable detector in
hours rather than weeks, and the marginal gain from the full set is small for
this class list.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets" / "detection" / "data.yaml"


def preflight(data_yaml: Path) -> bool:
    """Fail early and clearly rather than 40 minutes into a run."""
    ok = True
    print("=" * 70)
    print("  Pre-flight")
    print("=" * 70)

    try:
        import torch

        print(f"  torch           {torch.__version__}")
        print(f"  CUDA available  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / 1024**3
            print(f"  GPU             {props.name} ({vram:.1f} GB, SM {props.major}.{props.minor})")
            if props.major >= 12:
                print("  note            Blackwell (SM 12.x) needs a cu128 torch build")
            if vram < 8:
                print("  warning         under 8 GB: reduce --imgsz or --batch")
        else:
            print("  warning         no CUDA device; training on CPU is impractical")
            ok = False
    except ImportError:
        print("  torch           NOT INSTALLED  -> see docs/INSTALL.md")
        ok = False

    try:
        import ultralytics

        print(f"  ultralytics     {ultralytics.__version__}")
    except ImportError:
        print("  ultralytics     NOT INSTALLED  -> pip install ultralytics")
        ok = False

    if not data_yaml.is_file():
        print(f"  dataset         MISSING {data_yaml}")
        print("                  run: python tools/make_datasets.py structure")
        ok = False
    else:
        print(f"  dataset         {data_yaml}")
        import re

        text = data_yaml.read_text(encoding="utf-8")
        base = Path(re.search(r"^path:\s*(.+)$", text, re.M).group(1).strip())
        counts = {}
        for split in ("train", "val", "test"):
            images = base / "images" / split
            counts[split] = len(list(images.glob("*.*"))) if images.is_dir() else 0
            print(f"    {split:<6}        {counts[split]} images")
        if counts["train"] == 0:
            print("  ERROR           no training images. Populate detection/images/train")
            print("                  and detection/labels/train first (see docs/DATASETS.md)")
            ok = False
        elif counts["train"] < 500:
            print("  warning         very small training set; expect poor generalisation")
        if counts["val"] == 0:
            print("  warning         no validation split: you cannot measure overfitting")

    free = shutil.disk_usage(ROOT).free / 1024**3
    print(f"  free disk       {free:.1f} GB")
    if free < 20:
        print("  warning         checkpoints and cache need room")

    print("=" * 70)
    print("  READY" if ok else "  NOT READY - fix the items above")
    print("=" * 70)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the Sentinel detector")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11m.pt",
                        help="starting weights; yolo11s for speed, yolo11l for accuracy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1,
                        help="-1 auto-sizes to ~60%% of VRAM")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="sentinel")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--optimizer", default="AdamW",
                        help="AdamW at a low lr fine-tunes without wrecking the COCO "
                             "features; ultralytics 'auto' picks SGD lr 0.01 on large "
                             "sets, which dropped val mAP50-95 from 0.34 to 0.21 in 3 epochs")
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--check", action="store_true", help="pre-flight only")
    parser.add_argument("--register", action="store_true",
                        help="register the trained weights with a running Sentinel")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    if not preflight(args.data):
        return 1
    if args.check:
        return 0

    from ultralytics import YOLO

    model = YOLO(args.resume or args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(ROOT / "training" / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        resume=bool(args.resume),
        # 16 GB-friendly defaults
        amp=True,
        cache=False,
        # Surveillance footage is low, wide and often dim; these augmentations
        # match that domain rather than the web-photo defaults.
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.45,
        degrees=3.0, translate=0.1, scale=0.45, shear=1.5,
        perspective=0.0005, flipud=0.0, fliplr=0.5,
        mosaic=1.0, close_mosaic=12, mixup=0.1,
        optimizer=args.optimizer, lr0=args.lr0, warmup_epochs=args.warmup, cos_lr=True,
        plots=True, val=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nbest weights: {best}")

    # Deliberately NOT copied into storage/models: a fine-tune only replaces
    # the stock detector after training/compare_detectors.py shows it is not
    # worse on either evaluation set.
    print("next: python training/compare_detectors.py --candidate " + str(best))

    metrics = {}
    try:
        metrics = {k: float(v) for k, v in results.results_dict.items()}
        print("\nmetrics:")
        for key, value in metrics.items():
            print(f"  {key:<28} {value:.4f}")
    except Exception:
        pass

    if args.register and best.is_file():
        _register(args, best, metrics)
    return 0


def _register(args, weights: Path, metrics: dict) -> None:
    """Record the trained model in the running deployment's registry."""
    import getpass

    import httpx

    username = input("Sentinel username: ")
    password = getpass.getpass("Password: ")
    try:
        token = httpx.post(f"{args.api}/api/auth/login",
                           json={"username": username, "password": password},
                           timeout=20).json()["access_token"]
        response = httpx.post(
            f"{args.api}/api/system/models",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": weights.name,
                "task": "detection",
                "framework": "ultralytics",
                "version": args.name,
                "weights_path": str(weights),
                "params": {"imgsz": args.imgsz, "epochs": args.epochs, "base": args.model},
                "metrics": metrics,
                "notes": f"Fine-tuned from {args.model} on {args.data}",
            },
            timeout=20,
        )
        response.raise_for_status()
        print("registered with the model registry")
    except Exception as exc:
        print(f"could not register: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
