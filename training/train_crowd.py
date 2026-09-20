#!/usr/bin/env python
"""Crowd counting on UCF-QNRF with a density-map network (CSRNet), local GPU.

Why this exists: the crowd agent counts people by detecting them, one box each.
That works in a concourse and collapses in a packed crowd, where heads are a
few pixels wide and occlude each other - precisely the scenes crowd analytics
exists for. A density network predicts a per-pixel density whose integral is
the count, and does not need individual people to be separable.

    python training/train_crowd.py --prepare     # build resized images + density maps
    python training/train_crowd.py               # train and evaluate

Evaluation reports MAE / RMSE on the official UCF-QNRF test split (334 images)
AND the same metrics for the detector-based count the platform used before, on
the same images, so the improvement (or lack of one) is measured, not assumed.
The model is only promoted to storage/models/crowd/ if it beats that baseline.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw" / "ucf_qnrf"
CACHE = Path("V:/sentinel_cache/qnrf")
OUT = ROOT / "storage" / "models" / "crowd"
MAX_SIDE = 1536
DOWN = 8                    # CSRNet predicts at 1/8 resolution
DENSITY_SCALE = 100.0       # density maps are tiny; scale for stable gradients


# ------------------------------------------------------------- preparation
def _find_split(name: str) -> Path:
    for p in RAW.rglob("*"):
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    raise FileNotFoundError(f"UCF-QNRF '{name}' folder not found under {RAW}")


def prepare() -> None:
    import cv2
    from scipy.io import loadmat
    from scipy.ndimage import gaussian_filter

    for split in ("Train", "Test"):
        src = _find_split(split)
        dst = CACHE / split.lower()
        dst.mkdir(parents=True, exist_ok=True)
        images = sorted(src.glob("*.jpg"))
        print(f"{split}: {len(images)} images", flush=True)
        for i, img_path in enumerate(images, 1):
            out_img = dst / img_path.name
            out_den = dst / (img_path.stem + ".npy")
            if out_img.exists() and out_den.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ann = loadmat(str(img_path.with_name(img_path.stem + "_ann.mat")))
            points = ann["annPoints"].astype(np.float32)
            h, w = img.shape[:2]
            scale = min(1.0, MAX_SIDE / max(h, w))
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                points = points * scale
            h, w = img.shape[:2]
            # dimensions divisible by 8 so the density grid aligns exactly
            h8, w8 = h - h % DOWN, w - w % DOWN
            img = img[:h8, :w8]
            grid = np.zeros((h8 // DOWN, w8 // DOWN), dtype=np.float32)
            for x, y in points:
                gx, gy = int(x / DOWN), int(y / DOWN)
                if 0 <= gx < grid.shape[1] and 0 <= gy < grid.shape[0]:
                    grid[gy, gx] += 1.0
            density = gaussian_filter(grid, sigma=1.5, mode="constant")
            if grid.sum() > 0:                      # preserve the exact count
                density *= grid.sum() / max(1e-6, density.sum())
            cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            np.save(out_den, density)
            if i % 200 == 0:
                print(f"  {i}/{len(images)}", flush=True)
    print("prepared", flush=True)


# ------------------------------------------------------------------ model
# The architecture is defined once, in the app, so the trained weights and the
# code that serves them can never drift apart.
sys.path.insert(0, str(ROOT / "backend"))
from app.vision.crowd_density import build_csrnet  # noqa: E402


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img_bgr: np.ndarray):
    import torch

    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


class CropDataset:
    def __init__(self, folder: Path, crop: int) -> None:
        self.items = sorted(folder.glob("*.jpg"))
        self.crop = crop

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        import torch

        img = cv2.imread(str(self.items[i]))
        den = np.load(self.items[i].with_suffix(".npy"))
        h, w = img.shape[:2]
        c = min(self.crop, h - h % DOWN, w - w % DOWN)
        c -= c % DOWN
        y = random.randrange(0, (h - c) // DOWN + 1) * DOWN
        x = random.randrange(0, (w - c) // DOWN + 1) * DOWN
        img = img[y:y + c, x:x + c]
        den = den[y // DOWN:(y + c) // DOWN, x // DOWN:(x + c) // DOWN]
        if random.random() < 0.5:
            img, den = img[:, ::-1], den[:, ::-1]
        if c < self.crop:                           # pad small images
            pad = self.crop - c
            img = np.pad(img, ((0, pad), (0, pad), (0, 0)))
            den = np.pad(den, ((0, pad // DOWN), (0, pad // DOWN)))
        return to_tensor(np.ascontiguousarray(img)), torch.from_numpy(
            np.ascontiguousarray(den) * DENSITY_SCALE)[None]


# --------------------------------------------------------------- evaluate
def evaluate(model, folder: Path, device) -> dict:
    import cv2
    import torch

    errors = []
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for img_path in sorted(folder.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            gt = float(np.load(img_path.with_suffix(".npy")).sum())
            pred = float(model(to_tensor(img)[None].to(device)).float().sum()) / DENSITY_SCALE
            errors.append(pred - gt)
    model.train()
    e = np.array(errors)
    return {"mae": round(float(np.abs(e).mean()), 2),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 2), "images": len(e)}


def detector_baseline(folder: Path) -> dict:
    """What the platform did before: count YOLO person boxes."""
    import cv2
    from ultralytics import YOLO

    model = YOLO("yolo11m.pt")
    errors = []
    for img_path in sorted(folder.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        gt = float(np.load(img_path.with_suffix(".npy")).sum())
        res = model.predict(img, classes=[0], conf=0.25, imgsz=1536, max_det=3000,
                            device=0, verbose=False)[0]
        errors.append(len(res.boxes) - gt)
    e = np.array(errors)
    return {"mae": round(float(np.abs(e).mean()), 2),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 2), "images": len(e)}


# ------------------------------------------------------------------ train
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.prepare:
        prepare()
        return 0

    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        print("CUDA unavailable - refusing to train on CPU.")
        return 1
    device = torch.device("cuda")
    random.seed(0)
    torch.manual_seed(0)

    train_dir, test_dir = CACHE / "train", CACHE / "test"
    # Hold out 10% of the training images for model selection; the official
    # test split is only touched once, at the end.
    all_train = sorted(train_dir.glob("*.jpg"))
    val_names = {p.name for i, p in enumerate(all_train) if i % 10 == 0}
    val_dir = CACHE / "val"
    val_dir.mkdir(exist_ok=True)
    for p in all_train:
        if p.name in val_names and not (val_dir / p.name).exists():
            (val_dir / p.name).write_bytes(p.read_bytes())
            (val_dir / (p.stem + ".npy")).write_bytes(p.with_suffix(".npy").read_bytes())

    dataset = CropDataset(train_dir, args.crop)
    dataset.items = [p for p in dataset.items if p.name not in val_names]
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        persistent_workers=True, pin_memory=True, drop_last=True)
    print(f"train {len(dataset)} / val {len(val_names)} / test "
          f"{len(list(test_dir.glob('*.jpg')))} images", flush=True)

    model = build_csrnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    best, best_state, started = float("inf"), None, time.time()

    for epoch in range(1, args.epochs + 1):
        for img, den in loader:
            img, den = img.to(device, non_blocking=True), den.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.mse_loss(model(img).float(), den, reduction="sum") / img.size(0)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if epoch % 5 == 0 or epoch == 1:
            val = evaluate(model, val_dir, device)
            if val["mae"] < best:
                best = val["mae"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  epoch {epoch:3d}  loss {float(loss):.3f}  val MAE {val['mae']:.1f}  "
                  f"({(time.time() - started) / 60:.0f} min)", flush=True)

    model.load_state_dict(best_state)
    test = evaluate(model, test_dir, device)
    print("measuring detector-count baseline on the same test images ...", flush=True)
    baseline = detector_baseline(test_dir)

    promote = test["mae"] < baseline["mae"]
    result = {
        "model": "CSRNet (VGG16 frontend, dilated backend)",
        "dataset": "UCF-QNRF official test split",
        "density_net": test,
        "detector_count_baseline": baseline,
        "improvement_mae": round(baseline["mae"] - test["mae"], 2),
        "epochs": args.epochs,
        "train_minutes": round((time.time() - started) / 60, 1),
        "model_selection": "best val MAE on 10% of the training images",
        "promoted": promote,
        "density_scale": DENSITY_SCALE,
        "max_side": MAX_SIDE,
    }
    runs = ROOT / "training" / "runs" / "crowd"
    runs.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": result}, runs / "csrnet_qnrf.pt")
    if promote:
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "meta": result}, OUT / "csrnet_qnrf.pt")
        (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (runs / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
