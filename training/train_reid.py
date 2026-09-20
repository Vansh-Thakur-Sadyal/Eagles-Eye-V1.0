#!/usr/bin/env python
"""Train OSNet person re-identification on Market-1501 (local GPU).

    python training/train_reid.py                  # full run
    python training/train_reid.py --epochs 2       # smoke test

Starts from ImageNet OSNet weights and learns identity-discriminative features
with label-smoothed softmax + triplet loss. Evaluates rank-1 / mAP on the
standard query/gallery split, then writes the checkpoint into
storage/models/reid/, where the running platform picks it up automatically and
reports `weights: reid:<name>` instead of `weights: imagenet`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("V:/sentinel_cache")          # NVMe; contains market1501/
OUT_DIR = ROOT / "storage" / "models" / "reid"
LOG_DIR = ROOT / "training" / "runs" / "reid"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="osnet_x1_0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    import torch
    import torch.utils.data as tud

    # Windows spawns dataloader workers from scratch every epoch (each one
    # re-imports torch). torchreid does not expose persistent_workers, so turn
    # it on here - it cuts minutes of pure overhead per epoch.
    # Only for the TRAIN loader (the only one built with drop_last=True): each
    # Windows worker commits several GB just by importing torch, and keeping
    # the query + gallery workers alive as well exhausted the commit limit and
    # stalled the whole machine in paging.
    _orig_init = tud.DataLoader.__init__

    def _persistent_init(self, *a, **kw):
        if kw.get("num_workers", 0) > 0 and kw.get("drop_last", False):
            kw.setdefault("persistent_workers", True)
        _orig_init(self, *a, **kw)

    tud.DataLoader.__init__ = _persistent_init

    import torchreid

    if not torch.cuda.is_available():
        print("CUDA is not available - refusing to train Re-ID on CPU.")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    datamanager = torchreid.data.ImageDataManager(
        root=str(args.data),
        sources="market1501",
        targets="market1501",
        height=256,
        width=128,
        batch_size_train=args.batch,
        batch_size_test=256,
        transforms=["random_flip", "random_crop", "random_erase"],
        train_sampler="RandomIdentitySampler",   # needed for the triplet term
        num_instances=4,
        workers=args.workers,
    )

    model = torchreid.models.build_model(
        name=args.model,
        num_classes=datamanager.num_train_pids,
        loss="triplet",
        pretrained=True,
    ).cuda()

    optimizer = torchreid.optim.build_optimizer(model, optim="amsgrad", lr=args.lr)
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler="cosine", max_epoch=args.epochs
    )
    engine = torchreid.engine.ImageTripletEngine(
        datamanager, model, optimizer=optimizer, scheduler=scheduler,
        margin=0.3, weight_t=1.0, weight_x=1.0, label_smooth=True,
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline: the same network on ImageNet weights, before any training. A
    # trained model is only promoted to the live platform if it beats this.
    print("measuring ImageNet baseline ...", flush=True)
    baseline_rank1 = float(engine.test(dist_metric="euclidean", normalize_feature=True))
    print(f"baseline rank-1: {baseline_rank1:.4f}", flush=True)

    started = time.time()
    engine.run(
        save_dir=str(LOG_DIR),
        max_epoch=args.epochs,
        eval_freq=max(10, args.epochs // 4),
        print_freq=50,
        test_only=False,
    )
    minutes = (time.time() - started) / 60

    # Final evaluation, recorded explicitly so the numbers are not buried in logs.
    rank1 = engine.test(dist_metric="euclidean", normalize_feature=True)

    checkpoints = sorted(LOG_DIR.glob("model/model.pth.tar-*"),
                         key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoints:
        print("no checkpoint was written")
        return 1

    rank1 = float(rank1)
    # Promotion gate: must clearly beat the untrained baseline AND be a real
    # run, not a smoke test. Otherwise the checkpoint stays in training/runs/.
    promote = args.epochs >= 20 and rank1 >= baseline_rank1 + 0.10
    summary = {
        "model": args.model,
        "dataset": "market1501",
        "epochs": args.epochs,
        "rank1": round(rank1, 4),
        "baseline_rank1_imagenet": round(baseline_rank1, 4),
        "train_minutes": round(minutes, 1),
        "gpu": torch.cuda.get_device_name(0),
        "promoted": promote,
    }
    if promote:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUT_DIR / f"{args.model}_market1501.pth.tar"
        shutil.copy2(checkpoints[-1], target)
        summary["checkpoint"] = str(target)
        (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        summary["checkpoint"] = str(checkpoints[-1])
        summary["why_not_promoted"] = ("smoke test (<20 epochs)" if args.epochs < 20
                                       else "did not beat the ImageNet baseline by 10 points")
    (LOG_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":        # required on Windows for dataloader workers
    sys.exit(main())
