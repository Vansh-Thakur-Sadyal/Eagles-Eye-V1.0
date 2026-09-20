#!/usr/bin/env python
"""Curate an Open Images V7 subset for Sentinel's detector fine-tune.

Reads the downloaded Open Images annotations, keeps only the eleven classes
Sentinel uses, balances them, resizes images to 640px and writes a YOLO dataset
onto fast local storage (the source sits on a USB HDD, which would starve the
GPU if trained from directly).

Labels use **COCO class indices**, deliberately. The stock COCO-pretrained
YOLO11 and the fine-tuned model can then be validated on exactly the same data
with exactly the same metric - which is the only honest way to answer "did the
fine-tune actually help?".

    python training/prepare_open_images.py --per-class 6000 --person 18000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "raw" / "open_images_v7"
OUT = Path("V:/sentinel_cache/oi_yolo")

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
COCO_ID = {name: i for i, name in enumerate(COCO_NAMES)}

# Open Images display name -> Sentinel/COCO class
OI_TO_COCO = {
    "Person": "person", "Man": "person", "Woman": "person", "Boy": "person", "Girl": "person",
    "Backpack": "backpack",
    "Handbag": "handbag",
    "Suitcase": "suitcase", "Luggage and bags": "suitcase",
    "Bicycle": "bicycle",
    "Motorcycle": "motorcycle",
    "Car": "car",
    "Bus": "bus",
    "Truck": "truck",
    "Bottle": "bottle",
    "Laptop": "laptop",
}


def split_of(image_id: str, val_fraction: float) -> str:
    h = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=6000,
                        help="max images per non-person class")
    parser.add_argument("--person", type=int, default=18000,
                        help="max images whose only Sentinel class is person")
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    import pandas as pd

    classes = pd.read_csv(SRC / "annotations" / "classes.csv", header=None,
                          names=["label", "name"])
    wanted = classes[classes["name"].isin(OI_TO_COCO)]
    label_to_coco = {row.label: COCO_ID[OI_TO_COCO[row.name]] for row in wanted.itertuples()}
    print(f"mapped {len(label_to_coco)} Open Images labels -> "
          f"{len(set(label_to_coco.values()))} Sentinel classes")

    available = {p.stem for p in (SRC / "images" / "train").iterdir()}
    print(f"images on disk: {len(available):,}")

    print("reading train-bbox.csv (14.6M rows) ...")
    boxes = pd.read_csv(
        SRC / "annotations" / "train-bbox.csv",
        usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax",
                 "IsGroupOf", "IsDepiction"],
        dtype={"ImageID": "string", "LabelName": "string"},
    )
    boxes = boxes[boxes["ImageID"].isin(available)]
    ours = boxes[boxes["LabelName"].isin(label_to_coco)].copy()
    ours["cls"] = ours["LabelName"].map(label_to_coco).astype(int)

    # Group-of boxes (one box around a crowd) and depictions (drawings, posters)
    # teach the wrong thing for surveillance. Drop the image entirely if it has
    # one for our classes, so unlabelled people are never learned as background.
    bad_images = set(ours.loc[(ours["IsGroupOf"] == 1) | (ours["IsDepiction"] == 1), "ImageID"])
    ours = ours[~ours["ImageID"].isin(bad_images)]
    print(f"clean boxes: {len(ours):,} in {ours['ImageID'].nunique():,} images "
          f"(dropped {len(bad_images):,} images with group/depiction boxes)")

    # ---- balanced selection ------------------------------------------------
    per_image_classes = ours.groupby("ImageID")["cls"].agg(set)
    chosen: set = set()
    counts = {}
    rare_first = (
        ours.groupby("cls")["ImageID"].nunique().sort_values().index.tolist()
    )
    for cls in rare_first:
        if cls == COCO_ID["person"]:
            continue
        ids = [i for i, s in per_image_classes.items() if cls in s and i not in chosen]
        take = ids[: args.per_class]
        chosen.update(take)
        counts[COCO_NAMES[cls]] = len(take)
    person_only = [i for i, s in per_image_classes.items()
                   if s == {COCO_ID["person"]} and i not in chosen]
    chosen.update(person_only[: args.person])
    counts["person-only images"] = min(len(person_only), args.person)
    print("selection:", json.dumps(counts, indent=2))

    selected = ours[ours["ImageID"].isin(chosen)]
    by_image = {k: g for k, g in selected.groupby("ImageID")}

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    from PIL import Image

    def work(image_id: str) -> str:
        split = split_of(image_id, args.val_fraction)
        dst_img = OUT / "images" / split / f"{image_id}.jpg"
        dst_lbl = OUT / "labels" / split / f"{image_id}.txt"
        if dst_img.exists() and dst_lbl.exists():
            return split
        try:
            with Image.open(SRC / "images" / "train" / f"{image_id}.jpg") as im:
                im = im.convert("RGB")
                im.thumbnail((args.imgsz, args.imgsz))
                im.save(dst_img, quality=90)
        except Exception:
            return "error"
        lines = []
        for r in by_image[image_id].itertuples():
            cx, cy = (r.XMin + r.XMax) / 2, (r.YMin + r.YMax) / 2
            w, h = r.XMax - r.XMin, r.YMax - r.YMin
            if w <= 0.002 or h <= 0.002:
                continue
            lines.append(f"{r.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return split

    done = {"train": 0, "val": 0, "error": 0}
    ids = sorted(chosen)
    with ThreadPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(work, ids), 1):
            done[result] += 1
            if i % 5000 == 0:
                print(f"  {i:,}/{len(ids):,}  {done}", flush=True)

    names_yaml = "".join(f"  {i}: {n}\n" for i, n in enumerate(COCO_NAMES))
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(COCO_NAMES)}\nnames:\n{names_yaml}",
        encoding="utf-8",
    )
    (OUT / "manifest.json").write_text(json.dumps({
        "source": "Open Images V7 (downloaded subset)",
        "classes": sorted(set(OI_TO_COCO.values())),
        "label_space": "COCO-80 indices (only Sentinel classes labelled)",
        "selection": counts, "result": done,
        "excluded": "images containing group-of or depiction boxes for these classes",
    }, indent=2), encoding="utf-8")
    print("finished:", done)
    return 0 if done["train"] else 1


if __name__ == "__main__":
    sys.exit(main())
