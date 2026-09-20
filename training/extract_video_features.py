#!/usr/bin/env python
"""Extract clip features from surveillance video, on the local GPU.

Shared by the anomaly (UCF-Crime), violence (XD-Violence) and cross-dataset
evaluation (ShanghaiTech) pipelines.

Each video is split into N equal temporal segments (32 by default - the protocol
UCF-Crime was published with). From each segment 16 frames are sampled evenly,
resized, and passed through a Kinetics-400 pretrained R(2+1)D-18 with its
classifier removed, giving one 512-d feature per segment.

Decoding runs in a pool of CPU processes that feed the GPU, because decoding -
not the network - is the bottleneck on long videos. Output is resumable: a
video whose .npy already exists is skipped.

    python training/extract_video_features.py --dataset ucf_crime
    python training/extract_video_features.py --dataset xd_violence
    python training/extract_video_features.py --dataset shanghaitech_test --dense
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
FEATURES = Path("V:/sentinel_cache/features")

CLIP_LEN = 16
SIZE = 112
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


# ------------------------------------------------------------ video listing
def list_videos(dataset: str) -> List[Tuple[str, Path]]:
    """(video_id, path) for every video in a dataset."""
    exts = {".mp4", ".avi", ".mkv", ".mov"}
    if dataset == "ucf_crime":
        base = RAW / "ucf_crime"
    elif dataset == "xd_violence":
        base = RAW / "xd_violence"
    elif dataset == "shanghaitech_train":
        base = next((RAW / "shanghaitech").rglob("training"), RAW / "shanghaitech")
    else:
        raise ValueError(dataset)
    videos = sorted(p for p in base.rglob("*") if p.suffix.lower() in exts)
    return [(p.stem, p) for p in videos]


def list_frame_folders(dataset: str) -> List[Tuple[str, Path]]:
    """ShanghaiTech test videos ship as folders of JPEG frames."""
    root = RAW / "shanghaitech"
    frames_root = next((p for p in root.rglob("frames") if p.is_dir()
                        and "test" in str(p).lower()), None)
    if frames_root is None:
        return []
    return sorted((d.name, d) for d in frames_root.iterdir() if d.is_dir())


# ---------------------------------------------------------------- decoding
def _prep(frame: np.ndarray) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    scale = 128 / min(h, w)
    frame = cv2.resize(frame, (max(SIZE, int(w * scale)), max(SIZE, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    y, x = (h - SIZE) // 2, (w - SIZE) // 2
    frame = frame[y:y + SIZE, x:x + SIZE, ::-1].astype(np.float32) / 255.0   # BGR->RGB
    return (frame - MEAN) / STD


def _read_indices(source: Path, wanted: np.ndarray, is_folder: bool) -> Optional[np.ndarray]:
    """Return the requested frames, in order, as (len(wanted), SIZE, SIZE, 3)."""
    import cv2

    out: List[np.ndarray] = []
    if is_folder:
        files = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png"))
        if not files:
            return None
        for i in wanted:
            img = cv2.imread(str(files[min(int(i), len(files) - 1)]))
            if img is None:
                return None
            out.append(_prep(img))
        return np.stack(out)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return None
    wanted_set = {}
    for pos, i in enumerate(wanted):
        wanted_set.setdefault(int(i), []).append(pos)
    result: List[Optional[np.ndarray]] = [None] * len(wanted)
    last_good: Optional[np.ndarray] = None
    frame_no, target_max = 0, int(wanted.max())
    # Sequential grab/retrieve: decoding every frame is far faster than seeking
    # on x264, and grab() skips the colour conversion for frames we discard.
    while frame_no <= target_max:
        if not cap.grab():
            break
        if frame_no in wanted_set:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                last_good = _prep(frame)
                for pos in wanted_set[frame_no]:
                    result[pos] = last_good
        frame_no += 1
    cap.release()
    if last_good is None:
        return None
    # Pad any trailing indices the container reported but could not deliver.
    filler = last_good
    for pos in range(len(result)):
        if result[pos] is None:
            result[pos] = filler
        else:
            filler = result[pos]
    return np.stack(result)


def _frame_count(source: Path, is_folder: bool) -> int:
    if is_folder:
        return len(list(source.glob("*.jpg"))) + len(list(source.glob("*.png")))
    import cv2

    cap = cv2.VideoCapture(str(source))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def decode_job(args: Tuple[str, str, int, bool, bool]) -> Tuple[str, Optional[np.ndarray], int]:
    """Worker: returns (video_id, clips[S,16,112,112,3] or None, n_frames)."""
    video_id, path_str, segments, dense, is_folder = args
    path = Path(path_str)
    try:
        n = _frame_count(path, is_folder)
        if n < CLIP_LEN:
            return video_id, None, n
        if dense:
            # Non-overlapping 16-frame clips: one score per clip, used for
            # frame-level evaluation.
            count = n // CLIP_LEN
            starts = np.arange(count) * CLIP_LEN
            idx = (starts[:, None] + np.arange(CLIP_LEN)[None, :]).reshape(-1)
            segs = count
        else:
            edges = np.linspace(0, n, segments + 1).astype(int)
            idx = np.concatenate([
                np.linspace(edges[s], max(edges[s], edges[s + 1] - 1), CLIP_LEN).astype(int)
                for s in range(segments)
            ])
            segs = segments
        frames = _read_indices(path, idx, is_folder)
        if frames is None:
            return video_id, None, n
        return video_id, frames.reshape(segs, CLIP_LEN, SIZE, SIZE, 3).astype(np.float16), n
    except Exception:
        return video_id, None, 0


# ------------------------------------------------------------------ model
def build_backbone():
    import torch
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Identity()
    return model.cuda().eval().half()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["ucf_crime", "xd_violence", "shanghaitech_train",
                                 "shanghaitech_test"])
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--dense", action="store_true",
                        help="non-overlapping 16-frame clips (frame-level eval)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable - refusing to extract features on CPU.")
        return 1

    is_folder = args.dataset == "shanghaitech_test"
    items = list_frame_folders(args.dataset) if is_folder else list_videos(args.dataset)
    if args.limit:
        items = items[: args.limit]
    out_dir = FEATURES / (args.dataset + ("_dense" if args.dense else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(vid, p) for vid, p in items if not (out_dir / f"{vid}.npy").exists()]
    print(f"{args.dataset}: {len(items)} videos, {len(todo)} to extract -> {out_dir}", flush=True)
    if not todo:
        return 0

    model = build_backbone()
    meta_path = out_dir / "_frames.json"
    frame_counts = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    failed: List[str] = []
    started = time.time()

    jobs = [(vid, str(p), args.segments, args.dense, is_folder) for vid, p in todo]
    total, done = len(jobs), 0

    def consume(vid, clips, n) -> None:
        nonlocal done
        done += 1
        if clips is None:
            failed.append(vid)
        else:
            feats = []
            with torch.inference_mode():
                for i in range(0, len(clips), 32):
                    batch = torch.from_numpy(clips[i:i + 32]).cuda()     # S,T,H,W,C
                    batch = batch.permute(0, 4, 1, 2, 3).contiguous()    # S,C,T,H,W
                    feats.append(model(batch).float().cpu().numpy())
            np.save(out_dir / f"{vid}.npy", np.concatenate(feats).astype(np.float32))
            frame_counts[vid] = n
        if done % 25 == 0 or done == total:
            rate = done / max(1e-6, time.time() - started)
            eta = (total - done) / max(1e-6, rate) / 60
            print(f"  {done}/{total}  {rate:.2f} vid/s  eta {eta:.0f} min  "
                  f"failed {len(failed)}", flush=True)
            meta_path.write_text(json.dumps(frame_counts))

    # A malformed video can crash the native decoder and take the whole pool
    # down with it. Results arrive in order, so the first unfinished job is
    # the prime suspect: retry it alone; if it crashes again it is recorded as
    # failed, and the rest carry on in a fresh pool.
    pending = list(jobs)
    while pending:
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(decode_job, list(pending), chunksize=1):
                    consume(*result)
                    pending.pop(0)
        except BrokenProcessPool:
            suspect = pending.pop(0)
            print(f"  decoder crashed near {suspect[0]} - retrying it alone", flush=True)
            try:
                with ProcessPoolExecutor(max_workers=1) as solo:
                    consume(*solo.submit(decode_job, suspect).result())
            except BrokenProcessPool:
                print(f"  {suspect[0]} crashes the decoder - marked failed", flush=True)
                consume(suspect[0], None, 0)

    meta_path.write_text(json.dumps(frame_counts))
    (out_dir / "_failed.json").write_text(json.dumps(failed, indent=1))
    print(f"finished in {(time.time() - started) / 60:.1f} min, {len(failed)} failed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
