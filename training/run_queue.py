#!/usr/bin/env python
"""Run every training / evaluation job in order on the local GPU.

One card, so jobs run strictly one at a time. Each step declares the data it
needs; the queue waits for that data (extraction may still be running) and
skips a step only if its data can never arrive. Finished steps are recorded,
so re-running the queue resumes where it stopped.

    python training/run_queue.py            # run / resume everything
    python training/run_queue.py --status  # print progress

Progress: training/runs/queue_status.json   Logs: training/logs/<step>.log
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / "backend" / ".venv" / "Scripts" / "python.exe")
LOGS = ROOT / "training" / "logs"
STATUS = ROOT / "training" / "runs" / "queue_status.json"
DATA = Path("C:/sentinel_data")
EXTRACT_LOG = ROOT / "datasets" / "raw" / "_extract.log"


def extracted(name: str) -> Callable[[], bool]:
    return lambda: (DATA / name / ".done").exists()


def exists(path: Path) -> Callable[[], bool]:
    return lambda: path.exists()


def extraction_finished() -> bool:
    """True only when no extraction is running AND none is scheduled to start.

    Extraction runs in several batches, each of which logs "finished", so the
    log alone cannot say that all data has arrived. The marker file
    `_all_extraction_scheduled` is written once the last batch has been
    launched; until then, and while any batch is still running, keep waiting.
    """
    scheduled = (ROOT / "datasets" / "raw" / "_all_extraction_scheduled").exists()
    return scheduled and not process_running("extract_datasets")


def process_running(fragment: str) -> bool:
    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match '{fragment}' "
         f"-and $_.CommandLine -notmatch 'run_queue' }}).Count"],
        capture_output=True, text=True)
    try:
        return int(out.stdout.strip() or 0) > 0
    except ValueError:
        return False


class Step:
    def __init__(self, name: str, cmd: List[str], needs: Optional[Callable[[], bool]] = None,
                 done_when: Optional[Callable[[], bool]] = None, wait_for_process: str = ""):
        self.name, self.cmd, self.needs = name, cmd, needs
        self.done_when, self.wait_for_process = done_when, wait_for_process


RUNS = ROOT / "training" / "runs"
FEAT = Path("V:/sentinel_cache/features")

STEPS = [
    # already launched separately; the queue just waits for it
    Step("reid", [PY, "-u", "training/train_reid.py", "--epochs", "80"],
         done_when=exists(RUNS / "reid" / "summary.json"), wait_for_process="train_reid"),
    Step("features_ucf_crime", [PY, "-u", "training/extract_video_features.py", "--dataset", "ucf_crime"],
         needs=extracted("ucf_crime")),
    Step("anomaly_head", [PY, "-u", "training/train_anomaly.py", "--task", "anomaly"],
         needs=exists(FEAT / "ucf_crime" / "_failed.json")),
    Step("crowd_prepare", [PY, "-u", "training/train_crowd.py", "--prepare"],
         needs=extracted("ucf_qnrf")),
    Step("crowd_train", [PY, "-u", "training/train_crowd.py"],
         needs=extracted("ucf_qnrf"), done_when=exists(RUNS / "crowd" / "summary.json")),
    Step("features_shanghaitech_test", [PY, "-u", "training/extract_video_features.py",
                                        "--dataset", "shanghaitech_test", "--dense"],
         needs=extracted("shanghaitech")),
    # re-run so the saved metrics include the ShanghaiTech cross-dataset AUC
    Step("anomaly_head_with_crossdataset", [PY, "-u", "training/train_anomaly.py", "--task", "anomaly"],
         needs=exists(FEAT / "shanghaitech_test_dense" / "_failed.json")),
    Step("mot17_eval_stock", [PY, "-u", "training/evaluate_mot.py", "--dataset", "mot17"],
         needs=extracted("mot17")),
    Step("mot20_eval_stock", [PY, "-u", "training/evaluate_mot.py", "--dataset", "mot20"],
         needs=extracted("mot20")),
    Step("detector_finetune", [PY, "-u", "training/train_detector.py",
                               "--data", "V:/sentinel_cache/oi_yolo/data.yaml",
                               "--epochs", "60", "--batch", "16", "--workers", "10",
                               "--name", "sentinel", "--patience", "15"],
         needs=exists(Path("V:/sentinel_cache/oi_yolo/data.yaml")),
         done_when=exists(RUNS / "detect" / "sentinel" / "weights" / "best.pt")),
    Step("detector_compare", [PY, "-u", "training/compare_detectors.py", "--candidate",
                              str(RUNS / "detect" / "sentinel" / "weights" / "best.pt")],
         needs=lambda: (RUNS / "detect" / "sentinel" / "weights" / "best.pt").exists()
         and extracted("mot17")()),
    Step("features_xd_violence", [PY, "-u", "training/extract_video_features.py", "--dataset", "xd_violence"],
         needs=extracted("xd_violence")),
    Step("violence_head", [PY, "-u", "training/train_anomaly.py", "--task", "violence"],
         needs=exists(FEAT / "xd_violence" / "_failed.json")),
]


def load_status() -> dict:
    return json.loads(STATUS.read_text()) if STATUS.exists() else {}


def save_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2))


def run() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    status = load_status()
    for step in STEPS:
        entry = status.get(step.name, {})
        if entry.get("state") == "done":
            continue
        if step.done_when and step.done_when():
            status[step.name] = {**entry, "state": "done", "note": "output already present"}
            save_status(status)
            continue

        if step.wait_for_process and process_running(step.wait_for_process):
            status[step.name] = {"state": "waiting", "for": "already-running process"}
            save_status(status)
            while process_running(step.wait_for_process):
                time.sleep(60)
            if step.done_when and step.done_when():
                status[step.name] = {"state": "done", "finished": datetime.now().isoformat()}
                save_status(status)
                continue

        if step.needs and not step.needs():
            status[step.name] = {"state": "waiting", "for": "input data"}
            save_status(status)
            while not step.needs():
                if extraction_finished() and not step.needs():
                    break
                time.sleep(60)
            if not step.needs():
                status[step.name] = {"state": "skipped", "reason": "input data never arrived"}
                save_status(status)
                continue

        log = LOGS / f"{step.name}.log"
        status[step.name] = {"state": "running", "started": datetime.now().isoformat(),
                             "log": str(log)}
        save_status(status)
        started = time.time()
        with log.open("w", encoding="utf-8") as fh:
            code = subprocess.call(step.cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        status[step.name] = {
            "state": "done" if code == 0 else "failed",
            "exit_code": code,
            "minutes": round((time.time() - started) / 60, 1),
            "finished": datetime.now().isoformat(),
            "log": str(log),
        }
        save_status(status)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        for name, entry in load_status().items():
            print(f"{name:<34} {entry.get('state', '?'):<9} "
                  f"{entry.get('minutes', '')} {entry.get('reason', entry.get('for', ''))}")
        return 0
    return run()


if __name__ == "__main__":
    sys.exit(main())
