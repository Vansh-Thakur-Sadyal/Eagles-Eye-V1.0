#!/usr/bin/env bash
# Runs once the detector fine-tune + comparison are done (GPU free again).
cd "$(dirname "$0")/.."
PY=backend/.venv/Scripts/python.exe
until [ -f training/runs/detect/comparison.json ] || { [ -f training/logs/detector_compare.log ] && grep -q Traceback training/logs/detector_compare.log; }; do sleep 60; done
echo "$(date) detector stage finished"
while tasklist | grep -qi "train_detector\|compare_detectors"; do sleep 30; done
echo "$(date) LocateAnything GPU evaluation"
$PY -u tools/eval_locateanything.py --images 200 --mode slow > training/logs/eval_locateanything.log 2>&1
echo "$(date) full backend test suite"
(cd backend && PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider > ../training/logs/pytest_full.log 2>&1)
echo "$(date) edge e2e with the node on the GPU"
$PY -u tools/test_edge_e2e.py --node-device auto > training/logs/edge_e2e_gpu.log 2>&1
echo "$(date) all done"
