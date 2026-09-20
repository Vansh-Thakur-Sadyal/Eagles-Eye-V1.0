#!/usr/bin/env bash
# The two Training-Normal-Videos zips are truncated downloads (no central
# directory). Recover every intact, CRC-verified entry, then mark UCF-Crime
# extracted so the training queue can pick it up.
set -u
cd "$(dirname "$0")/.."
PY=backend/.venv/Scripts/python.exe
for part in 1 2; do
  "$PY" -u tools/recover_partial_zip.py \
    "C:/Users/thaku/Downloads/Training-Normal-Videos-Part-$part.zip" C:/sentinel_data/ucf_crime \
    > "training/logs/recover_normal$part.log" 2>&1 || echo "part $part exited non-zero"
done
n=$(find /c/sentinel_data/ucf_crime -name "Normal_Videos*.mp4" | wc -l)
echo "$(date) recovery finished: $n normal videos"
touch /c/sentinel_data/ucf_crime/.done
