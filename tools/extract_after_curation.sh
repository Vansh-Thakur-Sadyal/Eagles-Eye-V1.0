#!/usr/bin/env bash
# Start the USB-sourced extractions only once Open Images curation has finished
# with the USB disk - running them together made both several times slower.
RAW=/d/Eagle_Eye_V1.0/datasets/raw
while powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'prepare_open_images' -and \$_.CommandLine -notmatch 'Get-CimInstance' }) { exit 0 } else { exit 1 }" >/dev/null 2>&1; do
  sleep 60
done
touch "$RAW/_all_extraction_scheduled"
cd /d/Eagle_Eye_V1.0 && ONLY="ucf_qnrf shanghaitech mot17 mot20 xd_violence" bash tools/extract_datasets.sh
