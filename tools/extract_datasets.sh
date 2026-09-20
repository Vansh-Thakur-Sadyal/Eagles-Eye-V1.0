#!/usr/bin/env bash
# Extract every downloaded dataset archive onto the NVMe (C:\sentinel_data) and
# expose each one inside the project as datasets/raw/<name> via a junction.
#
# Why not extract straight into D:\Eagle_Eye_V1.0? D: is a USB hard drive.
# Writing ~180 GB of video to it and then reading it back for training would
# cost hours twice and leave the GPU idle. The junction keeps the project
# layout identical while the bytes sit on the SSD.
#
# Source archives are never modified or deleted. Re-running skips datasets that
# already have a .done marker.
set -u
SRC="/d/Eagle_Eye Dataset"
DL="/c/Users/thaku/Downloads"
DATA="/c/sentinel_data"
RAW="/d/Eagle_Eye_V1.0/datasets/raw"
LOG="$RAW/_extract.log"
mkdir -p "$DATA" "$RAW"

log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

link() {  # junction datasets/raw/<name> -> C:\sentinel_data\<name>
  local name="$1"
  if [ -d "$RAW/$name" ] && [ ! -L "$RAW/$name" ]; then
    powershell.exe -NoProfile -Command "if ((Get-Item 'D:\Eagle_Eye_V1.0\datasets\raw\\$name').LinkType -ne 'Junction') { Remove-Item -Recurse -Force 'D:\Eagle_Eye_V1.0\datasets\raw\\$name' }" >/dev/null 2>&1
  fi
  powershell.exe -NoProfile -Command "if (-not (Test-Path 'D:\Eagle_Eye_V1.0\datasets\raw\\$name')) { New-Item -ItemType Junction -Path 'D:\Eagle_Eye_V1.0\datasets\raw\\$name' -Target 'C:\sentinel_data\\$name' | Out-Null }" >/dev/null 2>&1
}

want() {  # ONLY="a b" restricts which datasets run
  [ -z "${ONLY:-}" ] && return 0
  case " $ONLY " in *" $1 "*) return 0;; esac
  return 1
}

unz() {  # unz <name> <zip...>
  local name="$1"; shift
  want "$name" || return 0
  if [ -f "$DATA/$name/.done" ]; then log "skip $name (done)"; link "$name"; return; fi
  mkdir -p "$DATA/$name"
  local failed=0
  for z in "$@"; do
    log "extract $name <- $(basename "$z")"
    unzip -q -o "$z" -d "$DATA/$name" || { log "WARN unzip failed: $z"; failed=1; }
  done
  link "$name"
  # A failed archive must not be reported as complete: downstream training
  # would silently run on partial data.
  if [ "$failed" = 1 ]; then log "INCOMPLETE $name (not marked done)"; return 1; fi
  touch "$DATA/$name/.done"; log "done $name"
}

log "=== extraction start (to NVMe) ==="
unz ucf_qnrf "$SRC/UCF-QNRF_ECCV18/UCF-QNRF_ECCV18.zip"

if want shanghaitech && [ ! -f "$DATA/shanghaitech/.done" ]; then
  mkdir -p "$DATA/shanghaitech"
  log "extract shanghaitech (split tar.gz)"
  cat "$SRC/ShanghaiTech Campus dataset/"shanghaitech.tar.gz.a? | tar -xz -C "$DATA/shanghaitech" \
    && touch "$DATA/shanghaitech/.done" && log "done shanghaitech" || log "WARN shanghaitech failed"
fi
want shanghaitech && link shanghaitech

unz mot17 "$SRC/MOT17/MOT17.zip" "$SRC/MOT17/MOT17Labels.zip"
unz mot20 "$SRC/MOT20/MOT20.zip" "$SRC/MOT20/MOT20Labels.zip"
unz ucf_crime "$DL"/Anomaly-Videos-Part-1.zip "$DL"/Anomaly-Videos-Part-2.zip \
              "$DL"/Anomaly-Videos-Part-3.zip "$DL"/Anomaly-Videos-Part-4.zip \
              "$DL"/Training-Normal-Videos-Part-1.zip "$DL"/Training-Normal-Videos-Part-2.zip
unz xd_violence "$SRC"/XD-Violence/*.zip
log "=== extraction finished ==="
