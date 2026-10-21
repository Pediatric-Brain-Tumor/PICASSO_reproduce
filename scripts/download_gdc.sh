#!/usr/bin/env bash
# Download the slides listed in manifest.csv from the GDC data API, in manifest order,
# N parallel workers, md5-verified, resumable (a slide with a .ok marker is skipped).
#   bash scripts/download_gdc.sh <manifest.csv> <slides_dir> [parallel=4]
set -euo pipefail
MANIFEST=$1; OUT=$2; PAR=${3:-4}
mkdir -p "$OUT/_logs"
fetch_one() {
  line=${1//$'\r'/}
  IFS=, read -r order fid fname proj case size md5 <<<"$line"
  dir="$2/$proj"; mkdir -p "$dir"
  dst="$dir/$fname"
  [ -f "$dst.ok" ] && return 0
  for attempt in 1 2 3 4 5; do
    if curl -s -f --retry 3 --retry-delay 10 --speed-limit 100000 --speed-time 120 \
         -C - -o "$dst.part" "https://api.gdc.cancer.gov/data/$fid"; then
      if [ "$(stat -c %s "$dst.part")" != "$size" ]; then echo "$(date +%FT%T) SIZEMISMATCH $order $fname attempt $attempt" >> "$2/_logs/download.log"; rm -f "$dst.part"; continue; fi
      got=$(md5sum "$dst.part" | cut -d' ' -f1)
      if [ "$got" = "$md5" ]; then
        mv "$dst.part" "$dst"; echo "$md5" > "$dst.ok"
        echo "$(date +%FT%T) OK   $order $proj $fname $size" >> "$2/_logs/download.log"; return 0
      fi
      echo "$(date +%FT%T) MD5MISMATCH $order $fname attempt $attempt" >> "$2/_logs/download.log"; rm -f "$dst.part"
    else
      echo "$(date +%FT%T) CURLFAIL $order $fname attempt $attempt" >> "$2/_logs/download.log"; sleep 30
    fi
  done
  echo "$(date +%FT%T) FAILED $order $proj $fname" >> "$2/_logs/download.log"; return 1
}
export -f fetch_one
tail -n +2 "$MANIFEST" | xargs -P "$PAR" -I{} bash -c 'fetch_one "$1" "$2"' _ {} "$OUT" || true
echo "$(date +%FT%T) ALL-DONE" >> "$OUT/_logs/download.log"
