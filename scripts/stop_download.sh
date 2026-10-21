#!/usr/bin/env bash
# Stop a running download_gdc.sh (the driver, its xargs workers and curl children).
me=$$
for pat in "scripts/download_gdc.sh manifest" "xargs -P" "fetch_one" "curl -s -f --retry 3 --retry-delay 10"; do
  for p in $(pgrep -f -- "$pat"); do
    [ "$p" = "$me" ] || [ "$p" = "$PPID" ] || kill "$p" 2>/dev/null || true
  done
done
sleep 2
echo "remaining downloader processes: $(pgrep -fc -- 'fetch_one|scripts/download_gdc.sh manifest')"
