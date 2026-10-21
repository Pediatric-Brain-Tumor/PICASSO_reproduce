#!/usr/bin/env bash
# Stage 3: Virchow2 CLS embedding of every tile, one process per GPU (resumable).
#   GPUS="0 1 2 3 4 5 6 7" bash pipeline/03_embed.sh
set -euo pipefail; cd "$(dirname "$0")/.."; source env.sh
arr=(${GPUS:-0}); n=${#arr[@]}
for i in "${!arr[@]}"; do
  CUDA_VISIBLE_DEVICES=${arr[$i]} python -m picasso.embed --config configs/data/embed.yaml \
    --set shard=$i num_shards=$n ${PROJECTS:+"projects=$PROJECTS"} > "$OUTPUT_ROOT/logs/embed_$i.log" 2>&1 &
done
wait
