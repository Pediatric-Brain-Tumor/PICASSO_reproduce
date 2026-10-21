#!/usr/bin/env bash
# Stage 6: (VAE latent, Virchow2 CLS) training pairs for Emb2Img, one process per GPU, then merge.
#   GPUS="0 1 2 3 4 5 6 7" bash pipeline/06_build_pairs.sh
set -euo pipefail; cd "$(dirname "$0")/.."; source env.sh
arr=(${GPUS:-0}); n=${#arr[@]}
for i in "${!arr[@]}"; do
  CUDA_VISIBLE_DEVICES=${arr[$i]} python -m picasso.emb2img.pairs --config configs/emb2img/pairs.yaml \
    --set shard=$i num_shards=$n > "$OUTPUT_ROOT/logs/pairs_$i.log" 2>&1 &
done
wait
python -m picasso.emb2img.pairs --config configs/emb2img/pairs.yaml --set merge=true
