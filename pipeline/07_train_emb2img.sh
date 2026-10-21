#!/usr/bin/env bash
# Stage 7: train Emb2Img with DDP (resumes from the last checkpoint), then sample + closed-loop eval.
# Effective batch = 128 x n_gpus x grad_accum; keep it at 1024 (8 GPUs: GRAD_ACCUM=1, 4 GPUs: 2).
#   GPUS="0,1,2,3,4,5,6,7" GRAD_ACCUM=1 bash pipeline/07_train_emb2img.sh
set -euo pipefail; cd "$(dirname "$0")/.."; source env.sh
GPUS=${GPUS:-0}; N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.run --nproc_per_node "$N" --master_port ${PORT:-29571} \
  -m picasso.emb2img.train --config configs/emb2img/train.yaml --set grad_accum=${GRAD_ACCUM:-1} "$@"
CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m picasso.emb2img.sample_eval --config configs/emb2img/sample.yaml
