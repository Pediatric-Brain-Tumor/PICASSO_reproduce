#!/usr/bin/env bash
# Stage 5: train the TopK SAE on one GPU, then evaluate it on the test split.
#   SEED=43 bash pipeline/05_train_sae.sh [--set key=value ...]
set -euo pipefail; cd "$(dirname "$0")/.."; source env.sh
SEED=${SEED:-42}; OUT=$OUTPUT_ROOT/sae/virchow2_cls_x32_k30_seed$SEED
python -m picasso.train --config configs/sae/train.yaml --set seed=$SEED output_dir="$OUT" "$@"
python -m picasso.evaluate --config configs/sae/evaluate.yaml --set checkpoint="$OUT/sae.pt" output_dir="$OUT/eval"
