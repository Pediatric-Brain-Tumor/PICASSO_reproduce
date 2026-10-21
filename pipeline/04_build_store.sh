#!/usr/bin/env bash
# Stage 4: one fp16 array per split with an 80/10/10 slide-level split.
#   bash pipeline/04_build_store.sh [--set key=value ...]
set -euo pipefail; cd "$(dirname "$0")/.."; source env.sh
python -m picasso.store --config configs/data/store.yaml ${PROJECTS:+--set "projects=$PROJECTS"} "$@"
