#!/usr/bin/env bash
# Stage 2: tissue mask + 256 px @ 0.5 um/px tile grid per slide (CPU, resumable).
#   PROJECTS="[TCGA-CHOL]" bash pipeline/02_tile.sh    # optional: a subset of projects
set -euo pipefail; cd "$(dirname "$0")/.."; source env.sh
python -m picasso.tiling --config configs/data/tiling.yaml ${PROJECTS:+--set "projects=$PROJECTS"}
