#!/usr/bin/env bash
# Stage 1: resumable, md5-checked download of all TCGA diagnostic slides from the GDC (open access).
# manifests/tcga_dx_all.csv lists 11,901 slides (13.2 TB). To refresh it from the GDC:
#   bash scripts/gdc_list.sh > list.tsv && python scripts/make_manifest.py --input list.tsv --out manifests/tcga_dx_all.csv
#   PAR=8 bash pipeline/01_download.sh
set -euo pipefail; cd "$(dirname "$0")/.."; source env.sh
MANIFEST=${MANIFEST:-manifests/tcga_dx_all.csv}
cp "$MANIFEST" "$DATA_ROOT/manifest.csv"
bash scripts/download_gdc.sh "$DATA_ROOT/manifest.csv" "$DATA_ROOT/slides" "${PAR:-4}"
