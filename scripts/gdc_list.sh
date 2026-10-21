#!/usr/bin/env bash
# List all open-access TCGA diagnostic H&E slides (SVS) from the GDC API as a TSV.
#   bash scripts/gdc_list.sh > gdc_tcga_dx_slides.tsv
set -euo pipefail
curl -s --max-time 300 'https://api.gdc.cancer.gov/files' -H 'Content-Type: application/json' -d '{
  "filters": {"op": "and", "content": [
    {"op": "in", "content": {"field": "cases.project.program.name", "value": ["TCGA"]}},
    {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
    {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}}]},
  "fields": "file_id,file_name,file_size,md5sum,cases.project.project_id,cases.submitter_id",
  "size": 20000, "format": "TSV"}'
