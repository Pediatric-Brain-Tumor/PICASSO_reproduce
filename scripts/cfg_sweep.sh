#!/usr/bin/env bash
# Guidance-scale sweep for an Emb2Img checkpoint on the full test set.
#   bash scripts/cfg_sweep.sh <checkpoint_dir> <out_dir> [n=256] [steps=100] [gpu=0]
set -euo pipefail; cd "$(dirname "$0")/.."; source env.sh
CK=$1; OUT=$2; N=${3:-256}; STEPS=${4:-100}; GPU=${5:-0}
for g in 1.0 1.5 2.0 2.5 3.0 4.0 5.0 6.5; do
  CUDA_VISIBLE_DEVICES=$GPU python -m picasso.emb2img.sample_eval --config configs/emb2img/sample.yaml \
    --set checkpoint=$CK pairs_dir=$DATA_ROOT/emb2img_pairs output_dir=$OUT/g$g n=$N num_steps=$STEPS guidance=$g 2>&1 \
    | grep -v -i warning | grep -E "cos_gen_cond_mean|Traceback|Error" | sed "s/^/g=$g /"
done
python - <<PY
import json, glob
rows=[(float(p.split('/g')[-1].split('/')[0]), json.load(open(p))) for p in glob.glob("$OUT/g*/summary.json")]
for g,s in sorted(rows): print(f"guidance {g:4.1f}: cos {s['cos_gen_cond_mean']:.4f} median {s['cos_gen_cond_median']:.4f} above_nn {s['frac_gen_above_nn']:.3f}")
PY
