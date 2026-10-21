# Source before running any stage:  source env.sh
# DATA_ROOT: slides, tiles, embeddings.  OUTPUT_ROOT: checkpoints, logs.
# STORE_DIR / PAIRS_DIR: training data of the SAE / Emb2Img (override them to train on a subset).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_ROOT="${DATA_ROOT:-$REPO/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/outputs}"
export STORE_DIR="${STORE_DIR:-$DATA_ROOT/store/virchow2}"      # SAE train/val/test arrays
export PAIRS_DIR="${PAIRS_DIR:-$DATA_ROOT/emb2img_pairs}"       # Emb2Img training pairs
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# Old system libstdc++ (no GLIBCXX_3.4.29) breaks conda's pillow/openslide once torch is imported:
# put conda's libstdc++ first in that case only.
SYS_STDCXX=$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6 /{p=$NF} END{print p}' || true)
if [ -n "${CONDA_PREFIX:-}" ] && [ -n "$SYS_STDCXX" ] && ! grep -aq GLIBCXX_3.4.29 "$SYS_STDCXX"; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT/logs"
