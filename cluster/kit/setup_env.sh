#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_PREFIX="${XMODAL_CONDA_ENV:-$PROJECT_DIR/.conda/envs/xmodal}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -p "$ENV_PREFIX" python=3.10 -y
fi
conda activate "$ENV_PREFIX"
cd "$PROJECT_DIR"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
if [[ -n "${XMODAL_WHEELHOUSE:-}" ]]; then
  python -m pip install --no-index --find-links "$XMODAL_WHEELHOUSE" \
    -r requirements-runtime.lock -r requirements-migration.txt
else
  python -m pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -r requirements-runtime.lock -r requirements-migration.txt
fi
python -m pip check
python -c 'import sys, importlib.metadata as m; print(sys.executable); print({p:m.version(p) for p in ("torch","ms-swift","transformers","bitsandbytes","huggingface-hub")})'
