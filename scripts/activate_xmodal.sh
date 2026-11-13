#!/usr/bin/env bash
# Source this file from any working directory to activate the project environment.
XMODAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$XMODAL_ROOT/.venv/bin/activate"
export TMPDIR="$XMODAL_ROOT/.runtime/tmp"
export XDG_CACHE_HOME="$XMODAL_ROOT/.runtime/cache"
export PIP_CACHE_DIR="$XMODAL_ROOT/.runtime/cache/pip"
export UV_CACHE_DIR="$XMODAL_ROOT/.runtime/cache/uv"
export HF_HOME="$XMODAL_ROOT/.runtime/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MODELSCOPE_CACHE="$XMODAL_ROOT/model_cache/modelscope"
export MPLCONFIGDIR="$XMODAL_ROOT/.runtime/cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"
