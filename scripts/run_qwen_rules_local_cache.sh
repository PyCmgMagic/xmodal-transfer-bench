#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/yunpan2/tanjiyang/xmodal"
export MODELSCOPE_CACHE="$ROOT/model_cache/modelscope"
export MODELSCOPE_HOME="$ROOT/model_cache/modelscope"
export HF_HOME="$ROOT/model_cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/model_cache/huggingface/datasets"
export XDG_CACHE_HOME="$ROOT/model_cache/xdg"

cd "$ROOT/rules_corrected_final"
source "$ROOT/.venv/bin/activate"
exec bash scripts/evaluate_rules.sh base --resume
