#!/usr/bin/env bash
# Usage: source cluster/ebcloud/environment.sh [qwen|glm]
XMODAL_ROOT="${XMODAL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
XMODAL_RUNTIME="${1:-qwen}"
case "$XMODAL_RUNTIME" in
  qwen|glm) ;;
  *) printf 'Expected runtime qwen or glm\n' >&2; return 2 ;;
esac
source "$XMODAL_ROOT/envs/$XMODAL_RUNTIME/bin/activate"
export XMODAL_ROOT
export TMPDIR="$XMODAL_ROOT/.runtime/tmp"
export XDG_CACHE_HOME="$XMODAL_ROOT/.runtime/cache"
export HF_HOME="$XMODAL_ROOT/.runtime/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MODELSCOPE_CACHE="$XMODAL_ROOT/model_cache/modelscope"
export MPLCONFIGDIR="$XMODAL_ROOT/.runtime/cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export QWEN_MODEL_PATH="$XMODAL_ROOT/model_cache/Qwen3-VL-4B-Instruct"
export GLM_MODEL_PATH="$XMODAL_ROOT/model_cache/modelscope/models/ZhipuAI--GLM-4.6V-Flash/snapshots/master"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"
