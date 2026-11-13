#!/usr/bin/env bash
# Usage: bash scripts/evaluate_entity_v19.sh qwen3|qwen3-lora|qwen25|glm [--resume|--dry-run]
set -euo pipefail

V19_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V19_DATA_ROOT="$V19_PROJECT_ROOT/entity_v19_1478"
V19_MODEL_NAME="${1:-qwen3}"
if (( $# > 0 )); then shift; fi
V19_MODE_ARGS=(--base-model-only)

case "$V19_MODEL_NAME" in
  qwen3)
    V19_MODEL_PATH="$V19_PROJECT_ROOT/model_cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    V19_OUTPUT_TAG=qwen3_base
    V19_ANSWER_FORMAT=strict
    ;;
  qwen3-lora)
    V19_MODEL_PATH="$V19_PROJECT_ROOT/model_cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    V19_OUTPUT_TAG=qwen3_lora8_ep1
    V19_ANSWER_FORMAT=strict
    V19_MODE_ARGS=(
      --text-adapter "$V19_DATA_ROOT/runs/experiments/qwen3_lora8_ep1-TEXT/checkpoint-77"
      --visual-adapter "$V19_DATA_ROOT/runs/experiments/qwen3_lora8_ep1-VISION/checkpoint-77"
    )
    ;;
  qwen25)
    V19_MODEL_PATH=/mnt/yunpan2/huggingface_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
    V19_OUTPUT_TAG=qwen25_base
    V19_ANSWER_FORMAT=strict
    ;;
  glm)
    V19_MODEL_PATH="$V19_PROJECT_ROOT/model_cache/modelscope/models/ZhipuAI--GLM-4.6V-Flash/snapshots/master"
    V19_OUTPUT_TAG=glm46_base
    V19_ANSWER_FORMAT=glm-box
    ;;
  *)
    echo 'Usage: bash scripts/evaluate_entity_v19.sh qwen3|qwen3-lora|qwen25|glm [--resume|--dry-run]' >&2
    exit 2
    ;;
esac

# Swift supplies a ModelScope cache_dir explicitly when loading local JSONL.
# HF_DATASETS_CACHE alone therefore does not redirect this dataset cache.
export MODELSCOPE_CACHE="$V19_PROJECT_ROOT/model_cache/modelscope"
export HF_HOME="$V19_PROJECT_ROOT/model_cache/huggingface"
export HF_DATASETS_CACHE="$V19_PROJECT_ROOT/model_cache/datasets"
export TMPDIR="$V19_PROJECT_ROOT/tmp"
mkdir -p "$MODELSCOPE_CACHE" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

exec "$V19_PROJECT_ROOT/.venv/bin/python" "$V19_DATA_ROOT/scripts/run_evaluation.py" \
  --data-root "$V19_DATA_ROOT" \
  --model "$V19_MODEL_PATH" \
  "${V19_MODE_ARGS[@]}" \
  --answer-format "$V19_ANSWER_FORMAT" \
  --directions T2T T2V V2T V2V \
  --gpu "${TEXT_GPU:-0}" --visual-gpu "${VISION_GPU:-1}" --parallel \
  --batch-size 1 --max-new-tokens 4 --max-pixels 1003520 \
  --swift-bin "$V19_PROJECT_ROOT/.venv/bin/swift" \
  --output-dir "$V19_DATA_ROOT/runs/evaluation/$V19_OUTPUT_TAG" \
  "$@"
