#!/usr/bin/env bash
# Usage: bash scripts/train_entity_v19.sh TEXT|VISION GPU [EXPERIMENT] [--dry-run]
set -euo pipefail

V19_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V19_DATA_ROOT="$V19_PROJECT_ROOT/entity_v19_1478"
V19_MODE="${1:-}"
V19_GPU="${2:-0}"
V19_EXPERIMENT="${3:-qwen3_lora8_ep1}"
V19_DRY_RUN="${4:-}"
case "$V19_MODE" in
  TEXT) V19_DIRECTION=T2T ;;
  VISION) V19_DIRECTION=V2V ;;
  *) echo 'Usage: bash scripts/train_entity_v19.sh TEXT|VISION GPU [EXPERIMENT] [--dry-run]' >&2; exit 2 ;;
esac
if [[ ! "$V19_EXPERIMENT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
   [[ -n "$V19_DRY_RUN" && "$V19_DRY_RUN" != --dry-run ]] || (( $# > 4 )); then
  echo 'Invalid experiment name or arguments.' >&2
  exit 2
fi
if [[ ! "$V19_GPU" =~ ^[0-9]+$ ]]; then
  echo 'Specify one numeric GPU ID.' >&2
  exit 2
fi

V19_MODEL_PATH="$V19_PROJECT_ROOT/model_cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
V19_TRAIN_FILE="$V19_DATA_ROOT/$V19_DIRECTION/train.jsonl"
V19_OUTPUT="$V19_DATA_ROOT/runs/experiments/$V19_EXPERIMENT-$V19_MODE"
test -f "$V19_MODEL_PATH/config.json"
test -f "$V19_TRAIN_FILE"
if [[ -e "$V19_OUTPUT" ]]; then
  echo "Output already exists: $V19_OUTPUT; choose another experiment name." >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="$V19_GPU"
export ROOT_IMAGE_DIR="$V19_DATA_ROOT"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export MODELSCOPE_CACHE="$V19_PROJECT_ROOT/model_cache/modelscope"
export HF_HOME="$V19_PROJECT_ROOT/model_cache/huggingface"
export HF_DATASETS_CACHE="$V19_PROJECT_ROOT/model_cache/datasets"
export TMPDIR="$V19_PROJECT_ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
unset NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT LOCAL_RANK RANK WORLD_SIZE
unset LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
for V19_ENV_KEY in ${!TORCHELASTIC_@}; do unset "$V19_ENV_KEY"; done

V19_ARGS=(
  "$V19_PROJECT_ROOT/.venv/bin/swift" pt
  --model "$V19_MODEL_PATH" --dataset "$V19_TRAIN_FILE"
  --tuner_type lora --target_modules all-linear
  --lora_rank 8 --lora_alpha 32 --lora_dropout 0.05
  --torch_dtype bfloat16 --freeze_vit true --freeze_aligner true
  --per_device_train_batch_size 1 --gradient_accumulation_steps "${GRAD_ACCUM:-16}"
  --learning_rate "${LEARNING_RATE:-1e-4}" --num_train_epochs "${EPOCHS:-1}"
  --gradient_checkpointing true --max_length "${MAX_LENGTH:-8192}"
  --warmup_ratio 0.05 --split_dataset_ratio 0
  --use_chat_template false --loss_scale all --strict true --enable_thinking false
  --save_strategy epoch --save_total_limit 2 --logging_steps 5
  --seed 42 --data_seed 42 --add_version false --output_dir "$V19_OUTPUT"
)
echo "Training mode: $V19_MODE; GPU: $V19_GPU; dataset: $V19_TRAIN_FILE"
echo "Output: $V19_OUTPUT"
if [[ "$V19_DRY_RUN" == --dry-run ]]; then
  printf '%q ' "${V19_ARGS[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$MODELSCOPE_CACHE" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$V19_OUTPUT/logs"
exec > >(tee "$V19_OUTPUT/logs/train.log") 2>&1
cd "$V19_DATA_ROOT"
"${V19_ARGS[@]}"
