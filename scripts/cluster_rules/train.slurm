#!/usr/bin/env bash
# Supply the actual --partition and, if required, --account to sbatch.
#SBATCH --job-name=xmodal-rules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

: "${PACKAGE_ROOT:?Set PACKAGE_ROOT to the extracted immutable data package}"
: "${WORK_ROOT:?Run prepare.py first, then set WORK_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH to the staged Qwen3-VL-4B-Instruct directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT outside the data package}"
: "${ARM:?Set ARM to original, repeat10, or multiform10}"
: "${MODALITY:?Set MODALITY to text or vision}"

# Activate your ms-swift environment before sbatch (Slurm exports it).
# Override time/memory/GPU defaults through sbatch options to match cluster policy.
SMOKE_ARGS=()
if [[ "${SMOKE:-0}" == 1 ]]; then
    SMOKE_ARGS=(--smoke)
fi
JOB_NAME="${ARM}_${MODALITY}_seed${SEED:-42}_${SLURM_JOB_ID}"
if [[ "${SMOKE:-0}" == 1 ]]; then
    JOB_NAME="smoke_${JOB_NAME}"
fi
srun --ntasks=1 python "$PACKAGE_ROOT/tools/train.py" \
    --root "$PACKAGE_ROOT" --work "$WORK_ROOT" --model "$MODEL_PATH" \
    --output "$OUTPUT_ROOT/$JOB_NAME" --arm "$ARM" --modality "$MODALITY" \
    --epochs "${EPOCHS:-10}" --seed "${SEED:-42}" "${SMOKE_ARGS[@]}" --execute
