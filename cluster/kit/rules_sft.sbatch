#!/usr/bin/env bash
# Replace the partition with one returned by: sinfo -o "%P %a %l %D %G"
#SBATCH --job-name=xmodal-rules-sft
#SBATCH --partition=REPLACE_WITH_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rules-sft-%j.out
#SBATCH --error=logs/rules-sft-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
ENV_PREFIX="${XMODAL_CONDA_ENV:-$PROJECT_DIR/.conda/envs/xmodal}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

DATA_DIR="${XMODAL_DATA_DIR:-$PROJECT_DIR/artifacts/dataset}"
MODEL_DIR="${XMODAL_BASE_MODEL:-$PROJECT_DIR/model_cache/Qwen3-VL-4B-Instruct}"
WORK_DIR="${XMODAL_WORK:-$PROJECT_DIR/runs/kit_rules_full_sft_ep10}"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$PROJECT_DIR/runs/qwen3_lora_cot_eval_20261025" "$HF_DATASETS_CACHE"
cd "$PROJECT_DIR"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  python scripts/train_rules_full_sft_ep10.py --train \
  --model "$MODEL_DIR" --source-data "$DATA_DIR/learning" \
  --source-manifest "$DATA_DIR/learning/manifest.json" \
  --image-root "$DATA_DIR/data/rules_repaired_single_image_v7/images" \
  --work "$WORK_DIR"
