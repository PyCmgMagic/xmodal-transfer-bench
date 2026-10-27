#!/usr/bin/env bash
#SBATCH --job-name=xmodal-fetch
#SBATCH --partition=REPLACE_WITH_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fetch-%j.out
#SBATCH --error=logs/fetch-%j.err
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit from project root}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${XMODAL_CONDA_ENV:-$PROJECT_DIR/.conda/envs/xmodal}"
cd "$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
read -r -a ASSETS <<< "${XMODAL_ASSETS:-dataset base}"
srun python scripts/fetch_hub_assets.py --assets "${ASSETS[@]}"
