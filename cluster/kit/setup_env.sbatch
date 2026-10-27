#!/usr/bin/env bash
#SBATCH --job-name=xmodal-setup
#SBATCH --partition=REPLACE_WITH_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
srun bash cluster/kit/setup_env.sh
