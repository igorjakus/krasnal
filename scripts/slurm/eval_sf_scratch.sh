#!/usr/bin/env bash
#SBATCH --partition=student-nvidia
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/slurm/%x-%j.log
set -euo pipefail
cd /Ziob/ijakus/krasnal
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec .venv-codex/bin/python scripts/eval_sf_scratch.py
