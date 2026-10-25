#!/usr/bin/env bash
#SBATCH --partition=student-nvidia
#SBATCH --nodelist=opus3
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=09:00:00
#SBATCH --output=experiments/slurm/%x-%j.log
set -euo pipefail
cd /Ziob/ijakus/krasnal-think
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
exec .venv-codex/bin/python scripts/sf_seminar_eval.py
