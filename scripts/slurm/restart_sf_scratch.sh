#!/usr/bin/env bash
#SBATCH --partition=student-nvidia
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/slurm/%x-%j.log
set -euo pipefail
cd /Ziob/ijakus/krasnal
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv-codex/bin/python scripts/restart_sf_scratch.py --mode "${MODE:?sf or star}"
