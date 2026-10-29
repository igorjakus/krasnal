#!/usr/bin/env bash
#SBATCH --partition=student-nvidia
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/slurm/%x-%j.log
set -euo pipefail
cd /Ziob/ijakus/krasnal
export PYTHONUNBUFFERED=1
exec .venv/bin/python scripts/run_sf_full.py
