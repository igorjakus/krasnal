#!/usr/bin/env bash
#SBATCH --partition=student-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/slurm/%x-%j.log
set -euo pipefail
cd /Ziob/ijakus/krasnal-think
exec .venv-codex/bin/python scripts/build_sf_overnight.py "${BUILD_SCOPE:-quick}"
