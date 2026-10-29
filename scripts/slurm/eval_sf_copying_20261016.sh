#!/usr/bin/env bash
#SBATCH --job-name=sf-copying-20261016
#SBATCH --partition=student-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/sf-copying-diagnostic-20261016/logs/slurm-%j.out
set -euo pipefail

cd /Ziob/ijakus/krasnal
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

exec .venv-codex/bin/python experiments/sf-copying-diagnostic-20261016/code/evaluate_sf_copying.py \
  --records data/sf-interleave-distinct-20261008-full-buffered/test/records.jsonl \
  --out experiments/sf-copying-diagnostic-20261016/copy-agreement.json \
  --batch-size 8 --cpu-threads "${SLURM_CPUS_PER_TASK:-8}" --bootstrap 4000 \
  --model finetune100m-sf experiments/sf-interleave-100m-full-20261009-humanloss/model.pt sf \
  --model scratch-sf experiments/sf-interleave-scratch-full-20261009-humanloss/model.pt sf \
  --model scratch-star experiments/sf-interleave-star-full-20261009-humanloss/model.pt star
