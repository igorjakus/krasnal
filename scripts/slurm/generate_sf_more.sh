#!/usr/bin/env bash
#SBATCH --partition=student-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=10G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/slurm/%x-%A_%a.log
set -euo pipefail
cd /Ziob/ijakus/krasnal
export HF_HOME=/Ziob/ijakus/krasnal/data/source-cache/sf-more-20261009/huggingface
export TMPDIR=/Ziob/ijakus/krasnal/data/source-cache/sf-more-20261009/tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 SF_GENERATOR_MODULE=sf_generate_20261009_base
mkdir -p "$HF_HOME" "$TMPDIR"
month=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" scripts/slurm/sf_more_months_20261009.txt)
test -n "$month"
echo "month=$month workers=${WORKERS:-24} node=$SLURMD_NODENAME HF_HOME=$HF_HOME"
exec .venv/bin/python scripts/generate_sf_more.py annotate \
    --month "$month" --out "data/sf-interleave-v2/raw/train/$month.jsonl" \
    --manifest "data/sf-interleave-v2/raw/train/$month.manifest.json" \
    --stockfish /Ziob/ijakus/bin/stockfish --nodes 30000 --threads 1 \
    --workers "${WORKERS:-24}" --game-limit 100000 \
    --min-elo 1100 --min-duration 180 --min-plies 20 --log-every 100 --resume
