#!/usr/bin/env bash
# sbatch --export=ALL,ANN_ROOT=data/advisor-trust-v1/raw scripts/slurm/advisor_build.sh
#SBATCH --job-name=advisor-build
#SBATCH --partition=student-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/slurm/%x-%j.log
set -euo pipefail
cd "${PROJECT_ROOT:-/Ziob/ijakus/krasnal}"
export PYTHONUNBUFFERED=1
ANN_ROOT=${ANN_ROOT:-data/advisor-trust-v1/raw}
CONFIG=${CONFIG:-configs/advisor-trust-v1.json}
OUT_DIR=${OUT_DIR:-$(.venv/bin/python -c "import json;print(json.load(open('$CONFIG'))['data'])")}
exec .venv/bin/python scripts/advisor_data.py build --config "$CONFIG" \
    --train "$ANN_ROOT"/train/*.jsonl --val "$ANN_ROOT"/val/*.jsonl \
    --out-dir "$OUT_DIR" --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
