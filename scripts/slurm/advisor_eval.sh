#!/usr/bin/env bash
# Etap 0 (no training): existing SF-full fine-tune under every advisor condition.
# MODELS="name=path name2=path2" overrides the default checkpoint list.
#SBATCH --job-name=advisor-eval
#SBATCH --partition=student-nvidia
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/slurm/%x-%j.log
set -euo pipefail
cd "${PROJECT_ROOT:-/Ziob/ijakus/krasnal}"
export PYTHONUNBUFFERED=1
RECORDS=${RECORDS:-data/advisor-trust-v1/test/2024-06.jsonl}
OUT=${OUT:-experiments/advisor-trust-v1/etap0/etap0.json}
MODELS=${MODELS:-"sf100m=experiments/sf-interleave-100m-full-20261009-humanloss/model.pt scratch-sf=experiments/sf-interleave-scratch-full-20261009-humanloss/model.pt"}
args=()
for pair in $MODELS; do args+=(--model "${pair%%=*}" "${pair#*=}"); done
exec .venv/bin/python scripts/advisor_eval.py --records "$RECORDS" --out "$OUT" \
    --device cuda --batch-size 16 --bootstrap 1000 "${args[@]}" ${EXTRA_ARGS:-}
