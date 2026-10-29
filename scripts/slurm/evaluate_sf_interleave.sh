#!/usr/bin/env bash
#SBATCH --partition=student-nvidia
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-06:00:00
#SBATCH --mem=32G
#SBATCH --output=experiments/slurm/%x-%j.log

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/Ziob/ijakus/krasnal}
RUN_ROOT=${RUN_ROOT:-experiments/sf-interleave-v1}
DATA_ROOT=${DATA_ROOT:-data/sf-interleave-v1}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-experiments/sf-interleave-v1/checkpoints/pretrain-model.pt}

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/metrics"

run_eval() {
    local split=$1
    local label=$2
    local checkpoint=$3
    local mode=$4
    .venv/bin/python scripts/evaluate_interleave.py \
        "$checkpoint" \
        --records "$DATA_ROOT/$split/records.jsonl" \
        --mode "$mode" \
        --limit 10000 \
        --batch-size 32 \
        --device cuda \
        --out "$RUN_ROOT/metrics/$split-$label.json"
}

base="$BASE_CHECKPOINT"
control="$RUN_ROOT/control-human/model.pt"
treatment="$RUN_ROOT/treatment-sf-human/model.pt"

for split in val test; do
    run_eval "$split" base-human "$base" human
    run_eval "$split" control-human "$control" human
    run_eval "$split" treatment-augmented "$treatment" augmented
done
