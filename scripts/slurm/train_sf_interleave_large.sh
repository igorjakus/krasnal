#!/usr/bin/env bash
#SBATCH --partition=student-nvidia
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --output=experiments/slurm/%x-%j.log

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/Ziob/ijakus/krasnal}
DATA_ROOT=${DATA_ROOT:-data/sf-interleave-v2}
RUN_ROOT=${RUN_ROOT:-experiments/sf-interleave-v2}
MODE=${MODE:?MODE must be human or augmented}
STEPS=${STEPS:?STEPS must be positive}
OUT=${OUT:?OUT must be set}
CHECKPOINT=${CHECKPOINT:-experiments/sf-interleave-v1/checkpoints/pretrain-model.pt}

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT"
exec .venv/bin/python scripts/train_interleave.py \
    "$CHECKPOINT" \
    --data-dir "$DATA_ROOT/train" \
    --mode "$MODE" \
    --out "$OUT" \
    --steps "$STEPS" \
    --batch-size 4 \
    --grad-accum 16 \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --human-weight 2.0 \
    --other-weight 1.0 \
    --eval-data-dir "$DATA_ROOT/val" \
    --eval-batches 50 \
    --save-every "${SAVE_EVERY:-10000}" \
    --device cuda
