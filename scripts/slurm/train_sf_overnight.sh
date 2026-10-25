#!/usr/bin/env bash
#SBATCH --partition=student-nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/slurm/%x-%j.log
set -euo pipefail
cd /Ziob/ijakus/krasnal-think
DATA_ROOT=data/sf-interleave-distinct-20261008-quick-buffered
RUN_ROOT=experiments/sf-interleave-distinct-20261008
MODE=${MODE:?}
GPUS=${GPUS:?}
OUT="$RUN_ROOT/$MODE"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
if [[ -e "$OUT/model.pt" ]]; then
    echo "Refusing to overwrite completed run $OUT" >&2
    exit 2
fi
resume_args=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    resume_args=(--resume-from "$RESUME_FROM")
fi
.venv-codex/bin/python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
    scripts/train_interleave.py experiments/sf-interleave-v1/checkpoints/pretrain-model.pt \
    --data-dir "$DATA_ROOT/train" --mode "$MODE" --out "$OUT" \
    --steps "${STEPS:?}" --batch-size 2 --grad-accum "$((16 / GPUS))" \
    --lr 1e-5 --min-lr 1e-6 --warmup 200 --save-every 1000 --log-every 50 \
    --human-weight 2 --other-weight 1 --no-compile --max-hours "${MAX_HOURS:-8}" \
    --eval-data-dir "$DATA_ROOT/val" --eval-batches 10 --device cuda "${resume_args[@]}"
mkdir -p "$RUN_ROOT/metrics"
for split in val test; do
    .venv-codex/bin/python scripts/evaluate_interleave.py "$OUT/model.pt" \
        --records "$DATA_ROOT/$split/records.jsonl" --mode "$MODE" \
        --limit 10000 --batch-size 8 --device cuda \
        --out "$RUN_ROOT/metrics/$split-$MODE.json"
done
if [[ "$MODE" == human ]]; then
    for split in val test; do
        .venv-codex/bin/python scripts/evaluate_interleave.py \
            experiments/sf-interleave-v1/checkpoints/pretrain-model.pt \
            --records "$DATA_ROOT/$split/records.jsonl" --mode human \
            --limit 10000 --batch-size 8 --device cuda \
            --out "$RUN_ROOT/metrics/$split-pretrain.json"
    done
fi
