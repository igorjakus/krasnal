#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-experiments/sf-interleave-v1}
DATA_ROOT=${DATA_ROOT:-data/sf-interleave-v1}
METRICS_ROOT=${METRICS_ROOT:-experiments/sf-interleave-v1/metrics-adrian}
CONTROL_CHECKPOINT=${CONTROL_CHECKPOINT:-$RUN_ROOT/control-human/model.pt}
TREATMENT_CHECKPOINT=${TREATMENT_CHECKPOINT:-$RUN_ROOT/treatment-sf-human/model.pt}
export RUN_ROOT DATA_ROOT METRICS_ROOT CONTROL_CHECKPOINT TREATMENT_CHECKPOINT

docker run --rm --gpus all \
    -e PYTHONPATH=/app/src \
    -e RUN_ROOT -e DATA_ROOT -e METRICS_ROOT \
    -e CONTROL_CHECKPOINT -e TREATMENT_CHECKPOINT \
    -v "$PWD:/app" \
    -w /app \
    krasnal-think:audit-runtime \
    bash -lc '
set -euo pipefail
run_eval() {
    local split=$1 label=$2 checkpoint=$3 mode=$4
    python scripts/evaluate_interleave.py "$checkpoint" \
        --records "$DATA_ROOT/$split/records.jsonl" \
        --mode "$mode" --limit 10000 --batch-size 32 --device cuda \
        --out "$METRICS_ROOT/$split-$label.json"
}
mkdir -p "$METRICS_ROOT"
for split in val test; do
    run_eval "$split" control-human "$CONTROL_CHECKPOINT" human
    run_eval "$split" control-augmented "$CONTROL_CHECKPOINT" augmented
    run_eval "$split" treatment-human "$TREATMENT_CHECKPOINT" human
    run_eval "$split" treatment-augmented "$TREATMENT_CHECKPOINT" augmented
done
'
