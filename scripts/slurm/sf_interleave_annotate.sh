#!/usr/bin/env bash
#SBATCH --partition=student-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --mem=60G
#SBATCH --output=experiments/slurm/%x-%A_%a.log

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/Ziob/ijakus/krasnal}
MONTHS_FILE=${MONTHS_FILE:-scripts/slurm/sf_interleave_months.txt}
DATA_ROOT=${DATA_ROOT:-data/sf-interleave-v2/raw/train}
ARRAY_OFFSET=${ARRAY_OFFSET:-0}
GAME_LIMIT=${GAME_LIMIT:-100000}
WORKERS=${WORKERS:-56}
THREADS=${THREADS:-1}
NODES=${NODES:-30000}
MIN_ELO=${MIN_ELO:-1100}
MIN_DURATION=${MIN_DURATION:-180}
MIN_PLIES=${MIN_PLIES:-20}
STOCKFISH=${STOCKFISH:-/Ziob/ijakus/bin/stockfish}

cd "$PROJECT_ROOT"
line=$((ARRAY_OFFSET + SLURM_ARRAY_TASK_ID + 1))
month=$(sed -n "${line}p" "$MONTHS_FILE")
if [[ -z "$month" ]]; then
    echo "No month at line $line in $MONTHS_FILE" >&2
    exit 2
fi

mkdir -p "$DATA_ROOT"
out="$DATA_ROOT/$month.jsonl"
manifest="$DATA_ROOT/$month.manifest.json"

echo "Annotating month=$month nodes=$NODES workers=$WORKERS game_limit=$GAME_LIMIT"
exec .venv/bin/python scripts/sf_interleave.py annotate \
    --month "$month" \
    --out "$out" \
    --manifest "$manifest" \
    --stockfish "$STOCKFISH" \
    --nodes "$NODES" \
    --workers "$WORKERS" \
    --threads "$THREADS" \
    --game-limit "$GAME_LIMIT" \
    --min-elo "$MIN_ELO" \
    --min-duration "$MIN_DURATION" \
    --min-plies "$MIN_PLIES" \
    --resume
