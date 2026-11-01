#!/usr/bin/env bash
# One array task per input JSONL listed (one per line) in $INPUTS_FILE.
#   MODE=all      evaluation records: every SF advisor in $ADVISORS for every game
#   MODE=assigned training records: per-game advisor from $CONFIG, SF only where needed
# sbatch --array=0-72%8 --export=ALL,MODE=assigned,INPUTS_FILE=...,OUT_ROOT=... scripts/slurm/advisor_annotate.sh
#SBATCH --job-name=advisor-annotate
#SBATCH --partition=student-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/slurm/%x-%A_%a.log
set -euo pipefail
cd "${PROJECT_ROOT:-/Ziob/ijakus/krasnal}"
export PYTHONUNBUFFERED=1
MODE=${MODE:?MODE=all|assigned}
INPUTS_FILE=${INPUTS_FILE:?}
OUT_ROOT=${OUT_ROOT:?}
CONFIG=${CONFIG:-configs/advisor-trust-v1.json}
ADVISORS=${ADVISORS:-"sfd1 sfd2 sfd4 sfd8"}
STOCKFISH=${STOCKFISH:-/Ziob/ijakus/bin/stockfish}
WORKERS=${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))}
input=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$INPUTS_FILE")
[[ -n "$input" ]] || { echo "no input for task ${SLURM_ARRAY_TASK_ID:-0}" >&2; exit 2; }
out="$OUT_ROOT/$(basename "$input")"
if [[ -f "$out" ]]; then echo "done already: $out"; exit 0; fi
mkdir -p "$OUT_ROOT"
extra=()
if [[ "$MODE" == all ]]; then for a in $ADVISORS; do extra+=(--advisor "$a"); done
else extra+=(--config "$CONFIG"); fi
exec .venv/bin/python scripts/advisor_data.py annotate --mode "$MODE" --in "$input" --out "$out" \
    --stockfish "$STOCKFISH" --workers "$WORKERS" "${extra[@]}"
