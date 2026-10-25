#!/usr/bin/env bash
#SBATCH --partition=student-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=60G
#SBATCH --output=experiments/slurm/%x-%j.log

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/Ziob/ijakus/krasnal-think}
OUT_ROOT=${OUT_ROOT:-data/sf-interleave-v2}

cd "$PROJECT_ROOT"
shopt -s nullglob
train_shards=(data/sf-interleave-v1/raw/train/*.jsonl data/sf-interleave-v2/raw/train/*.jsonl)
val_shards=(data/sf-interleave-v1/raw/val/*.jsonl)
test_shards=(data/sf-interleave-v1/raw/test/*.jsonl)
if (( ${#train_shards[@]} == 0 || ${#val_shards[@]} != 1 || ${#test_shards[@]} != 1 )); then
    echo "Unexpected shard layout" >&2
    exit 2
fi

args=(build --out-dir "$OUT_ROOT" --manifest "$OUT_ROOT/dataset.manifest.json")
args+=(--train-shards "${train_shards[@]}")
args+=(--val-shards "${val_shards[@]}")
args+=(--test-shards "${test_shards[@]}")
exec .venv-codex/bin/python scripts/sf_interleave.py "${args[@]}"
