#!/usr/bin/env bash
# Run inside the worker runtime: plan, run, append --sources ..., or status.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$#" -eq 0 ]; then set -- run; fi
exec "${KT_PYTHON:-python}" scripts/experiment.py "$@"
