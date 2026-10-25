#!/usr/bin/env python3
"""Hide forced-thinking diagnostics in the pretrain row without losing them."""

import argparse
import json
import os
from pathlib import Path

from curriculum import atomic_json, report, sha

FIELDS = (
    "thinking_top1",
    "candidate_recall",
    "selection",
    "selection_when_best_present",
    "raw_format_error",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result_path = root / "results/000-pretrain.json"
    original = result_path.read_bytes()
    result = json.loads(original)
    if result["stage"] != "000-pretrain":
        raise ValueError("expected the pretrain result")
    backup = root / "diagnostics/000-pretrain.forced-thinking.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        partial = backup.with_suffix(".json.partial")
        with partial.open("xb") as handle:
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(backup)
    diagnostic = json.loads(backup.read_text())
    if diagnostic["stage"] != "000-pretrain":
        raise ValueError("diagnostic backup is not the pretrain result")
    result["forced_thinking_diagnostic"] = {
        "note": "pretrain did not learn the thinking format; these are forced rollouts",
        "original_sha256": sha(backup),
        "values": {key: diagnostic.get(key) for key in FIELDS},
    }
    for key in FIELDS:
        result[key] = None
    atomic_json(result_path, result)
    report(root)


if __name__ == "__main__":
    main()
