#!/usr/bin/env python3
"""Create a checksum-backed manifest for a Stockfish-interleave evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/sf-interleave-v1"))
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/sf-interleave-v1/dataset.manifest.json"),
    )
    parser.add_argument("--metrics-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    metrics_dir = args.metrics_dir or args.root / "metrics"
    output = args.out or metrics_dir / "evaluation.manifest.json"
    dataset_manifest = json.loads(args.dataset_manifest.read_text())
    checkpoints = {
        "pretrained": args.root / "checkpoints/pretrain-model.pt",
        "control_training": args.root / "control-human/model.pt",
        "treatment_training": args.root / "treatment-sf-human/model.pt",
        "control_inference_copy": args.root / "checkpoints/control-human-inference.pt",
        "treatment_inference_copy": args.root / "checkpoints/treatment-sf-human-inference.pt",
    }
    checkpoint_info = {
        name: describe(path) for name, path in checkpoints.items() if path.is_file()
    }
    training = {}
    for name, relative in (
        ("control", "control-human/train-summary.json"),
        ("treatment", "treatment-sf-human/train-summary.json"),
    ):
        path = args.root / relative
        if path.is_file():
            training[name] = {"summary": json.loads(path.read_text()), **describe(path)}

    records = {}
    for split in ("val", "test"):
        path = Path("data/sf-interleave-v1") / split / "records.jsonl"
        if path.is_file():
            records[split] = describe(path)

    metrics = {}
    for path in sorted(metrics_dir.glob("*.json")):
        if path == output or not path.name.startswith(("val-", "test-")):
            continue
        metrics[path.name] = {**describe(path), "result": json.loads(path.read_text())}

    manifest = {
        "schema_version": 1,
        "kind": "stockfish_interleave_evaluation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_revision": dataset_manifest.get("code_revision"),
        "evaluation_host": "adrian; NVIDIA GeForce RTX 3090",
        "evaluator_script": describe(Path("scripts/evaluate_interleave.py")),
        "training_script": describe(Path("scripts/train_interleave.py")),
        "dataset_manifest": describe(args.dataset_manifest),
        "representation": dataset_manifest.get("representation"),
        "stockfish": dataset_manifest.get("stockfish"),
        "evaluation_protocol": {
            "augmented_mode_uses_teacher_forced_stockfish": True,
            "human_mode_uses_teacher_forced_stockfish": False,
            "stockfish_moves_are_true_prefix_labels": True,
            "primary_outcome": "human move predictions",
            "secondary_outcome": "Stockfish move predictions in augmented mode",
            "metrics": "token-weighted NLL, top-1 and top-5 accuracy",
            "record_limit_per_split": 10_000,
            "splits": records,
        },
        "training": training,
        "checkpoints": checkpoint_info,
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "metrics": len(metrics)}, sort_keys=True))


if __name__ == "__main__":
    main()
