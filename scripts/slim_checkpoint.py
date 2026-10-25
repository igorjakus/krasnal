#!/usr/bin/env python3
"""Write an inference-only copy of a Krasnal GPT checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PosixPath

import torch


def slim(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    inference_state = {
        key: checkpoint[key]
        for key in ("model", "config", "training_policy")
        if key in checkpoint
    }
    if "model" not in inference_state or "config" not in inference_state:
        raise ValueError(f"not a model checkpoint: {source}")
    torch.save(inference_state, partial)
    os.replace(partial, destination)
    print(f"{source} -> {destination} ({destination.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    slim(args.source, args.destination)


if __name__ == "__main__":
    main()
