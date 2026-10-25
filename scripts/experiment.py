#!/usr/bin/env python3
"""One configured pretrain -> (SFT -> DAgger -> eval) experiment, with resume/append."""

import argparse
import contextlib
import fcntl
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_config(path):
    config = json.loads(path.read_text())
    if config.get("schema") != 1:
        raise ValueError("unsupported experiment schema")
    if not config.get("sources"):
        raise ValueError("explicit input sources are required")
    if set(config) != {
        "schema",
        "root",
        "foundation",
        "sources",
        "prepare",
        "train",
        "eval",
        "device",
    }:
        raise ValueError("unknown or missing experiment configuration fields")
    if config["prepare"]["rounds"] < 1:
        raise ValueError("rounds must be positive")
    ratio = config["prepare"].get("dagger_ratio")
    if ratio is not None and not 0 < ratio <= 1:
        raise ValueError("dagger_ratio must be in (0,1]")
    if not 0 < config["train"]["teacher_fraction"] < 1:
        raise ValueError("teacher_fraction must be in (0,1)")
    return config


def flags(options):
    return [
        item
        for key, value in options.items()
        if value is not None
        for item in ("--" + key.replace("_", "-"), str(value))
    ]


def commands(config, sources=None):
    foundation = config["foundation"]
    prepare = [
        sys.executable,
        "scripts/curriculum.py",
        "prepare",
        "--root",
        config["root"],
        "--sources",
        *(sources or config["sources"]),
        "--block-size",
        str(foundation["block_size"]),
        *flags(config["prepare"]),
    ]
    run = [
        sys.executable,
        "scripts/curriculum.py",
        "run",
        "--root",
        config["root"],
        "--checkpoint",
        foundation["checkpoint"],
        "--human-val",
        foundation["human_val"],
        "--workers",
        str(config["prepare"]["workers"]),
        "--device",
        config["device"],
        *flags(config["train"]),
        *flags(config["eval"]),
    ]
    return prepare, run


def plan(config):
    prepare, run = commands(config)
    print("Foundation (human pretrain):", config["foundation"]["checkpoint"])
    print("Inputs:", ", ".join(config["sources"]))
    print(
        f"Chain: pretrain eval -> {config['prepare']['rounds']} x [SFT -> eval -> DAgger -> eval]"
    )
    ratio = config["prepare"].get("dagger_ratio")
    budget = (
        f"ceil({ratio:g} x round teacher records)"
        if ratio is not None
        else str(config["prepare"]["dagger_positions"])
    )
    print("DAgger positions per round:", budget)
    print("Teacher fraction inside DAgger pass:", config["train"]["teacher_fraction"])
    print("Prepare:", shlex.join(prepare))
    print("Run/resume:", shlex.join(run))
    root = Path(config["root"])
    if (root / "ledger.sqlite").exists():
        from curriculum import schedule

        stages = schedule(root, config["train"]["teacher_fraction"])
        print("\n| stage | teacher exposures | DAgger exposures |")
        print("|---|---:|---:|")
        for stage in stages:
            print(f"| {stage['stage']} | {stage['teacher']} | {stage['dagger']} |")
        teacher = sum(s["teacher"] for s in stages)
        dagger = sum(s["dagger"] for s in stages)
        if teacher + dagger:
            print(f"Overall DAgger share: {dagger / (teacher + dagger):.2%}")


@contextlib.contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def verify_foundation(config):
    import torch

    from krasnal_think.foundations import verify_foundation as verify_registered_foundation

    foundation = config["foundation"]
    registered = verify_registered_foundation(
        foundation["checkpoint"], foundation.get("foundation_id")
    )
    checkpoint = torch.load(foundation["checkpoint"], map_location="cpu", weights_only=True)
    if "training_policy" in checkpoint or "batch_in_epoch" in checkpoint:
        raise ValueError("foundation must be a human pretrain checkpoint, not an SFT/DAgger stage")
    if checkpoint["config"]["block_size"] != foundation["block_size"]:
        raise ValueError("foundation context length differs from configuration")
    if config["device"] == "cuda":
        active = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        if any(pid.strip() != str(os.getpid()) for pid in active if pid.strip()):
            raise RuntimeError("GPU already has a compute job; start after it finishes")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; restore worker GPU access before resuming")
    return {
        "kind": "human-pretrain",
        "foundation_id": registered.get("id"),
        "training_games": registered.get("training_games"),
        "gpu_count": 1 if config["device"] == "cuda" else 0,
        "checkpoint": foundation["checkpoint"],
        "step": checkpoint.get("step"),
        "model_config": checkpoint["config"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "append", "status"])
    parser.add_argument("--config", type=Path, default=Path("configs/curriculum-clean.json"))
    parser.add_argument("--sources", type=Path, nargs="+", help="fresh source files for append")
    args = parser.parse_args()
    os.chdir(REPO)
    config = load_config(args.config)
    if args.command == "plan":
        plan(config)
        return
    root = Path(config["root"])
    if args.command == "status":
        if (root / "status.json").exists():
            print((root / "status.json").read_text())
        if (root / "report.md").exists():
            print((root / "report.md").read_text())
        return
    if args.command == "append" and not args.sources:
        parser.error("append requires --sources with new data files")
    if args.command != "append" and args.sources:
        parser.error("--sources is only for append; initial sources belong in the config")
    from curriculum import atomic_json, sha

    prepare, run = commands(config, list(map(str, args.sources)) if args.sources else None)
    with exclusive(root / "experiment.lock"), exclusive(Path("experiments/gpu-pipeline.lock")):
        pinned = root / "experiment.json"
        if not pinned.exists() and (root / "run-config.json").exists():
            raise ValueError(
                "legacy experiment exists here; choose a new root for the clean protocol"
            )
        if pinned.exists() and json.loads(pinned.read_text()) != config:
            raise ValueError("experiment config is immutable; use a new root for a new protocol")
        if args.command == "append" and not pinned.exists():
            raise ValueError("run the initial experiment before appending data")
        foundation = verify_foundation(config)
        foundation["sha256"] = sha(config["foundation"]["checkpoint"])
        saved_foundation = root / "foundation.json"
        if saved_foundation.exists() and json.loads(saved_foundation.read_text()) != foundation:
            raise ValueError("the pinned pretrain checkpoint changed")
        atomic_json(pinned, config)
        atomic_json(saved_foundation, foundation)
        subprocess.run(prepare, check=True)
        plan(config)
        subprocess.run(run, check=True)


if __name__ == "__main__":
    main()
