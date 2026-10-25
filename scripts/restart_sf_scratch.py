"""Idempotent node-failure recovery for both scratch experiments."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sf", "star"), required=True)
    args = parser.parse_args()
    variant = "scratch" if args.mode == "sf" else "star"
    config = json.loads(Path(f"configs/sf-interleave-{variant}-full-20261009.json").read_text())
    root, data = Path(config["root"]), Path(config["data"])
    checkpoints = sorted(root.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    summary = root / "train-summary.json"
    if summary.exists() and (root / "model.pt").exists():
        if json.loads(summary.read_text())["steps"] >= 59631:
            print("Budget already complete; preserving outputs", flush=True)
            return
        checkpoint = root / "model.pt"
    else:
        checkpoint = checkpoints[-1] if checkpoints else None
    if checkpoint:
        trainer = "train_sf_star.py" if args.mode == "star" else "train_interleave.py"
        print("Resuming", args.mode, checkpoint, flush=True)
        command = [sys.executable, "scripts/" + trainer, str(checkpoint),
            "--resume-from", str(checkpoint), "--data-dir", str(data / "train"),
            "--eval-data-dir", str(data / "val"), "--eval-batches", "20",
            "--out", str(root), "--mode", "augmented", "--steps", "59631",
            "--batch-size", "2", "--grad-accum", "16", "--lr", "0.0003",
            "--min-lr", "0.00003", "--warmup", "1000", "--seed", "1234",
            "--sampling", "shuffled-blocks", "--human-weight", "1", "--other-weight", "0",
            "--save-every", "5000", "--max-hours", "40", "--no-compile", "--device", "cuda"]
    else:
        # A node died during initialization, before any atomic checkpoint.
        # Preserve its metadata/partial artifacts; restart identical random seed.
        if root.exists():
            archive = root.with_name(root.name + "-interrupted-start-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            root.rename(archive)
            print("Archived checkpoint-free interrupted start:", archive, flush=True)
        control = Path(str(root) + ".control.manifest.json")
        if control.exists():
            control.rename(Path(str(control) + ".interrupted-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
        trainer = "train_sf_star.py" if args.mode == "star" else "train_sf_scratch.py"
        command = [sys.executable, "scripts/" + trainer]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
