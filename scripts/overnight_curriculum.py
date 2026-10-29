#!/usr/bin/env python3
"""Run the pinned curriculum, then a bounded game screen, without an SSH session."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/curriculum-clean.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["root"])
    subprocess.run(
        [sys.executable, "scripts/experiment.py", "run", "--config", str(args.config)],
        check=True,
    )
    # All training subprocesses have ended before games may acquire the GPU.
    active = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if active:
        raise RuntimeError("another GPU process appeared; leave games for the monitor")
    with (root / "games.log").open("a") as log:
        subprocess.run(
            [
                sys.executable,
                "scripts/curriculum_games.py",
                "--curriculum-root",
                str(root),
                "--foundation-checkpoint",
                config["foundation"]["checkpoint"],
                "--out",
                str(root / "games-short"),
                "--minutes",
                "15",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("Training complete; bounded game screen saved.", flush=True)


if __name__ == "__main__":
    main()
