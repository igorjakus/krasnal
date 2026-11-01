#!/usr/bin/env python3
"""Plan and launch one advisor-trust fine-tune (same recipe as the SF-full run).

    python scripts/run_advisor_trust.py configs/advisor-trust-v1.json [--dry-run]

``world_size`` GPUs via torchrun; on Adrian (one RTX 3090) pass --world-size 1 and
double grad_accum to keep the global batch, e.g. --grad-accum 16.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from krasnal.data.foundations import verify_foundation

parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path)
parser.add_argument("--world-size", type=int, default=None)
parser.add_argument("--grad-accum", type=int, default=None)
parser.add_argument("--steps", type=int, default=None, help="override (smoke tests)")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

config = json.loads(args.config.read_text())
for key in ("world_size", "grad_accum"):
    if getattr(args, key) is not None:
        config[key] = getattr(args, key)
foundation = verify_foundation(config["checkpoint"], config["foundation_id"])
data = Path(config["data"])
dataset_path = data / "dataset.manifest.json"
dataset = json.loads(dataset_path.read_text())
assert dataset["kind"] == "advisor_trust_dataset"
assert dataset["config"]["train_mixture"] == config["train_mixture"], "dataset built from another mixture"
train = dataset["splits"]["train"]
global_batch = config["world_size"] * config["batch_size"] * config["grad_accum"]
blocks = math.ceil((train["tokens"] - 1) / 1024)
steps = args.steps or math.ceil(config["epochs"] * blocks / global_batch)
out = Path(config["root"])
if (out / "plan.json").exists():
    raise SystemExit(f"{out}/plan.json exists; use a new root or explicit recovery")
code = ["scripts/train_interleave.py", "scripts/run_advisor_trust.py", "scripts/advisor_data.py",
        "scripts/advisor_eval.py", "src/krasnal/data/advisors.py", "src/krasnal/data/interleave.py",
        "src/krasnal/models/reasoning.py", "src/krasnal/tokenization/reasoning.py",
        "src/krasnal/data/foundations.py", "configs/foundations.json"]
plan = {**config, "steps": steps, "global_batch": global_batch, "train_games": train["records"],
        "games_per_advisor": train["games_per_advisor"], "foundation": foundation,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "dataset_manifest_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code}}
print(json.dumps(plan, indent=2), flush=True)
if args.dry_run:
    sys.exit(0)
out.mkdir(parents=True, exist_ok=True)
(out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc_per_node={config['world_size']}", "scripts/train_interleave.py",
           config["checkpoint"], "--foundation-id", config["foundation_id"],
           "--data-dir", str(data / "train"), "--mode", "advisor", "--out", str(out),
           "--steps", str(steps), "--sampling", config["sampling"],
           "--eval-data-dir", str(data / "val"), "--eval-batches", "20",
           "--no-compile", "--device", "cuda", "--log-every", "100", "--warmup", "500"]
for key in ("batch_size", "grad_accum", "human_weight", "other_weight", "lr",
            "min_lr", "save_every", "max_hours"):
    command.extend(["--" + key.replace("_", "-"), str(config[key])])
subprocess.run(command, check=True)
