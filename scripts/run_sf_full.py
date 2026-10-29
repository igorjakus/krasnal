"""Resolve an immutable full-data treatment-only plan and run torchrun."""
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from krasnal.data.foundations import verify_foundation

config_path = Path("configs/sf-interleave-100m-full-20261009.json")
config = json.loads(config_path.read_text())
foundation = verify_foundation(config["checkpoint"], config["foundation_id"])
data = Path(config["data"])
dataset_path = data / "dataset.manifest.json"
dataset = json.loads(dataset_path.read_text())
assert dataset["token_namespace"]["stockfish_moves_distinct_from_human"]
train = dataset["splits"]["train"]["augmented"]
global_batch = config["world_size"] * config["batch_size"] * config["grad_accum"]
blocks = math.ceil((train["tokens"] - 1) / 1024)
steps = math.ceil(config["epochs"] * blocks / global_batch)
out = Path(config["root"])
out.mkdir(parents=True, exist_ok=True)
if (out / "plan.json").exists():
    raise SystemExit("Plan exists; use explicit checkpoint recovery, do not overwrite")
plan = {**config, "steps": steps, "train_games": train["records"],
        "human_moves_in_dataset": train["human_targets"], "full_blocks_per_epoch": blocks,
        "foundation": foundation,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset_manifest_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest()}
code = ["scripts/train_interleave.py", "scripts/run_sf_full.py", "scripts/eval_sf_full.py",
        "scripts/evaluate_interleave.py", "src/krasnal/models/reasoning.py",
        "src/krasnal/data/interleave.py", "src/krasnal/tokenization/reasoning.py",
        "src/krasnal/data/foundations.py", "configs/foundations.json"]
plan["code_sha256"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code}
(out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
print(json.dumps(plan), flush=True)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
command = [".venv/bin/python", "-m", "torch.distributed.run", "--standalone",
           "--nproc_per_node=" + str(config["world_size"]), "scripts/train_interleave.py",
           config["checkpoint"], "--foundation-id", config["foundation_id"],
           "--data-dir", str(data / "train"), "--mode", "augmented", "--out", str(out),
           "--steps", str(steps), "--sampling", config["sampling"],
           "--eval-data-dir", str(data / "val"), "--eval-batches", "20",
           "--no-compile", "--device", "cuda", "--log-every", "100", "--warmup", "500"]
for key in ("batch_size", "grad_accum", "human_weight", "other_weight", "lr",
            "min_lr", "save_every", "max_hours"):
    command.extend(["--" + key.replace("_", "-"), str(config[key])])
subprocess.run(command, check=True)
