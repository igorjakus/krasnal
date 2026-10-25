"""Random-initialized comparison; reuse the verified interleave primitives."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from krasnal.tokenization import reasoning as tokens
from krasnal.models.reasoning import GPT, GPTConfig
from krasnal.training.runtime import cosine_lr
from train_interleave import BlockSampler, _batch, _loss, _validation_loss, _atomic_save

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/sf-interleave-scratch-full-20261009.json"))
parser.add_argument("--smoke", action="store_true")
args_cli = parser.parse_args()
config = json.loads(args_cli.config.read_text())
data = Path(config["data"])
manifest_path = data / "dataset.manifest.json"
manifest = json.loads(manifest_path.read_text())
assert manifest["token_namespace"]["stockfish_moves_distinct_from_human"]
stats = manifest["splits"]["train"]["augmented"]
global_batch = config["batch_size"] * config["grad_accum"]
steps = math.ceil(config["epochs"] * math.ceil((stats["tokens"] - 1) / 1024) / global_batch)
out = Path(config["root"] + ("-smoke" if args_cli.smoke else ""))
if out.exists():
    raise SystemExit("Run exists: preserve it, use a fresh run or explicit recovery")
out.mkdir(parents=True)
plan = {**config, "steps": 2 if args_cli.smoke else steps,
        "train_games": stats["records"], "global_batch_size": global_batch,
        "pretrained_weights_loaded": False,
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                        ("scripts/train_sf_scratch.py", "scripts/train_interleave.py",
                         "src/krasnal/data/interleave.py", "src/krasnal/models/reasoning.py",
                         "src/krasnal/tokenization/reasoning.py")}}
(out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
torch.manual_seed(config["seed"])
torch.set_float32_matmul_precision("high")
model = GPT(GPTConfig(**config["model"])).to("cuda").train()
optimizer = model.configure_optimizers(0.1, config["lr"], (0.9, 0.95), fused=True)
stream = np.memmap(data / "train/augmented.bin", dtype=np.uint16, mode="r")
mask = np.memmap(data / "train/augmented.mask.bin", dtype=np.uint8, mode="r")
val = np.memmap(data / "val/augmented.bin", dtype=np.uint16, mode="r")
val_mask = np.memmap(data / "val/augmented.mask.bin", dtype=np.uint8, mode="r")
settings = argparse.Namespace(**config)
settings.out = out
settings.data_dir = data / "train"
settings.mode = "augmented"
settings.steps = plan["steps"]
settings.eval_batches = 2 if args_cli.smoke else 20
settings.checkpoint = None
accum = 1 if args_cli.smoke else config["grad_accum"]
sampler = BlockSampler(len(stream), 1024, config["seed"], 0, 1)
generator = np.random.default_rng(config["seed"])
start = time.time()
human_seen = 0
print("SCRATCH", model.num_params(), "params", settings.steps, "steps; no weights loaded", flush=True)
for step in range(settings.steps):
    lr = cosine_lr(step, config["warmup"], settings.steps, config["lr"], config["min_lr"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(accum):
        x, targets, weights = _batch(stream, mask, config["batch_size"], 1024, "cuda",
                                     generator, 1.0, 0.0, offsets=sampler.next(config["batch_size"]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = _loss(model, x, targets, weights) / accum
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; preserve checkpoints and stop")
        loss.backward()
        total += loss.item()
        human_seen += int(((targets >= tokens.MOVE_IDS.start) & (targets < tokens.MOVE_IDS.stop)).sum())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 100 == 0 or args_cli.smoke:
        print(f"step {step+1} loss {total:.4f} lr {lr:.2e} elapsed {time.time()-start:.0f}s", flush=True)
    if (step + 1) % config["save_every"] == 0 or args_cli.smoke:
        value = _validation_loss(model, val, val_mask, settings, "cuda")
        print(f"step {step+1} val-human-loss {value:.4f}", flush=True)
        _atomic_save(out / f"step-{step+1}.pt", model, optimizer, step+1, settings)
    if time.time() - start >= config["max_hours"] * 3600:
        break
_atomic_save(out / "model.pt", model, optimizer, step+1, settings)
summary = {"initialization": "random", "pretrained_weights_loaded": False,
           "steps": step+1, "requested_steps": settings.steps, "seconds": time.time()-start,
           "human_targets_seen": human_seen, "train_games": stats["records"],
           "parameters": model.num_params(), "global_batch_size": global_batch}
(out / "train-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
