#!/usr/bin/env python3
"""Stage 3: grouped policy optimization from an SFT checkpoint.

    uv run scripts/rl.py runs/sft/model.pt --annotations data/sft/annotations.jsonl
"""

import argparse
import os
import random
import time
from pathlib import Path

import chess.engine

from krasnal_think.annotate import Annotation
from krasnal_think.model import GPT
from krasnal_think.rl import update
from krasnal_think.train import cosine_lr, pick_device, save

ROOT = Path(__file__).resolve().parents[1]


def load_annotations(path: Path, limit: int) -> list[Annotation]:
    with path.open() as handle:
        return [Annotation.from_json(line) for _, line in zip(range(limit), handle, strict=False)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--annotations", type=Path,
                        default=ROOT / "data" / "sft" / "annotations.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "rl")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--efforts", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--nodes", type=int, default=5_000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--annotation-limit", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device(args.device)
    model = GPT.load(args.checkpoint, device)
    annotations = load_annotations(args.annotations, args.annotation_limit)
    optimizer = model.configure_optimizers(
        0.01, args.lr, (0.9, 0.95), fused=device.startswith("cuda")
    )
    engine = chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH_PATH", "stockfish"))
    rng = random.Random(args.seed)
    start = time.time()
    try:
        for step in range(args.steps):
            lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss, score = update(model, optimizer, annotations, engine, args, device, rng)
            if step % args.log_every == 0:
                print(f"step {step:>6} loss {loss:.4f} reward {score:.3f} lr {lr:.2e} "
                      f"{time.time() - start:.0f}s", flush=True)
    finally:
        engine.quit()
    save(args.out / "model.pt", model, args.steps)


if __name__ == "__main__":
    main()
