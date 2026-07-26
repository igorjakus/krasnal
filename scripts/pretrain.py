#!/usr/bin/env python3
"""Stage 1: next-move pretraining on human games.

uv run scripts/pretrain.py --size medium --steps 2000 --data data/smoke
"""

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from krasnal_think.data import get_batch, load_stream
from krasnal_think.model import GPT, GPTConfig
from krasnal_think.tokens import VOCAB_SIZE

ROOT = Path(__file__).resolve().parents[1]
SIZES = {
    "small": dict(n_layer=6, n_head=6, n_embd=384),
    "medium": dict(n_layer=8, n_head=8, n_embd=512),
    "large": dict(n_layer=12, n_head=12, n_embd=768),
    "xlarge": dict(n_layer=14, n_head=12, n_embd=768),
}


def learning_rate(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, stream, args, device) -> float:
    model.eval()
    generator = np.random.default_rng(0)
    losses = [
        model(*get_batch(stream, args.batch_size, args.block_size, device, generator))[1]
        for _ in range(args.eval_batches)
    ]
    model.train()
    return torch.stack(losses).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="medium", choices=SIZES)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "pretrain")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "pretrain")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    config = GPTConfig(
        vocab_size=VOCAB_SIZE, block_size=args.block_size, dropout=args.dropout, **SIZES[args.size]
    )
    model = GPT(config).to(device)
    print(f"{args.size}: {model.num_params() / 1e6:.1f}M params on {device}", flush=True)

    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (0.9, 0.95), fused=device.startswith("cuda")
    )
    step_model = torch.compile(model) if args.compile and device == "cuda" else model
    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    train_stream = load_stream(args.data / "train.bin")
    val_stream = load_stream(args.data / "val.bin")
    generator = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    start = time.time()
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args)

        train_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_stream, args.batch_size, args.block_size, device, generator)
            with autocast:
                _, loss = step_model(x, targets=y)
            (loss / args.grad_accum).backward()
            train_loss += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.eval_every == 0 or step == args.steps - 1:
            val_loss = estimate_loss(step_model, val_stream, args, device)
            print(
                f"step {step:>7} train {train_loss:.4f} val {val_loss:.4f} "
                f"lr {learning_rate(step, args):.2e} {time.time() - start:.0f}s",
                flush=True,
            )
            torch.save(
                {"model": model.state_dict(), "config": asdict(config), "step": step},
                args.out / "model.pt",
            )


if __name__ == "__main__":
    main()
