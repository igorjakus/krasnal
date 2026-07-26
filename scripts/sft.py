#!/usr/bin/env python3
"""Stage 2: fine-tune on Stockfish-annotated positions, at mixed effort levels.

uv run scripts/sft.py runs/pretrain/model.pt --annotations data/sft/annotations.jsonl
"""

import argparse
import itertools
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from krasnal_think.model import GPT
from krasnal_think.sft import batches, load
from krasnal_think.train import cosine_lr, pick_device, save

ROOT = Path(__file__).resolve().parents[1]


def stream(path: Path, args, device: str, seed: int, skip: int = 0, take: int | None = None):
    samples = itertools.islice(load(path, elo=args.elo, seed=seed), skip, None)
    if take is not None:
        samples = itertools.islice(samples, take)
    return batches(samples, args.batch_size, args.block_size, device, seed=seed)


@torch.no_grad()
def validate(model, path: Path, args, device: str) -> float:
    model.eval()
    losses = [
        model(x, targets=y)[1]
        for x, y in stream(path, args, device, seed=0, take=args.val_positions)
    ]
    model.train()
    return torch.stack(losses).mean().item() if losses else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--annotations", type=Path, default=ROOT / "data" / "sft" / "annotations.jsonl"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--elo", type=int, default=2600)
    parser.add_argument("--val-positions", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    torch.set_float32_matmul_precision("high")
    model = GPT.load(args.checkpoint, device).train()
    args.block_size = model.config.block_size
    print(f"{model.num_params() / 1e6:.1f}M params on {device}", flush=True)

    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (0.9, 0.95), fused=device.startswith("cuda")
    )
    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    # The schedule needs a horizon before the first epoch has been counted.
    total = args.epochs * sum(
        1 for _ in stream(args.annotations, args, device, seed=0, skip=args.val_positions)
    )
    print(f"{total:,} steps over {args.epochs} epochs", flush=True)

    step, start = 0, time.time()
    for epoch in range(args.epochs):
        for x, y in stream(args.annotations, args, device, seed=epoch, skip=args.val_positions):
            lr = cosine_lr(step, args.warmup, total, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with autocast:
                _, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                print(
                    f"epoch {epoch} step {step:>7} train {loss.item():.4f} lr {lr:.2e} "
                    f"{time.time() - start:.0f}s",
                    flush=True,
                )
            step += 1

        val = validate(model, args.annotations, args, device)
        print(f"epoch {epoch} val {val:.4f}", flush=True)
        save(args.out / "model.pt", model, step)


if __name__ == "__main__":
    main()
