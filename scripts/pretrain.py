#!/usr/bin/env python3
"""Stage 1: next-move pretraining on human games.

uv run scripts/pretrain.py --size medium --steps 2000 --data data/smoke

Continuation mode: resume a pretrained checkpoint with a depth-1 MTP module
(DeepSeek-V3 style, predicting t+2 given the realized t+1):

uv run scripts/pretrain.py --resume runs/pretrain/model.pt --mtp-weight 0.3 \
    --steps 40000 --lr 1e-4 --min-lr 1e-5 --data data/pretrain-eval --out runs/mtp-continue
"""

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from krasnal_think.data import get_batch, load_eval_stream, load_stream
from krasnal_think.model import GPT, GPTConfig
from krasnal_think.muon import OptimizerPair
from krasnal_think.tokens import VOCAB_SIZE
from krasnal_think.train import EMA, cosine_lr, pick_device, save

ROOT = Path(__file__).resolve().parents[1]
SIZES = {
    "small": dict(n_layer=6, n_head=6, n_embd=384),
    "medium": dict(n_layer=8, n_head=8, n_embd=512),
    "large": dict(n_layer=12, n_head=12, n_embd=768),
    "xlarge": dict(n_layer=14, n_head=12, n_embd=768),
}


@torch.no_grad()
def estimate_loss(model, stream, eval_stream, args, device) -> float:
    model.eval()
    generator = np.random.default_rng(0)
    losses = [
        model_batch(
            model,
            get_batch(stream, args.batch_size, args.block_size, device, generator, eval_stream),
            args.eval_weight,
        )[1]
        for _ in range(args.eval_batches)
    ]
    model.train()
    return torch.stack(losses).mean().item()


@torch.no_grad()
def estimate_mtp_loss(model, stream, args, device) -> float:
    from krasnal_think.data import mtp_targets

    model.eval()
    generator = np.random.default_rng(0)
    losses = []
    for _ in range(args.eval_batches):
        x, targets = get_batch(stream, args.batch_size, args.block_size, device, generator)
        losses.append(model.mtp_loss_only(x, mtp_targets(targets)))
    model.train()
    return torch.stack(losses).mean().item()


def model_batch(model, batch, eval_weight):
    x, targets = batch[:2]
    eval_targets = batch[2] if len(batch) > 2 else None
    return model(x, targets=targets, eval_targets=eval_targets, eval_weight=eval_weight)


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
    parser.add_argument("--eval-weight", type=float, default=0.25)
    parser.add_argument("--no-eval-head", action="store_true")
    parser.add_argument("--mtp-weight", type=float, default=0.0)
    parser.add_argument("--rope-values", action="store_true")
    parser.add_argument("--qk-norm", action="store_true")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--muon-lr-scale", type=float, default=2.0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        config = GPTConfig(**checkpoint["config"])
        config.mtp_depth = 1
        config.eval_classes = config.eval_classes or 29
        model = GPT(config)
        model.load_state_dict(checkpoint["model"], strict=False)
        print(
            f"resumed {args.resume} (step {checkpoint.get('step')}) + MTP module, "
            f"{model.num_params() / 1e6:.1f}M params",
            flush=True,
        )
    else:
        config = GPTConfig(
            vocab_size=VOCAB_SIZE,
            block_size=args.block_size,
            dropout=args.dropout,
            eval_classes=0 if args.no_eval_head else 29,
            mtp_depth=1 if args.mtp_weight > 0 else 0,
            rope_values=args.rope_values,
            qk_norm=args.qk_norm,
            **SIZES[args.size],
        )
        model = GPT(config)
    model = model.to(device)
    print(f"{args.size}: {model.num_params() / 1e6:.1f}M params on {device}", flush=True)

    if args.optimizer == "muon":
        optimizer = OptimizerPair(model, args.weight_decay, args.lr, args.lr)
        ema = EMA(model, decay=0.999, every=10) if args.ema else None
    else:
        optimizer = model.configure_optimizers(
            args.weight_decay, args.lr, (0.9, 0.95), fused=device.startswith("cuda")
        )
        ema = EMA(model, decay=0.999, every=10) if args.ema else None
    step_model = torch.compile(model) if args.compile and device == "cuda" else model
    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    train_stream = load_stream(args.data / "train.bin")
    val_stream = load_stream(args.data / "val.bin")
    eval_weight = 0.0 if args.no_eval_head else args.eval_weight
    train_evals = (
        load_eval_stream(args.data / "train.evals.bin")
        if (args.data / "train.evals.bin").exists() and not args.no_eval_head
        else None
    )
    val_evals = (
        load_eval_stream(args.data / "val.evals.bin")
        if (args.data / "val.evals.bin").exists() and not args.no_eval_head
        else None
    )
    generator = np.random.default_rng(args.seed)

    start = time.time()
    for step in range(args.steps):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        if isinstance(optimizer, OptimizerPair):
            # Muon's lr is transferred relative to AdamW's; both follow one schedule.
            optimizer.set_lr(lr * args.muon_lr_scale, lr)
        else:
            for group in optimizer.param_groups:
                group["lr"] = lr

        train_loss = 0.0
        for _ in range(args.grad_accum):
            batch = get_batch(
                train_stream,
                args.batch_size,
                args.block_size,
                device,
                generator,
                train_evals,
            )
            with autocast:
                x, targets = batch[:2]
                eval_targets = batch[2] if len(batch) > 2 else None
                _, loss = step_model(
                    x,
                    targets=targets,
                    eval_targets=eval_targets,
                    eval_weight=eval_weight,
                    mtp_weight=args.mtp_weight,
                )
            (loss / args.grad_accum).backward()
            train_loss += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model, step)

        if step % args.eval_every == 0 or step == args.steps - 1:
            val_loss = estimate_loss(step_model, val_stream, val_evals, args, device)
            mtp_val = (
                estimate_mtp_loss(step_model, val_stream, args, device)
                if args.mtp_weight > 0
                else float("nan")
            )
            ema_val = (
                _ema_val_loss(ema, step_model, val_stream, val_evals, args, device)
                if ema is not None
                else float("nan")
            )
            print(
                f"step {step:>7} train {train_loss:.4f} val {val_loss:.4f} "
                f"mtp {mtp_val:.4f} ema-val {ema_val:.4f} "
                f"lr {lr:.2e} {time.time() - start:.0f}s",
                flush=True,
            )
            save(args.out / "model.pt", model, step, ema)


@torch.no_grad()
def _ema_val_loss(ema, step_model, val_stream, val_evals, args, device) -> float:
    """Swap EMA weights in, measure val, swap raw weights back out."""
    raw = {k: v.detach().clone() for k, v in step_model.state_dict().items()}
    ema.copy_to(step_model)
    val_loss = estimate_loss(step_model, val_stream, val_evals, args, device)
    step_model.load_state_dict(raw)
    return val_loss


if __name__ == "__main__":
    main()
