#!/usr/bin/env python3
"""Measure move-token and auxiliary evaluation accuracy on a checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch

from krasnal.data.reasoning import get_batch, load_eval_stream, load_stream
from krasnal.models.reasoning import GPT
from krasnal.training.runtime import pick_device

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "pretrain-eval")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    model = GPT.load(args.checkpoint, device)
    model.eval()
    eval_logits = []

    def capture_eval(_module, _inputs, output):
        eval_logits.append(output)

    hook = model.eval_head.register_forward_hook(capture_eval) if model.eval_head else None
    stream = load_stream(args.data / "val.bin")
    eval_stream = load_eval_stream(args.data / "val.evals.bin") if model.eval_head else None
    generator = np.random.default_rng(0)
    move_correct = move_count = eval_correct = eval_count = eval_error = 0

    try:
        with torch.inference_mode():
            for _ in range(args.batches):
                batch = get_batch(
                    stream, args.batch_size, args.block_size, device, generator, eval_stream
                )
                x, targets = batch[:2]
                eval_targets = batch[2] if model.eval_head else None
                logits, _ = model(x, targets=targets, eval_targets=eval_targets)
                move_mask = targets != -1
                move_correct += (logits.argmax(-1)[move_mask] == targets[move_mask]).sum().item()
                move_count += move_mask.sum().item()
                if model.eval_head is None:
                    continue
                assert eval_targets is not None
                eval_mask = eval_targets != -1
                predictions = eval_logits.pop().argmax(-1)
                eval_correct += (predictions[eval_mask] == eval_targets[eval_mask]).sum().item()
                eval_error += (predictions[eval_mask] - eval_targets[eval_mask]).abs().sum().item()
                eval_count += eval_mask.sum().item()
    finally:
        if hook:
            hook.remove()

    print(f"move top-1: {move_correct / move_count:.3%} ({move_count:,} tokens)")
    if model.eval_head:
        print(f"eval bucket accuracy: {eval_correct / eval_count:.3%} ({eval_count:,} positions)")
        print(f"eval mean bucket error: {eval_error / eval_count:.2f}")


if __name__ == "__main__":
    main()
