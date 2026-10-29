#!/usr/bin/env python3
"""Continue the pretrained GPT on human-only or ``S W S B ...`` streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path, PosixPath

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from krasnal.data.reasoning import _CONDITIONING
from krasnal.tokenization import reasoning as tokens
from krasnal.data.interleave import role_mask_logits
from krasnal.data.foundations import verify_foundation
from krasnal.models.reasoning import GPT
from krasnal.training.runtime import cosine_lr, pick_device

ROOT = Path(__file__).resolve().parents[1]


def _atomic_save(path: Path, model, optimizer, step: int, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key.replace("_orig_mod.", ""): value for key, value in model.state_dict().items()}
    payload = {
        "model": state,
        "config": asdict(model.config),
        "step": step,
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(path)


def _batch(
    stream: np.ndarray,
    human_mask: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str,
    generator: np.random.Generator,
    human_weight: float,
    other_weight: float,
    offsets=None,
):
    max_offset = len(stream) - block_size - 1
    if max_offset < 1 or len(human_mask) != len(stream):
        raise ValueError("stream and mask are too short or have different lengths")
    if offsets is None:
        offsets = generator.integers(0, max_offset, size=batch_size)
    x = np.stack([stream[i : i + block_size] for i in offsets]).astype(np.int64)
    y = np.stack([stream[i + 1 : i + 1 + block_size] for i in offsets]).astype(np.int64)
    label_mask = np.stack(
        [human_mask[i + 1 : i + 1 + block_size] for i in offsets]
    ).astype(np.float32)
    targets = np.where(np.isin(y, _CONDITIONING), -1, y)
    weights = np.where(label_mask > 0, human_weight, other_weight)
    weights[targets == -1] = 0
    tensors = (
        torch.from_numpy(x),
        torch.from_numpy(targets),
        torch.from_numpy(weights),
    )
    if device.startswith("cuda"):
        return tuple(t.pin_memory().to(device, non_blocking=True) for t in tensors)
    return tuple(t.to(device) for t in tensors)


class BlockSampler:
    """Visit every full token block once per epoch, disjoint across DDP ranks."""
    def __init__(self, length, block_size, seed, rank, world_size, consumed=0):
        self.blocks = (length - 1 + block_size - 1) // block_size
        self.max_offset = length - block_size - 1
        self.block_size = block_size
        self.seed, self.rank, self.world = seed, rank, world_size
        self.order = None
        self.epoch = -1
        per_epoch = len(range(rank, self.blocks, world_size))
        if per_epoch == 0:
            raise ValueError("not enough blocks for all ranks")
        epoch, position = divmod(consumed, per_epoch)
        self._epoch(epoch)
        self.position = position

    def _epoch(self, epoch):
        self.epoch = epoch
        self.order = np.random.default_rng(self.seed + epoch).permutation(self.blocks)[self.rank::self.world]
        self.position = 0

    def next(self, size):
        selected = []
        while len(selected) < size:
            count = min(size - len(selected), len(self.order) - self.position)
            selected.extend(self.order[self.position:self.position + count])
            self.position += count
            if self.position == len(self.order):
                self._epoch(self.epoch + 1)
        # Include the tail using one overlapping final window rather than
        # silently dropping the last games of every epoch.
        return np.minimum(np.asarray(selected) * self.block_size, self.max_offset)


def _expand_pretrain(model: GPT) -> GPT:
    """Add separate SF rows while preserving every pretrained weight and ID."""
    if model.config.vocab_size == tokens.INTERLEAVED_VOCAB_SIZE:
        return model
    if model.config.vocab_size != tokens.VOCAB_SIZE:
        raise ValueError(f"unexpected pretrain vocab: {model.config.vocab_size}")
    expanded = GPT(replace(model.config, vocab_size=tokens.INTERLEAVED_VOCAB_SIZE))
    old_state = model.state_dict()
    new_state = expanded.state_dict()
    for key in new_state:
        if key not in ("wte.weight", "lm_head.weight"):
            new_state[key] = old_state[key]
    expanded.load_state_dict(new_state)
    with torch.no_grad():
        expanded.wte.weight[: tokens.VOCAB_SIZE].copy_(model.wte.weight)
        expanded.wte.weight[tokens.SF_MOVE_IDS.start : tokens.SF_MOVE_IDS.stop].copy_(
            model.wte.weight[tokens.MOVE_IDS.start : tokens.MOVE_IDS.stop]
        )
        # A tiny perturbation breaks exact equality without discarding the useful
        # pretrained representation of each UCI move.
        expanded.wte.weight[tokens.SF_MOVE_IDS.start : tokens.SF_MOVE_IDS.stop].add_(
            torch.randn_like(
                expanded.wte.weight[tokens.SF_MOVE_IDS.start : tokens.SF_MOVE_IDS.stop]
            ) * 1e-4
        )
    expanded.training_policy = getattr(model, "training_policy", {})
    return expanded.to(model.wte.weight.device)


def _loss(model, x, targets, weights):
    logits, _ = model(x, targets=targets)
    logits = role_mask_logits(logits, targets)
    per_token = F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), ignore_index=-1, reduction="none"
    ).view_as(targets)
    denominator = weights.sum().clamp_min(1.0)
    return (per_token * weights).sum() / denominator


def _validation_loss(model, stream, mask, args, device):
    model.eval()
    generator = np.random.default_rng(0)
    values = []
    with torch.inference_mode():
        for _ in range(args.eval_batches):
            x, targets, weights = _batch(
                stream,
                mask,
                args.batch_size,
                model.config.block_size,
                device,
                generator,
                args.human_weight,
                args.other_weight,
            )
            values.append(_loss(model, x, targets, weights))
    model.train()
    return torch.stack(values).mean().item() if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("augmented", "human"), default="augmented")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--human-weight", type=float, default=2.0)
    parser.add_argument("--other-weight", type=float, default=1.0)
    parser.add_argument("--eval-data-dir", type=Path, default=None)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=0,
                        help="Finish with a checkpoint before the overnight deadline")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--foundation-id", default=None)
    parser.add_argument("--sampling", choices=("random", "shuffled-blocks"), default="random")
    args = parser.parse_args()

    if args.steps < 1 or args.grad_accum < 1:
        raise SystemExit("steps and grad-accum must be positive")
    foundation = verify_foundation(args.checkpoint, args.foundation_id)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    dataset_manifest = args.data_dir.parent / "dataset.manifest.json"
    dataset = json.loads(dataset_manifest.read_text()) if dataset_manifest.is_file() else {}
    distinct_sf_tokens = dataset.get("token_namespace", {}).get(
        "stockfish_moves_distinct_from_human", False
    )
    model = GPT.load(args.resume_from or args.checkpoint, "cpu")
    if distinct_sf_tokens:
        model = _expand_pretrain(model)
    model = model.to(device).train()
    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (0.9, 0.95), fused=device.startswith("cuda")
    )
    start_step = 0
    if args.resume_from:
        with torch.serialization.safe_globals([PosixPath]):
            restored = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(restored["optimizer"])
        start_step = int(restored["step"])
        del restored
    step_model = torch.compile(model) if args.compile and device == "cuda" else model
    if distributed:
        step_model = DistributedDataParallel(
            step_model, device_ids=[int(os.environ["LOCAL_RANK"])],
            find_unused_parameters=bool(model.config.mtp_depth or model.config.eval_classes),
        )
    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    stem = args.mode
    train_stream = np.memmap(args.data_dir / f"{stem}.bin", dtype=np.uint16, mode="r")
    train_mask = np.memmap(args.data_dir / f"{stem}.mask.bin", dtype=np.uint8, mode="r")
    if len(train_stream) and int(train_stream.max()) >= model.config.vocab_size:
        raise ValueError("dataset token IDs exceed model vocabulary")
    val_stream = val_mask = None
    if args.eval_data_dir is not None:
        val_stream = np.memmap(args.eval_data_dir / f"{stem}.bin", dtype=np.uint16, mode="r")
        val_mask = np.memmap(args.eval_data_dir / f"{stem}.mask.bin", dtype=np.uint8, mode="r")

    if rank == 0:
        sources = [Path(__file__), ROOT / "src/krasnal/data/interleave.py",
                   ROOT / "src/krasnal/tokenization/reasoning.py", ROOT / "src/krasnal/models/reasoning.py",
                   ROOT / "src/krasnal/data/foundations.py", ROOT / "configs/foundations.json",
                   args.checkpoint, dataset_manifest]
        provenance = {"sha256": {}, "args": {k: str(v) if isinstance(v, Path) else v
                                              for k, v in vars(args).items()},
                      "world_size": world_size, "distinct_sf_tokens": distinct_sf_tokens,
                      "foundation": foundation}
        for source in sources:
            if source.is_file():
                digest = hashlib.sha256()
                with source.open("rb") as handle:
                    while chunk := handle.read(8 * 1024 * 1024):
                        digest.update(chunk)
                provenance["sha256"][str(source)] = digest.hexdigest()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "run.manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if distributed:
        dist.barrier()

    if rank == 0:
        print(
            f"{args.mode}: {model.num_params() / 1e6:.1f}M params, device={device}, "
            f"tokens={len(train_stream):,}, steps={args.steps:,}, world_size={world_size}",
            flush=True,
        )
    # Old checkpoints do not store the crop RNG. On recovery use a fresh,
    # deterministic stream instead of replaying the initial training crops.
    generator = np.random.default_rng(args.seed + rank + start_step)
    sampler = BlockSampler(len(train_stream), model.config.block_size, args.seed,
                           rank, world_size, start_step * args.batch_size * args.grad_accum) \
        if args.sampling == "shuffled-blocks" else None
    start = time.time()
    completed_steps = start_step
    human_targets_seen = 0
    for step in range(start_step, args.steps):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        human_fraction = 0.0
        for microstep in range(args.grad_accum):
            x, targets, weights = _batch(
                train_stream,
                train_mask,
                args.batch_size,
                model.config.block_size,
                device,
                generator,
                args.human_weight,
                args.other_weight,
                offsets=sampler.next(args.batch_size) if sampler else None,
            )
            sync = step_model.no_sync() if distributed and microstep + 1 < args.grad_accum else nullcontext()
            with sync:
                with autocast:
                    loss = _loss(step_model, x, targets, weights) / args.grad_accum
                loss.backward()
            total_loss += loss.item()
            human_fraction += float((weights > args.other_weight).float().mean().item())
            human_targets_seen += int(((targets >= tokens.MOVE_IDS.start) &
                                       (targets < tokens.MOVE_IDS.stop)).sum().item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        completed_steps = step + 1

        if rank == 0 and (step % args.log_every == 0 or step == args.steps - 1):
            print(
                f"step {step + 1:>7} loss {total_loss:.4f} "
                f"human-label-frac {human_fraction / args.grad_accum:.3f} "
                f"lr {lr:.2e} elapsed {time.time() - start:.0f}s",
                flush=True,
            )
        if args.eval_data_dir is not None and (step + 1) % args.save_every == 0:
            val_loss = _validation_loss(model, val_stream, val_mask, args, device)
            if rank == 0:
                print(f"step {step + 1:>7} val-loss {val_loss:.4f}", flush=True)
        if (step + 1) % args.save_every == 0:
            if rank == 0:
                _atomic_save(args.out / f"step-{step + 1}.pt", model, optimizer, step + 1, args)
            if distributed:
                dist.barrier()
        stop = torch.tensor(int(args.max_hours > 0 and time.time() - start >= args.max_hours * 3600), device=device)
        if distributed:
            dist.broadcast(stop, src=0)
        if stop.item():
            break

    exposure = torch.tensor(human_targets_seen, dtype=torch.int64, device=device)
    if distributed:
        dist.all_reduce(exposure)
    if rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return
    _atomic_save(args.out / "model.pt", model, optimizer, completed_steps, args)
    summary = {
        "mode": args.mode,
        "steps": completed_steps,
        "requested_steps": args.steps,
        "world_size": world_size,
        "global_batch_size": args.batch_size * args.grad_accum * world_size,
        "human_targets_seen": int(exposure.item()),
        "resume_start_step": start_step,
        "human_targets_seen_scope": "this invocation",
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "human_weight": args.human_weight,
        "other_weight": args.other_weight,
        "sampling": args.sampling,
        "foundation": foundation,
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "distinct_sf_tokens": distinct_sf_tokens,
        "vocab_size": model.config.vocab_size,
        "seconds": time.time() - start,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
