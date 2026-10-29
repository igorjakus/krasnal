#!/usr/bin/env python3
"""Stage 2: fine-tune on Stockfish-annotated positions, at mixed effort levels.

uv run scripts/sft.py runs/pretrain/model.pt --annotations data/sft/annotations.jsonl
"""

import argparse
import itertools
import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from krasnal.tokenization import reasoning as tokens
from krasnal.data.annotation import Annotation
from krasnal.data.reasoning import get_batch, load_stream
from krasnal.models.reasoning import GPT, KVCache
from krasnal.training.sft import batches, decision_weights, load, load_onpolicy, weighted_mix
from krasnal.training.runtime import cosine_lr, pick_device

ROOT = Path(__file__).resolve().parents[1]


def save_training(path, model, optimizer, step, epoch, batch_in_epoch, total_steps):
    """Atomically save enough state to resume from the last completed batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key.replace("_orig_mod.", ""): value for key, value in model.state_dict().items()}
    payload = {
        "model": state,
        "config": asdict(model.config),
        "step": step,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "total_steps": total_steps,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "training_policy": getattr(model, "training_policy", {}),
    }
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(path)


def stream(path: Path, args, device: str, seed: int, skip: int = 0, take: int | None = None):
    samples = itertools.islice(
        load(path, efforts=args.teacher_efforts, elo=args.elo, seed=seed, compact=args.compact),
        skip,
        None,
    )
    if take is not None:
        samples = itertools.islice(samples, take)
    return batches(samples, args.batch_size, args.block_size, device, seed=seed)


def teacher_samples(paths: list[Path], args, seed: int):
    """Chain shards without materializing a duplicated merged corpus."""
    for index, path in enumerate(paths):
        samples = load(
            path,
            efforts=args.teacher_efforts,
            elo=args.elo,
            seed=seed + index,
            compact=args.compact,
        )
        if index == 0 and args.validation_source is None:
            samples = itertools.islice(samples, args.val_positions, None)
        yield from samples


def training_stream(args, device: str, seed: int):
    teacher = teacher_samples(args.annotations, args, seed)
    if args.teacher_positions:
        teacher = itertools.islice(teacher, args.teacher_positions)
    if args.dagger_annotations is None:
        samples = teacher
    else:
        dagger = (
            load_onpolicy(args.dagger_annotations, elo=args.elo)
            if args.dagger_objective == "onpolicy-selection"
            else load(
                args.dagger_annotations,
                efforts=(2,) if args.compact else (1, 2, 3, 4),
                elo=args.elo,
                seed=seed + 10_000,
                compact=args.compact,
            )
        )
        samples = weighted_mix(
            teacher,
            dagger,
            teacher_fraction=args.teacher_fraction,
            seed=seed + 20_000,
        )
    return batches(samples, args.batch_size, args.block_size, device, seed=seed)


def sft_losses(model, x, targets, move_weight: float):
    """Balance the one decision token against the whole generated trace."""
    present = targets != -1
    positions = torch.arange(targets.size(1), device=targets.device).expand_as(targets)
    final = positions.masked_fill(~present, -1).max(dim=1).values
    rows = torch.arange(targets.size(0), device=targets.device)
    policy = getattr(model, "training_policy", {})
    if policy.get("compact"):
        # Project only supervised tokens, not the often much longer game history.
        logits, _ = model(x, targets=targets, supervised_only=True)
        flat_loss = F.cross_entropy(logits.float(), targets[present], reduction="none")
        losses = torch.zeros_like(targets, dtype=flat_loss.dtype).masked_scatter(present, flat_loss)
        weights = decision_weights(
            targets, policy["pv_root_weight"], policy["pv_decay"], policy["structure_weight"]
        )
        objective = ((losses * weights).sum(1) / weights.sum(1).clamp(min=1e-9)).mean()
        move_loss = losses[rows, final].mean()
        thinking_mask = present.clone()
        thinking_mask[rows, final] = False
        thinking_loss = losses[thinking_mask].mean()
        return objective, move_loss, thinking_loss
    logits, token_loss = model(x, targets=targets)
    move_loss = F.cross_entropy(logits[rows, final], targets[rows, final])
    thinking_targets = targets.clone()
    thinking_targets[rows, final] = -1
    if (thinking_targets != -1).any():
        thinking_loss = F.cross_entropy(
            logits.flatten(0, 1), thinking_targets.flatten(), ignore_index=-1
        )
    else:
        thinking_loss = torch.full((), float("nan"), device=targets.device)
    objective = (token_loss + move_weight * move_loss) / (1 + move_weight)
    return objective, move_loss, thinking_loss


@torch.no_grad()
def validate(model, path: Path, args, device: str) -> tuple[float, float, float]:
    model.eval()
    losses = [
        sft_losses(model, x, y, args.move_loss_weight)
        for x, y in stream(path, args, device, seed=0, take=args.val_positions)
    ]
    model.train()
    if not losses:
        return (float("nan"),) * 3
    return tuple(torch.stack([row[i] for row in losses]).nanmean().item() for i in range(3))


@torch.no_grad()
def pretrain_metrics(model, args, device: str, generator: np.random.Generator):
    """Human-move metrics on the unchanged pretrain validation stream."""
    model.eval()
    stream = load_stream(args.pretrain_val)
    eval_stream = (
        load_stream_eval(args.pretrain_val.parent / "val.evals.bin") if model.eval_head else None
    )
    correct = count = 0
    losses = []
    with torch.inference_mode():
        for _ in range(args.pretrain_batches):
            batch = get_batch(
                stream, args.batch_size, args.block_size, device, generator, eval_stream
            )
            x, targets = batch[:2]
            logits, loss = model(x, targets=targets)
            losses.append(loss)
            mask = targets != -1
            correct += (logits.argmax(-1)[mask] == targets[mask]).sum().item()
            count += mask.sum().item()
    model.train()
    return (
        correct / count if count else float("nan"),
        torch.stack(losses).mean().item() if losses else float("nan"),
    )


@torch.no_grad()
def raw_format_metrics(model, path: Path, args, device: str):
    """Greedy, unmasked rollouts: does the model itself know the grammar?"""
    model.eval()
    annotations = []
    with path.open() as handle:
        for line in itertools.islice(handle, args.format_positions):
            annotations.append(Annotation.from_json(line))
    totals = {
        e: {
            "n": 0,
            "error": 0,
            "opening": 0,
            "closing": 0,
            "budget": 0,
            "token_order": 0,
            "evaluation": 0,
        }
        for e in ((2,) if args.compact else range(1, 5))
    }
    for annotation in annotations:
        prefix = [tokens.BOS, tokens.elo_id(args.elo), tokens.elo_id(args.elo)]
        prefix += [tokens.move_id(move) for move in annotation.history]
        for effort in totals:
            cache = KVCache(model.config, 1, device, torch.float32)
            ids = prefix[-(model.config.block_size - tokens.EFFORT_BUDGET[effort] - 8) :]
            idx = torch.tensor([[*ids, tokens.effort_id(effort)]], device=device)
            logits, _ = model(idx, cache=cache)
            generated = []
            for _ in range(tokens.EFFORT_BUDGET[effort] + 3):
                token = int(logits[0, -1].argmax())
                generated.append(token)
                logits, _ = model(torch.tensor([[token]], device=device), cache=cache)
                if token == tokens.THINK_END:
                    generated.append(int(logits[0, -1].argmax()))
                    break
            m = totals[effort]
            m["n"] += 1
            opening = not generated or generated[0] != tokens.THINK
            closing = tokens.THINK_END not in generated
            close = generated.index(tokens.THINK_END) if not closing else len(generated)
            content = generated[1:close] if not opening else generated[:close]
            budget = len(content) > tokens.EFFORT_BUDGET[effort]
            order = evaluation = False
            variations = 0
            i = 0
            while i < len(content):
                if content[i] != tokens.PV:
                    order = True
                    break
                i += 1
                variations += 1
                moves = 0
                while i < len(content) and content[i] in tokens.MOVE_IDS:
                    moves += 1
                    i += 1
                if not moves:
                    order = True
                if args.compact and (moves > 3 or variations > 3):
                    order = True
                if i >= len(content) or content[i] not in DELTA_IDS:
                    evaluation = True
                    break
                if args.compact and content[i] not in {
                    tokens.ID[f"<delta:{bucket}>"] for bucket in tokens.COMPACT_DELTA_BUCKETS
                }:
                    evaluation = True
                i += 1
                if i >= len(content) or content[i] != tokens.PV_END:
                    order = True
                    break
                i += 1
            if args.compact and not variations:
                order = True
            for key, value in (
                ("opening", opening),
                ("closing", closing),
                ("budget", budget),
                ("token_order", order),
                ("evaluation", evaluation),
            ):
                m[key] += int(value)
            m["error"] += int(opening or closing or budget or order or evaluation)
    model.train()
    return {
        str(e): {
            key: value / row["n"] if row["n"] else None for key, value in row.items() if key != "n"
        }
        for e, row in totals.items()
    }


DELTA_IDS = set(i for i, name in enumerate(tokens.VOCAB) if name.startswith("<delta:"))


def load_stream_eval(path: Path) -> np.ndarray:
    return np.memmap(path, dtype=np.uint8, mode="r")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--annotations",
        type=Path,
        nargs="+",
        default=[ROOT / "data" / "sft" / "annotations.jsonl"],
    )
    parser.add_argument("--dagger-annotations", type=Path, default=None)
    parser.add_argument(
        "--dagger-objective",
        choices=["teacher-trace", "onpolicy-selection"],
        default="teacher-trace",
    )
    parser.add_argument(
        "--teacher-fraction",
        type=float,
        default=0.75,
        help="expected teacher fraction while both finite streams have records",
    )
    parser.add_argument(
        "--teacher-efforts",
        default="0,0,0,0,1,2,3,4",
        help="comma-separated effort sampling weights for teacher data; use 0 for direct-only",
    )
    parser.add_argument(
        "--teacher-positions",
        type=int,
        default=0,
        help="cap teacher records per epoch; 0 means all (use 3x DAgger for a true 75/25 pass)",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs all requested epochs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="restore optimizer, epoch and batch position from the checkpoint",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--move-loss-weight",
        type=float,
        default=1.0,
        help="extra weight for the final played move relative to the full trace loss",
    )
    parser.add_argument("--elo", type=int, default=2600)
    parser.add_argument("--val-positions", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--format-positions", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--pretrain-val", type=Path, default=ROOT / "data" / "pretrain-eval" / "val.bin"
    )
    parser.add_argument("--pretrain-batches", type=int, default=150)
    parser.add_argument(
        "--pretrain-check-every",
        type=int,
        default=0,
        help="optional extra human-move guardrail checks between full evaluations",
    )
    parser.add_argument(
        "--min-pretrain-top1",
        type=float,
        default=0.40,
        help="abort SFT if the pretrain move top-1 falls below this",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--validation-source", type=Path)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--pv-root-weight", type=float, default=0.5)
    parser.add_argument("--pv-decay", type=float, default=0.5)
    parser.add_argument("--structure-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--continue-optimizer", action="store_true")
    args = parser.parse_args()
    if not (0 < args.pv_decay <= 1 and args.pv_root_weight > 0 and args.structure_weight > 0):
        parser.error("PV decay must be in (0,1], weights positive")
    torch.manual_seed(args.seed)
    try:
        args.teacher_efforts = tuple(int(item) for item in args.teacher_efforts.split(","))
    except ValueError as exc:
        raise SystemExit("--teacher-efforts must be comma-separated integers") from exc
    if not args.teacher_efforts or any(item < 0 or item > 4 for item in args.teacher_efforts):
        raise SystemExit("--teacher-efforts values must be in 0..4")

    device = pick_device(args.device)
    torch.set_float32_matmul_precision("high")
    model = GPT.load(args.checkpoint, device).train()
    model.training_policy = {
        key: getattr(args, key)
        for key in ("compact", "pv_root_weight", "pv_decay", "structure_weight", "dagger_objective")
    }
    args.block_size = model.config.block_size
    print(f"{model.num_params() / 1e6:.1f}M params on {device}", flush=True)

    if args.eval_only:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        val, val_move, val_thinking = validate(model, args.annotations[0], args, device)
        human_top1, human_nll = pretrain_metrics(model, args, device, np.random.default_rng(0))
        metrics = {
            "step": checkpoint.get("step", -1),
            "final": True,
            "sft_nll": val,
            "move_nll": val_move,
            "thinking_nll": val_thinking,
            "human_top1_acc": human_top1,
            "human_nll": human_nll,
            "format": raw_format_metrics(model, args.annotations[0], args, device),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(f"eval {json.dumps(metrics, sort_keys=True)}", flush=True)
        return

    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (0.9, 0.95), fused=device.startswith("cuda")
    )
    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    generator = np.random.default_rng(0)
    # The schedule needs a horizon before the first epoch has been counted.
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if "optimizer" not in resume_state:
            raise SystemExit("--resume requires a training-state checkpoint")
        if resume_state.get("training_policy", {}) != model.training_policy:
            raise SystemExit("resume training policy differs from checkpoint")
        optimizer.load_state_dict(resume_state["optimizer"])
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        if resume_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in resume_state["cuda_rng_state"]])
    elif args.continue_optimizer:
        previous = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if "optimizer" in previous:
            optimizer.load_state_dict(previous["optimizer"])
    total = (
        resume_state["total_steps"]
        if resume_state is not None
        else (
            args.max_steps
            if args.max_steps
            else args.epochs * sum(1 for _ in training_stream(args, "cpu", seed=args.seed))
        )
    )
    print(f"{total:,} steps over {args.epochs} epochs", flush=True)

    step = resume_state["step"] if resume_state is not None else 0
    start_epoch = resume_state.get("epoch", 0) if resume_state is not None else 0
    resume_batch = resume_state.get("batch_in_epoch", 0) if resume_state is not None else 0
    start = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_stream = training_stream(args, device, seed=args.seed + epoch)
        for batch_index, (x, y) in enumerate(epoch_stream):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            lr = cosine_lr(step, args.warmup, total, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with autocast:
                loss, move_loss, thinking_loss = sft_losses(model, x, y, args.move_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                print(
                    f"epoch {epoch} step {step:>7} loss {loss.item():.4f} "
                    f"move {move_loss.item():.4f} think {thinking_loss.item():.4f} "
                    f"lr {lr:.2e} "
                    f"{time.time() - start:.0f}s",
                    flush=True,
                )
            step += 1
            if args.save_every and step % args.save_every == 0:
                save_training(
                    args.out / "latest.pt", model, optimizer, step, epoch, batch_index + 1, total
                )

            if args.eval_every and step % args.eval_every == 0:
                val, val_move, val_thinking = validate(model, args.annotations[0], args, device)
                human_top1, human_nll = pretrain_metrics(model, args, device, generator)
                fmt = raw_format_metrics(model, args.annotations[0], args, device)
                metrics = {
                    "step": step,
                    "sft_nll": val,
                    "move_nll": val_move,
                    "thinking_nll": val_thinking,
                    "human_top1_acc": human_top1,
                    "human_nll": human_nll,
                    "format": fmt,
                }
                args.out.mkdir(parents=True, exist_ok=True)
                with (args.out / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                print(f"eval {json.dumps(metrics, sort_keys=True)}", flush=True)
                save_training(
                    args.out / f"step-{step}.pt",
                    model,
                    optimizer,
                    step,
                    epoch,
                    batch_index + 1,
                    total,
                )
                if human_top1 < args.min_pretrain_top1:
                    print(
                        f"guardrail tripped: human top-1 {human_top1:.3%} below "
                        + f"{args.min_pretrain_top1:.0%}; stopping",
                        flush=True,
                    )
                    save_training(
                        args.out / "model.pt",
                        model,
                        optimizer,
                        step,
                        epoch,
                        batch_index + 1,
                        total,
                    )
                    return

            if args.pretrain_check_every and step % args.pretrain_check_every == 0:
                top1, _ = pretrain_metrics(model, args, device, generator)
                print(
                    f"epoch {epoch} step {step:>7} human-top1 {top1:.3%} "
                    f"{time.time() - start:.0f}s",
                    flush=True,
                )
                if top1 < args.min_pretrain_top1:
                    print(
                        f"guardrail tripped: pretrain top-1 {top1:.3%} below "
                        f"{args.min_pretrain_top1:.0%}; stopping and keeping this checkpoint",
                        flush=True,
                    )
                    save_training(
                        args.out / "model.pt",
                        model,
                        optimizer,
                        step,
                        epoch,
                        batch_index + 1,
                        total,
                    )
                    return

            if args.max_steps and step >= args.max_steps:
                save_training(
                    args.out / "model.pt",
                    model,
                    optimizer,
                    step,
                    epoch,
                    batch_index + 1,
                    total,
                )
                return

        val, val_move, val_thinking = validate(
            model, args.validation_source or args.annotations[0], args, device
        )
        print(
            f"epoch {epoch} val {val:.4f} move {val_move:.4f} think {val_thinking:.4f}",
            flush=True,
        )
        save_training(args.out / "model.pt", model, optimizer, step, epoch + 1, 0, total)


if __name__ == "__main__":
    main()
