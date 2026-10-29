#!/usr/bin/env python3
"""Evaluate human moves with true Stockfish moves teacher-forced."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from krasnal.tokenization import reasoning as tokens
from krasnal.data.interleave import record_to_tokens
from krasnal.models.reasoning import GPT
from krasnal.training.runtime import pick_device


def _targets(
    ids: list[int], interleaved: bool
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return (human labels, logits positions, SF labels/logits positions)."""
    human, human_positions, sf = [], [], []
    if interleaved:
        for index in range(3, len(ids) - 1, 2):
            sf.append(ids[index])
            human.append(ids[index + 1])
        # In ``S W S B`` logits at 2, 4, ... predict S; logits at 3, 5, ...
        # see S and predict the following human move.
        sf_positions = list(range(2, len(ids) - 2, 2))
        human_positions = list(range(3, len(ids) - 2, 2))
    else:
        human = ids[3:-1]
        human_positions = list(range(2, len(ids) - 2))
        sf_positions = []
    return human, human_positions, sf, sf_positions


def _evaluate(args):
    device = pick_device(args.device)
    model = GPT.load(args.checkpoint, device).eval()
    interleaved = args.mode == "augmented"
    if interleaved and model.config.vocab_size != tokens.INTERLEAVED_VOCAB_SIZE:
        raise ValueError("augmented evaluation requires the distinct-Stockfish-token model")
    records = []
    with args.records.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= args.limit:
                    break
    if not records:
        raise SystemExit("no records to evaluate")

    totals = {
        "human_nll": 0.0,
        "human_top1": 0,
        "human_top5": 0,
        "human_count": 0,
        "sf_nll": 0.0,
        "sf_top1": 0,
        "sf_top5": 0,
        "sf_count": 0,
    }
    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    with torch.inference_mode():
        for offset in range(0, len(records), args.batch_size):
            chunk = records[offset : offset + args.batch_size]
            encoded = [record_to_tokens(record, interleaved)[0] for record in chunk]
            too_long = [len(ids) for ids in encoded if len(ids) > model.config.block_size]
            if too_long:
                raise ValueError(
                    "game stream exceeds model block size "
                    f"{model.config.block_size}: {max(too_long)}"
                )
            width = max(len(ids) for ids in encoded)
            x = torch.full((len(encoded), width - 1), tokens.PAD, dtype=torch.long, device=device)
            for row, ids in enumerate(encoded):
                x[row, : len(ids) - 1] = torch.tensor(ids[:-1], device=device)
            # Supplying dummy targets asks GPT for logits at every position; their
            # loss is discarded.  Padding is after EOS and cannot affect earlier
            # causal positions.
            dummy_targets = torch.zeros_like(x)
            with autocast:
                logits, _ = model(x, targets=dummy_targets)
            for row, ids in enumerate(encoded):
                human, human_positions, sf, sf_positions = _targets(ids, interleaved)
                row_logits = logits[row]
                for label, positions, name, forbidden in (
                    (human, human_positions, "human", tokens.SF_MOVE_IDS),
                    (sf, sf_positions, "sf", tokens.MOVE_IDS),
                ):
                    if not label:
                        continue
                    selected = row_logits[positions].float()
                    if model.config.vocab_size == tokens.INTERLEAVED_VOCAB_SIZE:
                        if name == "human":
                            # Match the original pretrain output vocabulary;
                            # new SF and padding rows are not human candidates.
                            selected[:, tokens.VOCAB_SIZE:] = float("-inf")
                        else:
                            selected[:, forbidden.start : forbidden.stop] = float("-inf")
                    gold = torch.tensor(label, device=device)
                    log_probs = selected.log_softmax(-1)
                    totals[f"{name}_nll"] += -float(
                        log_probs.gather(1, gold[:, None]).sum()
                    )
                    totals[f"{name}_top1"] += int((selected.argmax(-1) == gold).sum())
                    totals[f"{name}_top5"] += int(
                        (selected.topk(5).indices == gold[:, None]).any(-1).sum()
                    )
                    totals[f"{name}_count"] += len(label)

    human_count = totals["human_count"]
    sf_count = totals["sf_count"]
    result = {
        "checkpoint": str(args.checkpoint),
        "records": len(records),
        "mode": args.mode,
        "teacher_forced_stockfish": interleaved,
        "stockfish_move_namespace": "distinct" if interleaved else "not_used",
        "vocab_size": model.config.vocab_size,
        "human_moves": human_count,
        "human_nll": totals["human_nll"] / human_count,
        "human_top1": totals["human_top1"] / human_count,
        "human_top5": totals["human_top5"] / human_count,
    }
    if sf_count:
        result.update(
            {
                "sf_moves": sf_count,
                "sf_nll": totals["sf_nll"] / sf_count,
                "sf_top1": totals["sf_top1"] / sf_count,
                "sf_top5": totals["sf_top5"] / sf_count,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--mode", choices=("augmented", "human"), default="augmented")
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.limit < 1 or args.batch_size < 1:
        raise SystemExit("limit and batch-size must be positive")
    result = _evaluate(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
