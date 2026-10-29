#!/usr/bin/env python3
"""Compare the learned final move with choosing the lowest generated PV delta."""

import argparse
import json
from pathlib import Path

import torch

from krasnal.tokenization import reasoning as tokens
from krasnal.data.annotation import Annotation
from krasnal.models.reasoning import GPT
from krasnal.inference.reasoning import Player


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    model = GPT.load(args.checkpoint, args.device).eval()
    player = Player(model, device=args.device, effort=2, temperature=0, distinct_pv_roots=True)
    counts = {
        "positions": 0,
        "candidate_recall": 0,
        "learned_final_top1": 0,
        "lowest_delta_top1": 0,
        "lowest_delta_tied": 0,
        "gold_among_lowest_delta_ties": 0,
    }
    with args.source.open() as handle, torch.inference_mode():
        for line in handle:
            annotation = Annotation.from_json(line)
            board = annotation.board()
            if board.is_game_over(claim_draw=True):
                continue
            player.reset(board)
            final = player.move()
            candidates = [(tokens.VOCAB[root], delta) for root, delta in player.candidates]
            if not candidates:
                continue
            gold = annotation.variations[0][0]
            lowest = min(delta for _, delta in candidates)
            tied = [root for root, delta in candidates if delta == lowest]
            counts["positions"] += 1
            counts["candidate_recall"] += int(any(root == gold for root, _ in candidates))
            counts["learned_final_top1"] += int(final == gold)
            counts["lowest_delta_top1"] += int(tied[0] == gold)
            counts["lowest_delta_tied"] += int(len(tied) > 1)
            counts["gold_among_lowest_delta_ties"] += int(gold in tied)
            if counts["positions"] == args.positions:
                break
    if counts["positions"] != args.positions:
        raise RuntimeError(f"only {counts['positions']}/{args.positions} usable positions")
    print(json.dumps({**counts, "rates": {key: value / counts["positions"] for key, value in counts.items() if key != "positions"}}, indent=2))


if __name__ == "__main__":
    main()
