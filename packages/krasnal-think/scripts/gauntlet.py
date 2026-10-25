#!/usr/bin/env python3
"""Does thinking make it stronger? Head-to-head, plus a Stockfish ladder.

uv run scripts/gauntlet.py runs/pretrain/model.pt --games 200
"""

import argparse
import time
from pathlib import Path

import torch

from krasnal_think.arena import Stockfish, match
from krasnal_think.model import GPT
from krasnal_think.play import Player


def timed_move_cost(player: Player, moves: int = 20) -> float:
    player.reset()
    start = time.time()
    for _ in range(moves):
        player.move()
    return (time.time() - start) / moves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--efforts", type=int, nargs=2, default=[0, 4])
    parser.add_argument("--ladder", type=int, nargs="*", default=[1320, 1600, 1900, 2200])
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT.load(args.checkpoint, device)
    low, high = (
        Player(model, device=device, effort=e, temperature=args.temperature) for e in args.efforts
    )

    for name, player in zip(args.efforts, (low, high), strict=True):
        print(f"effort {name}: {timed_move_cost(player) * 1000:.0f} ms/move", flush=True)

    print(
        f"\neffort {args.efforts[1]} vs effort {args.efforts[0]}: {match(high, low, args.games)}",
        flush=True,
    )

    for elo in args.ladder:
        engine = Stockfish(elo, nodes=args.nodes)
        for effort, player in zip(args.efforts, (low, high), strict=True):
            print(
                f"effort {effort} vs Stockfish {elo}: "
                f"{match(player, engine, args.games // len(args.ladder))}",
                flush=True,
            )
        engine.close()


if __name__ == "__main__":
    main()
