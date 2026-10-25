#!/usr/bin/env python3
"""Short, resumable, colour-paired game screen for the curriculum checkpoints."""

import argparse
import fcntl
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import chess
import torch

from krasnal.eval.arena import Stockfish, play_game
from krasnal.models.reasoning import GPT
from krasnal.inference.reasoning import Player


OPENINGS = {
    "italian": "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "ruy_lopez": "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "sicilian": "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",
    "french": "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "queens_gambit": "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "kings_indian": "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
    "english": "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5",
    "reti": "g1f3 d7d5 d2d4 g8f6 c2c4 e7e6",
}

PLAYERS = {
    "pretrain_direct": ("runs/pretrain/model.pt", 0),
    "sft02_direct": ("experiments/curriculum-v4/round-002/sft/model.pt", 0),
    "sft02_thinking": ("experiments/curriculum-v4/round-002/sft/model.pt", 2),
    "sft08_direct": ("experiments/curriculum-v4/round-008/sft/model.pt", 0),
    "final_direct": ("experiments/curriculum-v4/round-010/dagger/model.pt", 0),
    "final_thinking": ("experiments/curriculum-v4/round-010/dagger/model.pt", 2),
}

MATCHES = [
    ("final_direct", "pretrain_direct"),
    ("final_direct", "sft02_direct"),
    ("final_direct", "sft08_direct"),
    ("final_thinking", "final_direct"),
    ("final_thinking", "sft02_thinking"),
    ("pretrain_direct", "stockfish_1600"),
    ("final_direct", "stockfish_1600"),
    ("final_thinking", "stockfish_1600"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def board_from_opening(moves: str) -> chess.Board:
    board = chess.Board()
    for uci in moves.split():
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal opening move {uci}: {moves}")
        board.push(move)
    return board


def render(root: Path, state: dict):
    lines = [
        "# Curriculum short game screen",
        "",
        f"Status: {state['status']}. Fixed openings: {len(OPENINGS)}, colour-swapped pairs: {len(OPENINGS)} per match.",
        f"Max plies: {state['config']['max_plies']}; Stockfish: Elo 1600, {state['config']['nodes']} nodes/move.",
        "Greedy model moves (temperature 0). Draws include the max-plies cutoff.",
        "",
        "| A | B | Games | W-D-L for A | A score |",
        "|---|---|---:|---:|---:|",
    ]
    for a, b in MATCHES:
        values = [
            result["a_score"]
            for key, result in state["games"].items()
            if key.startswith(f"{a}__{b}__")
        ]
        wins = sum(score == 1 for score in values)
        draws = sum(score == 0.5 for score in values)
        losses = sum(score == 0 for score in values)
        score = f"{sum(values) / len(values):.1%}" if values else "—"
        lines.append(
            f"| {a} | {b} | {len(values)}/{2 * len(OPENINGS)} | "
            f"{wins}-{draws}-{losses} | {score} |"
        )
    lines += [
        "",
        "Small fixed-opening samples are a screen, not a precise Elo estimate.",
        "Individual games and checkpoint hashes are in state.json.",
        "",
    ]
    temporary = root / "report.md.partial"
    temporary.write_text("\n".join(lines))
    os.replace(temporary, root / "report.md")


def main():
    global PLAYERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/curriculum-v4/games-short"))
    parser.add_argument("--minutes", type=float, default=40)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--nodes", type=int, default=20_000)
    parser.add_argument("--curriculum-root", type=Path, help="use this run's final checkpoint")
    parser.add_argument("--foundation-checkpoint", type=Path)
    args = parser.parse_args()
    if args.curriculum_root is not None:
        final = str(args.curriculum_root / "round-010/dagger/model.pt")
        PLAYERS = {**PLAYERS, "final_direct": (final, 0), "final_thinking": (final, 2)}
    if args.foundation_checkpoint is not None:
        # Registered runs compare against their actual foundation, not retired SFTs.
        PLAYERS = {
            "pretrain_direct": (str(args.foundation_checkpoint), 0),
            "final_direct": PLAYERS["final_direct"],
            "final_thinking": PLAYERS["final_thinking"],
        }
        global MATCHES
        MATCHES = [
            ("final_direct", "pretrain_direct"),
            ("final_thinking", "final_direct"),
            ("pretrain_direct", "stockfish_1600"),
            ("final_direct", "stockfish_1600"),
            ("final_thinking", "stockfish_1600"),
        ]
    if args.minutes <= 0 or args.max_plies < 20 or args.nodes <= 0:
        parser.error("minutes and nodes must be positive; max-plies must be at least 20")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refuse slow CPU game run")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = {path for path, _ in PLAYERS.values()}
        hashes = {path: sha256(Path(path)) for path in paths}
        config = {
            "checkpoints": {name: hashes[path] for name, (path, _) in PLAYERS.items()},
            "openings": OPENINGS,
            "matches": [list(pair) for pair in MATCHES],
            "max_plies": args.max_plies,
            "nodes": args.nodes,
            "temperature": 0,
        }
        state_path = args.out / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["config"] != config:
                raise ValueError("existing game protocol differs; choose a new --out")
        else:
            state = {"config": config, "games": {}, "status": "running"}
            atomic_json(state_path, state)
        state["status"] = "running"
        atomic_json(state_path, state)
        deadline = time.monotonic() + args.minutes * 60
        boards = {name: board_from_opening(moves) for name, moves in OPENINGS.items()}

        for a, b in MATCHES:
            if all(
                f"{a}__{b}__{opening}__{colour}" in state["games"]
                for opening in OPENINGS
                for colour in ("white", "black")
            ):
                continue
            models = {}

            def make_player(name):
                if name == "stockfish_1600":
                    return Stockfish(elo=1600, nodes=args.nodes)
                path, effort = PLAYERS[name]
                if path not in models:
                    models[path] = GPT.load(path, "cuda").eval()
                return Player(models[path], device="cuda", effort=effort, temperature=0)

            first, second = make_player(a), make_player(b)
            try:
                for opening, board in boards.items():
                    needed = [
                        colour
                        for colour in ("white", "black")
                        if f"{a}__{b}__{opening}__{colour}" not in state["games"]
                    ]
                    if not needed:
                        continue
                    if time.monotonic() >= deadline:
                        state["status"] = "time_limit"
                        atomic_json(state_path, state)
                        render(args.out, state)
                        return
                    for colour in needed:
                        key = f"{a}__{b}__{opening}__{colour}"
                        white, black = (first, second) if colour == "white" else (second, first)
                        started = time.monotonic()
                        with torch.inference_mode():
                            white_score = play_game(white, black, max_plies=args.max_plies, board=board)
                        score = white_score if colour == "white" else 1 - white_score
                        state["games"][key] = {
                            "a_score": score,
                            "seconds": round(time.monotonic() - started, 2),
                        }
                        atomic_json(state_path, state)
                        render(args.out, state)
                        print(f"{key}: {score:g} ({state['games'][key]['seconds']}s)", flush=True)
            finally:
                if isinstance(first, Stockfish):
                    first.close()
                if isinstance(second, Stockfish):
                    second.close()
                del first, second, models
                gc.collect()
                torch.cuda.empty_cache()
        state["status"] = "complete"
        atomic_json(state_path, state)
        render(args.out, state)


if __name__ == "__main__":
    main()
