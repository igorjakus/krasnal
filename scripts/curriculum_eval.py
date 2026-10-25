#!/usr/bin/env python3
"""Freeze Stockfish labels once; evaluate identical human validation batches."""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import chess.engine
from curriculum import atomic_json, sha

from krasnal.data.annotation import Annotation

ENGINE = None
NODES = None


def start_engine(executable, nodes):
    global ENGINE, NODES
    ENGINE = chess.engine.SimpleEngine.popen_uci(executable)
    ENGINE.configure({"Threads": 1, "Hash": 32})
    NODES = nodes


def gold_record(line):
    row = Annotation.from_json(line)
    board = row.board()
    ENGINE.configure({"Clear Hash": None})
    result = ENGINE.analyse(board, chess.engine.Limit(nodes=NODES))
    score = result["score"].pov(board.turn)
    return Annotation(
        row.history, [[m.uci() for m in result["pv"][:3]]], [(score.score(), score.mate())]
    ).to_json()


def freeze(args):
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:
        identity = engine.id
    partial = args.out.with_suffix(".partial")
    count = 0
    with args.source.open() as source, partial.open("w") as output:
        with mp.Pool(args.workers, start_engine, (args.stockfish, args.nodes)) as pool:
            for line in pool.imap(gold_record, source, chunksize=4):
                output.write(line + "\n")
                count += 1
        output.flush()
        os.fsync(output.fileno())
    partial.replace(args.out)
    atomic_json(
        args.out.with_suffix(".manifest.json"),
        {
            "sha256": sha(args.out),
            "source_sha256": sha(args.source),
            "positions": count,
            "engine": identity,
            "nodes": args.nodes,
            "hash_mb": 32,
            "threads": 1,
            "clear_hash_per_position": True,
        },
    )


def human(args):
    from types import SimpleNamespace

    import numpy as np
    import torch
    from sft import raw_format_metrics

    from krasnal.data.reasoning import get_batch, load_stream
    from krasnal.models.reasoning import GPT

    model = GPT.load(args.checkpoint, args.device).eval()
    stream = load_stream(args.source)
    generator = np.random.default_rng(1729)
    correct = count = nll = 0
    with torch.inference_mode():
        for _ in range(args.batches):
            x, y = get_batch(stream, 32, model.config.block_size, args.device, generator)
            logits, _ = model(x, targets=y)
            mask = y != -1
            correct += ((logits.argmax(-1) == y) & mask).sum().item()
            count += mask.sum().item()
            nll += torch.nn.functional.cross_entropy(
                logits.flatten(0, 1).float(), y.flatten(), ignore_index=-1, reduction="sum"
            ).item()
    format_metrics = {}
    if args.format_source:
        fmt_args = SimpleNamespace(compact=True, elo=2600, format_positions=args.format_positions)
        format_metrics = raw_format_metrics(model, args.format_source, fmt_args, args.device)
    atomic_json(
        args.out,
        {
            "human_top1": correct / count,
            "human_nll": nll / count,
            "human_targets": count,
            "human_seed": 1729,
            "human_block_size": model.config.block_size,
            "human_batches": args.batches,
            "raw_format_error": format_metrics.get("2", {}).get("error"),
            "raw_format": format_metrics,
        },
    )


def direct(args):
    """A human pretrain has no thinking grammar; measure its legal direct policy."""
    import torch

    from krasnal.models.reasoning import GPT
    from krasnal.inference.reasoning import Player

    model = GPT.load(args.checkpoint, args.device).eval()
    player = Player(model, device=args.device, effort=0, temperature=0)
    correct = count = 0
    with args.source.open() as handle, torch.inference_mode():
        for line in handle:
            row = Annotation.from_json(line)
            player.reset(row.board())
            correct += int(player.move() == row.variations[0][0])
            count += 1
    if not count:
        raise ValueError("empty evaluation set")
    atomic_json(
        args.out,
        {
            "positions": count,
            "direct_top1": correct / count,
            "thinking_top1": None,
            "candidate_recall": None,
            "selection": None,
            "selection_when_best_present": None,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    gold = sub.add_parser("gold")
    gold.add_argument("--source", type=Path, required=True)
    gold.add_argument("--out", type=Path, required=True)
    gold.add_argument("--nodes", type=int, default=30000)
    gold.add_argument("--workers", type=int, default=8)
    gold.add_argument("--stockfish", default="stockfish")
    gold.set_defaults(function=freeze)
    hum = sub.add_parser("human")
    hum.add_argument("--source", type=Path, required=True)
    hum.add_argument("--checkpoint", type=Path, required=True)
    hum.add_argument("--out", type=Path, required=True)
    hum.add_argument("--device", default="cuda")
    hum.add_argument("--batches", type=int, default=30)
    hum.add_argument("--format-source", type=Path)
    hum.add_argument("--format-positions", type=int, default=100)
    hum.set_defaults(function=human)
    baseline = sub.add_parser("direct")
    baseline.add_argument("--source", type=Path, required=True)
    baseline.add_argument("--checkpoint", type=Path, required=True)
    baseline.add_argument("--out", type=Path, required=True)
    baseline.add_argument("--device", default="cuda")
    baseline.set_defaults(function=direct)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.function(args)


if __name__ == "__main__":
    main()
