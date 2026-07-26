#!/usr/bin/env python3
"""Annotate sampled positions with Stockfish, one JSONL record per position.

    uv run scripts/annotate.py --positions 2_000_000 --workers 20

Analysis runs at maximum effort; lower effort levels are derived by truncation
when the SFT stream is built.
"""

import argparse
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import chess
import chess.engine
import duckdb
from huggingface_hub import hf_hub_download

from krasnal_think.annotate import Annotation, analyse

HF_REPO = "thomasd1/aix-lichess-database"
ROOT = Path(__file__).resolve().parents[1]
_engine: chess.engine.SimpleEngine | None = None
_nodes = 100_000


def _start_worker(path: str, nodes: int, threads: int):
    global _engine, _nodes
    _engine = chess.engine.SimpleEngine.popen_uci(path)
    _engine.configure({"Threads": threads})
    _nodes = nodes


def _annotate(history: list[str]):
    assert _engine is not None
    board = chess.Board()
    for uci in history:
        board.push(chess.Move.from_uci(uci))
    variations, scores = analyse(_engine, board, _nodes)
    return Annotation(history, variations, scores).to_json()


def sample_positions(months, min_elo, per_game, min_ply, limit, seed):
    """Yield move histories, sampled uniformly over plies past the opening."""
    rng = random.Random(seed)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute("INSTALL aixchess FROM community")
    con.execute("LOAD aixchess")
    yielded = 0
    for month in months:
        path = hf_hub_download(
            HF_REPO,
            f"high_compression/aix_lichess_{month}_high.parquet",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        cursor = con.execute(f"""
            SELECT to_uci(movedata) AS uci FROM '{path}'
            WHERE white_rating >= {min_elo} AND black_rating >= {min_elo}
              AND termination = 'Normal' AND result IN ('1-0', '0-1', '1/2-1/2')
        """)
        while rows := cursor.fetchmany(10_000):
            for (uci,) in rows:
                moves = uci.split() if uci else []
                if len(moves) <= min_ply + 2:
                    continue
                for ply in rng.sample(
                    range(min_ply, len(moves) - 1), min(per_game, len(moves) - 1 - min_ply)
                ):
                    yield moves[:ply]
                    yielded += 1
                    if yielded >= limit:
                        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--threads", type=int, default=1, help="engine threads per worker")
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument("--per-game", type=int, default=2)
    parser.add_argument("--min-ply", type=int, default=10)
    parser.add_argument("--min-elo", type=int, default=1800)
    parser.add_argument("--months", nargs="*", default=["2018-01", "2018-02", "2018-03"])
    parser.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH", "stockfish"))
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sft" / "annotations.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    positions = sample_positions(
        args.months, args.min_elo, args.per_game, args.min_ply, args.positions, args.seed
    )
    start, done = time.time(), 0
    with (
        mp.Pool(args.workers, _start_worker, (args.stockfish, args.nodes, args.threads)) as pool,
        args.out.open("w") as out,
    ):
        for line in pool.imap_unordered(_annotate, positions, chunksize=8):
            out.write(line + "\n")
            done += 1
            if done % 5_000 == 0:
                rate = done / (time.time() - start)
                print(
                    f"{done:,}/{args.positions:,}  {rate:.0f}/s  "
                    f"eta {(args.positions - done) / rate / 3600:.1f}h",
                    flush=True,
                )
    print(f"{done:,} positions in {time.time() - start:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
