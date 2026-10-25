#!/usr/bin/env python3
"""Stream Lichess games from the Aix parquet dumps into flat uint16 token shards.

uv run scripts/download_pretrain.py --target-games 1000        # smoke test
uv run scripts/download_pretrain.py --target-games 100_000_000
"""

import argparse
import os
import time
from pathlib import Path

import duckdb
from huggingface_hub import hf_hub_download

from krasnal_think.data import EvalShardWriter, ShardWriter, encode_evals, encode_game

HF_REPO = "thomasd1/aix-lichess-database"
SKIP_MONTHS = {"2016-12", "2019-12", "2020-07", "2020-12", "2021-01"}
ROOT = Path(__file__).resolve().parents[1]

QUERY = """
    SELECT white_rating, black_rating, to_uci(movedata) AS uci, evals
    FROM '{path}'
    WHERE white_rating >= {min_elo} AND black_rating >= {min_elo}
      AND time_initial::INTEGER + 40 * time_increment::INTEGER >= {min_duration}
      AND termination = 'Normal'
      AND result IN ('1-0', '0-1', '1/2-1/2')
      AND evals IS NOT NULL
    LIMIT {limit}
"""


def expand_months(start: str, end: str) -> list[str]:
    year, month = (int(part) for part in start.split("-"))
    end_year, end_month = (int(part) for part in end.split("-"))
    out = []
    while (year, month) <= (end_year, end_month):
        out.append(f"{year}-{month:02d}")
        month, year = (1, year + 1) if month == 12 else (month + 1, year)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-games", type=int, default=100_000_000)
    parser.add_argument("--min-elo", type=int, default=1100)
    parser.add_argument("--min-duration", type=int, default=180, help="excludes bullet")
    parser.add_argument("--start-month", default="2017-04")
    parser.add_argument("--end-month", default="2026-03")
    parser.add_argument("--val-games", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "pretrain")
    parser.add_argument("--memory-limit", default="8GB")
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("INSTALL aixchess FROM community")
    con.execute("LOAD aixchess")

    val = ShardWriter(args.out / "val.bin")
    train = ShardWriter(args.out / "train.bin")
    val_evals = EvalShardWriter(args.out / "val.evals.bin")
    train_evals = EvalShardWriter(args.out / "train.evals.bin")
    games, start = 0, time.time()

    for month in expand_months(args.start_month, args.end_month):
        if games >= args.target_games + args.val_games:
            break
        if month in SKIP_MONTHS:
            continue
        try:
            path = hf_hub_download(
                HF_REPO,
                f"high_compression/aix_lichess_{month}_high.parquet",
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as exc:
            print(f"[{month}] skipped: {exc}", flush=True)
            continue

        remaining = args.target_games + args.val_games - games
        cursor = con.execute(
            QUERY.format(
                path=path,
                min_elo=args.min_elo,
                min_duration=args.min_duration,
                limit=remaining,
            )
        )
        month_games = 0
        while rows := cursor.fetchmany(50_000):
            for white_elo, black_elo, uci, evals in rows:
                if not uci:
                    continue
                writer = val if games < args.val_games else train
                eval_writer = val_evals if games < args.val_games else train_evals
                moves = uci.split()
                ids = encode_game(uci, white_elo, black_elo)
                labels = encode_evals(evals, len(moves))
                writer.write(ids)
                eval_writer.write(labels)
                games += 1
                month_games += 1
        print(f"[{month}] +{month_games:,} games (total {games:,})", flush=True)

    val.close()
    train.close()
    val_evals.close()
    train_evals.close()
    print(
        f"{games:,} games, {train.count:,} train + {val.count:,} val tokens "
        f"in {time.time() - start:.0f}s -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
