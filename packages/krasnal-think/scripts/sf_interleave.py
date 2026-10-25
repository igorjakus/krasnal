#!/usr/bin/env python3
"""Build reusable ``S W S B ...`` game shards with Stockfish labels.

Annotation is intentionally game-based.  A shard is tied to one or more AIX
months, and the resulting manifest records the exact engine, filters, source
months, row ids and file hashes.  Splitting by disjoint months prevents an
original game from crossing train/validation/test.

Examples:

    python scripts/sf_interleave.py annotate --month 2024-01 \
        --out data/sf-interleave/raw/train-2024-01.jsonl --game-limit 100000

    python scripts/sf_interleave.py build --train-shards ... --val-shards ... \
        --test-shards ... --out-dir data/sf-interleave
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

import chess
import chess.engine
import duckdb
import numpy as np
from huggingface_hub import hf_hub_download

from krasnal_think import tokens
from krasnal_think.interleave import game_id, record_to_tokens

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "thomasd1/aix-lichess-database"
_engine: chess.engine.SimpleEngine | None = None
_nodes = 30_000


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("KRASNAL_CODE_REVISION", "unknown")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _start_worker(stockfish: str, nodes: int, threads: int):
    global _engine, _nodes
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    _engine.configure({"Threads": threads, "Hash": 16})
    _nodes = nodes


def _annotate_game(task: tuple[str, int, int, int, str]) -> dict:
    """Calculate one SF move at every prefix of one original game."""
    assert _engine is not None
    source_month, row, white_elo, black_elo, uci = task
    moves = uci.split()
    board = chess.Board()
    sf_moves = []
    for human_uci in moves:
        human_move = chess.Move.from_uci(human_uci)
        if human_move not in board.legal_moves:
            raise ValueError(f"illegal source move {human_uci} at ply {len(sf_moves)}")
        info = _engine.analyse(board, chess.engine.Limit(nodes=_nodes))
        pv = info.get("pv") or []
        if not pv or pv[0] not in board.legal_moves:
            raise ValueError(f"Stockfish returned no legal move at ply {len(sf_moves)}")
        sf_moves.append(pv[0].uci())
        board.push(human_move)
    return {
        "id": game_id(source_month, row, white_elo, black_elo, moves),
        "month": source_month,
        "row": row,
        "w": white_elo,
        "b": black_elo,
        "h": " ".join(moves),
        "s": " ".join(sf_moves),
    }


def _source_games(args):
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = true")
    con.execute("SET threads = 1")
    con.execute("INSTALL aixchess FROM community")
    con.execute("LOAD aixchess")
    for month in args.month:
        path = hf_hub_download(
            HF_REPO,
            f"high_compression/aix_lichess_{month}_high.parquet",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        query = f"""
            SELECT white_rating, black_rating, to_uci(movedata) AS uci
            FROM '{path}'
            WHERE white_rating >= {args.min_elo}
              AND black_rating >= {args.min_elo}
              AND time_initial::INTEGER + 40 * time_increment::INTEGER >= {args.min_duration}
              AND termination = 'Normal'
              AND result IN ('1-0', '0-1', '1/2-1/2')
              AND movedata IS NOT NULL
            LIMIT {args.game_limit}
        """
        cursor = con.execute(query)
        row = 0
        while rows := cursor.fetchmany(2_000):
            for white_elo, black_elo, uci in rows:
                if not uci:
                    continue
                moves = uci.split()
                if len(moves) < args.min_plies:
                    continue
                yield month, row, int(white_elo), int(black_elo), uci
                row += 1


def _read_existing(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open() as handle:
        for line in handle:
            if line.strip():
                completed.add(json.loads(line)["id"])
    return completed


def annotate(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    if args.resume:
        completed = _read_existing(partial)
        if args.out.exists():
            completed |= _read_existing(args.out)
    else:
        if args.out.exists() or partial.exists():
            raise SystemExit(f"refusing to overwrite {args.out}; use --resume or a new path")
        completed = set()

    start = time.time()
    attempted = skipped = done = 0
    with mp.Pool(
        args.workers,
        _start_worker,
        (args.stockfish, args.nodes, args.threads),
    ) as pool, partial.open("a") as output:
        tasks = (
            task
            for task in _source_games(args)
            if game_id(task[0], task[1], task[2], task[3], task[4].split()) not in completed
        )
        for record in pool.imap_unordered(_annotate_game, tasks, chunksize=1):
            attempted += 1
            if record["id"] in completed:
                skipped += 1
                continue
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            completed.add(record["id"])
            done += 1
            if done % args.log_every == 0:
                rate = done / max(time.time() - start, 1e-9)
                print(f"{args.month} {done:,} records {rate:.2f} games/s", flush=True)
        output.flush()
        os.fsync(output.fileno())
    partial.replace(args.out)

    manifest = {
        "schema_version": 2,
        "kind": "stockfish_interleave_raw",
        "source": {
            "repo": HF_REPO,
            "months": args.month,
            "game_limit_per_month": args.game_limit,
            "min_elo": args.min_elo,
            "min_duration": args.min_duration,
            "min_plies": args.min_plies,
        },
        "stockfish": {
            "path": args.stockfish,
            "binary_sha256": _sha256(Path(args.stockfish))
            if Path(args.stockfish).is_file()
            else None,
            "nodes": args.nodes,
            "threads_per_engine": args.threads,
            "workers": args.workers,
        },
        "code_revision": _git_revision(),
        "records": len(completed),
        "sha256": _sha256(args.out),
        "output": str(args.out),
        "seconds": time.time() - start,
        "resume": args.resume,
        "complete": True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


def _load_shard(path: Path):
    seen = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in seen:
                raise ValueError(f"duplicate id {record['id']} inside {path}")
            seen.add(record["id"])
            moves = record["h"].split()
            sf_moves = record["s"].split()
            if len(moves) != len(sf_moves):
                raise ValueError(f"length mismatch in {path}:{line_number}")
            yield record


def _write_split(records, split_dir: Path, interleaved: bool):
    split_dir.mkdir(parents=True, exist_ok=True)
    stem = "augmented" if interleaved else "human"
    token_partial = split_dir / f"{stem}.bin.partial"
    mask_partial = split_dir / f"{stem}.mask.bin.partial"
    offsets_partial = split_dir / f"{stem}.offsets.npy.partial"
    offsets = []
    record_count = token_count = human_targets = 0
    with token_partial.open("wb") as token_out, mask_partial.open("wb") as mask_out:
        for record in records:
            ids, mask = record_to_tokens(record, interleaved)
            start = token_count
            np.asarray(ids, dtype=np.uint16).tofile(token_out)
            mask.tofile(mask_out)
            token_count += len(ids)
            offsets.append((start, token_count))
            human_targets += int(mask.sum())
            record_count += 1
    with offsets_partial.open("wb") as handle:
        np.save(handle, np.asarray(offsets, dtype=np.uint64))
    token_path = split_dir / f"{stem}.bin"
    mask_path = split_dir / f"{stem}.mask.bin"
    offsets_path = split_dir / f"{stem}.offsets.npy"
    token_partial.replace(token_path)
    mask_partial.replace(mask_path)
    offsets_partial.replace(offsets_path)
    return {
        "records": record_count,
        "tokens": token_count,
        "human_targets": human_targets,
        "files": {
            token_path.name: {"bytes": token_path.stat().st_size, "sha256": _sha256(token_path)},
            mask_path.name: {"bytes": mask_path.stat().st_size, "sha256": _sha256(mask_path)},
            offsets_path.name: {
                "bytes": offsets_path.stat().st_size,
                "sha256": _sha256(offsets_path),
            },
        },
    }


def _write_both(records, split_dir: Path, records_path: Path | None = None):
    """Write both training streams in one pass without materializing a split."""
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "augmented": {
            "token": split_dir / "augmented.bin",
            "mask": split_dir / "augmented.mask.bin",
            "offsets": split_dir / "augmented.offsets.npy",
        },
        "human": {
            "token": split_dir / "human.bin",
            "mask": split_dir / "human.mask.bin",
            "offsets": split_dir / "human.offsets.npy",
        },
    }
    handles = {}
    offsets = {name: [] for name in paths}
    counts = {
        name: {"records": 0, "tokens": 0, "human_targets": 0} for name in paths
    }
    records_handle = None
    try:
        for name, file_paths in paths.items():
            handles[name] = {
                "token": (file_paths["token"].with_suffix(".bin.partial")).open("wb", buffering=4 * 1024 * 1024),
                "mask": (file_paths["mask"].with_suffix(".bin.partial")).open("wb", buffering=4 * 1024 * 1024),
            }
        if records_path is not None:
            records_path.parent.mkdir(parents=True, exist_ok=True)
            records_handle = records_path.with_suffix(records_path.suffix + ".partial").open("w")
        for record in records:
            if records_handle is not None:
                records_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            for name, interleaved in (("augmented", True), ("human", False)):
                ids, mask = record_to_tokens(record, interleaved)
                start = counts[name]["tokens"]
                handles[name]["token"].write(np.asarray(ids, dtype=np.uint16).tobytes())
                handles[name]["mask"].write(mask.tobytes())
                counts[name]["tokens"] += len(ids)
                counts[name]["human_targets"] += int(mask.sum())
                counts[name]["records"] += 1
                offsets[name].append((start, counts[name]["tokens"]))
            if counts["human"]["records"] % 10000 == 0:
                print(f"{split_dir}: {counts['human']['records']:,} games encoded", flush=True)
    finally:
        for pair in handles.values():
            pair["token"].close()
            pair["mask"].close()
        if records_handle is not None:
            records_handle.close()
    for name, file_paths in paths.items():
        token_partial = file_paths["token"].with_suffix(".bin.partial")
        mask_partial = file_paths["mask"].with_suffix(".bin.partial")
        offsets_partial = file_paths["offsets"].with_suffix(".npy.partial")
        with offsets_partial.open("wb") as handle:
            np.save(handle, np.asarray(offsets[name], dtype=np.uint64))
        token_partial.replace(file_paths["token"])
        mask_partial.replace(file_paths["mask"])
        offsets_partial.replace(file_paths["offsets"])
        counts[name]["files"] = {
            file_paths["token"].name: {
                "bytes": file_paths["token"].stat().st_size,
                "sha256": _sha256(file_paths["token"]),
            },
            file_paths["mask"].name: {
                "bytes": file_paths["mask"].stat().st_size,
                "sha256": _sha256(file_paths["mask"]),
            },
            file_paths["offsets"].name: {
                "bytes": file_paths["offsets"].stat().st_size,
                "sha256": _sha256(file_paths["offsets"]),
            },
        }
    if records_path is not None:
        partial = records_path.with_suffix(records_path.suffix + ".partial")
        partial.replace(records_path)
        counts["records_file"] = {
            "path": str(records_path),
            "bytes": records_path.stat().st_size,
            "sha256": _sha256(records_path),
        }
    return counts


def _records_for_split(paths, split, all_ids, all_months):
    for path in paths:
        for record in _load_shard(path):
            if record["id"] in all_ids:
                raise ValueError(f"game leakage/duplicate across splits: {record['id']}")
            all_ids.add(record["id"])
            all_months.setdefault(split, set()).add(record["month"])
            yield record


def build(args):
    split_paths = {
        "train": [Path(path) for path in args.train_shards],
        "val": [Path(path) for path in args.val_shards],
        "test": [Path(path) for path in args.test_shards],
    }
    if not split_paths["train"] or not split_paths["val"] or not split_paths["test"]:
        raise SystemExit("train, val and test each require at least one raw shard")

    all_ids: set[str] = set()
    all_months: dict[str, set[str]] = {}
    split_stats = {}
    engine_configs = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, paths in split_paths.items():
        for path in paths:
            raw_manifest = path.with_name(path.stem + ".manifest.json")
            if raw_manifest.exists():
                config = json.loads(raw_manifest.read_text()).get("stockfish", {})
                binary = Path(config.get("path", ""))
                if "binary_sha256" not in config and binary.is_file():
                    config["binary_sha256"] = _sha256(binary)
                engine_configs.append(config)
        records_path = args.out_dir / split / "records.jsonl" if split != "train" else None
        records = _records_for_split(paths, split, all_ids, all_months)
        split_outputs = _write_both(records, args.out_dir / split, records_path)
        split_stats[split] = {
            "months": sorted(all_months[split]),
            "raw_shards": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in paths
            ],
            **split_outputs,
        }

    month_sets = list(all_months.values())
    overlap = set()
    for index, left in enumerate(month_sets):
        for right in month_sets[index + 1 :]:
            overlap |= left & right
    if overlap:
        raise ValueError(f"source-month leakage across splits: {sorted(overlap)}")
    comparable_engine_configs = [
        {key: value for key, value in config.items() if key != "workers"}
        for config in engine_configs
    ]
    unique_engine_configs = {
        json.dumps(config, sort_keys=True) for config in comparable_engine_configs
    }
    if len(unique_engine_configs) > 1:
        raise ValueError("Stockfish parameters differ across raw shards")
    stockfish_config = (
        json.loads(next(iter(unique_engine_configs))) if unique_engine_configs else None
    )
    if stockfish_config is not None:
        stockfish_config["workers_per_shard"] = sorted(
            {config.get("workers") for config in engine_configs}
        )
    manifest = {
        "schema_version": 2,
        "kind": "stockfish_interleave_dataset",
        "representation": "S W S B ...",
        "token_namespace": {
            "human_vocab_size": tokens.VOCAB_SIZE,
            "stockfish_move_start": tokens.SF_MOVE_START,
            "stockfish_move_count": len(tokens.SF_MOVE_IDS),
            "interleaved_vocab_size": tokens.INTERLEAVED_VOCAB_SIZE,
            "stockfish_moves_distinct_from_human": True,
        },
        "sf_is_teacher_forced_at_eval": True,
        "split_isolation": "source months are disjoint; game ids are unique",
        "checkpoint_source_overlap": "unknown: base pretraining corpus metadata has no game ids",
        "code_revision": _git_revision(),
        "stockfish": stockfish_config,
        "total_records": len(all_ids),
        "splits": split_stats,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


def parser():
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    ann = sub.add_parser("annotate")
    ann.add_argument("--month", action="append", required=True)
    ann.add_argument("--out", type=Path, required=True)
    ann.add_argument("--manifest", type=Path, required=True)
    ann.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH", "stockfish"))
    ann.add_argument("--nodes", type=int, default=30_000)
    ann.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ann.add_argument("--threads", type=int, default=1)
    ann.add_argument("--game-limit", type=int, default=100_000)
    ann.add_argument("--min-elo", type=int, default=1100)
    ann.add_argument("--min-duration", type=int, default=180)
    ann.add_argument("--min-plies", type=int, default=20)
    ann.add_argument("--log-every", type=int, default=1_000)
    ann.add_argument("--resume", action="store_true")
    ann.set_defaults(function=annotate)

    build_parser = sub.add_parser("build")
    build_parser.add_argument("--train-shards", nargs="+", required=True)
    build_parser.add_argument("--val-shards", nargs="+", required=True)
    build_parser.add_argument("--test-shards", nargs="+", required=True)
    build_parser.add_argument("--out-dir", type=Path, required=True)
    build_parser.add_argument("--manifest", type=Path, required=True)
    build_parser.set_defaults(function=build)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
