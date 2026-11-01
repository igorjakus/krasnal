#!/usr/bin/env python3
"""Data for the advisor-trust experiment (see docs/advisor-trust.md).

``annotate`` adds Stockfish advice at extra budgets to existing interleave
records (which already hold ``h`` human moves and ``s`` SF-30k moves):

    # evaluation split: every listed advisor for every game
    python scripts/advisor_data.py annotate --mode all \
        --in data/sf-interleave-v1/raw/test/2024-06.jsonl \
        --out data/advisor-trust-v1/test/2024-06.jsonl \
        --advisor sfd1 --advisor sfd2 --advisor sfd4 --advisor sfd8

    # training split: draw one advisor per game, run SF only where needed
    python scripts/advisor_data.py annotate --mode assigned \
        --config configs/advisor-trust-v1.json \
        --in data/sf-interleave-v1/raw/train/2024-01.jsonl \
        --out data/advisor-trust-v1/raw/train/2024-01.jsonl

``build`` writes ``advisor.bin``/``advisor.mask.bin``/``advisor.offsets.npy``
streams that ``scripts/train_interleave.py --mode advisor`` consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import numpy as np

from krasnal.data.advisors import (
    advice_for,
    assign_advisor,
    encode_advised_game,
    sf_limit,
    validate_advisor,
)

ROOT = Path(__file__).resolve().parents[1]
_engine: chess.engine.SimpleEngine | None = None
_job: dict = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("KRASNAL_CODE_REVISION", "unknown")


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    config = json.loads(path.read_text())
    for name in config["train_mixture"]:
        validate_advisor(name)
    if "oracle100" in config["train_mixture"]:
        raise ValueError("oracle100 is pure label leakage; evaluation only")
    return config


# --------------------------------------------------------------------- annotate

def _init_worker(job: dict):
    global _engine, _job
    _job = job
    if job["needs_engine"]:
        _engine = chess.engine.SimpleEngine.popen_uci(job["stockfish"])
        _engine.configure({"Threads": job["threads"], "Hash": 16})


def _sf_moves(moves: list[str], limit: dict) -> str:
    assert _engine is not None
    board = chess.Board()
    advice = []
    for move in moves:
        info = _engine.analyse(board, chess.engine.Limit(**limit))
        pv = info.get("pv") or []
        if not pv or pv[0] not in board.legal_moves:
            raise ValueError(f"Stockfish returned no legal move at ply {len(advice)}")
        advice.append(pv[0].uci())
        board.push_uci(move)
    return " ".join(advice)


def _annotate(line: str) -> str:
    record = json.loads(line)
    if len(record["h"].split()) != len(record["s"].split()):
        raise ValueError(f"length mismatch in {record['id']}")
    if _job["mode"] == "assigned":
        record["adv"] = assign_advisor(record["id"], _job["mixture"], _job["seed"])
        wanted = [record["adv"]] if sf_limit(record["adv"]) else []
    else:
        wanted = _job["advisors"]
    annotations = record.setdefault("a", {})
    for name in wanted:
        if name not in annotations:
            annotations[name] = _sf_moves(record["h"].split(), sf_limit(name))
    if not annotations:
        del record["a"]
    return json.dumps(record, separators=(",", ":"))


def _done_ids(*paths: Path) -> set[str]:
    """Ids already written; a torn last line from a killed job is dropped in place."""
    done = set()
    for path in paths:
        if not path.exists():
            continue
        good = []
        with path.open() as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["id"])
                    good.append(line)
                except (json.JSONDecodeError, KeyError):
                    pass
        if path.suffix == ".partial" and sum(map(len, good)) != path.stat().st_size:
            path.write_text("".join(good))
    return done


def annotate(args):
    config = _load_config(args.config)
    advisors = [validate_advisor(name) for name in args.advisor]
    if args.mode == "all":
        if not advisors or any(not sf_limit(name) for name in advisors):
            raise SystemExit("--mode all needs one or more sfd<N>/sfn<N> --advisor")
        needs_engine = True
    else:
        if not config:
            raise SystemExit("--mode assigned needs --config")
        needs_engine = any(sf_limit(name) for name in config["train_mixture"])
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    done = _done_ids(partial)
    job = {"mode": args.mode, "advisors": advisors, "stockfish": args.stockfish,
           "threads": args.threads, "needs_engine": needs_engine,
           "mixture": config.get("train_mixture"), "seed": config.get("seed", 0)}

    def pending():
        with args.input.open() as handle:
            for index, line in enumerate(handle):
                if args.limit and index >= args.limit:
                    return
                if line.strip() and json.loads(line)["id"] not in done:
                    yield line

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start, written = time.time(), 0
    with mp.Pool(args.workers, _init_worker, (job,)) as pool, partial.open("a") as out:
        for row in pool.imap_unordered(_annotate, pending(), chunksize=4):
            out.write(row + "\n")
            written += 1
            if written % args.log_every == 0:
                out.flush()
                print(f"{args.input.name}: {written:,} new records, "
                      f"{written / (time.time() - start):.1f} games/s", flush=True)
        out.flush()
        os.fsync(out.fileno())
    partial.replace(args.out)
    manifest = {
        "kind": "advisor_trust_annotations",
        "mode": args.mode,
        "input": {"path": str(args.input), "sha256": _sha256(args.input)},
        "output": {"path": str(args.out), "sha256": _sha256(args.out),
                   "records": len(_done_ids(args.out))},
        "advisors": advisors,
        "config": config,
        "config_sha256": _sha256(args.config) if args.config else None,
        "stockfish": {"path": args.stockfish, "threads_per_engine": args.threads, "hash_mb": 16,
                      "binary_sha256": _sha256(Path(args.stockfish))
                      if Path(args.stockfish).is_file() else None} if needs_engine else None,
        "code_revision": _git_revision(),
        "limit": args.limit,
        "seconds_this_invocation": time.time() - start,
    }
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


# ------------------------------------------------------------------------ build

def _encode(line: str):
    record = json.loads(line)
    advisor = assign_advisor(record["id"], _job["mixture"], _job["seed"])
    if record.get("adv", advisor) != advisor:
        raise ValueError(f"{record['id']}: annotated with {record['adv']}, config says {advisor}")
    human = record["h"].split()
    advice = advice_for(record, advisor, _job["seed"])
    ids, mask = encode_advised_game(human, advice, int(record["w"]), int(record["b"]))
    agree = sum(hint == move for hint, move in zip(advice, human))
    return record["id"], record.get("month"), advisor, ids, mask, agree, len(human)


def _write_split(paths: list[Path], split_dir: Path, job: dict, workers: int, seen: set):
    split_dir.mkdir(parents=True, exist_ok=True)
    files = {kind: split_dir / f"advisor.{kind}" for kind in ("bin", "mask.bin", "offsets.npy")}
    partial = {kind: path.with_name(path.name + ".partial") for kind, path in files.items()}
    offsets, tokens_written = [], 0
    per_advisor: Counter = Counter()
    agree: Counter = Counter()
    plies: Counter = Counter()
    months = set()

    def lines():
        for path in paths:
            with path.open() as handle:
                yield from (line for line in handle if line.strip())

    with mp.Pool(workers, _init_worker, ({**job, "needs_engine": False},)) as pool, \
            partial["bin"].open("wb") as token_out, partial["mask.bin"].open("wb") as mask_out:
        for game, month, advisor, ids, mask, agreed, n in pool.imap(_encode, lines(), chunksize=64):
            if game in seen:
                raise ValueError(f"duplicate or cross-split game {game}")
            seen.add(game)
            months.add(month)
            token_out.write(np.asarray(ids, dtype=np.uint16).tobytes())
            mask_out.write(mask.tobytes())
            offsets.append((tokens_written, tokens_written + len(ids)))
            tokens_written += len(ids)
            per_advisor[advisor] += 1
            agree[advisor] += agreed
            plies[advisor] += n
            if len(offsets) % 100_000 == 0:
                print(f"{split_dir}: {len(offsets):,} games", flush=True)
    with partial["offsets.npy"].open("wb") as handle:
        np.save(handle, np.asarray(offsets, dtype=np.uint64))
    for kind in files:
        partial[kind].replace(files[kind])
    return {
        "records": len(offsets),
        "tokens": tokens_written,
        "human_targets": sum(plies.values()),
        "months": sorted(m for m in months if m),
        "games_per_advisor": dict(sorted(per_advisor.items())),
        "advice_equals_human_move": {
            name: agree[name] / max(plies[name], 1) for name in sorted(plies)
        },
        "inputs": [{"path": str(p), "sha256": _sha256(p)} for p in paths],
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                  for path in files.values()},
    }


def build(args):
    config = _load_config(args.config)
    job = {"mixture": config["train_mixture"], "seed": config.get("seed", 0)}
    if (args.out_dir / "dataset.manifest.json").exists():
        raise SystemExit("dataset exists; build into a new directory")
    seen: set[str] = set()
    splits = {"train": _write_split(args.train, args.out_dir / "train", job, args.workers, seen),
              "val": _write_split(args.val, args.out_dir / "val", job, args.workers, seen)}
    overlap = set(splits["train"]["months"]) & set(splits["val"]["months"])
    if overlap:
        raise ValueError(f"source months overlap between train and val: {sorted(overlap)}")
    manifest = {
        "kind": "advisor_trust_dataset",
        "representation": "A W A B ... (A = advice in SF namespace or NONE 4080)",
        "token_namespace": {"stockfish_moves_distinct_from_human": True, "none_id": 4080},
        "config": config,
        "config_sha256": _sha256(args.config),
        "code_revision": _git_revision(),
        "splits": splits,
    }
    (args.out_dir / "dataset.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def parser():
    root = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)
    ann = sub.add_parser("annotate")
    ann.add_argument("--mode", choices=("all", "assigned"), required=True)
    ann.add_argument("--in", dest="input", type=Path, required=True)
    ann.add_argument("--out", type=Path, required=True)
    ann.add_argument("--manifest", type=Path, default=None)
    ann.add_argument("--advisor", action="append", default=[])
    ann.add_argument("--config", type=Path, default=None)
    ann.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH", "stockfish"))
    ann.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ann.add_argument("--threads", type=int, default=1)
    ann.add_argument("--limit", type=int, default=0, help="first N games only (smoke tests)")
    ann.add_argument("--log-every", type=int, default=2_000)
    ann.set_defaults(function=annotate)
    bld = sub.add_parser("build")
    bld.add_argument("--config", type=Path, required=True)
    bld.add_argument("--train", type=Path, nargs="+", required=True)
    bld.add_argument("--val", type=Path, nargs="+", required=True)
    bld.add_argument("--out-dir", type=Path, required=True)
    bld.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    bld.set_defaults(function=build)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
