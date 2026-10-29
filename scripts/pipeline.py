#!/usr/bin/env python3
"""Small, resumable pipeline utilities shared by teacher and DAgger runs."""

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import chess

from krasnal.data.annotation import Annotation


def history_key(history: list[str]) -> int:
    digest = hashlib.blake2b(" ".join(history).encode(), digest_size=8).digest()
    return int.from_bytes(digest)


def valid(annotation: Annotation) -> bool:
    if (
        not annotation.history
        or not annotation.variations
        or len(annotation.variations) != len(annotation.scores)
    ):
        return False
    try:
        board = chess.Board()
        for uci in annotation.history:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                return False
            board.push(move)
        for pv in annotation.variations:
            if not pv:
                return False
            shadow = board.copy()
            for uci in pv:
                move = chess.Move.from_uci(uci)
                if move not in shadow.legal_moves:
                    return False
                shadow.push(move)
    except (ValueError, AssertionError):
        return False
    return True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shard(args):
    if args.out.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite an existing validated shard or manifest")
    seen: set[int] = set()
    existing_records = 0
    for path in args.against:
        with path.open() as handle:
            for line in handle:
                try:
                    annotation = Annotation.from_json(line)
                except (ValueError, KeyError, TypeError):
                    continue
                seen.add(history_key(annotation.history))
                existing_records += 1

    counts = {"input": 0, "accepted": 0, "duplicate": 0, "invalid": 0}
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.input.open() as source, partial.open("x") as output:
            for line in source:
                counts["input"] += 1
                try:
                    annotation = Annotation.from_json(line)
                except (ValueError, KeyError, TypeError):
                    counts["invalid"] += 1
                    continue
                key = history_key(annotation.history)
                if key in seen:
                    counts["duplicate"] += 1
                    continue
                if not valid(annotation):
                    counts["invalid"] += 1
                    continue
                seen.add(key)
                output.write(annotation.to_json() + "\n")
                counts["accepted"] += 1
            output.flush()
            os.fsync(output.fileno())
        partial.replace(args.out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    manifest = {
        "schema": 1,
        "created_unix": time.time(),
        "host": socket.gethostname(),
        "kind": args.kind,
        "source": {
            "path": str(args.input),
            "sha256": file_sha256(args.input),
            "parameters": json.loads(args.parameters),
        },
        "dedupe_against": [str(path) for path in args.against],
        "existing_records_scanned": existing_records,
        "counts": counts,
        "output": {"path": str(args.out), "sha256": file_sha256(args.out)},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temp_manifest = args.manifest.with_suffix(args.manifest.suffix + ".partial")
    temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temp_manifest.replace(args.manifest)
    print(json.dumps(manifest, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    validate_parser = subparsers.add_parser("validate-shard")
    validate_parser.add_argument("--input", type=Path, required=True)
    validate_parser.add_argument("--out", type=Path, required=True)
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--against", type=Path, nargs="*", default=[])
    validate_parser.add_argument("--kind", choices=("teacher", "dagger"), required=True)
    validate_parser.add_argument("--parameters", default="{}")
    validate_parser.set_defaults(function=validate_shard)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
