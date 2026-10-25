#!/usr/bin/env python3
"""Immutable ten-round datasets, resumable SFT/DAgger execution, append-only reports.

Run on the GPU worker inside the runtime container; no SSH dependency on the Mac.
See docs/curriculum.md for preparation, restart and extension commands.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline import valid

from krasnal.data.annotation import Annotation
from krasnal.tokenization.reasoning import to_centipawns


def checked_record(task):
    line, block_size = task
    try:
        annotation = Annotation.from_json(line)
        if len(annotation.history) + 32 > block_size or not valid(annotation):
            return None
        paired = sorted(
            zip(annotation.variations, annotation.scores, strict=True),
            key=lambda item: to_centipawns(*item[1]),
            reverse=True,
        )
        annotation.variations = [pv for pv, _ in paired]
        annotation.scores = [score for _, score in paired]
        return annotation.to_json()
    except (ValueError, KeyError, TypeError, AssertionError):
        return None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare(args):
    """One transaction registers unique histories across every data generation."""
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / "pipeline.lock"):
        db = sqlite3.connect(root / "ledger.sqlite")
        db.execute("CREATE TABLE IF NOT EXISTS histories (key BLOB PRIMARY KEY) WITHOUT ROWID")
        db.execute("CREATE TABLE IF NOT EXISTS generations (id INTEGER PRIMARY KEY, manifest TEXT)")
        existing = db.execute("SELECT id, manifest FROM generations ORDER BY id").fetchall()
        if existing and json.loads(existing[0][1])["block_size"] != args.block_size:
            raise ValueError("block size must match the original curriculum")
        inputs = [{"path": str(p.resolve()), "sha256": sha(p)} for p in args.sources]
        for _, saved in existing:
            manifest = json.loads(saved)
            if manifest["sources"] == inputs:
                if (
                    manifest["rounds"] != args.rounds
                    or manifest["dagger_positions"] != args.dagger_positions
                    or manifest.get("dagger_ratio") != args.dagger_ratio
                ):
                    raise ValueError("same sources already prepared with different round settings")
                atomic_json(root / manifest["directory"] / "manifest.json", manifest)
                print("already prepared:", manifest["directory"], flush=True)
                return
        generation = len(existing)
        directory = f"data-{generation:03d}"
        folder = root / directory
        if folder.exists():
            folder.rename(root / f"{directory}.interrupted-{time.time_ns()}")
        folder.mkdir()
        counts = {"input": 0, "duplicate": 0, "rejected": 0, "train": 0, "eval": 0, "validation": 0}
        paths = [folder / f"round-{r:02d}.jsonl" for r in range(args.rounds)]
        round_counts = [0] * args.rounds
        eval_path = folder / "eval-source.jsonl"
        val_path = folder / "validation.jsonl"
        db.execute("BEGIN")
        try:
            with contextlib.ExitStack() as stack:
                pool = stack.enter_context(mp.Pool(args.workers))
                outputs = [stack.enter_context(p.open("w")) for p in paths]
                evaluation = stack.enter_context(eval_path.open("w"))
                validation = stack.enter_context(val_path.open("w"))
                for source in args.sources:
                    with source.open() as handle:
                        for line in pool.imap(
                            checked_record,
                            ((line, args.block_size) for line in handle),
                            chunksize=128,
                        ):
                            counts["input"] += 1
                            if line is None:
                                counts["rejected"] += 1
                                continue
                            annotation = Annotation.from_json(line)
                            key = hashlib.sha256(" ".join(annotation.history).encode()).digest()
                            inserted = db.execute(
                                "INSERT OR IGNORE INTO histories VALUES (?)", (key,)
                            )
                            if not inserted.rowcount:
                                counts["duplicate"] += 1
                                continue
                            if not generation and counts["eval"] < args.eval_positions:
                                evaluation.write(annotation.to_json() + "\n")
                                counts["eval"] += 1
                            elif not generation and counts["validation"] < args.val_positions:
                                validation.write(annotation.to_json() + "\n")
                                counts["validation"] += 1
                            else:
                                r = counts["train"] % args.rounds
                                outputs[r].write(annotation.to_json() + "\n")
                                round_counts[r] += 1
                                counts["train"] += 1
                            if counts["input"] % 100000 == 0:
                                print(json.dumps(counts), flush=True)
                for handle in [*outputs, evaluation, validation]:
                    handle.flush()
                    os.fsync(handle.fileno())
            budgets = [
                max(1, math.ceil(n * args.dagger_ratio))
                if args.dagger_ratio is not None
                else args.dagger_positions
                for n in round_counts
            ]
            if any(n < budget for n, budget in zip(round_counts, budgets, strict=True)):
                raise ValueError("each round must contain at least dagger-positions fresh records")
            if not generation and (
                counts["eval"] != args.eval_positions or counts["validation"] != args.val_positions
            ):
                raise ValueError("not enough held-out data")
            manifest = {
                "schema": 1,
                "generation": generation,
                "directory": directory,
                "sources": inputs,
                "counts": counts,
                "rounds": args.rounds,
                "block_size": args.block_size,
                "dagger_positions": args.dagger_positions,
                "dagger_ratio": args.dagger_ratio,
                "shards": [
                    {
                        "path": str(p.relative_to(root)),
                        "count": n,
                        "sha256": sha(p),
                        "dagger_positions": budget,
                    }
                    for p, n, budget in zip(paths, round_counts, budgets, strict=True)
                ],
                "eval_source": str(eval_path.relative_to(root)) if not generation else None,
                "validation": str(val_path.relative_to(root)) if not generation else None,
                "eval_sha256": sha(eval_path) if not generation else None,
                "validation_sha256": sha(val_path) if not generation else None,
            }
            db.execute("INSERT INTO generations VALUES (?, ?)", (generation, json.dumps(manifest)))
            db.commit()
            atomic_json(folder / "manifest.json", manifest)
            print(json.dumps(manifest, indent=2), flush=True)
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()


def execute(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("a") as handle:
        subprocess.run(list(map(str, command)), stdout=handle, stderr=subprocess.STDOUT, check=True)


def _read_jsonl(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def report(root):
    rows = []
    for p in sorted(root.glob("results/*.json")):
        rows.append(json.loads(p.read_text()))
    columns = [
        "stage",
        "teacher_seen",
        "dagger_seen",
        "human_top1",
        "direct_top1",
        "thinking_top1",
        "candidate_recall",
        "selection",
        "selection_when_best_present",
        "raw_format_error",
        "seconds",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for key in columns:
            value = row.get(key)
            if value is None:
                cells.append("—")
            elif key in columns[3:-1]:
                cells.append(f"{100 * value:.2f}%")
            elif key == "seconds":
                cells.append(f"{value:.0f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    temporary_md = root / f"report.md.partial-{os.getpid()}"
    temporary_csv = root / f"report.csv.partial-{os.getpid()}"
    temporary_md.write_text(table)
    temporary_md.replace(root / "report.md")
    import csv

    with temporary_csv.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(root / "report.csv")
    print(table, flush=True)


def schedule(root, teacher_fraction=0.75):
    """Describe the complete chain, including appended data generations."""
    with sqlite3.connect(root / "ledger.sqlite") as db:
        manifests = [
            json.loads(row[0]) for row in db.execute("SELECT manifest FROM generations ORDER BY id")
        ]
    stages = [{"stage": "000-pretrain", "checkpoint": "foundation", "teacher": 0, "dagger": 0}]
    index = 0
    for manifest in manifests:
        for shard in manifest["shards"]:
            index += 1
            count = shard.get("dagger_positions", manifest["dagger_positions"])
            replay = min(round(count * teacher_fraction / (1 - teacher_fraction)), shard["count"])
            stages += [
                {
                    "stage": f"{index:03d}-a-sft",
                    "dataset": shard["path"],
                    "teacher": shard["count"],
                    "dagger": 0,
                },
                {
                    "stage": f"{index:03d}-b-dagger",
                    "dataset": shard["path"],
                    "teacher": replay,
                    "dagger": count,
                },
            ]
    return stages


def run(args):
    root = args.root.resolve()
    with lock(root / "pipeline.lock"):
        db = sqlite3.connect(root / "ledger.sqlite")
        manifests = [
            json.loads(row[0]) for row in db.execute("SELECT manifest FROM generations ORDER BY id")
        ]
        db.close()
        if not manifests:
            raise ValueError("prepare data first")
        config = {
            k: v for k, v in vars(args).items() if k not in ("function", "root", "adopt_code")
        }
        config = json.loads(json.dumps(config, default=str))
        config["checkpoint_sha256"] = sha(args.checkpoint)
        config["human_sha256"] = sha(args.human_val)
        config["format"] = {
            "version": "compact-v1",
            "efforts": [0, 2],
            "max_pvs": 3,
            "max_plies": 3,
            "delta_edges_cp": [5, 40, 180],
            "pv_root_weight": args.pv_root_weight,
            "pv_decay": args.pv_decay,
            "structure_weight": args.structure_weight,
            "dagger_objective": args.dagger_objective,
        }
        config["code_sha256"] = {
            str(p): sha(p)
            for p in [
                Path("scripts") / name
                for name in (
                    "curriculum.py",
                    "curriculum_eval.py",
                    "sft.py",
                    "dagger.py",
                    "experiment.py",
                )
            ]
            + sorted(Path("src/krasnal").rglob("*.py"))
            + sorted(Path("src/krasnal_think").glob("*.py"))
        }
        config_path = root / "run-config.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text())
            if previous != config:
                before = {k: v for k, v in previous.items() if k != "code_sha256"}
                after = {k: v for k, v in config.items() if k != "code_sha256"}
                changed = {
                    key
                    for key in previous["code_sha256"]
                    if previous["code_sha256"][key] != config["code_sha256"].get(key)
                }
                if not (
                    args.adopt_code
                    and before == after
                    and changed <= {"scripts/curriculum.py", "scripts/dagger.py"}
                ):
                    raise ValueError(
                        "immutable run configuration differs; use the original arguments"
                    )
                stamp = time.time_ns()
                atomic_json(root / f"run-config.before-repair-{stamp}.json", previous)
                atomic_json(
                    root / f"code-repair-{stamp}.json",
                    {
                        "reason": (
                            "skip terminal/empty-root on-policy positions; "
                            "backfill incomplete round"
                        ),
                        "changed": sorted(changed),
                        "previous_code_sha256": previous["code_sha256"],
                        "new_code_sha256": config["code_sha256"],
                    },
                )
        atomic_json(config_path, config)
        atomic_json(root / "schedule.json", schedule(root, args.teacher_fraction))
        for manifest in manifests:
            for shard in manifest["shards"]:
                if sha(root / shard["path"]) != shard["sha256"]:
                    raise ValueError("training shard changed: " + shard["path"])
        first = manifests[0]
        validation = root / first["validation"]
        source = root / first["eval_source"]
        if sha(validation) != first["validation_sha256"] or sha(source) != first["eval_sha256"]:
            raise ValueError("held-out data changed")
        gold = root / "eval-gold.jsonl"
        py = sys.executable
        if not gold.with_suffix(".manifest.json").exists():
            if gold.exists():
                gold.rename(gold.with_name(gold.name + f".interrupted-{time.time_ns()}"))
            execute(
                [
                    py,
                    "scripts/curriculum_eval.py",
                    "gold",
                    "--source",
                    source,
                    "--out",
                    gold,
                    "--nodes",
                    args.nodes,
                    "--workers",
                    args.workers,
                    "--stockfish",
                    args.stockfish,
                ],
                root / "gold.log",
            )
        gold_manifest = json.loads(gold.with_suffix(".manifest.json").read_text())
        if sha(gold) != gold_manifest["sha256"]:
            raise ValueError("frozen evaluation labels changed")

        def collect(checkpoint, dataset, folder, count, frozen=False):
            path = folder / "candidates.jsonl"
            if path.exists() and not frozen:
                with path.open() as handle:
                    previous = [json.loads(line) for line in handle]
                empty = [row for row in previous if not row["roots"]]
                if empty:
                    backfill = folder / "candidates.backfill.jsonl"
                    if not backfill.exists():
                        execute(
                            [
                                py,
                                "scripts/dagger.py",
                                checkpoint,
                                "--source",
                                dataset,
                                "--out",
                                folder / "unused.jsonl",
                                "--metrics-out",
                                folder / "unused.json",
                                "--positions",
                                len(empty),
                                "--source-offset",
                                count,
                                "--effort",
                                "2",
                                "--compact",
                                "--device",
                                args.device,
                                "--candidates-out",
                                backfill,
                            ],
                            folder / "collect.log",
                        )
                    replacements = list(_read_jsonl(backfill))
                    if len(replacements) != len(empty) or any(not r["roots"] for r in replacements):
                        raise ValueError("backfill did not replace all empty-root positions")
                    repaired = [row for row in previous if row["roots"]] + replacements
                    if len({tuple(row["history"]) for row in repaired}) != count:
                        raise ValueError("backfill histories are not distinct")
                    archive = path.with_name(f"candidates.empty-roots-{time.time_ns()}.jsonl")
                    temporary = path.with_suffix(".jsonl.repaired.partial")
                    with temporary.open("x") as output:
                        for row in repaired:
                            output.write(json.dumps(row, sort_keys=True) + "\n")
                        output.flush()
                        os.fsync(output.fileno())
                    path.rename(archive)
                    temporary.replace(path)
                    atomic_json(
                        folder / "backfill.manifest.json",
                        {
                            "archived_candidates": str(archive),
                            "archived_sha256": sha(archive),
                            "backfill": str(backfill),
                            "backfill_sha256": sha(backfill),
                            "repaired_sha256": sha(path),
                            "replaced_empty": len(empty),
                        },
                    )
            if not path.exists():
                execute(
                    [
                        py,
                        "scripts/dagger.py",
                        checkpoint,
                        "--source",
                        dataset,
                        "--out",
                        folder / "unused.jsonl",
                        "--metrics-out",
                        folder / "unused.json",
                        "--positions",
                        count,
                        "--effort",
                        "2",
                        "--compact",
                        "--device",
                        args.device,
                        "--candidates-out",
                        path,
                        *(["--frozen-gold"] if frozen else []),
                    ],
                    folder / "collect.log",
                )
            with path.open() as handle:
                if sum(1 for _ in handle) != count:
                    raise ValueError("incomplete collection: " + str(path))
            return path

        def relabel(candidates, folder, count):
            output, metrics = folder / "labeled.jsonl", folder / "decision.json"
            if metrics.exists() and json.loads(metrics.read_text())["positions"] != count:
                stamp = time.time_ns()
                metrics.rename(folder / f"decision.incomplete-{stamp}.json")
                if output.exists():
                    output.rename(folder / f"labeled.incomplete-{stamp}.jsonl")
            if not metrics.exists():
                execute(
                    [
                        py,
                        "scripts/dagger.py",
                        "--source",
                        gold,
                        "--candidates-in",
                        candidates,
                        "--out",
                        output,
                        "--metrics-out",
                        metrics,
                        "--positions",
                        count,
                        "--nodes",
                        args.nodes,
                        "--workers",
                        args.workers,
                        "--stockfish",
                        args.stockfish,
                    ],
                    folder / "relabel.log",
                )
            result = json.loads(metrics.read_text())
            if result["positions"] != count:
                raise ValueError("relabel dropped positions; inspect " + str(folder))
            return result

        def evaluate(checkpoint, name, teacher_seen, dagger_seen):
            result_path = root / "results" / (name + ".json")
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result["human_top1"] < 0.40:
                    raise RuntimeError("saved result is below human guardrail")
                return result
            started = time.time()
            folder = root / "evaluation" / name
            baseline = name == "000-pretrain"
            if baseline:
                decision_path = folder / "direct.json"
                if not decision_path.exists():
                    execute(
                        [
                            py,
                            "scripts/curriculum_eval.py",
                            "direct",
                            "--checkpoint",
                            checkpoint,
                            "--source",
                            gold,
                            "--out",
                            decision_path,
                            "--device",
                            args.device,
                        ],
                        folder / "direct.log",
                    )
                candidates = None
            else:
                candidates = collect(checkpoint, gold, folder, first["counts"]["eval"], True)
            # CPU Stockfish and GPU human evaluation use different resources concurrently.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = (
                    pool.submit(relabel, candidates, folder, first["counts"]["eval"])
                    if not baseline
                    else None
                )
                human = folder / "human.json"
                if not human.exists():
                    execute(
                        [
                            py,
                            "scripts/curriculum_eval.py",
                            "human",
                            "--checkpoint",
                            checkpoint,
                            "--source",
                            args.human_val,
                            "--out",
                            human,
                            "--device",
                            args.device,
                            "--batches",
                            args.human_batches,
                            *(
                                [
                                    "--format-source",
                                    gold,
                                    "--format-positions",
                                    args.format_positions,
                                ]
                                if not baseline
                                else []
                            ),
                        ],
                        folder / "human.log",
                    )
                decision = future.result() if future else json.loads(decision_path.read_text())
            result = {
                **decision,
                **json.loads(human.read_text()),
                "stage": name,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha(checkpoint),
                "teacher_seen": teacher_seen,
                "dagger_seen": dagger_seen,
                "seconds": time.time() - started,
                "eval_sha256": gold_manifest["sha256"],
            }
            atomic_json(result_path, result)
            report(root)
            if result["human_top1"] < 0.40:
                raise RuntimeError("human top1 below 40%; saved checkpoint and stopped")
            return result

        def train(checkpoint, dataset, folder, count, dagger=None, dagger_count=0):
            output = folder / "model.pt"
            if output.exists():
                return output
            latest = folder / "latest.pt"
            source_checkpoint = latest if latest.exists() else checkpoint
            cmd = [
                py,
                "scripts/sft.py",
                source_checkpoint,
                "--annotations",
                dataset,
                "--validation-source",
                validation,
                "--val-positions",
                first["counts"]["validation"],
                "--out",
                folder,
                "--epochs",
                "1",
                "--batch-size",
                args.batch_size,
                "--compact",
                "--dagger-objective",
                args.dagger_objective,
                "--pv-root-weight",
                args.pv_root_weight,
                "--pv-decay",
                args.pv_decay,
                "--structure-weight",
                args.structure_weight,
                "--teacher-efforts",
                "0,2",
                "--lr",
                args.lr,
                "--min-lr",
                args.min_lr,
                "--warmup",
                "50",
                "--eval-every",
                "0",
                "--pretrain-check-every",
                "0",
                "--save-every",
                args.save_every,
                "--seed",
                "42",
                "--device",
                args.device,
                "--resume" if latest.exists() else "--continue-optimizer",
            ]
            if dagger:
                cmd += [
                    "--dagger-annotations",
                    dagger,
                    "--teacher-positions",
                    count,
                    "--teacher-fraction",
                    args.teacher_fraction,
                ]
            execute(cmd, folder / "train.log")
            if not output.exists():
                raise RuntimeError("trainer did not finish")
            atomic_json(
                folder / "completed.json",
                {"teacher_count": count, "dagger_count": dagger_count, "sha256": sha(output)},
            )
            return output

        checkpoint = args.checkpoint.resolve()
        teacher_seen = dagger_seen = 0
        baseline = evaluate(checkpoint, "000-pretrain", 0, 0)
        if baseline["human_block_size"] != first["block_size"]:
            raise ValueError("prepare --block-size must match the pretrain model")
        index = 0
        for manifest in manifests:
            for shard in manifest["shards"]:
                index += 1
                dataset = root / shard["path"]
                folder = root / f"round-{index:03d}"
                atomic_json(
                    root / "status.json",
                    {
                        "round": index,
                        "stage": "sft",
                        "started": time.time(),
                        "checkpoint": str(checkpoint),
                    },
                )
                checkpoint = train(checkpoint, dataset, folder / "sft", shard["count"])
                teacher_seen += shard["count"]
                evaluate(checkpoint, f"{index:03d}-a-sft", teacher_seen, dagger_seen)
                count = shard.get("dagger_positions", manifest["dagger_positions"])
                replay = min(
                    round(count * args.teacher_fraction / (1 - args.teacher_fraction)),
                    shard["count"],
                )
                atomic_json(
                    root / "status.json",
                    {"round": index, "stage": "dagger", "checkpoint": str(checkpoint)},
                )
                candidates = collect(checkpoint, dataset, folder / "onpolicy", count)
                relabel(candidates, folder / "onpolicy", count)
                labeled = folder / "onpolicy" / "labeled.jsonl"
                checkpoint = train(
                    checkpoint,
                    dataset,
                    folder / "dagger",
                    replay,
                    labeled,
                    count,
                )
                teacher_seen += replay
                dagger_seen += count
                evaluate(checkpoint, f"{index:03d}-b-dagger", teacher_seen, dagger_seen)
        atomic_json(
            root / "status.json",
            {
                "stage": "complete",
                "rounds": index,
                "checkpoint": str(checkpoint),
                "teacher_seen": teacher_seen,
                "dagger_seen": dagger_seen,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--sources", type=Path, nargs="+", required=True)
    prep.add_argument("--rounds", type=int, default=10)
    prep.add_argument("--eval-positions", type=int, default=1000)
    prep.add_argument("--val-positions", type=int, default=1000)
    prep.add_argument("--dagger-positions", type=int, default=10000)
    prep.add_argument(
        "--dagger-ratio",
        type=float,
        default=None,
        help="DAgger positions per teacher record; overrides fixed position budget",
    )
    prep.add_argument("--block-size", type=int, default=1024)
    prep.add_argument("--workers", type=int, default=8)
    prep.set_defaults(function=prepare)
    runner = sub.add_parser("run")
    runner.add_argument("--root", type=Path, required=True)
    runner.add_argument("--checkpoint", type=Path, required=True)
    runner.add_argument("--human-val", type=Path, required=True)
    runner.add_argument("--human-batches", type=int, default=30)
    runner.add_argument("--format-positions", type=int, default=100)
    runner.add_argument("--nodes", type=int, default=30000)
    runner.add_argument("--workers", type=int, default=8)
    runner.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH", "stockfish"))
    runner.add_argument("--batch-size", type=int, default=64)
    runner.add_argument("--save-every", type=int, default=1000)
    runner.add_argument("--lr", type=float, default=5e-6)
    runner.add_argument("--min-lr", type=float, default=5e-7)
    runner.add_argument("--teacher-fraction", type=float, default=0.75)
    runner.add_argument(
        "--dagger-objective",
        choices=["teacher-trace", "onpolicy-selection"],
        default="teacher-trace",
    )
    runner.add_argument("--pv-root-weight", type=float, default=0.5)
    runner.add_argument("--pv-decay", type=float, default=0.5)
    runner.add_argument("--structure-weight", type=float, default=0.1)
    runner.add_argument("--device", default="cuda")
    runner.add_argument(
        "--adopt-code",
        action="store_true",
        help="one-time audited repair of curriculum/dagger code fingerprints",
    )
    runner.set_defaults(function=run)
    status = sub.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    status.set_defaults(function=lambda args: report(args.root))
    args = parser.parse_args()
    if (
        hasattr(args, "dagger_ratio")
        and args.dagger_ratio is not None
        and not 0 < args.dagger_ratio <= 1
    ):
        parser.error("dagger-ratio must be in (0,1]")
    if hasattr(args, "teacher_fraction"):
        if not 0 < args.teacher_fraction < 1:
            parser.error("teacher-fraction must be in (0,1)")
        if not (args.pv_root_weight > 0 and 0 < args.pv_decay <= 1 and args.structure_weight > 0):
            parser.error("loss weights must be positive; pv-decay must be in (0,1]")
    for name in (
        "rounds",
        "eval_positions",
        "val_positions",
        "dagger_positions",
        "workers",
        "human_batches",
        "nodes",
        "batch_size",
        "save_every",
    ):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(name + " must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
