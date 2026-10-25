#!/usr/bin/env python3
"""Aggregate on-policy candidate sets and relabel them with Stockfish."""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import chess
import chess.engine

from krasnal.tokenization import reasoning as tokens
from krasnal.data.annotation import Annotation

ROOT = Path(__file__).resolve().parents[1]
_engine = None
_nodes = 30_000


def _start_worker(stockfish: str, nodes: int):
    global _engine, _nodes
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    _nodes = nodes


def _value(score: tuple[int | None, int | None]) -> int:
    return tokens.to_centipawns(*score)


def _pv_diversity(variations: list[list[str]], ply: int) -> float | None:
    moves = [pv[ply] for pv in variations if len(pv) > ply]
    if len(moves) < 2:
        return None
    pairs = [(moves[i], moves[j]) for i in range(len(moves)) for j in range(i + 1, len(moves))]
    return sum(left != right for left, right in pairs) / len(pairs)


def _parse_variations(generated: list[str]) -> list[list[str]]:
    variations = []
    current = None
    for token in generated:
        if token == "<pv>":
            current = []
        elif token == "</pv>":
            if current:
                variations.append(current)
            current = None
        elif current is not None and not token.startswith("<"):
            current.append(token)
    return variations


def _relabel(task):
    if isinstance(task, dict):
        history = task["history"]
        roots = task["roots"]
        variations = task["variations"]
        direct = task["direct"]
        thinking = task["thinking"]
    else:
        history, roots, variations, direct, thinking = task
    board = chess.Board()
    for uci in history:
        board.push_uci(uci)
    pov = board.turn
    # Clear hash between positions: labels must not depend on worker scheduling.
    _engine.configure({"Clear Hash": None})
    gold = task.get("gold") if isinstance(task, dict) else None
    global_info = None if gold else _engine.analyse(board, chess.engine.Limit(nodes=_nodes))
    global_pv = gold["pv"] if gold else [move.uci() for move in global_info.get("pv", [])[:7]]
    if not global_pv:
        return None
    global_best = global_pv[0]
    if gold:
        global_value = _value(tuple(gold["score"]))
    else:
        global_score = global_info["score"].pov(pov)
        global_value = _value((global_score.score(), global_score.mate()))
    candidates = []
    for root in roots:
        move = chess.Move.from_uci(root)
        if move not in board.legal_moves:
            continue
        child = board.copy()
        child.push(move)
        info = _engine.analyse(child, chess.engine.Limit(nodes=_nodes))
        score = info["score"].pov(pov)
        continuation = [move.uci() for move in info.get("pv", [])[:6]]
        candidates.append(([root, *continuation], (score.score(), score.mate())))
    candidates.sort(key=lambda item: _value(item[1]), reverse=True)
    if not candidates:
        return None
    annotation = Annotation(
        history,
        [line for line, _ in candidates],
        [score for _, score in candidates],
    ).to_json()
    generated = set(roots)
    scored_generated = [item for item in candidates if item[0][0] in generated]
    best_generated = scored_generated[0][0][0] if scored_generated else None
    if isinstance(task, dict) and "trace" in task:
        record = json.loads(annotation)
        record["onpolicy"] = {
            "version": 1,
            "trace": task["trace"],
            "target": best_generated,
        }
        annotation = json.dumps(record)
    score_by_root = {line[0]: _value(score) for line, score in candidates}
    rank_by_root = {
        root: rank
        for rank, (root, _score) in enumerate(
            sorted(score_by_root.items(), key=lambda item: item[1], reverse=True), 1
        )
    }

    def candidate_value(root):
        if root in score_by_root:
            return score_by_root[root]
        try:
            move = chess.Move.from_uci(root)
        except ValueError:
            return None
        if move not in board.legal_moves:
            return None
        child = board.copy()
        child.push(move)
        info = _engine.analyse(child, chess.engine.Limit(nodes=_nodes))
        score = info["score"].pov(pov)
        value = _value((score.score(), score.mate()))
        score_by_root[root] = value
        return value

    direct_value = candidate_value(direct)
    thinking_value = candidate_value(thinking)
    thinking_rank = rank_by_root.get(thinking)
    direct_gap = max(global_value - direct_value, 0) if direct_value is not None else None
    thinking_gap = max(global_value - thinking_value, 0) if thinking_value is not None else None
    metrics = {
        "direct_top1": int(direct == global_best),
        "thinking_top1": int(thinking == global_best),
        "candidate_recall": int(global_best in generated),
        "selection": int(best_generated is not None and thinking == best_generated),
        "selection_when_best_present": int(thinking == global_best)
        if global_best in generated
        else None,
        "final_outside_candidates": int(thinking not in generated),
        "direct_cp_gap": direct_gap,
        "thinking_cp_gap": thinking_gap,
        "thinking_vs_direct_cp": (
            thinking_value - direct_value
            if thinking_value is not None and direct_value is not None
            else None
        ),
        "thinking_rank": thinking_rank,
        "thinking_mrr": 1 / thinking_rank if thinking_rank else 0.0,
        "global_best_rank": rank_by_root.get(global_best),
        "candidate_count": len(roots),
        "distinct_roots": len(generated),
        "pv_diversity_1": _pv_diversity(variations, 0),
        "pv_diversity_2": _pv_diversity(variations, 1),
        "pv_diversity_3": _pv_diversity(variations, 2),
    }
    return annotation, metrics


def _tasks(args, model):
    from krasnal.inference.reasoning import Player

    rng = random.Random(args.seed)
    thinking_player = Player(
        model,
        device=args.device,
        effort=args.effort,
        temperature=0,
        distinct_pv_roots=True,
        compact=getattr(args, "compact", False),
    )
    direct_player = Player(model, device=args.device, effort=0, temperature=0)
    accepted = skipped = 0
    with args.source.open() as handle:
        for line in itertools.islice(handle, args.source_offset, None):
            annotation = Annotation.from_json(line)
            board = annotation.board()
            if board.is_game_over(claim_draw=True):
                skipped += 1
                continue
            direct_player.reset(board)
            direct = direct_player.move()
            thinking_player.reset(board)
            thinking = thinking_player.move()
            roots = [tokens.VOCAB[token] for token, _ in thinking_player.candidates]
            if not roots:
                skipped += 1
                continue
            variations = _parse_variations(thinking_player.thinking)
            best = annotation.variations[0][0]
            if rng.random() < args.inject_best_prob and best not in roots:
                roots.append(best)
            yield {
                "history": annotation.history,
                "roots": roots,
                "variations": variations,
                "direct": direct,
                "thinking": thinking,
                "trace": thinking_player.thinking[:-1],
                **(
                    {"gold": {"pv": annotation.variations[0], "score": annotation.scores[0]}}
                    if getattr(args, "frozen_gold", False)
                    else {}
                ),
            }
            accepted += 1
            if accepted == args.positions:
                break
    if accepted != args.positions:
        raise RuntimeError(
            f"only {accepted}/{args.positions} usable positions after offset "
            f"{args.source_offset}; skipped {skipped} terminal/empty-root positions"
        )
    if skipped:
        print(f"skipped {skipped} terminal/empty-root positions", flush=True)


def _candidate_rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def _collect(args):
    from krasnal.models.reasoning import GPT

    if args.checkpoint is None or args.candidates_out is None:
        raise SystemExit("collection requires checkpoint and --candidates-out")
    args.candidates_out.parent.mkdir(parents=True, exist_ok=True)
    model = GPT.load(args.checkpoint, args.device).eval()
    partial = args.candidates_out.with_suffix(args.candidates_out.suffix + ".partial")
    if partial.exists():
        partial.rename(partial.with_name(partial.name + f".interrupted-{time.time_ns()}"))
    try:
        with partial.open("x") as output:
            for index, task in enumerate(_tasks(args, model), 1):
                output.write(json.dumps(task, sort_keys=True) + "\n")
                if index % 500 == 0:
                    print(f"collected {index:,}/{args.positions:,}", flush=True)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(args.candidates_out)
    except BaseException:
        # Retain incomplete output for diagnosis/recovery; never discard collected data.
        raise


def _relabel_rows(args, tasks):
    start = time.time()
    with mp.Pool(args.workers, _start_worker, (args.stockfish, args.nodes)) as pool:
        done = 0
        totals = {
            "direct_top1": 0,
            "thinking_top1": 0,
            "candidate_recall": 0,
            "selection": 0,
            "selection_when_best_present": 0,
            "final_outside_candidates": 0,
            "direct_cp_gap": 0,
            "thinking_cp_gap": 0,
            "thinking_vs_direct_cp": 0,
            "thinking_rank": 0,
            "thinking_mrr": 0,
            "global_best_rank": 0,
            "candidate_count": 0,
            "distinct_roots": 0,
            "pv_diversity_1": 0,
            "pv_diversity_2": 0,
            "pv_diversity_3": 0,
        }
        metric_counts = dict.fromkeys(totals, 0)
        partial = args.out.with_suffix(args.out.suffix + ".partial")
        if partial.exists():
            partial.rename(partial.with_name(partial.name + f".interrupted-{time.time_ns()}"))
        with partial.open("x") as output:
            for record in pool.imap(_relabel, tasks, chunksize=4):
                if record is None:
                    continue
                annotation, metrics = record
                output.write(annotation + "\n")
                done += 1
                for key, value in metrics.items():
                    if value is not None:
                        totals[key] += value
                        metric_counts[key] += 1
                if done % 500 == 0:
                    rate = done / (time.time() - start)
                    print(f"{done:,}/{args.positions:,} {rate:.2f}/s", flush=True)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(args.out)
    summary = {
        "positions": done,
        "seconds": time.time() - start,
        **{
            key: value / metric_counts[key] if metric_counts[key] else None
            for key, value in totals.items()
        },
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.metrics_out.with_suffix(".json.partial")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.metrics_out)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=10_000)
    parser.add_argument("--source-offset", type=int, default=0)
    parser.add_argument("--effort", type=int, default=4)
    parser.add_argument(
        "--inject-best-prob",
        type=float,
        default=0.0,
        help="diagnostic only; keep at zero for unbiased candidate recall",
    )
    parser.add_argument("--nodes", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH", "stockfish"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--frozen-gold", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--candidates-out", type=Path)
    mode.add_argument("--candidates-in", type=Path)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.candidates_out is not None:
        _collect(args)
        return
    if args.candidates_in is not None:
        _relabel_rows(args, _candidate_rows(args.candidates_in))
        return
    raise SystemExit("choose --candidates-out or --candidates-in")


if __name__ == "__main__":
    main()
