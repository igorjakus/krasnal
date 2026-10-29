#!/usr/bin/env python3
"""Measure whether human-move predictions track teacher-forced Stockfish moves."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from krasnal.tokenization import reasoning as tokens
from krasnal.data.interleave import record_to_tokens
from krasnal.models.reasoning import GPT
from krasnal.training.runtime import pick_device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(checkpoint: Path, records_path: Path, conditioning: str, batch_size: int,
             device_name: str | None, cpu_threads: int) -> dict:
    device = pick_device(device_name)
    if device != "cpu":
        raise ValueError("copy diagnostic is intentionally CPU-only")
    torch.set_num_threads(cpu_threads)
    model = GPT.load(checkpoint, device).eval()
    if model.config.vocab_size != tokens.INTERLEAVED_VOCAB_SIZE:
        raise ValueError("expected checkpoint with distinct Stockfish token namespace")

    records = [json.loads(line) for line in records_path.open()
               if line.strip()]
    if not records:
        raise ValueError("test records are empty")

    per_game = []
    totals = {key: 0 for key in (
        "positions", "human_correct", "model_matches_sf", "human_matches_sf",
        "disagreements", "model_matches_sf_on_disagreement",
        "sf_logit_margin_sum", "sf_logit_margin_positive",
    )}
    autocast = nullcontext()
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            chunk = records[offset:offset + batch_size]
            encoded = []
            for record in chunk:
                ids, _ = record_to_tokens(record, interleaved=True)
                if conditioning == "star":
                    ids = [tokens.SF_MOVE_IDS.stop if value in tokens.SF_MOVE_IDS else value
                           for value in ids]
                encoded.append(ids)
            width = max(map(len, encoded))
            x = torch.full((len(encoded), width - 1), tokens.PAD,
                           dtype=torch.long, device=device)
            for row, ids in enumerate(encoded):
                x[row, :len(ids) - 1] = torch.tensor(ids[:-1], dtype=torch.long, device=device)
            with autocast:
                logits, _ = model(x, targets=torch.zeros_like(x))

            move_logits = logits[:, :, tokens.MOVE_IDS.start:tokens.MOVE_IDS.stop].float()
            # Restrict the diagnostic's selected action to the human-move vocabulary.
            move_logits = move_logits[:, :, :]
            for row, record in enumerate(chunk):
                human = record["h"].split()
                sf = record["s"].split()
                if len(human) != len(sf):
                    raise ValueError("human/SF move counts differ")
                # At these positions the model has seen the SF note and predicts
                # the following human move: BOS,EloW,EloB,SF1,H1,SF2,H2,...
                positions = list(range(3, 3 + 2 * len(human), 2))
                if positions and positions[-1] >= logits.shape[1]:
                    raise ValueError("game exceeds model context length")
                row_logits = move_logits[row, positions]
                predicted = row_logits.argmax(-1).cpu().numpy() + tokens.MOVE_IDS.start
                human_ids = np.asarray([tokens.move_id(move) for move in human])
                sf_ids = np.asarray([tokens.move_id(move) for move in sf])
                disagreement = human_ids != sf_ids
                copied = predicted == sf_ids
                correct = predicted == human_ids
                margin = (row_logits.gather(1, torch.as_tensor(
                    sf_ids - tokens.MOVE_IDS.start, device=device
                )[:, None]).squeeze(1) - row_logits.gather(1, torch.as_tensor(
                    human_ids - tokens.MOVE_IDS.start, device=device
                )[:, None]).squeeze(1)).cpu().numpy()

                n = len(human)
                d = int(disagreement.sum())
                copied_d = int((copied & disagreement).sum())
                per_game.append((copied_d, d))
                totals["positions"] += n
                totals["human_correct"] += int(correct.sum())
                totals["model_matches_sf"] += int(copied.sum())
                totals["human_matches_sf"] += int((human_ids == sf_ids).sum())
                totals["disagreements"] += d
                totals["model_matches_sf_on_disagreement"] += copied_d
                totals["sf_logit_margin_sum"] += float(margin.sum())
                totals["sf_logit_margin_positive"] += int((margin > 0).sum())

    n = totals["positions"]
    d = totals["disagreements"]
    totals_float = {
        "human_move_top1": totals["human_correct"] / n,
        "predicted_move_matches_stockfish": totals["model_matches_sf"] / n,
        "human_and_stockfish_same_move": totals["human_matches_sf"] / n,
        "predicted_stockfish_on_human_stockfish_disagreements": (
            totals["model_matches_sf_on_disagreement"] / d if d else None
        ),
        "mean_stockfish_vs_human_move_logit_margin": totals["sf_logit_margin_sum"] / n,
        "stockfish_move_logit_exceeds_human_move_fraction": (
            totals["sf_logit_margin_positive"] / n
        ),
    }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "conditioning": conditioning,
        "precision": "float32",
        "records": len(records),
        "records_sha256": sha256_file(records_path),
        "human_moves": n,
        "moves_where_human_differs_from_stockfish": d,
        **totals_float,
        "counts": totals,
        "per_game_disagreement_counts": per_game,
    }


def paired_bootstrap(sf: dict, star: dict, repetitions: int, seed: int) -> dict:
    if sf["records"] != star["records"]:
        raise ValueError("SF and star evaluations are not paired on identical games")
    sf_values = np.asarray(sf["per_game_disagreement_counts"], dtype=np.int64)
    star_values = np.asarray(star["per_game_disagreement_counts"], dtype=np.int64)
    if not np.array_equal(sf_values[:, 1], star_values[:, 1]):
        raise ValueError("SF/star disagreement sets differ")
    rng = np.random.default_rng(seed)
    n_games = len(sf_values)
    differences = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        sample = rng.integers(0, n_games, size=n_games)
        denominator = sf_values[sample, 1].sum()
        if denominator:
            differences[i] = (sf_values[sample, 0].sum() - star_values[sample, 0].sum()) / denominator
        else:
            differences[i] = np.nan
    return {
        "sf_minus_star_copy_rate_on_disagreements": (
            sf["predicted_stockfish_on_human_stockfish_disagreements"]
            - star["predicted_stockfish_on_human_stockfish_disagreements"]
        ),
        "paired_game_bootstrap_95pct_ci": [
            float(np.nanquantile(differences, 0.025)),
            float(np.nanquantile(differences, 0.975)),
        ],
        "bootstrap_repetitions": repetitions,
        "bootstrap_unit": "game",
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "CONDITIONING"), required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        raise SystemExit("batch size and CPU threads must be positive")
    results = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if args.out.exists():
        try:
            previous = json.loads(args.out.read_text())
            cached = previous.get("models", {})
        except (OSError, json.JSONDecodeError):
            cached = {}
    records_sha = sha256_file(args.records)
    for name, checkpoint, conditioning in args.model:
        if conditioning not in {"sf", "star"}:
            raise SystemExit("conditioning must be sf or star")
        checkpoint_path = Path(checkpoint)
        cached_result = cached.get(name)
        checkpoint_sha = sha256_file(checkpoint_path)
        if (cached_result and cached_result.get("checkpoint_sha256") == checkpoint_sha
                and cached_result.get("checkpoint") == str(checkpoint_path)
                and cached_result.get("conditioning") == conditioning
                and cached_result.get("records") == sum(1 for line in args.records.open() if line.strip())
                and cached_result.get("records_sha256", records_sha) == records_sha):
            result = cached_result
            result["records_sha256"] = records_sha
            print(f"reusing verified completed model: {name}", flush=True)
        else:
            result = evaluate(checkpoint_path, args.records, conditioning,
                              args.batch_size, args.device, args.cpu_threads)
            print(f"completed model: {name}", flush=True)
        results[name] = result
        # Persist each completed model atomically so a preemption loses no work.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        partial = args.out.with_suffix(args.out.suffix + ".partial")
        partial.write_text(json.dumps({"models": results}, indent=2) + "\n")
        partial.replace(args.out)
    sf_name = next((name for name, result in results.items()
                    if result["conditioning"] == "sf" and "scratch" in name.lower()), None)
    star_name = next((name for name, result in results.items()
                      if result["conditioning"] == "star"), None)
    summary = {}
    if sf_name and star_name:
        summary["scratch_sf_vs_star"] = paired_bootstrap(
            results[sf_name], results[star_name], args.bootstrap, seed=20261016
        )
    output = {
        "records": str(args.records),
        "records_sha256": sha256_file(args.records),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "implementation_sha256": {
            "model.py": sha256_file(Path("src/krasnal/models/reasoning.py")),
            "interleave.py": sha256_file(Path("src/krasnal/data/interleave.py")),
            "tokens.py": sha256_file(Path("src/krasnal/tokenization/reasoning.py")),
        },
        "metrics_note": "Model top-1 is restricted to human move tokens. SF suggestions are teacher-forced for conditioning=sf and replaced by constant star for conditioning=star.",
        "models": results,
        "paired_comparisons": summary,
    }
    rendered = json.dumps(output, indent=2) + "\n"
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(rendered)
    partial.replace(args.out)
    print(rendered, end="")


if __name__ == "__main__":
    main()
