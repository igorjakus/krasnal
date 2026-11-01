#!/usr/bin/env python3
"""Etap 0 / advisor-trust evaluation: how does the model use advice of varying quality?

For every checkpoint and every advisor condition, all games get advice from that
single advisor (teacher-forced), and we score the following human move.

Per human move we record:
  nll, top1, top5      human-move prediction (original human vocabulary, as in
                       scripts/evaluate_interleave.py)
  agree                advice == human move (how useful the advisor actually is)
  copy                 model top-1 == advice
  p_hint               model probability of the advised move as the human move

``p_hint`` vs ``agree`` is the calibration question: an ideally calibrated
predictor puts, on average, as much probability on the advice as the rate at
which humans of that rating actually play it. ``by_ply`` shows whether trust
changes as the model sees more of the same advisor in the game (in-context
reliability inference). Paired per-game bootstrap compares each condition
with the empty-note condition ``none``.

    python scripts/advisor_eval.py --records data/advisor-trust-v1/test/2024-06.jsonl \
        --model sf100m experiments/sf-interleave-100m-full-20261009-humanloss/model.pt \
        --out experiments/advisor-trust-v1/etap0/sf100m.json --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from krasnal.data.advisors import advice_for, encode_advised_game, validate_advisor
from krasnal.models.reasoning import GPT
from krasnal.tokenization import reasoning as tokens
from krasnal.training.runtime import pick_device

DEFAULT_CONDITIONS = ["none", "sf30k", "sfd8", "sfd4", "sfd2", "sfd1", "random",
                      "oracle20", "oracle40", "oracle60", "oracle80", "oracle100"]
ELO_EDGES = [1200, 1400, 1600, 1800, 2000, 2200, 2400]
PLY_EDGES = [2, 5, 10, 20, 40, 80]
FIELDS = ("nll", "top1", "top5", "agree", "copy", "p_hint")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _label(edges: list[int], index: int) -> str:
    low = "" if index == 0 else str(edges[index - 1])
    high = "" if index == len(edges) else str(edges[index] - 1)
    return f"{low}-{high}" if low and high else (f"<{edges[0]}" if not low else f">={low}")


def _summary(rows: dict[str, np.ndarray], select: np.ndarray) -> dict:
    out = {"positions": int(select.sum())}
    if not out["positions"]:
        return out
    for field in FIELDS:
        values = rows[field][select]
        values = values[~np.isnan(values)]
        out[field] = float(values.mean()) if len(values) else None
    hinted = select & ~np.isnan(rows["p_hint"])
    wrong = hinted & (rows["agree"] == 0)
    right = hinted & (rows["agree"] == 1)
    out["copy_when_advice_wrong"] = float(rows["copy"][wrong].mean()) if wrong.any() else None
    out["p_hint_when_advice_wrong"] = float(rows["p_hint"][wrong].mean()) if wrong.any() else None
    out["p_hint_when_advice_right"] = float(rows["p_hint"][right].mean()) if right.any() else None
    if hinted.any():
        # Positive: the model trusts the advice more than humans actually follow it.
        out["overtrust"] = float(rows["p_hint"][hinted].mean() - rows["agree"][hinted].mean())
    return out


def evaluate(model: GPT, records: list[dict], advisor: str, batch_size: int,
             device: str, seed: int) -> dict[str, np.ndarray]:
    columns = {name: [] for name in (*FIELDS, "elo", "ply", "game")}
    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            chunk = records[offset:offset + batch_size]
            encoded, advice_lists = [], []
            for record in chunk:
                advice = advice_for(record, advisor, seed)
                ids, _ = encode_advised_game(record["h"].split(), advice,
                                             int(record["w"]), int(record["b"]))
                if len(ids) > model.config.block_size:
                    raise ValueError(f"game {record['id']} exceeds block size")
                encoded.append(ids)
                advice_lists.append(advice)
            width = max(map(len, encoded))
            x = torch.full((len(encoded), width - 1), tokens.PAD, dtype=torch.long, device=device)
            for row, ids in enumerate(encoded):
                x[row, :len(ids) - 1] = torch.tensor(ids[:-1], device=device)
            with autocast:
                logits, _ = model(x, targets=torch.zeros_like(x))
            for row, record in enumerate(chunk):
                human = record["h"].split()
                n = len(human)
                # BOS,EloW,EloB,A1,H1,A2,H2,... : logits at 3+2k predict human move k.
                positions = torch.arange(3, 3 + 2 * n, 2, device=device)
                log_probs = logits[row, positions, :tokens.VOCAB_SIZE].float().log_softmax(-1)
                gold = torch.tensor([tokens.move_id(m) for m in human], device=device)
                advice = advice_lists[row]
                hint_ids = torch.tensor([tokens.move_id(a) if a else 0 for a in advice],
                                        device=device)
                has_hint = torch.tensor([a is not None for a in advice], device=device)
                top5 = log_probs.topk(5, -1).indices
                argmax = top5[:, 0]
                p_hint = log_probs.gather(1, hint_ids[:, None]).squeeze(1).exp()
                columns["nll"].append((-log_probs.gather(1, gold[:, None]).squeeze(1)).cpu().numpy())
                columns["top1"].append((argmax == gold).float().cpu().numpy())
                columns["top5"].append((top5 == gold[:, None]).any(-1).float().cpu().numpy())
                nan = torch.full_like(p_hint, float("nan"))
                columns["agree"].append(torch.where(has_hint, (hint_ids == gold).float(), nan).cpu().numpy())
                columns["copy"].append(torch.where(has_hint, (argmax == hint_ids).float(), nan).cpu().numpy())
                columns["p_hint"].append(torch.where(has_hint, p_hint, nan).cpu().numpy())
                elos = np.where(np.arange(n) % 2 == 0, int(record["w"]), int(record["b"]))
                columns["elo"].append(np.searchsorted(ELO_EDGES, elos, side="right"))
                columns["ply"].append(np.searchsorted(PLY_EDGES, np.arange(n), side="right"))
                columns["game"].append(np.full(n, offset + row))
    return {name: np.concatenate(values) for name, values in columns.items()}


def _paired_bootstrap(base: dict, other: dict, games: int, reps: int, seed: int) -> dict:
    out = {}
    rng = np.random.default_rng(seed)
    counts = np.bincount(base["game"], minlength=games)
    samples = [rng.integers(0, games, games) for _ in range(reps)]
    for field in ("nll", "top1"):
        delta = np.bincount(other["game"], other[field], games) - np.bincount(base["game"], base[field], games)
        values = [delta[s].sum() / counts[s].sum() for s in samples]
        out[f"delta_{field}"] = {"mean": float(delta.sum() / counts.sum()),
                                 "ci95": [float(np.quantile(values, 0.025)),
                                          float(np.quantile(values, 0.975))]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True,
                        help="JSONL with h, s, w, b, id and a{sfd<N>...} annotations")
    parser.add_argument("--model", nargs=2, action="append", metavar=("NAME", "CHECKPOINT"),
                        required=True)
    parser.add_argument("--condition", action="append", default=None,
                        help=f"advisor conditions (default: {' '.join(DEFAULT_CONDITIONS)})")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="random/oracle advice seed")
    args = parser.parse_args()

    conditions = [validate_advisor(c) for c in (args.condition or DEFAULT_CONDITIONS)]
    if "none" not in conditions:
        conditions.insert(0, "none")
    device = pick_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    with args.records.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        records = records[:args.limit]
    missing = {c for c in conditions if c.startswith("sf") and c != "sf30k"
               and any(c not in r.get("a", {}) for r in records)}
    if missing:
        raise SystemExit(f"records lack annotations for {sorted(missing)}; run advisor_data.py annotate")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "records": str(args.records), "records_sha256": _sha256(args.records),
        "games": len(records), "conditions": conditions, "seed": args.seed,
        "elo_buckets": [_label(ELO_EDGES, i) for i in range(len(ELO_EDGES) + 1)],
        "ply_buckets": [_label(PLY_EDGES, i) for i in range(len(PLY_EDGES) + 1)],
        "evaluator_sha256": _sha256(Path(__file__)),
        "models": {},
    }
    for name, checkpoint in args.model:
        model = GPT.load(checkpoint, device).eval()
        if model.config.vocab_size != tokens.INTERLEAVED_VOCAB_SIZE:
            raise SystemExit(f"{name}: checkpoint has no advice token namespace")
        entry = {"checkpoint": checkpoint, "checkpoint_sha256": _sha256(Path(checkpoint)),
                 "conditions": {}}
        rows = {}
        for condition in conditions:
            start = time.time()
            rows[condition] = data = evaluate(model, records, condition, args.batch_size,
                                              device, args.seed)
            everything = np.ones(len(data["nll"]), dtype=bool)
            entry["conditions"][condition] = {
                "overall": _summary(data, everything),
                "by_elo": {result["elo_buckets"][i]: _summary(data, data["elo"] == i)
                           for i in range(len(ELO_EDGES) + 1)},
                "by_ply": {result["ply_buckets"][i]: _summary(data, data["ply"] == i)
                           for i in range(len(PLY_EDGES) + 1)},
                "seconds": time.time() - start,
            }
            if condition != "none" and args.bootstrap:
                entry["conditions"][condition]["vs_none"] = _paired_bootstrap(
                    rows["none"], data, len(records), args.bootstrap, seed=20261101)
            overall = entry["conditions"][condition]["overall"]
            print(f"{name:>12} {condition:>10}  nll {overall['nll']:.4f}  top1 {overall['top1']:.4f}"
                  f"  agree {overall['agree'] if overall['agree'] is not None else float('nan'):.3f}"
                  f"  copy {overall['copy'] if overall['copy'] is not None else float('nan'):.3f}"
                  f"  p_hint {overall['p_hint'] if overall['p_hint'] is not None else float('nan'):.3f}",
                  flush=True)
        np.savez_compressed(args.out.with_name(args.out.stem + f".{name}.rows.npz"),
                            **{f"{c}/{k}": v for c, d in rows.items() for k, v in d.items()})
        result["models"][name] = entry
        partial = args.out.with_suffix(args.out.suffix + ".partial")
        partial.write_text(json.dumps(result, indent=2) + "\n")
        partial.replace(args.out)
        del model
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
