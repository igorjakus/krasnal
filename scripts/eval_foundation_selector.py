"""All-legal one-ply value-head selection on the registered PGN foundation.

No training, no true evals in inputs, no rollout tree. Shared prefix KV cache,
then one candidate token per continuation. Eval targets are White-relative.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import chess
import numpy as np
import torch

from krasnal.tokenization import reasoning as tokens
from krasnal.models.reasoning import GPT, KVCache

EXPECTED = "b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def hidden(model, ids, cache=None):
    x = model.drop(model.wte(ids))
    for block in model.blocks:
        x = block(x, cache)
    if cache is not None:
        cache.advance(ids.shape[1])
    if model.config.hyper_connections > 1:
        x = x.mean(dim=-1)
    return model.norm_f(x)


@torch.inference_mode()
def candidates(model, ids, move_ids, chunk=32):
    device = next(model.parameters()).device
    length = len(ids)
    assert length + 1 <= model.config.block_size, "Do not silently truncate prefix"
    cfg = replace(model.config, block_size=length+1)
    base = KVCache(cfg, 1, device, torch.float32)
    h = hidden(model, torch.tensor([ids], device=device), base)
    policy = model.lm_head(h[:, -1])[0]
    values = []
    for start in range(0, len(move_ids), chunk):
        moves = move_ids[start:start+chunk]
        cache = KVCache(cfg, len(moves), device, torch.float32)
        cache.k[:, :, :, :length] = base.k[:, :, :, :length].expand(-1, len(moves), -1, -1, -1)
        cache.v[:, :, :, :length] = base.v[:, :, :, :length].expand(-1, len(moves), -1, -1, -1)
        cache.length = length
        h = hidden(model, torch.tensor(moves, device=device)[:, None], cache)
        values.append(model.eval_head(h[:, -1]).float().softmax(-1))
    return policy, torch.cat(values)


def smoke(model):
    device = next(model.parameters()).device
    ids = [tokens.BOS, tokens.elo_id(2600), tokens.elo_id(2600), tokens.effort_id(0),
           tokens.move_id("e2e4"), tokens.move_id("e7e5")]
    moves = [tokens.move_id(m) for m in ("g1f3", "f1c4", "b1c3")]
    with torch.inference_mode():
        policy, probs = candidates(model, ids, moves, chunk=2)
        full = torch.tensor([ids+[m] for m in moves], device=device)
        h = hidden(model, full)
        reference = model.eval_head(h[:, -1]).float().softmax(-1)
        delta = float((probs-reference).abs().max())
        assert torch.allclose(probs, reference, atol=1e-5, rtol=1e-4), delta
        direct, _ = model(torch.tensor([ids], device=device))
        assert torch.allclose(policy, direct[0, -1], atol=1e-4, rtol=1e-4)
    # White-relative scores: Black must minimize, not maximize.
    example = np.asarray([.2, .8])
    assert int(example.argmax()) == 1 and int(example.argmin()) == 0
    print("SMOKE cached/uncached eval and policy match; max probability delta", delta, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument("--raw-eval", action="store_true", help="Do not override terminal scores")
    args = parser.parse_args()
    assert not args.out.exists(), "Do not overwrite another evaluation"
    assert sha(args.checkpoint) == EXPECTED
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    model = GPT.load(args.checkpoint, "cuda").eval()
    assert model.config.eval_classes == 29
    smoke(model)
    edges = np.asarray(tokens.CP_EDGES, dtype=np.float32)
    # Frozen rule before evaluation: outer CP bins represented by +/-3000cp.
    cp = torch.tensor(np.r_[-3000., (edges[:-1]+edges[1:])/2, 3000.], device="cuda")
    value = torch.sigmoid(cp*.00368208)
    assert len(cp) == len(value) == 29
    rows = [json.loads(line) for line in args.gold.read_text().splitlines()][:args.positions]
    records, elapsed = [], []
    started = time.time()
    for i, row in enumerate(rows):
        b = chess.Board()
        for uci in row["h"].split():
            b.push_uci(uci)
        assert not b.is_game_over(claim_draw=True)
        legal = list(b.legal_moves)
        ids = [tokens.BOS, tokens.elo_id(2600), tokens.elo_id(2600), tokens.effort_id(0)]
        ids += [tokens.move_id(uci) for uci in row["h"].split()]
        move_ids = [tokens.move_id(m.uci()) for m in legal]
        torch.cuda.synchronize()
        start = time.monotonic()
        policy, probs = candidates(model, ids, move_ids)
        scores_cp = (probs*cp).sum(-1).cpu().numpy()
        scores_value = (probs*value).sum(-1).cpu().numpy()
        torch.cuda.synchronize()
        elapsed.append(time.monotonic()-start)
        policy_scores = policy[move_ids].cpu().numpy()
        terminal = []
        for j, move in enumerate(legal):
            b.push(move)
            outcome = b.outcome(claim_draw=True)
            if outcome is not None:
                white = .5 if outcome.winner is None else float(outcome.winner == chess.WHITE)
                if not args.raw_eval:
                    scores_value[j] = white
                    scores_cp[j] = 0 if outcome.winner is None else (1e9 if white == 1 else -1e9)
                terminal.append(move.uci())
            b.pop()
        sign = 1 if b.turn == chess.WHITE else -1
        # Stable policy tie-break, no SF labels involved.
        cp_best = max(range(len(legal)), key=lambda j: (sign*scores_cp[j], policy_scores[j]))
        val_best = max(range(len(legal)), key=lambda j: (sign*scores_value[j], policy_scores[j]))
        policy_best = int(policy_scores.argmax())
        chosen = {"policy": legal[policy_best].uci(), "eval_expected_cp": legal[cp_best].uci(),
                  "eval_expected_logistic": legal[val_best].uci()}
        gold = row["v"][0].split()[0]
        gold_index = [m.uci() for m in legal].index(gold)
        rank_cp = sorted(range(len(legal)), key=lambda j: (sign*scores_cp[j], policy_scores[j]), reverse=True)
        records.append({"index": i, "history": row["h"], "gold_move": gold,
                        "selected": chosen, "matches_gold": {k: v == gold for k, v in chosen.items()},
                        "gold_eval_cp_rank": rank_cp.index(gold_index)+1, "legal_count": len(legal),
                        "white_to_move": b.turn == chess.WHITE, "terminal_candidates": terminal,
                        "eval_cp_choice_policy_rank": int((policy_scores > policy_scores[cp_best]).sum())+1})
        if (i+1) % 100 == 0:
            print("positions", i+1, "seconds", time.time()-started, flush=True)
    methods = ("policy", "eval_expected_cp", "eval_expected_logistic")
    n = len(records)
    summary = {method: {"sf30k_top1": sum(r["matches_gold"][method] for r in records)/n,
                        "matches": sum(r["matches_gold"][method] for r in records), "positions": n}
               for method in methods}
    for method in methods[1:]:
        summary[method]["paired_vs_policy"] = {
            "improved": sum(r["matches_gold"][method] and not r["matches_gold"]["policy"] for r in records),
            "worsened": sum(not r["matches_gold"][method] and r["matches_gold"]["policy"] for r in records),
            "same_choice": sum(r["selected"][method] == r["selected"]["policy"] for r in records)}
    result = {"checkpoint_sha256": EXPECTED, "gold_sha256": sha(args.gold), "source_sha256": sha(__file__),
              "input": "BOS,Elo2600,Elo2600,effort0,complete UCI prefix,candidate token; no eval inputs",
              "score": "White-relative expected CP, bin midpoints, outer representatives +/-3000cp; Black minimizes",
              "secondary_score": "Expected logistic-transformed representative CP; not calibrated WDL",
              "terminal_rules": ("No terminal override in any method" if args.raw_eval else
                                 "Value selectors use exact terminal outcomes; raw policy baseline has no terminal override"),
              "summary": summary, "mean_all_candidates_ms": float(np.mean(elapsed))*1000,
              "gold_cp_rank_mean": float(np.mean([r["gold_eval_cp_rank"] for r in records])),
              "seconds": time.time()-started, "records": records,
              "limitations": "Exact frozen SF top1, not Elo or all-move quality. No training. One-ply, not minimax. Both methods fixed in advance, no score tuning on eval."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(args.out)+".partial")
    temp.write_text(json.dumps(result, indent=2)+"\n")
    temp.replace(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
