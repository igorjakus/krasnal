"""Bounded, paired games. UCI-setting-relative rating, not a human rating.

Fixed sample, no significance-based early stopping. Opening-pair bootstrap;
unfinished games yield rating bounds rather than fabricated draws.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from krasnal.tokenization import reasoning as tokens
from krasnal.models.reasoning import GPT
from eval_foundation_selector import EXPECTED, candidates, hidden, sha, smoke

INTERLEAVE_SHA = "0605bf201b31397e50ed36705d4a5d3b8ae7f86607da1b45493f18b1c1544f33"
MODES = ("foundation_policy", "foundation_value", "sf30k_control", "interleave_sf_assisted")
RATINGS = (1320, 1700, 2100, 2500, 2900)


def atomic(path, value):
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def fit(scores, anchors, weights=None):
    """Fractional-score logistic quasi-MLE. Draw variance comes from bootstrap."""
    scores = np.asarray(scores)
    anchors = np.asarray(anchors)
    if weights is None:
        weights = np.ones_like(scores)
    total = np.sum(scores * weights, axis=-1)
    count = np.sum(weights, axis=-1)
    lo = np.full_like(total, -10000., dtype=float)
    hi = np.full_like(total, 10000., dtype=float)
    for _ in range(55):
        mid = (lo + hi) / 2
        expected = 1 / (1 + 10 ** ((anchors - np.expand_dims(mid, -1)) / 400))
        below = np.sum(expected * weights, axis=-1) < total
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    rating = (lo + hi) / 2
    return np.where(total == 0, -np.inf, np.where(total == count, np.inf, rating))


def endpoint(value):
    return float(value) if np.isfinite(value) else ("+infinity" if value > 0 else "-infinity")


def report(rows, plan):
    result = {"rating_scale": "Equivalent Stockfish 17.1 UCI_Elo under this fixed 20k-node protocol; NOT FIDE/Lichess Elo",
              "ci": "95% percentile bootstrap of opening pairs, 10000 resamples, seed 20261015; conditional on engine/settings/opening population",
              "limitations": ["Stockfish UCI_Elo calibration is not preserved by arbitrary node limits.",
                              "Human validation opening prefixes are not an independent standardized tournament opening book.",
                              "Logistic rating model can misfit engine styles; per-anchor results are primary.",
                              "Interleave is SF-assisted at every ply; not unaided neural strength.",
                              "Unfinished games remain censored; no automatic cutoff draws."], "modes": {}}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        completed = [r for r in selected if r["score"] is not None]
        groups = {}
        per_anchor = {}
        for anchor in RATINGS:
            rr = [r for r in selected if r["anchor"] == anchor]
            per_anchor[str(anchor)] = {"games": len(rr), "wins": sum(r["score"] == 1 for r in rr),
                "draws": sum(r["score"] == .5 for r in rr), "losses": sum(r["score"] == 0 for r in rr),
                "unfinished": sum(r["score"] is None for r in rr)}
        for row in selected:
            groups.setdefault(row["opening_id"], []).append(row)
        full_pairs = [pair for pair in groups.values() if len(pair) == 2 and all(r["score"] is not None for r in pair)]
        entry = {"games": len(selected), "completed": len(completed), "complete_pairs": len(full_pairs), "per_anchor": per_anchor}
        if full_pairs:
            scores = np.array([sum(r["score"] for r in pair)/2 for pair in full_pairs])
            anchors = np.array([pair[0]["anchor"] for pair in full_pairs])
            entry["complete_pair_point_rating"] = endpoint(fit(scores, anchors))
            rng = np.random.default_rng(20261015)
            # Stratify by fixed anchor allocation; preserve paired colours.
            sampled = np.empty((10000, len(scores)), dtype=int)
            for anchor in RATINGS:
                indices = np.flatnonzero(anchors == anchor)
                if len(indices):
                    sampled[:, indices] = rng.choice(indices, size=(10000, len(indices)))
            estimates = np.sort(fit(scores[sampled], anchors[sampled]))
            entry["bootstrap_95ci_approximate"] = [endpoint(estimates[249]), endpoint(estimates[9749])]
            # Unlike a pure empirical bootstrap this also handles zero wins/losses.
            # Each independently sampled opening pair contributes one bounded [0,1] score.
            width = np.sqrt(np.log(2/.05)/(2*len(scores)))
            entry["conservative_95ci_hoeffding_pairs"] = [
                endpoint(fit(np.full_like(scores, max(0, scores.mean()-width)), anchors)),
                endpoint(fit(np.full_like(scores, min(1, scores.mean()+width)), anchors))]
            if len(full_pairs) < 100:
                entry["warning"] = "Incomplete fixed sample; interim CI only, no optional-stopping guarantee."
        if any(r["score"] is None for r in selected):
            entry["censoring_rating_bounds"] = [endpoint(fit([0 if r["score"] is None else r["score"] for r in selected], [r["anchor"] for r in selected])),
                endpoint(fit([1 if r["score"] is None else r["score"] for r in selected], [r["anchor"] for r in selected]))]
            entry["warning"] = "Complete-case estimate can be biased by censoring; prioritize censoring bounds."
        result["modes"][mode] = entry
    result["planned_games"] = plan["planned_games"]
    result["recorded_games"] = len(rows)
    return result


def make_openings(path):
    stream = np.memmap(path, dtype=np.uint16, mode="r")
    starts = np.flatnonzero(stream == tokens.BOS)
    rng = np.random.default_rng(20261015)
    openings, seen = [], set()
    for start in rng.permutation(starts):
        moves = []
        board = chess.Board()
        for token in stream[int(start)+1:int(start)+40]:
            if token == tokens.EOS or token == tokens.BOS:
                break
            if token in tokens.MOVE_IDS:
                move = chess.Move.from_uci(tokens.VOCAB[int(token)])
                if move not in board.legal_moves:
                    break
                board.push(move)
                moves.append(move.uci())
                if len(moves) == 8:
                    key = " ".join(board.fen().split()[:4])
                    if key not in seen and not board.is_game_over(claim_draw=True):
                        seen.add(key)
                        openings.append(moves)
                    break
        if len(openings) == 100:
            return openings
    raise RuntimeError("Not enough distinct legal held-out opening prefixes")


@torch.inference_mode()
def choose(model, mode, board, ids, values):
    legal = list(board.legal_moves)
    move_ids = [tokens.move_id(m.uci()) for m in legal]
    if mode == "foundation_value":
        policy, probs = candidates(model, ids, move_ids)
        scores = (probs * values).sum(-1).cpu().numpy()
        sign = 1 if board.turn == chess.WHITE else -1
        for i, move in enumerate(legal):
            board.push(move)
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                scores[i] = .5 if outcome.winner is None else float(outcome.winner == chess.WHITE)
            board.pop()
        j = max(range(len(legal)), key=lambda i: (sign * scores[i], float(policy[move_ids[i]])))
    else:
        assert len(ids) <= model.config.block_size
        h = hidden(model, torch.tensor([ids], device="cuda"))
        policy = model.lm_head(h[:, -1])[0]
        j = int(policy[move_ids].argmax())
    return legal[j]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--foundation", type=Path, required=True)
    p.add_argument("--interleave", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--continue-from", type=Path)
    p.add_argument("--extension-deadline", type=float)
    args = p.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    assert sha(args.foundation) == EXPECTED
    foundation = GPT.load(args.foundation, "cuda").eval()
    interleave = None
    smoke(foundation)
    assert np.isclose(float(fit([.5, .5], [1700, 1700])), 1700)
    assert np.isneginf(fit([0], [1700])) and np.isposinf(fit([1], [1700]))
    opponent = chess.engine.SimpleEngine.popen_uci(str(args.engine), timeout=20)
    teacher = chess.engine.SimpleEngine.popen_uci(str(args.engine), timeout=20)
    opponent.configure({"Threads": 1, "Hash": 32, "UCI_LimitStrength": True})
    teacher.configure({"Threads": 1, "Hash": 32, "UCI_LimitStrength": False})
    edges = np.array(tokens.CP_EDGES)
    values = torch.sigmoid(torch.tensor(np.r_[-3000., (edges[:-1]+edges[1:])/2, 3000.], device="cuda") * .00368208)
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan["source_sha256"] == sha(__file__)
    elif args.continue_from:
        parent = args.continue_from / "plan.json"
        plan = json.loads(parent.read_text())
        assert plan["foundation_sha256"] == EXPECTED and plan["interleave_sha256"] == INTERLEAVE_SHA
        assert args.extension_deadline is not None and time.time() < args.extension_deadline
        assert args.extension_deadline <= time.time()+45*60
        plan = {**plan, "created": time.time(), "deadline": args.extension_deadline,
                "source_sha256": sha(__file__), "parent_root": str(args.continue_from),
                "parent_plan_sha256": sha(parent), "original_deadline": plan["deadline"],
                "extension_authorization": "User explicitly approved additional45min on2026-10-16; finish only missing200 human-SF games, no repeats",
                "resources": "One RTX3090, immutable45min extension deadline including preparation/smoke; original run untouched"}
        if not args.smoke:
            atomic(plan_path, plan)
    else:
        plan = {"created": time.time(), "deadline": time.time()+6*3600, "planned_games": 800,
                "foundation_sha256": EXPECTED, "interleave_sha256": INTERLEAVE_SHA,
                "interleave_bytes": 1227294188,
                "source_sha256": sha(__file__), "val_sha256": sha(args.val), "engine_sha256": sha(args.engine),
                "engine_id": opponent.id, "modes": MODES, "anchors": RATINGS,
                "openings": make_openings(args.val), "opening_source": "100 seed-selected distinct 8-ply historical human validation prefixes, no eval/strength selection",
                "protocol": "20 opening pairs per anchor/mode; same openings across modes, legal argmax, Elo conditioning2600/2600; FP32; 20k opponent nodes, 30k teacher nodes; max480 plies then censored; game claim_draw=True; no resign/adjudication",
                "interpretation": "Interleave uses SF tokens every ply without effort0, as trained; foundation uses effort0. Interleave eval head unused by FT, so not used here.",
                "resources": "One RTX3090, 2 torch CPU threads plus two single-thread engines; 6h immutable wall-clock deadline"}
        if not args.smoke:
            atomic(plan_path, plan)
    rows_path = root / "games.jsonl"
    if args.continue_from and not rows_path.exists() and not args.smoke:
        shutil.copyfile(args.continue_from / "games.jsonl", rows_path)
        shutil.copyfile(args.continue_from / "games.pgn", root / "games.pgn")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    if args.continue_from and args.smoke:
        rows = [json.loads(line) for line in (args.continue_from / "games.jsonl").read_text().splitlines()]
    if args.continue_from:
        assert len([r for r in rows if r["mode"] != "interleave_sf_assisted"]) == 600
    done = {(r["opening_id"], r["mode"], r["model_white"]) for r in rows}
    try:
        for schedule_id in range(len(MODES)*len(plan["openings"])):
            opening_id = schedule_id % len(plan["openings"])
            mode = MODES[schedule_id // len(plan["openings"])]
            opening = plan["openings"][opening_id]
            if args.smoke and opening_id > 0:
                continue
            if mode == "interleave_sf_assisted" and interleave is None:
                if args.smoke and not args.interleave.exists():
                    print("SMOKE foundation/control passed; interleave pending transfer, verified at first load", flush=True)
                    return
                while not args.interleave.exists():
                    partial = Path(str(args.interleave)+".partial")
                    # Full source checkpoint size pinned before transfer. Never load partial tensors.
                    if partial.exists() and partial.stat().st_size == plan["interleave_bytes"]:
                        assert sha(partial) == INTERLEAVE_SHA
                        partial.replace(args.interleave)
                        break
                    if time.time() >= plan["deadline"]:
                        atomic(root / "complete.json", {"reason": "deadline_waiting_checkpoint", "games": len(rows)})
                        return
                    time.sleep(5)
                assert sha(args.interleave) == INTERLEAVE_SHA
                interleave = GPT.load(args.interleave, "cuda").eval()
                assert interleave.config.vocab_size == tokens.INTERLEAVED_VOCAB_SIZE
            anchor = RATINGS[opening_id % len(RATINGS)]
            for _ in (None,):
                for model_white in (True, False):
                    if (opening_id, mode, model_white) in done:
                        continue
                    if time.time() >= plan["deadline"]:
                        atomic(root / "complete.json", {"reason": "deadline", "time": time.time(), "games": len(rows)})
                        return
                    game_start = time.time()
                    board = chess.Board()
                    ids = [tokens.BOS, tokens.elo_id(2600), tokens.elo_id(2600)]
                    if mode != "interleave_sf_assisted":
                        ids.append(tokens.effort_id(0))
                    game = chess.pgn.Game()
                    game.headers.update({"Event": "Foundation paired benchmark", "White": mode if model_white else f"SF{anchor}",
                                         "Black": f"SF{anchor}" if model_white else mode})
                    node = game
                    teacher_game, opponent_game = object(), object()
                    opponent.configure({"UCI_Elo": anchor})
                    for uci in opening:
                        if mode == "interleave_sf_assisted":
                            suggestion = teacher.play(board, chess.engine.Limit(nodes=30000), game=teacher_game).move
                            ids.append(tokens.sf_move_id(suggestion.uci()))
                        move = chess.Move.from_uci(uci)
                        board.push(move)
                        ids.append(tokens.move_id(uci))
                        node = node.add_variation(move)
                    stop = "natural"
                    while not board.is_game_over(claim_draw=True):
                        if time.time() >= plan["deadline"]:
                            stop = "deadline"
                            break
                        if board.ply() >= 480:
                            stop = "max_ply_censored"
                            break
                        suggestion = None
                        if mode == "interleave_sf_assisted":
                            suggestion = teacher.play(board, chess.engine.Limit(nodes=30000), game=teacher_game).move
                            ids.append(tokens.sf_move_id(suggestion.uci()))
                        if board.turn == model_white:
                            if mode == "sf30k_control":
                                move = teacher.play(board, chess.engine.Limit(nodes=30000), game=teacher_game).move
                            else:
                                model = interleave if mode == "interleave_sf_assisted" else foundation
                                move = choose(model, mode, board, ids, values)
                        else:
                            move = opponent.play(board, chess.engine.Limit(nodes=20000), game=opponent_game).move
                        assert move in board.legal_moves
                        board.push(move)
                        ids.append(tokens.move_id(move.uci()))
                        node = node.add_variation(move)
                        if args.smoke and board.ply() >= 12:
                            stop = "smoke_censored"
                            break
                    outcome = board.outcome(claim_draw=True)
                    score = None if outcome is None else (.5 if outcome.winner is None else float(outcome.winner == model_white))
                    game.headers["Result"] = board.result(claim_draw=True)
                    row = {"opening_id": opening_id, "mode": mode, "anchor": anchor, "model_white": model_white,
                           "score": score, "plies": board.ply(), "termination": outcome.termination.name if outcome else stop,
                           "seconds": time.time()-game_start, "time": time.time(), "final_fen": board.fen()}
                    if args.smoke:
                        print("SMOKE", row, flush=True)
                        if mode == MODES[-1] and not model_white:
                            print("SMOKE all 4 modes both colours legal, teacher history and context checked", flush=True)
                            return
                        continue
                    with (root / "games.pgn").open("a") as handle:
                        handle.write(str(game)+"\n\n")
                    with rows_path.open("a") as handle:
                        handle.write(json.dumps(row)+"\n")
                        handle.flush()
                    rows.append(row)
                    atomic(root / "status.json", {"games": len(rows), "planned": 800, "last": row, "deadline": plan["deadline"]})
                    if len(rows) % 40 == 0 or len(rows) == plan["planned_games"]:
                        atomic(root / "result.json", report(rows, plan))
                    print(json.dumps(row), flush=True)
        atomic(root / "complete.json", {"reason": "fixed_sample_complete", "time": time.time(), "games": len(rows)})
    finally:
        if not args.smoke and rows:
            atomic(root / "result.json", report(rows, plan))
        opponent.quit()
        teacher.quit()


if __name__ == "__main__":
    main()
