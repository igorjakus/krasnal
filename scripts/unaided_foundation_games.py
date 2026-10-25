"""Frozen 2x2 chess benchmark: two checkpoints x policy/value, NO teacher.

The only engine is the opponent. Its move/info is never supplied to choose();
only actually played moves enter the history. Interleave plain-history input is
an explicit out-of-training-format ablation, not a matched causal FT experiment.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from krasnal_think import tokens
from krasnal_think.model import GPT
import foundation_games as fg
from eval_foundation_selector import EXPECTED, sha, smoke

MODES = ("foundation_policy", "foundation_value", "interleave_policy", "interleave_value")


def summary(rows, plan):
    fg.MODES = MODES
    result = fg.report(rows, plan)
    result["external_advice_calls"] = sum(r["external_advice_calls"] for r in rows)
    assert result["external_advice_calls"] == 0
    result["limitations"] = [
        "Stockfish UCI_Elo at20k nodes is not calibrated FIDE/Lichess Elo; CI excludes calibration uncertainty.",
        "Opening prefixes come from historical human validation, not an independent standardized engine opening book.",
        "Logistic rating model can misfit opponent styles. Report per-anchor WDL.",
        "Interleave checkpoint gets plain actual-move history WITHOUT SF tokens and WITHOUT effort0. This is OOD relative to its SWSB training.",
        "Interleave eval head was not supervised during FT, though its shared trunk changed.",
        "No teacher/engine advice. The opponent engine only returns its own moves on opponent turns.",
        "Cutoffs/deadline are censored, never automatic draws. Incomplete samples have provisional CIs."]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--foundation", type=Path, required=True)
    p.add_argument("--interleave", type=Path, required=True)
    p.add_argument("--openings-plan", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    assert sha(args.foundation) == EXPECTED
    assert sha(args.interleave) == fg.INTERLEAVE_SHA
    models = {"foundation": GPT.load(args.foundation, "cuda").eval(),
              "interleave": GPT.load(args.interleave, "cuda").eval()}
    assert all(m.config.eval_classes == 29 for m in models.values())
    assert models["interleave"].config.vocab_size == tokens.INTERLEAVED_VOCAB_SIZE
    smoke(models["foundation"])
    smoke(models["interleave"])
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan["source_sha256"] == sha(__file__)
        assert plan["engine_sha256"] == sha(args.engine)
    else:
        parent = json.loads(args.openings_plan.read_text())
        openings = parent["openings"]
        assert len(openings) == 100
        plan = {"created": time.time(), "deadline": time.time()+2*3600,
                "planned_games": 800, "modes": MODES, "anchors": fg.RATINGS,
                "openings": openings, "openings_plan_sha256": sha(args.openings_plan),
                "foundation_sha256": EXPECTED, "interleave_sha256": fg.INTERLEAVE_SHA,
                "model_configs": {name: asdict(m.config) for name, m in models.items()},
                "engine_sha256": sha(args.engine), "source_sha256": sha(__file__),
                "sources": {str(Path(f).relative_to(Path(__file__).resolve().parents[1])): sha(f)
                            for f in (Path(__file__).resolve(), Path(fg.__file__).resolve(),
                                      Path(__file__).resolve().with_name("eval_foundation_selector.py"),
                                      Path(__file__).resolve().parents[1]/"src/krasnal_think/model.py",
                                      Path(tokens.__file__).resolve())},
                "protocol": "Fresh200games/mode;100fixed8-plyopenings,20pairs/rating,colour swapped; SF17.1 opponent20k nodes Threads1 Hash32MiB; FP32; Elo2600/2600; legal argmax or all-legal child value; max480plies, claim_draw=True; no resign/adjudication.",
                "input": {"foundation": "BOS,Elo2600,Elo2600,effort0,actual moves",
                          "interleave": "BOS,Elo2600,Elo2600,actual moves ONLY; NO SF tokens, NO effort token; OOD ablation"},
                "engine_access": "Single engine, opponent only. No teacher instance and no engine object/info in neural chooser.",
                "resources": "OneRTX3090,2torchthreads,1single-threadopponent; immutable2h wall budget; no training"}
        if not args.smoke:
            fg.atomic(plan_path, plan)
    rows_path = root / "games.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {(r["opening_id"], r["mode"], r["model_white"]) for r in rows}
    edges = np.asarray(tokens.CP_EDGES)
    values = torch.sigmoid(torch.tensor(np.r_[-3000., (edges[:-1]+edges[1:])/2, 3000.], device="cuda")*.00368208)
    opponent = chess.engine.SimpleEngine.popen_uci(str(args.engine), timeout=20)
    opponent.configure({"Threads": 1, "Hash": 32, "UCI_LimitStrength": True})
    print("AUDIT no teacher created; engine is OPPONENT ONLY; inputs exclude all SF tokens", flush=True)
    try:
        for opening_id, opening in enumerate(plan["openings"]):
            if args.smoke and opening_id > 0:
                break
            anchor = fg.RATINGS[opening_id % 5]
            # Rotate order to balance time drift; all modes share each opening.
            rotated = MODES[opening_id % 4:] + MODES[:opening_id % 4]
            for mode in rotated:
                family, method = mode.split("_")
                model = models[family]
                for model_white in (True, False):
                    if (opening_id, mode, model_white) in done:
                        continue
                    if time.time() >= plan["deadline"]:
                        fg.atomic(root / "complete.json", {"reason": "deadline", "games": len(rows), "time": time.time()})
                        return
                    started = time.time()
                    b = chess.Board()
                    ids = [tokens.BOS, tokens.elo_id(2600), tokens.elo_id(2600)]
                    if family == "foundation":
                        ids.append(tokens.effort_id(0))
                    game = chess.pgn.Game()
                    game.headers.update({"Event": "Unaided2x2", "White": mode if model_white else f"SF{anchor}",
                                         "Black": f"SF{anchor}" if model_white else mode})
                    node = game
                    for uci in opening:
                        move = chess.Move.from_uci(uci)
                        assert move in b.legal_moves
                        b.push(move)
                        ids.append(tokens.move_id(uci))
                        node = node.add_variation(move)
                    opponent.configure({"UCI_Elo": anchor})
                    engine_game = object()
                    model_calls = opponent_calls = 0
                    stop = "natural"
                    while not b.is_game_over(claim_draw=True):
                        if time.time() >= plan["deadline"] or b.ply() >= 480:
                            stop = "deadline_censored" if time.time() >= plan["deadline"] else "max_ply_censored"
                            break
                        assert not any(t in tokens.SF_MOVE_IDS for t in ids)
                        assert len(ids) == 3 + (family == "foundation") + b.ply()
                        if b.turn == model_white:
                            # No engine object, scores or advice can cross this call boundary.
                            move = fg.choose(model, "foundation_value" if method == "value" else "foundation_policy", b, ids, values)
                            model_calls += 1
                        else:
                            move = opponent.play(b, chess.engine.Limit(nodes=20000), game=engine_game).move
                            opponent_calls += 1
                        assert move in b.legal_moves
                        b.push(move)
                        ids.append(tokens.move_id(move.uci()))
                        node = node.add_variation(move)
                        if args.smoke and b.ply() >= 12:
                            stop = "smoke_censored"
                            break
                    outcome = b.outcome(claim_draw=True)
                    score = None if outcome is None else (.5 if outcome.winner is None else float(outcome.winner == model_white))
                    game.headers["Result"] = b.result(claim_draw=True)
                    row = {"opening_id": opening_id, "mode": mode, "anchor": anchor, "model_white": model_white,
                           "score": score, "plies": b.ply(), "termination": outcome.termination.name if outcome else stop,
                           "seconds": time.time()-started, "time": time.time(), "final_fen": b.fen(),
                           "model_calls": model_calls, "opponent_calls": opponent_calls, "external_advice_calls": 0,
                           "sf_tokens_in_input": 0}
                    print(json.dumps(row), flush=True)
                    if args.smoke:
                        continue
                    with (root / "games.pgn").open("a") as f:
                        f.write(str(game)+"\n\n")
                    with rows_path.open("a") as f:
                        f.write(json.dumps(row)+"\n")
                        f.flush()
                    rows.append(row)
                    fg.atomic(root / "status.json", {"games": len(rows), "planned": 800, "last": row, "deadline": plan["deadline"]})
                    if len(rows) % 40 == 0:
                        fg.atomic(root / "result.json", summary(rows, plan))
        if args.smoke:
            print("SMOKE8 short games,2checkpoints x2heads x2colours;no teacher and zeroSFtokens; all legal", flush=True)
        else:
            fg.atomic(root / "complete.json", {"reason": "fixed_sample_complete", "games": len(rows), "time": time.time()})
    finally:
        if rows and not args.smoke:
            fg.atomic(root / "result.json", summary(rows, plan))
        opponent.quit()


if __name__ == "__main__":
    main()
