"""Assisted human-SF robustness: depth1/4/8 and training-budget30k hints.

Two distinct engine processes: unrestricted adviser, fixed-budget opponent.
Teacher-only controls receive the same adviser budget at every ply as the model.
No training, no unaided Elo claim, bounded fixed sample with paired openings.
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

from krasnal.tokenization import reasoning as tokens
from krasnal.models.reasoning import GPT
import foundation_games as fg
from eval_foundation_selector import sha

BUDGETS = {"d1": {"depth": 1}, "d4": {"depth": 4}, "d8": {"depth": 8}, "n30k": {"nodes": 30000}}
MODES = tuple(f"{system}_{budget}" for budget in BUDGETS for system in ("human_sf", "teacher"))


def summary(rows, plan):
    fg.MODES = MODES
    result = fg.report(rows, plan)
    for entry in result["modes"].values():
        if entry["games"] == 100 and entry["complete_pairs"] == 50:
            entry.pop("warning", None)
        elif entry["games"]:
            entry["warning"] = "Incomplete fixed100-game sample or censoring: provisional CI; use censoring bounds when present."
    for mode, entry in result["modes"].items():
        rr = [r for r in rows if r["mode"] == mode]
        calls = sum(r["teacher_calls"] for r in rr)
        entry["teacher_calls"] = calls
        entry["teacher_seconds"] = sum(r["teacher_seconds"] for r in rr)
        entry["model_calls"] = sum(r["model_calls"] for r in rr)
        entry["model_seconds"] = sum(r["model_seconds"] for r in rr)
        entry["follow_rate"] = (sum(r["follow_count"] for r in rr)/entry["model_calls"] if entry["model_calls"] else None)
        entry["mean_reached_depth"] = sum(r["depth_sum"] for r in rr)/calls if calls else None
        entry["mean_teacher_nodes"] = sum(r["nodes_sum"] for r in rr)/calls if calls else None
    result["limitations"] = [
        "ALL human-SF systems are engine-assisted. Teacher-only controls isolate the adviser strength baseline, not neural standalone Elo.",
        "Changing teacher depth is a quality/compute shift relative to training30k nodes; actual reached depth/nodes and time are recorded.",
        "Depth and node limits are different budgets; do not label30k as a fixed depth or assume monotonic game strength from nominal depth alone.",
        "UCI_Elo with20k opponent nodes is only a protocol-relative scale, not calibrated FIDE/Lichess; bootstrap/Hoeffding exclude calibration and logistic-model error.",
        "50 opening pairs per mode is a pilot,100games; finite CIs can still be wide and extrema/extrapolation unreliable.",
        "Historical human validation opening book is not an independent standardized engine book. No testing-based tuning or optional-stopping significance claim.",
        "Cutoff/deadline games are censored, never counted as draws. Incomplete samples remain provisional."]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--foundation", type=Path, required=True)
    p.add_argument("--openings-plan", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    assert sha(args.checkpoint) == fg.INTERLEAVE_SHA
    assert sha(args.foundation) == fg.EXPECTED
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    model = GPT.load(args.checkpoint, "cuda").eval()
    assert model.config.vocab_size == tokens.INTERLEAVED_VOCAB_SIZE
    plan_path = root/"plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan["source_sha256"] == sha(__file__) and plan["engine_sha256"] == sha(args.engine)
    else:
        parent = json.loads(args.openings_plan.read_text())
        assert len(parent["openings"]) == 100
        rng = np.random.default_rng(20261023)
        indices = sorted(int(i) for anchor in range(5) for i in rng.choice(np.arange(anchor,100,5),10,replace=False))
        code = Path(__file__).resolve().parents[1]
        sources = [Path(__file__).resolve(), Path(fg.__file__).resolve(), code/"scripts/eval_foundation_selector.py", code/"src/krasnal/models/reasoning.py", Path(tokens.__file__).resolve()]
        plan = {"created": time.time(), "deadline": time.time()+7200, "planned_games": 800,
                "checkpoint_sha256": fg.INTERLEAVE_SHA, "foundation_sha256": fg.EXPECTED,
                "model_config": asdict(model.config), "source_sha256": sha(__file__),
                "sources": {str(f.relative_to(code)): sha(f) for f in sources},
                "engine_sha256": sha(args.engine), "openings_plan_sha256": sha(args.openings_plan),
                "opening_ids": indices, "openings": parent["openings"], "modes": MODES, "budgets": BUDGETS,
                "protocol": "Fresh100games/mode,50sameopenings/2colours,10pairs/rating1320/1700/2100/2500/2900. HumanSFlegalpolicyargmax,FP32,Elo2600/2600,noeffort. Adviser at EVERY actual-ply prefix including opening/opponent moves; no future history. Teacher-only control computes adviser at all prefixes too, plays its move directly on its turns. Opponent SF20knodes,Threads1 Hash32; adviser unrestrictedSkill20,Threads1 Hash16. No evalhead experiment or training. max480plies,claim_draw=True,no resign/adjudication,censored cutoffs. Immutable2h wallclock deadline.",
                "resources": "OneRTX3090,2torchthreads,2single-threadengines; detached; no other jobs modified"}
        if not args.smoke:
            fg.atomic(plan_path, plan)
    rows_path = root/"games.jsonl"
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {(r["mode"],r["opening_id"],r["model_white"]) for r in rows}
    adviser = chess.engine.SimpleEngine.popen_uci(str(args.engine), timeout=20)
    opponent = chess.engine.SimpleEngine.popen_uci(str(args.engine), timeout=20)
    adviser.configure({"Threads":1,"Hash":16,"UCI_LimitStrength":False,"Skill Level":20})
    opponent.configure({"Threads":1,"Hash":32,"UCI_LimitStrength":True})
    print("AUDIT ASSISTED humanSF; separate adviser/opponent; control receives same adviser budget; only hint MOVE token enters model",flush=True)
    try:
        for ordinal, opening_id in enumerate(plan["opening_ids"]):
            if args.smoke and ordinal:
                break
            rotated = MODES[ordinal%8:] + MODES[:ordinal%8]
            for mode in rotated:
                budget_name = mode.rsplit("_",1)[1]
                system = "human_sf" if mode.startswith("human_sf_") else "teacher"
                budget = BUDGETS[budget_name]
                for white in (True,False):
                    if (mode,opening_id,white) in done:
                        continue
                    if time.time() >= plan["deadline"]:
                        fg.atomic(root/"complete.json",{"reason":"deadline","games":len(rows),"time":time.time()})
                        return
                    start = time.time()
                    b = chess.Board()
                    ids = [tokens.BOS,tokens.elo_id(2600),tokens.elo_id(2600)]
                    game = chess.pgn.Game()
                    anchor = fg.RATINGS[opening_id%5]
                    game.headers.update({"Event":"Assisted teacher-depth robustness","White":mode if white else f"SF{anchor}","Black":f"SF{anchor}" if white else mode})
                    node = game
                    adviser_game,opponent_game = object(),object()
                    opponent.configure({"UCI_Elo":anchor})
                    teacher_calls=depth_sum=nodes_sum=model_calls=opponent_calls=follow_count=0
                    teacher_seconds=model_seconds=0.
                    traces=[]

                    def hint():
                        nonlocal teacher_calls,depth_sum,nodes_sum,teacher_seconds
                        t=time.monotonic()
                        result=adviser.play(b,chess.engine.Limit(**budget),game=adviser_game,info=chess.engine.INFO_BASIC|chess.engine.INFO_PV)
                        teacher_seconds+=time.monotonic()-t
                        move=result.move
                        assert move in b.legal_moves
                        depth=int(result.info.get("depth",0)); nodes=int(result.info.get("nodes",0))
                        teacher_calls+=1;depth_sum+=depth;nodes_sum+=nodes
                        traces.append({"ply":b.ply(),"fen":b.fen(),"hint":move.uci(),"depth":depth,"nodes":nodes})
                        if "depth" in budget:
                            assert depth <= budget["depth"],(budget,depth)
                        return move

                    for uci in plan["openings"][opening_id]:
                        suggestion=hint()
                        if system=="human_sf": ids.append(tokens.sf_move_id(suggestion.uci()))
                        move=chess.Move.from_uci(uci);assert move in b.legal_moves
                        b.push(move);node=node.add_variation(move)
                        if system=="human_sf": ids.append(tokens.move_id(uci))
                    stop="natural"
                    while not b.is_game_over(claim_draw=True):
                        if time.time()>=plan["deadline"] or b.ply()>=480:
                            stop="deadline_censored" if time.time()>=plan["deadline"] else "max_ply_censored"
                            break
                        suggestion=hint()
                        if system=="human_sf":
                            ids.append(tokens.sf_move_id(suggestion.uci()))
                            assert len(ids)==4+2*b.ply()
                        if b.turn==white:
                            if system=="human_sf":
                                t=time.monotonic()
                                move=fg.choose(model,"foundation_policy",b,ids,None)
                                model_seconds+=time.monotonic()-t;model_calls+=1
                            else:
                                move=suggestion
                            follow_count+=int(move==suggestion)
                        else:
                            move=opponent.play(b,chess.engine.Limit(nodes=20000),game=opponent_game).move
                            opponent_calls+=1
                        assert move in b.legal_moves
                        b.push(move);node=node.add_variation(move)
                        if system=="human_sf":ids.append(tokens.move_id(move.uci()))
                        if args.smoke and b.ply()>=12:
                            stop="smoke_censored";break
                    outcome=b.outcome(claim_draw=True)
                    score=None if outcome is None else (.5 if outcome.winner is None else float(outcome.winner==white))
                    row={"mode":mode,"system":system,"budget":budget,"opening_id":opening_id,"anchor":anchor,"model_white":white,
                         "score":score,"plies":b.ply(),"final_fen":b.fen(),"termination":outcome.termination.name if outcome else stop,
                         "teacher_calls":teacher_calls,"depth_sum":depth_sum,"nodes_sum":nodes_sum,"teacher_seconds":teacher_seconds,
                         "model_calls":model_calls,"model_seconds":model_seconds,"opponent_calls":opponent_calls,"follow_count":follow_count,
                         "seconds":time.time()-start,"time":time.time(),"assisted":True}
                    print(json.dumps(row),flush=True)
                    if args.smoke:continue
                    game.headers["Result"]=b.result(claim_draw=True)
                    with (root/"games.pgn").open("a") as f:f.write(str(game)+"\n\n")
                    with rows_path.open("a") as f:f.write(json.dumps(row)+"\n")
                    with (root/"advice.jsonl").open("a") as f:
                        for trace in traces:f.write(json.dumps({"mode":mode,"opening_id":opening_id,"model_white":white,**trace})+"\n")
                    rows.append(row)
                    fg.atomic(root/"status.json",{"games":len(rows),"planned":800,"last":row,"deadline":plan["deadline"]})
                    if len(rows)%80==0:fg.atomic(root/"result.json",summary(rows,plan))
        if args.smoke:print("SMOKE16shortgames=4adviserbudgets x2systems x2colours,legal,exactprefixorder,depthlimits verified",flush=True)
        else:fg.atomic(root/"complete.json",{"reason":"fixed_sample_complete","games":len(rows),"time":time.time()})
    finally:
        if rows and not args.smoke:fg.atomic(root/"result.json",summary(rows,plan))
        adviser.quit();opponent.quit()


if __name__=="__main__":main()
