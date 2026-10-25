"""Streaming e2e replay of saved PGNs and adviser traces; no engine/GPU calls."""
import argparse
import hashlib
import json
from pathlib import Path
import chess
import chess.pgn


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--limit",type=int)
    p.add_argument("--out",type=Path)
    args=p.parse_args();root=args.root
    plan=json.loads((root/"plan.json").read_text())
    rows=[json.loads(x) for x in (root/"games.jsonl").read_text().splitlines()]
    if args.limit:rows=rows[:args.limit]
    assert len({(r["mode"],r["opening_id"],r["model_white"]) for r in rows})==len(rows)
    hints=0;censored=0
    with (root/"games.pgn").open() as pg, (root/"advice.jsonl").open() as adv:
        for r in rows:
            g=chess.pgn.read_game(pg);assert g is not None and not g.errors
            b=g.board();moves=list(g.mainline_moves())
            assert [m.uci() for m in moves[:8]]==plan["openings"][r["opening_id"]]
            assert r["anchor"]==[1320,1700,2100,2500,2900][r["opening_id"]%5]
            assert (g.headers["White"]==r["mode"])==r["model_white"]
            assert len(moves)==r["plies"]==r["teacher_calls"]
            follow=0;models=0;opponents=0;depth_sum=0;nodes_sum=0
            for ply,move in enumerate(moves):
                line=adv.readline();assert line,"Missing adviser trace"
                t=json.loads(line)
                assert (t["mode"],t["opening_id"],t["model_white"])==(r["mode"],r["opening_id"],r["model_white"])
                assert t["ply"]==ply and t["fen"]==b.fen()
                hint=chess.Move.from_uci(t["hint"])
                assert hint in b.legal_moves and move in b.legal_moves
                if "depth" in r["budget"]:assert t["depth"]<=r["budget"]["depth"]
                depth_sum+=t["depth"];nodes_sum+=t["nodes"];hints+=1
                if ply>=8:
                    if b.turn==r["model_white"]:
                        follow+=int(move==hint)
                        if r["system"]=="human_sf":models+=1
                        else:assert move==hint,"Teacher-only control did not play adviser move"
                    else:opponents+=1
                b.push(move)
            assert b.fen()==r["final_fen"] and follow==r["follow_count"]
            assert (models,opponents)==(r["model_calls"],r["opponent_calls"])
            assert (depth_sum,nodes_sum)==(r["depth_sum"],r["nodes_sum"])
            o=b.outcome(claim_draw=True)
            if o is None:
                assert r["score"] is None;censored+=1
            else:
                score=.5 if o.winner is None else float(o.winner==r["model_white"])
                assert score==r["score"] and o.termination.name==r["termination"]
        if not args.limit:
            assert chess.pgn.read_game(pg) is None and not adv.read().strip()
    result={"games_verified":len(rows),"adviser_prefixes_verified":hints,"censored":censored,
            "checks":"PGN legality,outcome,FEN,plies,opening/anchor/colour; adviser trace matches exact pre-move board;legalhint and depthcap;teacher-control actually plays hint;follow/role/node/depth counters",
            "scope":"prefix" if args.limit else "full_saved_data","source_sha256":sha(Path(__file__)),
            "counts":{mode:sum(r["mode"]==mode for r in rows) for mode in plan["modes"]}}
    if not args.limit:
        result["files"]={name:sha(root/name) for name in ("games.jsonl","games.pgn","advice.jsonl","plan.json")}
    if args.out:
        tmp=Path(str(args.out)+".partial");tmp.write_text(json.dumps(result,indent=2)+"\n");tmp.replace(args.out)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
