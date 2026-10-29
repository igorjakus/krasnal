"""Bounded Mateusz/Lichess SV baseline, pinned data, resumable GPU training.

Imports the unchanged pilot's architecture/objective/decision policy so this
run changes data/exposure budget, not several model choices simultaneously.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import chess
import chess.engine
import chess.pgn
import duckdb
import numpy as np
import torch

from compare_sv_data import (
    FOUNDATION_SHA, PIECES, SV, atomic_json, bag_records, choose, cp_value,
    decision_eval, encode, evaluate, key, sha, targets,
)

OPENINGS = (
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5",
    "g1f3 d7d5 d2d4 g8f6 c2c4 e7e6",
)


def digest(fen):
    return hashlib.sha256(fen.encode()).digest()[:16]


def encode_key(fen):
    """Fast encode of an already canonical key, identical to pilot encode."""
    board, side, rights, ep = fen.split()
    squares = []
    for c in board.replace("/", ""):
        squares.extend([0] * int(c) if c.isdigit() else [PIECES.index(c)])
    assert len(squares) == 64
    return squares + [13 + (side == "b")] + [15 + int(c in rights) for c in "KQkq"] + [
        17 if ep == "-" else 18 + chess.parse_square(ep), 82]


def decode_key(tokens):
    b = chess.Board(None)
    for i, token in enumerate(tokens[:64]):
        if token:
            b.set_piece_at(chess.square(i % 8, 7 - i // 8), chess.Piece.from_symbol(PIECES[token]))
    b.turn = int(tokens[64]) == 13
    b.set_castling_fen("".join(c for c, t in zip("KQkq", tokens[65:69]) if t == 16) or "-")
    b.ep_square = None if tokens[69] == 17 else int(tokens[69]) - 18
    return key(b.fen())


def exclusions(cfg):
    pilot = Path(cfg["pilot"])
    excluded = {digest(f) for f, _ in bag_records(pilot / "raw/test.bag")}
    for source in ("mateusz", "chessbench"):
        x = np.load(pilot / f"eval-{source}.npz")["x"]
        for tokens in x:
            excluded.add(digest(decode_key(tokens)))
    for row in map(json.loads, Path(cfg["gold"]).read_text().splitlines()):
        b = chess.Board()
        excluded.add(digest(key(b.fen())))
        for uci in row["h"].split():
            b.push_uci(uci)
            excluded.add(digest(key(b.fen())))
        for move in list(b.legal_moves):
            b.push(move)
            excluded.add(digest(key(b.fen())))
            b.pop()
    # Fixed game opening positions must not become training examples either.
    for opening in OPENINGS:
        b = chess.Board()
        for uci in opening.split():
            b.push_uci(uci)
            excluded.add(digest(key(b.fen())))
    return excluded


def prepare(root, cfg, deadline):
    manifest_path = root / "data-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for row in manifest["shards"]:
            for name in ("x", "y"):
                assert sha(root / row[name]) == row[name + "_sha256"]
        return manifest
    excluded = exclusions(cfg)
    seen = set()
    pilot = Path(cfg["pilot"])
    upstream = json.loads((pilot / "data-manifest.json").read_text())
    assert upstream["mateusz_revision"] == cfg["hf_revision"]
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='2GB'")
    con.execute(f"SET temp_directory='{root}/duckdb-tmp'")
    per_part = cfg["train_records"] // 10
    assert per_part * 10 == cfg["train_records"]
    records, depths, audit = [], np.zeros(256, dtype=np.int64), []
    for part, source in enumerate(upstream["mateusz_sources"]):
        if time.time() >= deadline:
            raise TimeoutError("Global deadline during preparation")
        path = pilot / "raw" / source["path"]
        assert sha(path) == source["sha256"], "Source differs from pinned HF snapshot"
        sample = math.ceil(per_part * 1.1)
        query = (f"SELECT fen, cp, mate, depth FROM read_parquet('{path}') "
                 f"USING SAMPLE reservoir({sample} ROWS) REPEATABLE ({cfg['seed'] + part}) "
                 "ORDER BY hash(fen)")
        cursor = con.execute(query)
        xp, yp = root / f"train-x-{part:02d}.npy.partial", root / f"train-y-{part:02d}.npy.partial"
        x = np.lib.format.open_memmap(xp, mode="w+", dtype=np.uint8, shape=(per_part, 71))
        y = np.lib.format.open_memmap(yp, mode="w+", dtype=np.float32, shape=(per_part,))
        count, dropped = 0, 0
        while count < per_part:
            if time.time() >= deadline:
                raise TimeoutError("Global deadline during preparation")
            batch = cursor.fetchmany(8192)
            if not batch:
                raise RuntimeError(f"Insufficient unique eligible records in source shard {part}")
            for fen, cp, mate, depth in batch:
                canonical = key(fen)
                h = digest(canonical)
                if h in excluded or h in seen or (cp is None and mate is None):
                    dropped += 1
                    continue
                if mate == 0:
                    # Sign of mate=0 is lost; infer the winning side from terminal board.
                    b = chess.Board(canonical + " 0 1")
                    if not b.is_checkmate():
                        dropped += 1
                        continue
                    value = 0.0
                else:
                    value = cp_value(cp, mate, canonical.split()[1] == "w")
                seen.add(h)
                x[count] = encode_key(canonical)
                y[count] = value
                depths[min(255, max(0, int(depth)))] += 1
                if len(audit) < 64 and cp is not None and abs(cp) >= 300:
                    audit.append({"fen": canonical, "cp_white": cp, "target_stm": value})
                count += 1
                if count % 100000 == 0:
                    state = {"phase": "preparing", "shard": part, "records": part*per_part + count}
                    atomic_json(root / "status.json", state)
                    print(json.dumps(state), flush=True)
                if count == per_part:
                    break
        x.flush()
        y.flush()
        del x, y
        xf, yf = root / f"train-x-{part:02d}.npy", root / f"train-y-{part:02d}.npy"
        os.replace(xp, xf)
        os.replace(yp, yf)
        records.append({"x": xf.name, "y": yf.name, "x_sha256": sha(xf), "y_sha256": sha(yf),
                        "records": count, "filtered": dropped, "source": source})
    assert len(seen) == cfg["train_records"] and not seen & excluded
    manifest = {"shards": records, "records": len(seen), "excluded_hashes": len(excluded),
                "hash_identity": "SHA256 canonical legal-EP four-field FEN, first 128 bits",
                "train_eval_overlap": 0, "depth_histogram": depths.tolist(),
                "source_revision": upstream["mateusz_revision"],
                "pilot_manifest_sha256": sha(pilot / "data-manifest.json"),
                "gold_sha256": sha(cfg["gold"]), "perspective_audit_positions": audit,
                "validation_sha256": sha(pilot / "eval-mateusz.npz"),
                "cb_test_sha256": sha(pilot / "eval-chessbench.npz")}
    atomic_json(manifest_path, manifest)
    return manifest


def save_checkpoint(path, model, opt, step, rng, order, offset, elapsed):
    tmp = Path(str(path) + ".partial")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                "rng": rng.get_state(), "order": order.cpu(), "offset": offset,
                "train_phase_seconds": elapsed}, tmp)
    os.replace(tmp, path)


def render(root, cfg, status, curves):
    lines = ["# Mateusz/Lichess SV baseline", "", f"Status: {status}.", "",
             "10.7M parameters, scratch initialization; 20M available training positions.",
             "Validation selects best checkpoint; SF30k is diagnostic, not a selection objective.", "",
             "| Step | Exposures | Validation MAE | CB test MAE | SF30k top1 | Train phase seconds |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in curves:
        sf = row.get("sf30k", {}).get("sf30k_top1")
        lines.append(f"| {row['step']} | {row['step']*cfg['batch_size']} | "
                     f"{row['validation']['expected_value_mae']:.5f} | "
                     f"{row['chessbench']['expected_value_mae']:.5f} | "
                     f"{sf if sf is not None else '—'} | {row['seconds']:.1f} |")
    lines += ["", "Single seed. No absolute Elo inferred from top1, loss or a bounded game screen."]
    tmp = root / "report.md.partial"
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, root / "report.md")


def train(root, cfg, deadline, manifest):
    device = "cuda"
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    model = SV().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(.9, .95),
                           weight_decay=.1, fused=True)
    x = torch.empty((manifest["records"], 71), device=device, dtype=torch.uint8)
    y = torch.empty(manifest["records"], device=device, dtype=torch.float32)
    pos = 0
    for shard in manifest["shards"]:
        n = shard["records"]
        # Arrays remain byte-sized on GPU; cast only selected batches to int64.
        x[pos:pos+n].copy_(torch.from_numpy(np.load(root / shard["x"])))
        y[pos:pos+n].copy_(torch.from_numpy(np.load(root / shard["y"])))
        pos += n
    rng = torch.Generator(device=device).manual_seed(cfg["seed"])
    order = torch.randperm(len(y), generator=rng, device=device)
    offset, first_step, previous_seconds = 0, 0, 0.
    latest = root / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        first_step = state["step"]
        rng.set_state(state["rng"].cpu())
        order, offset = state["order"].to(device), state["offset"]
        previous_seconds = state["train_phase_seconds"]
    evals = {name: np.load(Path(cfg["pilot"]) / f"eval-{name}.npz")
             for name in ("mateusz", "chessbench")}
    curve_path = root / "curve.json"
    curves = json.loads(curve_path.read_text()) if curve_path.exists() else []
    best = min((r["validation"]["expected_value_mae"] for r in curves), default=float("inf"))
    start = time.time()
    phase_deadline = min(deadline - cfg["games_seconds"] - 120,
                         start + max(0, cfg["train_seconds"] - previous_seconds))
    step = first_step

    def snapshot(full):
        nonlocal best
        row = {"step": step, "seconds": previous_seconds + time.time()-start,
               "validation": evaluate(model, evals["mateusz"], device),
               "chessbench": evaluate(model, evals["chessbench"], device)}
        if full and time.time() < deadline:
            row["sf30k"] = decision_eval(model, cfg, device, deadline)
        model.train()
        value = row["validation"]["expected_value_mae"]
        if value < best:
            best = value
            torch.save({"model": model.state_dict(), "step": step,
                        "selection": "minimum Mateusz validation value MAE", "validation_mae": best},
                       root / "best.pt.partial")
            os.replace(root / "best.pt.partial", root / "best.pt")
        curves.append(row)
        atomic_json(curve_path, curves)
        render(root, cfg, "training", curves)
        print("EVAL", json.dumps(row), flush=True)

    if not curves:
        snapshot(False)
    while step < cfg["updates"] and time.time() < phase_deadline:
        if offset + cfg["batch_size"] > len(y):
            order = torch.randperm(len(y), generator=rng, device=device)
            offset = 0
        idx = order[offset:offset+cfg["batch_size"]]
        offset += cfg["batch_size"]
        lr = cfg["lr"] * min(1., (step+1)/cfg["warmup"])
        if step >= cfg["warmup"]:
            progress = (step-cfg["warmup"])/(cfg["updates"]-cfg["warmup"])
            lr = cfg["lr"] * (.1 + .9*.5*(1+math.cos(math.pi*progress)))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[idx].long()).float()
            loss = -(targets(y[idx])*logits.log_softmax(-1)).sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss; preserve last checkpoint")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        step += 1
        elapsed = previous_seconds + time.time()-start
        if step % 100 == 0:
            state = {"phase": "training", "step": step, "exposures": step*cfg["batch_size"],
                     "loss": loss.item(), "lr": lr, "train_phase_seconds": elapsed,
                     "deadline": deadline}
            atomic_json(root / "status.json", state)
            with open(root / "updates.jsonl", "a") as f:
                f.write(json.dumps(state)+"\n")
            print(json.dumps(state), flush=True)
        if step % cfg["save_every"] == 0:
            save_checkpoint(latest, model, opt, step, rng, order, offset, elapsed)
            # Preserve fixed intermediate points for future matched-compute comparisons.
            if step in cfg["keep_steps"]:
                torch.save({"model": model.state_dict(), "step": step}, root / f"step-{step:06d}.pt")
        if step % cfg["eval_every"] == 0:
            snapshot(step in cfg["decision_steps"])
    elapsed = previous_seconds + time.time()-start
    save_checkpoint(latest, model, opt, step, rng, order, offset, elapsed)
    if not curves or curves[-1]["step"] != step:
        snapshot(False)
    model.load_state_dict(torch.load(root / "best.pt", map_location=device, weights_only=False)["model"])
    final = {"steps": step, "exposures": step*cfg["batch_size"],
             "completed_updates": step == cfg["updates"], "train_phase_seconds": elapsed,
             "parameters": sum(p.numel() for p in model.parameters()),
             "best_checkpoint_sha256": sha(root / "best.pt"), "latest_checkpoint_sha256": sha(latest),
             "validation": evaluate(model, evals["mateusz"], device),
             "chessbench": evaluate(model, evals["chessbench"], device),
             "sf30k": decision_eval(model, cfg, device, deadline)}
    atomic_json(root / "result.json", final)
    return model.eval(), final, curves


@torch.no_grad()
def games(root, cfg, model, deadline):
    end = min(deadline, time.time()+cfg["games_seconds"])
    pilot = SV().cuda().eval()
    pilot_state = Path(cfg["pilot"]) / "mateusz-model.pt"
    assert sha(pilot_state) == cfg["pilot_checkpoint_sha256"]
    pilot.load_state_dict(torch.load(pilot_state, map_location="cuda", weights_only=False)["model"])
    engine = chess.engine.SimpleEngine.popen_uci(cfg["stockfish"])
    engine.configure({"Threads": 1, "Hash": 32, "UCI_LimitStrength": True, "UCI_Elo": 1600})
    state = {"pilot_checkpoint_sha256": sha(pilot_state), "stockfish_sha256": sha(cfg["stockfish"]),
             "stockfish_options": {"UCI_Elo": 1600, "UCI_LimitStrength": True, "nodes": 20000},
             "games": [], "note": "Bounded colour-paired screen, not calibrated Elo; cutoffs reported separately"}
    try:
        for opponent in ("pilot", "sf1600"):
            for oi, opening in enumerate(OPENINGS):
                for our_white in (True, False):
                    if time.time() >= end:
                        return state
                    b = chess.Board()
                    for uci in opening.split():
                        b.push_uci(uci)
                    game = chess.pgn.Game.from_board(b)
                    node = game.end()
                    reason = "ply_cap"
                    for ply in range(cfg["max_game_plies"]):
                        if time.time() >= end:
                            reason = "time_budget"
                            break
                        if b.is_game_over(claim_draw=True):
                            reason = "rules"
                            break
                        ours = b.turn == our_white
                        if ours:
                            move = choose(model, b, "cuda")
                        elif opponent == "pilot":
                            move = choose(pilot, b, "cuda")
                        else:
                            move = engine.play(b, chess.engine.Limit(nodes=20000)).move
                        assert move in b.legal_moves
                        b.push(move)
                        node = node.add_variation(move)
                    outcome = b.outcome(claim_draw=True)
                    if outcome:
                        reason = "rules"
                    score = None if not outcome else (.5 if outcome.winner is None else float(outcome.winner == our_white))
                    game.headers["White"] = "baseline" if our_white else opponent
                    game.headers["Black"] = opponent if our_white else "baseline"
                    game.headers["Result"] = b.result(claim_draw=True) if outcome else "*"
                    with open(root / "games.pgn", "a") as f:
                        f.write(str(game)+"\n\n")
                    state["games"].append({"opponent": opponent, "opening": oi, "our_white": our_white,
                                           "score": score, "termination": reason, "plies": len(b.move_stack)})
                    atomic_json(root / "games.json", state)
                    print("GAME", json.dumps(state["games"][-1]), flush=True)
    finally:
        engine.quit()
        atomic_json(root / "games.json", state)
    return state


def smoke():
    torch.set_num_threads(2)
    b = chess.Board()
    for uci in "e2e4 a7a6 e4e5 d7d5".split():
        b.push_uci(uci)
    for fen in (chess.STARTING_FEN, b.fen(), "7k/8/8/8/8/8/8/K7 b - - 0 1"):
        k = key(fen)
        assert np.array_equal(encode_key(k), encode(fen))
        assert decode_key(encode(fen)) == k
    model = SV()
    x = torch.tensor(np.stack([encode(chess.STARTING_FEN)]*4).astype(np.int64))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = -(targets(torch.tensor([0., .3, .7, 1.]))*model(x).log_softmax(-1)).sum(-1).mean()
    loss.backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    opt.step()
    assert choose(model.eval(), chess.Board(), "cpu") in chess.Board().legal_moves
    print("BASELINE SMOKE passed: encode/decode incl EP/turn/castling, gradient, optimizer, legal decision", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--e2e-smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    cfg = json.loads(args.config.read_text())
    if args.e2e_smoke:
        cfg = {**cfg, "root": cfg["root"] + "-smoke2", "train_records": 10000,
               "updates": 4, "warmup": 2, "save_every": 2, "eval_every": 2,
               "keep_steps": [2, 4], "decision_steps": [], "train_seconds": 120,
               "global_seconds": 900, "games_seconds": 2, "max_game_plies": 4}
    root = Path(cfg["root"])
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = root / "plan.json"
        if plan_path.exists():
            if not args.resume:
                raise RuntimeError("Existing plan; require explicit --resume, never reset budget")
            plan = json.loads(plan_path.read_text())
            assert plan["config"] == cfg
            deadline = plan["deadline"]
        else:
            if args.resume:
                raise RuntimeError("No saved plan to resume")
            assert sha(cfg["foundation_registry_path"]) == FOUNDATION_SHA
            deadline = time.time()+cfg["global_seconds"]
            plan = {"config": cfg, "started_at": time.time(), "deadline": deadline,
                    "initialization": "random; user-approved independent small SV baseline; no foundation weights loaded",
                    "foundation_registry_sha256": FOUNDATION_SHA,
                    "objective": "SV 128 bins, HL-Gauss sigma .75 bins, mean CE; pilot architecture unchanged",
                    "sampling": "20M canonical-unique positions, equal 2M/shard; epoch permutation without replacement",
                    "code_sha256": {str(p): sha(p) for p in (Path(__file__), Path("scripts/compare_sv_data.py"),
                                                             Path("src/krasnal/models/reasoning.py"))}}
            atomic_json(plan_path, plan)
        if time.time() >= deadline:
            raise TimeoutError("Saved global budget exhausted; no restart or extension")
        try:
            atomic_json(root / "status.json", {"phase": "preparing", "deadline": deadline})
            manifest = prepare(root, cfg, deadline)
            if time.time() >= deadline:
                raise TimeoutError("Preparation exhausted saved global budget")
            assert torch.cuda.is_available()
            model, result, curves = train(root, cfg, deadline, manifest)
            atomic_json(root / "status.json", {"phase": "games", "deadline": deadline})
            game_results = games(root, cfg, model, deadline) if time.time() < deadline else {"games": []}
            result["games"] = {op: {"completed": sum(g["opponent"] == op and g["score"] is not None for g in game_results["games"]),
                                           "wins": sum(g["opponent"] == op and g["score"] == 1 for g in game_results["games"]),
                                           "draws": sum(g["opponent"] == op and g["score"] == .5 for g in game_results["games"]),
                                           "losses": sum(g["opponent"] == op and g["score"] == 0 for g in game_results["games"]),
                                           "cutoffs": sum(g["opponent"] == op and g["score"] is None for g in game_results["games"])}
                               for op in ("pilot", "sf1600")}
            atomic_json(root / "result.json", result)
            render(root, cfg, "finished" if result["completed_updates"] else "training capped", curves)
            with open(root / "report.md", "a") as f:
                f.write("\nBest-checkpoint final results (validation-selected):\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n")
            atomic_json(root / "status.json", {"phase": "finished", "completed_updates": result["completed_updates"],
                                              "step": result["steps"], "deadline": deadline})
        except Exception as exc:
            atomic_json(root / "status.json", {"phase": "failed", "error": repr(exc), "deadline": deadline})
            raise


if __name__ == "__main__":
    main()
