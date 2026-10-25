"""Bounded, scratch-initialized SV data comparison. No search or text traces.

Use --smoke for an executable end-to-end check (not a unit-test suite).
Bag range decoding follows Apache Beam TupleCoder(StrUtf8Coder, FloatCoder)
and DeepMind's Apache-2.0 bagz layout; no Beam/JAX runtime is required.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import urllib.request

import chess
import duckdb
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from krasnal_think.model import Block, GPTConfig

GCS = "https://storage.googleapis.com/searchless_chess/data"
HF = "https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized"
PIECES = ".PNBRQKpnbrqk"
FOUNDATION_SHA = "b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def key(fen):
    # Ignore unavailable clocks in both arms; normalize unusable EP squares.
    b = chess.Board(fen if len(fen.split()) == 6 else fen + " 0 1")
    return " ".join(b.fen(en_passant="legal").split()[:4])


def encode(fen):
    fields = key(fen).split()
    board = []
    for c in fields[0].replace("/", ""):
        board.extend([0] * int(c) if c.isdigit() else [PIECES.index(c)])
    assert len(board) == 64
    # 64 squares, side, four castling rights, EP square, value readout token.
    return np.asarray(board + [13 + (fields[1] == "b")]
                      + [15 + int(c in fields[2]) for c in "KQkq"]
                      + [17 if fields[3] == "-" else 18 + chess.parse_square(fields[3])]
                      + [82], dtype=np.uint8)


def cp_value(cp, mate, white_turn):
    # Lichess labels are White-relative; DeepMind SV labels are side-to-move.
    white = (1.0 if mate > 0 else 0.0) if mate is not None else (
        1.0 / (1.0 + math.exp(-0.00368208 * max(-10000, min(10000, cp)))))
    return white if white_turn else 1.0 - white


def fetch(url, first=None, last=None, etag=None):
    headers = {"User-Agent": "Krasnal-SV-data-pilot/1"}
    if first is not None:
        headers["Range"] = f"bytes={first}-{last}"
    if etag:
        headers["If-Match"] = etag
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
        if first is not None:
            assert r.status == 206, "Server ignored byte range; refusing huge download"
        return r.read(), dict(r.headers)


def download(url, path):
    if path.exists():
        return
    print("download", url, flush=True)
    tmp = Path(str(path) + ".partial")
    with urllib.request.urlopen(url, timeout=90) as r, open(tmp, "wb") as out:
        while block := r.read(1024 * 1024):
            out.write(block)
    os.replace(tmp, path)


def decode_record(record):
    length, shift, pos = 0, 0, 0
    while True:
        byte = record[pos]
        pos += 1
        length |= (byte & 127) << shift
        if byte < 128:
            break
        shift += 7
    assert pos + length + 8 == len(record), "Unexpected Beam record format"
    fen = record[pos:pos + length].decode()
    value = struct.unpack(">d", record[-8:])[0]
    assert 0 <= value <= 1
    return key(fen), value


def bag_records(path):
    raw = path.read_bytes()
    start = struct.unpack("<Q", raw[-8:])[0]
    limits = np.frombuffer(raw[start:], dtype="<u8")
    previous = 0
    for end in limits:
        yield decode_record(raw[previous:int(end)])
        previous = int(end)


class SV(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=71,
                        vocab_size=83, qk_norm=False)
        self.embedding = nn.Embedding(83, 384)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(6))
        self.norm = nn.RMSNorm(384)
        self.head = nn.Linear(384, 128, bias=False)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h[:, -1]))


def targets(y):
    edges = torch.linspace(0, 1, 129, device=y.device)
    cdf = .5 * (1 + torch.erf((edges[None] - y[:, None]) / (.75 / 128 * math.sqrt(2))))
    p = cdf[:, 1:] - cdf[:, :-1]
    return p / p.sum(-1, keepdim=True)


def pack(rows, path):
    x = np.stack([encode(fen) for fen, _ in rows])
    y = np.asarray([v for _, v in rows], dtype=np.float32)
    np.savez(path, x=x, y=y)
    return {"records": len(rows), "sha256": sha(path),
            "target_mean": float(y.mean()), "target_std": float(y.std())}


def prepare(root, cfg, deadline):
    rawdir = root / "raw"
    rawdir.mkdir(exist_ok=True)
    gold_path = Path(cfg["gold"])
    gold = [json.loads(line) for line in gold_path.read_text().splitlines()]
    excluded = set()
    for row in gold:
        b = chess.Board()
        # Block complete histories and all evaluated child positions.
        excluded.add(key(b.fen()))
        for uci in row["h"].split():
            b.push_uci(uci)
            excluded.add(key(b.fen()))
        for m in list(b.legal_moves):
            b.push(m)
            excluded.add(key(b.fen()))
            b.pop()
    download(GCS + "/test/state_value_data.bag", rawdir / "test.bag")
    test = dict(bag_records(rawdir / "test.bag"))
    excluded.update(test)
    rng = np.random.default_rng(cfg["seed"])
    items = list(test.items())
    eval_cb = [items[i] for i in rng.choice(len(items), cfg["eval_records"], replace=False)]
    revision = cfg["hf_revision"]
    candidates = {}
    source_files = []
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='2GB'")
    # Equal contribution from all 10 physical shards, not sorted first-N data.
    for part in range(10):
        if time.time() >= deadline:
            raise TimeoutError("Global deadline during preparation")
        path = rawdir / f"train-{part:05d}.parquet"
        download(f"{HF}/resolve/{revision}/{path.name}", path)
        rows = con.execute(f"SELECT fen, cp, mate, depth FROM read_parquet('{path}') "
                           f"USING SAMPLE reservoir(130000 ROWS) REPEATABLE ({42 + part})").fetchall()
        source_files.append({"path": path.name, "sha256": sha(path), "sample_rows": len(rows)})
        for fen, cp, mate, depth in rows:
            canonical = key(fen)
            if canonical in excluded or canonical in candidates:
                continue
            if cp is None and mate is None:
                continue
            candidates[canonical] = (cp_value(cp, mate, canonical.split()[1] == "w"), depth)
        print("Mateusz shard", part, "unique candidates", len(candidates), flush=True)
    fens = list(candidates)
    rng.shuffle(fens)
    eval_m = [(f, candidates[f][0]) for f in fens[:cfg["eval_records"]]]
    excluded.update(f for f, _ in eval_m)
    train_m = [(f, candidates[f][0]) for f in fens[cfg["eval_records"]:]
               if f not in excluded][:cfg["train_records"]]
    assert len(train_m) == cfg["train_records"]
    depths = [candidates[f][1] for f, _ in train_m]
    del candidates, fens
    url = GCS + "/train/state_value_data.bag"
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        size, etag = int(r.headers["Content-Length"]), r.headers["ETag"]
    tail, _ = fetch(url, size - 8, size - 1, etag)
    index_start = struct.unpack("<Q", tail)[0]
    n = (size - index_start) // 8
    # Uniformly chosen non-overlapping blocks across the full shuffled bag.
    block_n = 65536
    starts = np.arange(0, n - block_n, block_n)
    rng.shuffle(starts)
    train_cb, seen, ranges = [], set(), []
    for first in starts:
        if time.time() >= deadline:
            raise TimeoutError("Global deadline during preparation")
        first = int(first)
        off_start = index_start + max(0, first - 1) * 8
        off_end = index_start + (first + block_n) * 8 - 1
        offsets, _ = fetch(url, off_start, off_end, etag)
        limits = np.frombuffer(offsets, dtype="<u8")
        previous = int(limits[0]) if first else 0
        ends = limits[1:] if first else limits
        body, _ = fetch(url, previous, int(ends[-1]) - 1, etag)
        ranges.append({"first_record": first, "records": block_n,
                       "index_sha256": hashlib.sha256(offsets).hexdigest(),
                       "body_sha256": hashlib.sha256(body).hexdigest()})
        base = previous
        for end in ends:
            end = int(end)
            fen, value = decode_record(body[previous-base:end-base])
            previous = end
            if fen in excluded or fen in seen:
                continue
            seen.add(fen)
            train_cb.append((fen, value))
            if len(train_cb) == cfg["train_records"]:
                break
        print("ChessBench sampled", len(train_cb), flush=True)
        if len(train_cb) == cfg["train_records"]:
            break
    assert len(train_cb) == cfg["train_records"]
    train_m_keys = {f for f, _ in train_m}
    assert not train_m_keys & excluded and not seen & excluded
    manifest = {"chessbench_train_source": {"url": url, "etag": etag,
                "bytes": size, "records": n, "sampled_ranges": ranges},
                "mateusz_revision": revision, "mateusz_sources": source_files,
                "mateusz_depth_percentiles": np.percentile(depths, [0, 25, 50, 75, 100]).tolist(),
                "train_cross_source_overlap": len(train_m_keys & seen),
                "excluded_canonical_positions": len(excluded), "train_eval_overlap": 0,
                "gold_sha256": sha(gold_path), "chessbench_test_sha256": sha(rawdir / "test.bag"),
                "splits": {}}
    for name, rows in (("train-chessbench", train_cb), ("train-mateusz", train_m),
                       ("eval-chessbench", eval_cb), ("eval-mateusz", eval_m)):
        manifest["splits"][name] = pack(rows, root / f"{name}.npz")
    atomic_json(root / "data-manifest.json", manifest)


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    ce, error, n = 0., 0., 0
    centers = (torch.arange(128, device=device) + .5) / 128
    for start in range(0, len(data["y"]), 512):
        x = torch.as_tensor(data["x"][start:start+512].astype(np.int64), device=device)
        y = torch.as_tensor(data["y"][start:start+512], device=device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x).float()
        p = logits.softmax(-1)
        ce += float(-(targets(y) * logits.log_softmax(-1)).sum())
        error += float(((p * centers).sum(-1) - y).abs().sum())
        n += len(y)
    model.train()
    return {"soft_target_cross_entropy": ce / n, "expected_value_mae": error / n, "n": n}


@torch.no_grad()
def choose(model, b, device):
    moves, inputs, scores = [], [], []
    us = b.turn
    for move in list(b.legal_moves):
        b.push(move)
        outcome = b.outcome(claim_draw=True)
        score = (.5 if outcome.winner is None else float(outcome.winner == us)) if outcome else None
        moves.append(move)
        scores.append(score)
        inputs.append(encode(b.fen()))
        b.pop()
    x = torch.as_tensor(np.stack(inputs).astype(np.int64), device=device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        pred = model(x).float().softmax(-1)
    centers = (torch.arange(128, device=device) + .5) / 128
    values = (1 - (pred * centers).sum(-1)).cpu().numpy()
    for i, score in enumerate(scores):
        if score is not None:
            values[i] = score
    return moves[int(values.argmax())]


@torch.no_grad()
def decision_eval(model, cfg, device, deadline):
    model.eval()
    rows = [json.loads(line) for line in Path(cfg["gold"]).read_text().splitlines()]
    hits, n, elapsed = 0, 0, []
    for row in rows:
        if time.time() > deadline:
            break
        b = chess.Board()
        for uci in row["h"].split():
            b.push_uci(uci)
        if b.is_game_over(claim_draw=True):
            continue
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.monotonic()
        move = choose(model, b, device)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed.append(time.monotonic() - start)
        hits += move.uci() == row["v"][0].split()[0]
        n += 1
    return {"sf30k_top1": hits / n if n else None, "n": n,
            "move_ms_mean": 1000 * float(np.mean(elapsed)) if n else None,
            "note": "Exact gold top1; no all-move regret or Elo claim"}


def train(root, cfg, deadline, device):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    evals = {name: np.load(root / f"eval-{name}.npz") for name in ("chessbench", "mateusz")}
    results = {}
    for arm in ("chessbench", "mateusz"):
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"]) if device == "cuda" else None
        model = SV().to(device)
        # Identical initialization, independent fresh AdamW and sampling RNG.
        init_path = root / f"{arm}-initial.pt"
        torch.save(model.state_dict(), init_path)
        flat = b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())
        init_sha = hashlib.sha256(flat).hexdigest()
        data = np.load(root / f"train-{arm}.npz")
        x = torch.as_tensor(data["x"].astype(np.int64), device=device)
        y = torch.as_tensor(data["y"], device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(.9, .95),
                               weight_decay=.1, fused=device == "cuda")
        rng = torch.Generator(device=device).manual_seed(cfg["seed"])
        start = time.time()
        steps = 0
        atomic_json(root / "status.json", {"phase": "training", "arm": arm, "step": 0})
        for step in range(cfg["updates"]):
            if time.time() >= min(deadline, start + cfg["arm_seconds"]):
                break
            lr = cfg["lr"] * min(1., (step + 1) / 100)
            if step >= 100:
                lr = cfg["lr"] * (.1 + .9 * .5 * (1 + math.cos(math.pi * (step-100) / max(1, cfg["updates"]-100))))
            for group in opt.param_groups:
                group["lr"] = lr
            idx = torch.randint(len(y), (cfg["batch_size"],), generator=rng, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x[idx]).float()
                loss = -(targets(y[idx]) * logits.log_softmax(-1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            steps = step + 1
            if step % 100 == 0:
                record = {"arm": arm, "step": step+1, "loss": loss.item(), "lr": lr,
                          "seconds": time.time()-start}
                with open(root / "updates.jsonl", "a") as f:
                    f.write(json.dumps(record) + "\n")
                atomic_json(root / "status.json", {"phase": "training", **record})
                print(json.dumps(record), flush=True)
            if (step + 1) % 1024 == 0:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "step": step+1, "rng": rng.get_state()}, root / f"{arm}-latest.pt.partial")
                os.replace(root / f"{arm}-latest.pt.partial", root / f"{arm}-latest.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": steps, "rng": rng.get_state()}, root / f"{arm}-model.pt")
        result = {"steps": steps, "exposures": steps*cfg["batch_size"],
                  "train_seconds": time.time()-start, "parameters": sum(p.numel() for p in model.parameters()),
                  "initial_tensor_sha256": init_sha, "completed_updates": steps == cfg["updates"],
                  "checkpoint_sha256": sha(root / f"{arm}-model.pt"), "value_eval": {}}
        if time.time() < deadline:
            result["value_eval"] = {name: evaluate(model, data, device) for name, data in evals.items()}
            result["decision_eval"] = decision_eval(model, cfg, device, deadline)
        results[arm] = result
        atomic_json(root / "results.json", results)
        print("RESULT", arm, json.dumps(result), flush=True)
        del model, opt, x, y
        if device == "cuda":
            torch.cuda.empty_cache()
        if time.time() >= deadline:
            break
    if len(results) == 2:
        assert results["chessbench"]["initial_tensor_sha256"] == results["mateusz"]["initial_tensor_sha256"]
    return results


def smoke():
    torch.set_num_threads(2)
    assert len(encode(chess.STARTING_FEN)) == 71
    assert cp_value(100, None, True) > .5 > cp_value(100, None, False)
    assert cp_value(None, 2, True) == 1 and cp_value(None, 2, False) == 0
    assert np.array_equal(encode(chess.STARTING_FEN), encode(chess.STARTING_FEN.rsplit(" ", 2)[0]))
    model = SV()
    x = torch.tensor(np.stack([encode(chess.STARTING_FEN)] * 2).astype(np.int64))
    y = torch.tensor([0., 1.])
    loss = -(targets(y) * model(x).log_softmax(-1)).sum(-1).mean()
    loss.backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    assert torch.allclose(targets(y).sum(-1), torch.ones(2))
    assert choose(model.eval(), chess.Board(), "cpu") in chess.Board().legal_moves
    url = GCS + "/test/state_value_data.bag"
    raw, _ = fetch(url, 0, 199)
    # First tuple length is a one-byte varint for these standard FEN records.
    fen, value = decode_record(raw[:1 + raw[0] + 8])
    assert len(encode(fen)) == 71 and 0 <= value <= 1
    print("SMOKE passed: encode, perspective/mate, HL-Gauss, backward, legal SV decision, real Beam record", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["root"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise RuntimeError("Existing run: do not reset deadline or silently retrain")
    assert sha(cfg["foundation_registry_path"]) == FOUNDATION_SHA
    start = time.time()
    deadline = start + cfg["global_seconds"]
    atomic_json(root / "plan.json", {**cfg, "started_at": start, "deadline": deadline,
                "initialization": "random; user-approved new SV architecture, foundation not loaded",
                "foundation_registry_sha256": FOUNDATION_SHA,
                "objective": "128-bin HL-Gauss SV, sigma .75 bins; equal updates/exposures",
                "input": "canonical 4-field FEN, no clocks in either arm; 71 tokens",
                "sampling": "Mateusz uniform reservoir per all 10 shards; CB random shuffled-bag blocks; train draws with replacement",
                "label_perspective": "both side-to-move; Mateusz cp White-relative converted with Lichess logistic",
                "limitations": "single seed; different source distribution/teacher/version/budget; cross-source losses are not Elo",
                "code_sha256": {str(p): sha(p) for p in (Path(__file__), Path("src/krasnal_think/model.py"))}})
    atomic_json(root / "status.json", {"phase": "preparing", "deadline": deadline})
    try:
        prepare(root, cfg, deadline)
        if time.time() >= deadline:
            raise TimeoutError("Preparation exhausted total budget")
        assert torch.cuda.is_available(), "No CUDA; do not run full training on CPU"
        results = train(root, cfg, deadline, "cuda")
        atomic_json(root / "status.json", {"phase": "finished", "results": results,
                                          "seconds": time.time()-start})
        lines = ["# SV data comparison", "", "Single-seed bounded pilot, not a DeepMind Elo reproduction.", "",
                 "| Data | Updates | SF30k top1 | CB value MAE | Mateusz value MAE | Training seconds |",
                 "|---|---:|---:|---:|---:|---:|"]
        for name, r in results.items():
            ev = r["value_eval"]
            lines.append(f"| {name} | {r['steps']} | {r.get('decision_eval', {}).get('sf30k_top1')} | "
                         f"{ev.get('chessbench', {}).get('expected_value_mae')} | "
                         f"{ev.get('mateusz', {}).get('expected_value_mae')} | {r['train_seconds']:.1f} |")
        lines += ["", "No measured Elo or games. Matched comparisons require equal completed updates.",
                  "Clocks removed from both inputs; best-depth Lichess labels still mix engine versions."]
        (root / "report.md").write_text("\n".join(lines) + "\n")
    except Exception as exc:
        atomic_json(root / "status.json", {"phase": "failed", "error": repr(exc), "deadline": deadline})
        raise


if __name__ == "__main__":
    main()
