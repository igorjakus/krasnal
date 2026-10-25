"""Independent human-only baseline audit; no training or canonical metric edits."""
import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from krasnal_think import tokens
from krasnal_think.model import GPT


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--historical-val", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    checkpoint_sha = sha(args.checkpoint)
    assert checkpoint_sha == "b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81"
    model = GPT.load(args.checkpoint, "cpu").eval()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    selected = np.sort(np.random.default_rng(0).choice(len(records), min(args.limit, len(records)), replace=False))
    records = [records[int(i)] for i in selected]
    result = {"checkpoint_sha256": checkpoint_sha, "records_sha256": sha(args.records),
              "sample_indices": selected.tolist(), "precision": "float32", "device": "cpu",
              "model_config": asdict(model.config), "model_parameters": model.num_params(), "variants": {}}
    start = time.time()
    with torch.inference_mode():
        for effort in (False, True):
            correct = count = 0
            nll = 0.0
            for offset in range(0, len(records), args.batch_size):
                encoded = [[tokens.BOS, tokens.elo_id(int(r["w"])), tokens.elo_id(int(r["b"])),
                    *([tokens.effort_id(0)] if effort else []),
                    *[tokens.move_id(m) for m in r["h"].split()], tokens.EOS]
                    for r in records[offset:offset + args.batch_size]]
                width = max(map(len, encoded)) - 1
                assert width <= model.config.block_size
                x = torch.full((len(encoded), width), tokens.PAD, dtype=torch.long)
                y = torch.full_like(x, -1)
                for row, ids in enumerate(encoded):
                    x[row, :len(ids)-1] = torch.tensor(ids[:-1])
                    y[row, :len(ids)-1] = torch.tensor(ids[1:])
                logits, _ = model(x, targets=torch.zeros_like(x))
                mask = (y >= tokens.MOVE_IDS.start) & (y < tokens.MOVE_IDS.stop)
                scores, gold = logits[mask].float(), y[mask]
                correct += int((scores.argmax(-1) == gold).sum())
                count += int(mask.sum())
                nll += float(-scores.log_softmax(-1).gather(1, gold[:, None]).sum())
            value = {"games": len(records), "human_moves": count, "correct": correct,
                     "human_top1": correct/count, "human_nll": nll/count}
            result["variants"]["with_effort0" if effort else "without_effort0"] = value
            print("with_effort0" if effort else "without_effort0", value, "elapsed", time.time()-start, flush=True)
        if args.historical_val:
            stream = np.memmap(args.historical_val, dtype=np.uint16, mode="r")
            positions = np.flatnonzero(stream == tokens.BOS)
            result["historical_prefix_examples"] = [stream[int(p):int(p)+5].tolist() for p in positions[:5]]
            random = np.random.default_rng(0)
            conditioning = [tokens.BOS, *[tokens.elo_id(e) for e in (900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500)],
                            *[tokens.effort_id(e) for e in range(tokens.EFFORT_LEVELS)]]
            move_correct = move_count = all_correct = all_count = 0
            for _ in range(8):
                offsets = random.integers(0, len(stream)-1024-1, size=1)
                x = torch.tensor(np.stack([stream[i:i+1024] for i in offsets]).astype(np.int64))
                y = torch.tensor(np.stack([stream[i+1:i+1025] for i in offsets]).astype(np.int64))
                logits, _ = model(x, targets=torch.zeros_like(x))
                correct = logits.argmax(-1) == y
                moves = (y >= tokens.MOVE_IDS.start) & (y < tokens.MOVE_IDS.stop)
                all_labels = ~torch.isin(y, torch.tensor(conditioning))
                move_correct += int(correct[moves].sum()); move_count += int(moves.sum())
                all_correct += int(correct[all_labels].sum()); all_count += int(all_labels.sum())
            result["historical_val_probe"] = {"batches": 8, "move_only_top1": move_correct/move_count,
                "move_count": move_count, "historical_labels_top1_including_eos": all_correct/all_count,
                "historical_labels_count": all_count}
            print("historical_val_probe", result["historical_val_probe"], flush=True)
    result["seconds"] = time.time()-start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(".json.partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    partial.replace(args.out)


if __name__ == "__main__":
    main()
