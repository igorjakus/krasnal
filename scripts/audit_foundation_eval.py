"""Read-only checkpoint evaluation: reproduce historical crops, add baselines.

No training, no true evals in input. Report bucket errors rather than pretending
that nonuniform CP classes represent equally spaced centipawns.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from krasnal_think.data import get_batch, load_eval_stream, load_stream
from krasnal_think.model import GPT
from krasnal_think import tokens

EXPECTED = "b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()
    assert sha(args.checkpoint) == EXPECTED
    assert not args.out.exists(), "Preserve previous audit"
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    model = GPT.load(args.checkpoint, "cuda").eval()
    assert model.eval_head is not None and model.config.eval_classes == 29
    stream = load_stream(args.data / "val.bin")
    eval_stream = load_eval_stream(args.data / "val.evals.bin")
    cache = []
    hook = model.eval_head.register_forward_hook(lambda _m, _i, out: cache.append(out))
    confusion = torch.zeros((29, 29), dtype=torch.int64, device="cuda")
    correct = near1 = near2 = top3 = count = move_correct = move_count = 0
    error = ce = 0.
    generator = np.random.default_rng(0)
    examples = []
    start = time.time()
    try:
        with torch.inference_mode():
            for _ in range(args.batches):
                x, target, ev = get_batch(stream, 1, 1024, "cuda", generator, eval_stream)
                logits, _ = model(x, targets=target, eval_targets=ev)
                move_mask = target != -1
                move_correct += int((logits.argmax(-1)[move_mask] == target[move_mask]).sum())
                move_count += int(move_mask.sum())
                ev_logits = cache.pop()[ev != -1].float()
                truth = ev[ev != -1]
                pred = ev_logits.argmax(-1)
                err = (truth - pred).abs()
                count += len(truth)
                correct += int((err == 0).sum())
                near1 += int((err <= 1).sum())
                near2 += int((err <= 2).sum())
                top3 += int((ev_logits.topk(3, -1).indices == truth[:, None]).any(-1).sum())
                error += float(err.sum())
                ce += float(F.cross_entropy(ev_logits, truth, reduction="sum"))
                confusion += torch.bincount(truth*29 + pred, minlength=29*29).reshape(29, 29)
                if len(examples) < 12:
                    # Reconstruct only complete histories visible since a BOS.
                    ids = x[0].cpu().tolist()
                    labels = ev[0].cpu().tolist()
                    ev_all = cache  # cache must stay empty after each forward
                    assert not ev_all
                    last_bos = None
                    for pos, token in enumerate(ids):
                        if token == tokens.BOS:
                            last_bos = pos
                        if last_bos is None or labels[pos] == -1 or pos < last_bos + 8:
                            continue
                        if len(examples) >= 12:
                            break
                        before = ids[last_bos:pos+1]
                        moves = [tokens.VOCAB[i] for i in before if i in tokens.MOVE_IDS]
                        idx = int((ev[0, :pos+1] != -1).sum()) - 1
                        examples.append({"uci_history": " ".join(moves), "true_bucket": labels[pos],
                                         "predicted_bucket": int(pred[idx]), "perspective": "White"})
                        # Spread examples across games rather than adjacent moves.
                        last_bos = None
    finally:
        hook.remove()
    matrix = confusion.cpu().numpy()
    frequencies = matrix.sum(1)
    majority = int(frequencies.argmax())
    recalled = np.divide(np.diag(matrix), frequencies, out=np.zeros(29), where=frequencies > 0)
    payload = {"checkpoint_sha256": EXPECTED, "parameters": model.num_params(),
               "eval_classes": 29, "cp_edges": tokens.CP_EDGES, "perspective": "White post-move",
               "protocol": "historical 100 seeded random 1024-token val crops, batch1, seed0, FP32",
               "batches": args.batches, "eval_positions": count,
               "eval_accuracy": correct/count, "eval_within_one_bucket": near1/count,
               "eval_within_two_buckets": near2/count, "eval_top3": top3/count,
               "eval_mean_bucket_error": error/count, "eval_cross_entropy": ce/count,
               "macro_bucket_recall": float(recalled[frequencies > 0].mean()),
               "always_training_majority_bucket": 14,
               "training_majority_baseline_accuracy": float(frequencies[14]/count),
               "best_constant_on_this_eval_accuracy": float(frequencies[majority]/count),
               "best_constant_bucket": majority, "eval_target_histogram": frequencies.tolist(),
               "confusion_true_rows_pred_columns": matrix.tolist(),
               "historical_move_top1_including_eos": move_correct/move_count,
               "historical_move_labels": move_count, "seconds": time.time()-start,
               "validation_tokens_sha256": sha(args.data / "val.bin"),
               "validation_evals_sha256": sha(args.data / "val.evals.bin"),
               "source_sha256": sha(__file__), "examples": examples,
               "limitations": "Bucket accuracy, not exact cp or game strength; crops can overlap and start mid-game. Historical protocol, not new independent full-game evaluation."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.out)+".partial")
    tmp.write_text(json.dumps(payload, indent=2)+"\n")
    tmp.replace(args.out)
    print(json.dumps({k: v for k, v in payload.items() if k not in (
        "confusion_true_rows_pred_columns", "examples", "eval_target_histogram")}), flush=True)


if __name__ == "__main__":
    main()
