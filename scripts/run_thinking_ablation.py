"""Sequential stage A with immutable manifest, deadlines and fixed evaluation."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--mix", action="store_true")
    p.add_argument("--continuation-from", type=Path)
    p.add_argument("--evaluate-only", type=Path)
    a = p.parse_args()
    root, source = a.root.resolve(), a.source.resolve()
    checkpoint = source / "round-003/dagger/model.pt"
    foundation = Path("runs/foundations/human-pretrain-100m-20260730/model.pt")
    assert sha(checkpoint) == "086781ca9846cb4216e2b1d48781f9a7bba890fa4f13bcc5f70f2d578d684025"
    assert sha(foundation) == "b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81"
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        current = {str(x): sha(x) for x in [*Path("scripts").glob("*.py"), *Path("src").rglob("*.py")]}
        if current != plan["source_hashes"]:
            atomic(root / ("rl-eval-source-audit.json" if a.evaluate_only else "repair-import-audit.json"), {"time": time.time(),
                   "reason": "RL evaluation snapshot" if a.evaluate_only else "Add omitted pipeline.py dependency; reuse completed eval collection",
                   "original_source_hashes": plan["source_hashes"], "current_source_hashes": current,
                   "deadline_unchanged": plan["deadline"]})
    else:
        plan = {"start": time.time(), "deadline": time.time() + 6 * 3600,
                "checkpoint_sha256": sha(checkpoint), "foundation_sha256": sha(foundation),
                "source_hashes": {str(x): sha(x) for x in [*Path("scripts").glob("*.py"), *Path("src").rglob("*.py")]},
                "data_sha256": sha(source / "data-000/round-03.jsonl"),
                "gold_sha256": sha(source / "eval-gold.jsonl"), "gpu_count": 1,
                "updates": 1000, "batch": 64, "teacher_positions": 73728, "seed": 42,
                "objective": "A: original per-example weighted trace CE vs final-only CE; identical teacher context/batches",
                "optimizer": "continue checkpoint optimizer, identical seed42 RNG reset for both arms",
                "remaining_stages": "B mixed replay and C RL not launched by this stage-A driver"}
        atomic(plan_path, plan)
        if a.continuation_from:
            prior = a.continuation_from.resolve()
            previous = json.loads((prior / "plan.json").read_text())
            plan["authorized_continuation_of"] = str(prior)
            plan["previous_deadline"] = previous["deadline"]
            plan["remaining_stages"] = "B mixed replay, then separately audited C RL"
            atomic(plan_path, plan)
            for name in ("baseline", "teacher-trace", "final-only"):
                assert (prior / name / "eval/result.json").exists()
                (root / name).symlink_to(prior / name, target_is_directory=True)

    def run(cmd, folder, limit):
        remaining = plan["deadline"] - time.time()
        if remaining <= 0:
            raise RuntimeError("global six-hour deadline exhausted")
        folder.parent.mkdir(parents=True, exist_ok=True)
        with folder.open("a") as log:
            print("RUN", *map(str, cmd), flush=True)
            subprocess.run(list(map(str, [sys.executable, *cmd])), stdout=log,
                           stderr=subprocess.STDOUT, timeout=min(limit, remaining), check=True)

    def evaluate(cp, name):
        folder = root / name / "eval"
        if (folder / "result.json").exists():
            return
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "candidates.jsonl").exists():
            run(["scripts/dagger.py", cp, "--source", source / "eval-gold.jsonl",
             "--out", folder / "unused.jsonl", "--metrics-out", folder / "unused.json",
             "--positions", 1000, "--effort", 2, "--compact", "--device", "cuda",
             "--candidates-out", folder / "candidates.jsonl", "--frozen-gold"], folder / "collect.log", 1200)
        if sum(1 for _ in (folder / "candidates.jsonl").open()) != 1000:
            raise RuntimeError("Incomplete eval collection: preserve and inspect")
        if not (folder / "human.json").exists():
            run(["scripts/curriculum_eval.py", "human", "--checkpoint", cp,
             "--source", "data/pretrain-eval/val.bin", "--out", folder / "human.json",
             "--device", "cuda", "--batches", 30, "--format-source", source / "eval-gold.jsonl",
             "--format-positions", 100], folder / "human.log", 600)
        if not (folder / "decision.json").exists():
            run(["scripts/dagger.py", "--source", source / "eval-gold.jsonl",
             "--candidates-in", folder / "candidates.jsonl", "--out", folder / "labeled.jsonl",
             "--metrics-out", folder / "decision.json", "--positions", 1000, "--nodes", 30000,
             "--workers", 4, "--stockfish", "/usr/games/stockfish"], folder / "relabel.log", 1200)
        atomic(folder / "result.json", {**json.loads((folder / "decision.json").read_text()),
               **json.loads((folder / "human.json").read_text()), "checkpoint_sha256": sha(cp)})

    if a.evaluate_only:
        evaluate(a.evaluate_only, "rl")
        return
    evaluate(checkpoint, "baseline")
    for name in ("teacher-trace", "final-only"):
        out = root / name / "train"
        cp = out / "model.pt"
        if not cp.exists():
            run(["scripts/diagnostic_sft.py", checkpoint, "--annotations", source / "data-000/round-03.jsonl",
                 "--validation-source", source / "data-000/validation.jsonl", "--out", out,
                 "--epochs", 1, "--max-steps", 1000, "--teacher-positions", 73728,
                 "--batch-size", 64, "--compact", "--teacher-efforts", "0,2", "--seed", 42,
                 "--lr", 5e-6, "--min-lr", 5e-7, "--warmup", 50, "--device", "cuda",
                 "--continue-optimizer", "--eval-every", 0, "--pretrain-check-every", 0,
                 "--save-every", 1000, *(["--final-move-only"] if name == "final-only" else [])],
                out / "train.log", 1800)
        evaluate(cp, name)
    atomic(root / "stage-a-complete.json", {"finished": time.time(), "deadline": plan["deadline"]})
    if a.mix:
        out = root / "mixed" / "train"
        cp = out / "model.pt"
        if not cp.exists():
            atomic(root / "mix-plan.json", {"sampling": [0.5, 0.4, 0.1],
                   "onpolicy": "first4096 unique per round1-4, sampled with replacement",
                   "teacher": "first8192 next disjoint shard, sampled with replacement",
                   "human": "random full-prefix final move from pretrain-eval/train.bin, never val.bin",
                   "human_data_sha256": sha(Path("data/pretrain-eval/train.bin")),
                   "updates": 1000, "checkpoint_sha256": sha(checkpoint),
                   "deadline": plan["deadline"], "source_sha256": sha(Path("scripts/diagnostic_mix.py"))})
            run(["scripts/diagnostic_mix.py", checkpoint, "--replay-root", source,
                 "--annotations", source / "data-000/round-03.jsonl",
                 "--validation-source", source / "data-000/validation.jsonl", "--out", out,
                 "--epochs", 1, "--max-steps", 1000, "--batch-size", 64, "--compact",
                 "--teacher-efforts", "0,2", "--seed", 42, "--lr", 5e-6,
                 "--min-lr", 5e-7, "--warmup", 50, "--device", "cuda",
                 "--continue-optimizer", "--eval-every", 0, "--pretrain-check-every", 0,
                 "--save-every", 1000], out / "train.log", 1800)
        evaluate(cp, "mixed")
        atomic(root / "stage-b-complete.json", {"finished": time.time(), "deadline": plan["deadline"]})


if __name__ == "__main__":
    main()
