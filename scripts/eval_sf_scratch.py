"""Scratch learning curve after the existing evaluator releases its GPU."""
import hashlib
import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/sf-interleave-scratch-full-20261009.json")
parser.add_argument("--evaluator", default="scripts/evaluate_interleave.py")
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
root, data = Path(config["root"]), Path(config["data"])
metrics = root / "metrics"
metrics.mkdir(parents=True, exist_ok=True)
baseline_root = Path("experiments/sf-interleave-100m-full-20261009-humanloss/metrics")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, result):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(path)


def publish():
    atomic(metrics / "transfer.manifest.json", {"sha256": {
        p.name: sha(p) for p in metrics.glob("*.json") if p.name != "transfer.manifest.json"}})


for split in ("val", "test"):
    baseline = json.loads((baseline_root / f"{split}-pretrain100m.json").read_text())
    assert baseline["checkpoint_sha256"] == "b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81"
    assert baseline["records_sha256"] == sha(data / split / "records.jsonl")
    atomic(metrics / f"{split}-pretrain100m.json", baseline)
publish()


def evaluate(checkpoint, split, label):
    output = metrics / f"{split}-{label}.json"
    if output.exists():
        return json.loads(output.read_text())
    for batch in (8, 2, 1):
        temporary = output.with_suffix(".json.partial")
        try:
            subprocess.run([".venv-codex/bin/python", args.evaluator,
                            str(checkpoint), "--records", str(data / split / "records.jsonl"),
                            "--mode", "augmented", "--limit", "10000", "--batch-size", str(batch),
                            "--device", "cuda", "--out", str(temporary)], check=True, timeout=1800)
            result = json.loads(temporary.read_text())
            result.update(checkpoint_sha256=sha(checkpoint),
                          records_sha256=sha(data / split / "records.jsonl"))
            atomic(output, result)
            publish()
            return result
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            print("Retry", label, repr(exc), flush=True)
    raise RuntimeError("Evaluation failed")


deadline = datetime.now() + timedelta(hours=45)
while datetime.now() < deadline:
    for checkpoint in sorted(root.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[1])):
        evaluate(checkpoint, "val", checkpoint.stem)
    if (root / "train-summary.json").exists():
        evaluate(root / "model.pt", "val", "final")
        candidates = [json.loads(p.read_text()) for p in metrics.glob("val-*.json")
                      if p.name != "val-pretrain100m.json"]
        selected = min(candidates, key=lambda m: m["human_nll"])
        test = evaluate(Path(selected["checkpoint"]), "test", "selected")
        baseline = json.loads((metrics / "test-pretrain100m.json").read_text())
        comparison = {
            "status": "complete", "initialization": "random", "pretrained_weights_loaded": False,
            "conditioning": config.get("conditioning", "true_stockfish"),
            "selected_checkpoint": selected["checkpoint"], "selection_metric": "validation human NLL",
            "test_top1_delta_pp": 100 * (test["human_top1"] - baseline["human_top1"]),
            "test_nll_delta": test["human_nll"] - baseline["human_nll"],
            "baseline": baseline, "treatment": test,
            "train_summary": json.loads((root / "train-summary.json").read_text())}
        if config.get("matched_treatment_root"):
            peer_metrics = Path(config["matched_treatment_root"]) / "metrics"
            peer_path = peer_metrics / "comparison.json"
            if not peer_path.exists():
                raise RuntimeError("SF scratch comparison missing; preserve results and retry evaluator")
            peer = json.loads(peer_path.read_text())
            peer_manifest = json.loads((peer_metrics / "transfer.manifest.json").read_text())
            assert sha(peer_path) == peer_manifest["sha256"]["comparison.json"]
            treatment = peer["treatment"]
            assert treatment["records_sha256"] == test["records_sha256"]
            comparison["matched_scratch_sf"] = {
                "comparison_sha256": sha(peer_path),
                "sf_minus_star_top1_pp": 100 * (treatment["human_top1"] - test["human_top1"]),
                "sf_minus_star_nll": treatment["human_nll"] - test["human_nll"],
                "sf_steps": peer["train_summary"]["steps"],
                "star_steps": comparison["train_summary"]["steps"],
                "same_final_step_budget": peer["train_summary"]["steps"] == comparison["train_summary"]["steps"],
                "sf_selected_checkpoint": peer["selected_checkpoint"],
                "validation_common_steps": []}
            for path in metrics.glob("val-step-*.json"):
                peer_file = peer_metrics / path.name
                if peer_file.exists():
                    assert sha(peer_file) == peer_manifest["sha256"][peer_file.name]
                    sf_value, star_value = json.loads(peer_file.read_text()), json.loads(path.read_text())
                    comparison["matched_scratch_sf"]["validation_common_steps"].append({
                        "step": int(path.stem.split("-")[-1]),
                        "sf_human_nll": sf_value["human_nll"], "star_human_nll": star_value["human_nll"],
                        "sf_human_top1": sf_value["human_top1"], "star_human_top1": star_value["human_top1"]})
        atomic(metrics / "comparison.json", comparison)
        publish()
        break
    time.sleep(60)
else:
    raise SystemExit("Deadline reached; metrics preserved")
