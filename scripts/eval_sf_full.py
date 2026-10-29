"""Evaluate the full-data learning curve, select on val, test once, publish."""
import hashlib
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

config = json.loads(Path("configs/sf-interleave-100m-full-20261009.json").read_text())
root = Path(config["root"])
data = Path(config["data"])
metrics = root / "metrics"
metrics.mkdir(parents=True, exist_ok=True)
deadline = datetime.now() + timedelta(hours=45)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, payload):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    partial.replace(path)


def publish():
    hashes = {p.name: sha(p) for p in metrics.glob("*.json")
              if p.name != "transfer.manifest.json"}
    atomic(metrics / "transfer.manifest.json", {"sha256": hashes})


def evaluate(checkpoint, mode, split, label):
    output = metrics / f"{split}-{label}.json"
    if output.is_file():
        return json.loads(output.read_text())
    for batch in (8, 2, 1):
        temporary = output.with_suffix(".json.partial")
        command = [".venv/bin/python", "scripts/evaluate_interleave.py", str(checkpoint),
                   "--records", str(data / split / "records.jsonl"), "--mode", mode,
                   "--limit", str(config["eval_limit"]), "--batch-size", str(batch),
                   "--device", "cuda", "--out", str(temporary)]
        try:
            print("Evaluating", label, split, flush=True)
            subprocess.run(command, check=True, timeout=1800)
            result = json.loads(temporary.read_text())
            result.update(checkpoint_sha256=sha(checkpoint),
                          records_sha256=sha(data / split / "records.jsonl"),
                          dataset_manifest_sha256=sha(data / "dataset.manifest.json"))
            atomic(output, result)
            publish()
            return result
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            print("Retry", label, split, repr(exc), flush=True)
    raise RuntimeError("Evaluation failed at all batch sizes")


for split in ("val", "test"):
    evaluate(Path(config["checkpoint"]), "human", split, "pretrain100m")
while datetime.now() < deadline:
    candidates = sorted(root.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    for checkpoint in candidates:
        evaluate(checkpoint, "augmented", "val", checkpoint.stem)
    if (root / "train-summary.json").is_file() and (root / "model.pt").is_file():
        evaluate(root / "model.pt", "augmented", "val", "final")
        results = [(json.loads(p.read_text()), p) for p in metrics.glob("val-*.json")
                   if p.name != "val-pretrain100m.json"]
        selected, val_file = min(results, key=lambda pair: pair[0]["human_nll"])
        checkpoint = Path(selected["checkpoint"])
        test = evaluate(checkpoint, "augmented", "test", "selected")
        baseline = json.loads((metrics / "test-pretrain100m.json").read_text())
        assert test["records_sha256"] == baseline["records_sha256"]
        result = {"status": "complete", "selected_checkpoint": str(checkpoint),
                  "selection_metric": "validation human NLL", "validation_result": val_file.name,
                  "test_top1_delta_pp": 100 * (test["human_top1"] - baseline["human_top1"]),
                  "test_nll_delta": test["human_nll"] - baseline["human_nll"],
                  "train_summary": json.loads((root / "train-summary.json").read_text()),
                  "baseline": baseline, "treatment": test}
        atomic(metrics / "comparison.json", result)
        publish()
        print(json.dumps(result), flush=True)
        break
    publish()
    time.sleep(60)
else:
    raise SystemExit("Evaluation deadline reached; existing metrics preserved")
