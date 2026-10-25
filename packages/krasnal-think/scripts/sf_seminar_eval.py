"""Independent early-checkpoint evaluations, bounded by the seminar deadline."""
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path("experiments/sf-interleave-distinct-20261008")
data = Path("data/sf-interleave-distinct-20261008-quick-buffered")
metrics = root / "metrics"
metrics.mkdir(parents=True, exist_ok=True)
deadline = datetime(2026, 10, 9, 8, 0, tzinfo=ZoneInfo("Europe/Warsaw"))


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def publish_manifest():
    files = {p.name: checksum(p) for p in metrics.glob("*.json")
             if p.name != "transfer.manifest.json"}
    # Final training jobs may concurrently publish their own metrics. This
    # manifest only hashes completed JSON files; incomplete files are omitted.
    valid = {}
    for name, digest in files.items():
        try:
            json.loads((metrics / name).read_text())
            valid[name] = digest
        except (ValueError, OSError):
            pass
    partial = metrics / "transfer.manifest.json.partial"
    partial.write_text(json.dumps({"sha256": valid}, indent=2) + "\n")
    partial.replace(metrics / "transfer.manifest.json")


def evaluate(checkpoint, mode, label):
    for split in ("val", "test"):
        output = metrics / f"{split}-{label}.json"
        if output.exists():
            continue
        temporary = output.with_suffix(".json.partial")
        success = False
        for batch in (8, 2, 1):
            command = [".venv-codex/bin/python", "scripts/evaluate_interleave.py",
                       str(checkpoint), "--records", str(data / split / "records.jsonl"),
                       "--mode", mode, "--limit", "10000", "--batch-size", str(batch),
                       "--device", "cuda", "--out", str(temporary)]
            print("Evaluating", label, split, "batch", batch, flush=True)
            try:
                subprocess.run(command, check=True, timeout=1200)
                result = json.loads(temporary.read_text())
                result["checkpoint_sha256"] = checksum(checkpoint)
                result["dataset_manifest_sha256"] = checksum(data / "dataset.manifest.json")
                result["records_sha256"] = checksum(data / split / "records.jsonl")
                result["evaluated_at"] = datetime.now(ZoneInfo("Europe/Warsaw")).isoformat()
                temporary.write_text(json.dumps(result, indent=2) + "\n")
                temporary.replace(output)
                publish_manifest()
                success = True
                break
            except (subprocess.SubprocessError, ValueError, OSError) as exc:
                print("Evaluation retry:", label, split, repr(exc), flush=True)
        if not success:
            return False
    return True


base = Path("experiments/sf-interleave-v1/checkpoints/pretrain-model.pt")
baseline_done = evaluate(base, "human", "pretrain-early")
targets = [("human", 1000), ("augmented", 2000),
           ("human", 5000), ("augmented", 10000),
           ("human", 10000), ("augmented", 20000)]
done = set()
while datetime.now(ZoneInfo("Europe/Warsaw")) < deadline:
    if not baseline_done:
        baseline_done = evaluate(base, "human", "pretrain-early")
    for mode, step in targets:
        checkpoint = root / mode / f"step-{step}.pt"
        key = (mode, step)
        if key not in done and checkpoint.exists():
            if evaluate(checkpoint, mode, f"{mode}-step{step}"):
                done.add(key)
    publish_manifest()
    if baseline_done and len(done) == len(targets):
        break
    time.sleep(30)
print("Early evaluation finished", sorted(done), flush=True)
