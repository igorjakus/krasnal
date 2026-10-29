"""Build a broad ready-shard subset immediately, or all completed shards."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from sf_interleave import build

scope = sys.argv[1]
assert scope in ("quick", "full")
paths = []
for root in ("data/sf-interleave-v1/raw/train", "data/sf-interleave-v2/raw/train"):
    for manifest in sorted(Path(root).glob("*.manifest.json")):
        raw = manifest.with_name(manifest.name.replace(".manifest.json", ".jsonl"))
        meta = json.loads(manifest.read_text())
        if raw.is_file() and meta.get("records", 0) > 0:
            paths.append(raw)
# Fixed evenly spaced months rather than only the oldest games.
if scope == "quick":
    import numpy as np
    paths = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, min(12, len(paths)), dtype=int)]
out = Path("data/sf-interleave-distinct-20261008-" + scope + "-buffered")
if (out / "dataset.manifest.json").exists():
    raise SystemExit("Completed dataset already exists; preserve it")
print("Building", scope, "from", len(paths), "complete shards", flush=True)
build(SimpleNamespace(
    train_shards=paths,
    val_shards=sorted(Path("data/sf-interleave-v1/raw/val").glob("*.jsonl")),
    test_shards=sorted(Path("data/sf-interleave-v1/raw/test").glob("*.jsonl")),
    out_dir=out, manifest=out / "dataset.manifest.json",
))
