"""Sequential CPU-only extra shards; durable isolated Docker worker."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if sha(config["stockfish"]) != config["stockfish_sha256"]:
        raise SystemExit("Wrong Stockfish binary")
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    sources = list(Path(__file__).resolve().parent.glob("*.py"))
    sources += list(Path(__file__).resolve().parents[1].joinpath("src").rglob("*.py"))
    current = {"config_sha256": sha(args.config), "source_sha256": {str(p): sha(p) for p in sources},
               "engine_sha256": sha(config["stockfish"]), "container": config["container"],
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "initialization": "CPU annotation only; no model training", "configuration": config}
    audit = root / "generation.plan.json"
    if audit.exists():
        if json.loads(audit.read_text()) != current:
            raise SystemExit("Pinned generation plan changed; preserve and inspect")
    else:
        partial = audit.with_suffix(".json.partial")
        partial.write_text(json.dumps(current, indent=2) + "\n")
        partial.replace(audit)
    out = root / "smoke" if args.smoke else Path(config["out"])
    cache = Path(config["cache"]).resolve()
    env = {**os.environ, "HF_HOME": str(cache / "huggingface"),
           "HF_HUB_CACHE": str(cache / "huggingface/hub"),
           "HUGGINGFACE_HUB_CACHE": str(cache / "huggingface/hub"),
           "TMPDIR": str(cache / "tmp"), "SF_GENERATOR_MODULE": "sf_interleave",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    months = config["months"][:1] if args.smoke else config["months"]
    for month in months:
        print("Starting", month, "smoke" if args.smoke else "full", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("generate_sf_more.py")),
            "annotate", "--month", month, "--out", str(out / f"{month}.jsonl"),
            "--manifest", str(out / f"{month}.manifest.json"), "--stockfish", config["stockfish"],
            "--nodes", str(config["nodes"]), "--threads", str(config["threads"]),
            "--workers", "2" if args.smoke else str(config["workers"]),
            "--game-limit", "3" if args.smoke else str(config["game_limit"]),
            "--min-elo", "1100", "--min-duration", "180", "--min-plies", "20",
            "--log-every", "1" if args.smoke else "100", "--resume"], env=env, check=True)
    print("Completed", len(months), "months", flush=True)


if __name__ == "__main__":
    main()
