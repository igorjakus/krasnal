"""Resume unbuilt training months with bounded memory and persistent HF cache."""
import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path

generator = importlib.import_module(os.environ.get("SF_GENERATOR_MODULE", "sf_interleave"))
original_connect = generator.duckdb.connect


def bounded_connect(*args, **kwargs):
    connection = original_connect(*args, **kwargs)
    connection.execute("SET memory_limit = '2GB'")
    connection.execute("SET max_temp_directory_size = '12GB'")
    return connection


def main():
    args = generator.parser().parse_args()
    if args.command != "annotate" or args.month == ["2024-05"] or args.month == ["2024-06"]:
        raise SystemExit("Only new training annotations; never modify held-out months")
    if set(args.month) & {"2024-05", "2024-06"}:
        raise SystemExit("Held-out month forbidden")
    # The old annotator must not be resumed over a completed output: doing so
    # could replace it with an empty partial. Skip only verified completed shards.
    if args.out.exists():
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("complete") and generator._sha256(args.out) == manifest["sha256"]:
            print("Already complete and SHA-256 verified:", args.out, flush=True)
            return
        raise SystemExit("Existing final output without valid manifest: preserve and inspect")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out.parent).free < 30 * 1024 ** 3:
        raise SystemExit("Less than 30 GiB shared disk free; preserve files and stop")
    engine_sha = generator._sha256(Path(args.stockfish))
    if engine_sha != "38faa5883b03652f847a87ed168b1bfee81b361db9584dae59a51cb91e69d9d6":
        raise SystemExit("Engine differs from existing corpus")
    if args.nodes != 30000 or args.threads != 1:
        raise SystemExit("Keep existing 30k-node, one-thread engine quality")
    sources = [Path(__file__), Path(generator.__file__)]
    provenance = {
        "months": args.month, "output": str(args.out), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hf_home": os.environ.get("HF_HOME"), "engine_sha256": engine_sha,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "duckdb_memory_limit": "2GB", "workers": args.workers,
        "existing_partial_records": len(generator._read_existing(args.out.with_suffix(".jsonl.partial")))
    }
    audit = args.manifest.with_name(args.manifest.name + ".launch.json")
    if audit.exists():
        audit = audit.with_name(audit.name + "." + os.environ.get("SLURM_JOB_ID", "retry"))
    audit.write_text(json.dumps(provenance, indent=2) + "\n")
    generator.duckdb.connect = bounded_connect
    generator.annotate(args)


if __name__ == "__main__":
    main()
