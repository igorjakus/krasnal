"""Compute-matched empty-note control; never feed Stockfish identities."""
import runpy
import sys
import hashlib
import json
from pathlib import Path

import train_interleave
from krasnal_think import tokens

STAR_ID = tokens.SF_MOVE_IDS.stop
assert STAR_ID == 4080 and STAR_ID < tokens.INTERLEAVED_VOCAB_SIZE
original_batch = train_interleave._batch


def star_batch(*args, **kwargs):
    x, targets, weights = original_batch(*args, **kwargs)
    sf = (x >= tokens.SF_MOVE_IDS.start) & (x < tokens.SF_MOVE_IDS.stop)
    x[sf] = STAR_ID
    # SF labels are never supervised and never supplied as model inputs.
    sf_targets = (targets >= tokens.SF_MOVE_IDS.start) & (targets < tokens.SF_MOVE_IDS.stop)
    if (weights[sf_targets] != 0).any():
        raise ValueError("Star control requires zero Stockfish target weight")
    return x, targets, weights


def main():
    train_interleave._batch = star_batch
    if "--resume-from" in sys.argv:
        # Generic recovery keeps the same input transformation and shuffled cursor.
        train_interleave.main()
    else:
        if "--config" not in sys.argv:
            sys.argv.extend(["--config", "configs/sf-interleave-star-full-20261009.json"])
        config_path = Path(sys.argv[sys.argv.index("--config") + 1])
        config = json.loads(config_path.read_text())
        assert config["conditioning"] == "constant_star" and config["star_id"] == STAR_ID
        sources = [str(Path(__file__)), str(config_path),
                   "scripts/train_sf_scratch.py", "scripts/train_interleave.py",
                   "scripts/evaluate_star_interleave.py", "scripts/eval_sf_star_curve.py"]
        provenance = Path(config["root"] + ("-smoke" if "--smoke" in sys.argv else "") + ".control.manifest.json")
        provenance.parent.mkdir(parents=True, exist_ok=True)
        if provenance.exists():
            raise SystemExit("Control manifest exists; preserve artifacts and use explicit recovery")
        provenance.write_text(json.dumps({"conditioning": "constant_star", "star_id": STAR_ID,
            "sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}}, indent=2) + "\n")
        runpy.run_path(str(Path(__file__).with_name("train_sf_scratch.py")), run_name="__main__")


if __name__ == "__main__":
    main()
