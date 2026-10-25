"""Move the exact retired checkpoint only after its active Docker run exits."""
import argparse
import hashlib
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--wait-container")
args = parser.parse_args()
if args.wait_container:
    while True:
        check = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               check=True, capture_output=True, text=True)
        if args.wait_container not in check.stdout.splitlines():
            break
        time.sleep(60)
if not args.source.is_file():
    raise SystemExit("Source absent; nothing moved")
digest = hashlib.sha256()
with args.source.open("rb") as handle:
    while chunk := handle.read(8 * 1024 * 1024):
        digest.update(chunk)
if digest.hexdigest() != "52929453aaac2e62bb8ce1b61c084d7491e2dac462d10a1d3b64218e415b7f84":
    raise SystemExit("Source changed: refusing to archive a different model")
if args.archive.exists():
    raise SystemExit("Archive exists: refusing overwrite")
args.archive.parent.mkdir(parents=True, exist_ok=True)
args.source.rename(args.archive)
print("Archived retired foundation:", args.source, "->", args.archive, flush=True)
