"""Hash-pinned foundation selection shared by launchers on every worker."""
import hashlib
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / "configs/foundations.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_foundation(path, foundation_id=None, registry=REGISTRY):
    if not registry.is_file():
        if foundation_id:
            raise ValueError("Missing canonical foundation registry")
        return {}
    settings = json.loads(registry.read_text())
    digest = sha256(path)
    if digest in settings["retired"]:
        raise ValueError("RETIRED pretrain rejected: " + settings["retired"][digest]["reason"])
    if foundation_id:
        entry = settings["foundations"][foundation_id]
        if digest != entry["sha256"]:
            raise ValueError(f"Foundation SHA-256 mismatch for {foundation_id}: {digest}")
        return {"id": foundation_id, **entry}
    return {"sha256": digest}
