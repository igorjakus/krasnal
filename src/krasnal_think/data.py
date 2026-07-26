"""Stage-1 data: a flat uint16 token stream sampled with random crops.

Games are written back to back as [BOS, elo_w, elo_b, effort:0, moves..., EOS].
Conditioning tokens are never predicted; since they only ever occur in a prefix,
the dataloader masks them by identity and no parallel mask array is needed.
"""

from pathlib import Path

import numpy as np
import torch

from . import tokens

IGNORE = -1
_CONDITIONING = np.array(
    [tokens.BOS]
    + [tokens.ID[f"<elo:{i}>"] for i in range(len(tokens.ELO_EDGES) + 1)]
    + [tokens.ID[f"<effort:{i}>"] for i in range(tokens.EFFORT_LEVELS)],
    dtype=np.uint16,
)


def encode_game(uci: str, white_elo: int, black_elo: int) -> list[int]:
    return [
        tokens.BOS,
        tokens.elo_id(white_elo),
        tokens.elo_id(black_elo),
        tokens.effort_id(0),
        *(tokens.move_id(move) for move in uci.split()),
        tokens.EOS,
    ]


class ShardWriter:
    """Appends token ids to a flat uint16 file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("wb")
        self.count = 0

    def write(self, ids: list[int]):
        self.file.write(np.asarray(ids, dtype=np.uint16).tobytes())
        self.count += len(ids)

    def close(self):
        self.file.close()


def load_stream(path: Path) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    stream: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str,
    generator: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = generator.integers(0, len(stream) - block_size - 1, size=batch_size)
    x = np.stack([stream[i : i + block_size] for i in offsets])
    y = np.stack([stream[i + 1 : i + 1 + block_size] for i in offsets])
    targets = np.where(np.isin(y, _CONDITIONING), IGNORE, y.astype(np.int64))
    batch = (torch.from_numpy(x.astype(np.int64)), torch.from_numpy(targets))
    if device.startswith("cuda"):
        return tuple(t.pin_memory().to(device, non_blocking=True) for t in batch)
    return tuple(t.to(device) for t in batch)
