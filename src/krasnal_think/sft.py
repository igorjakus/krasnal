"""Stage 2: build training sequences from annotations at a chosen effort level.

An effort level is a truncation of the stored analysis, so all five levels come
from one pass of the engine. Only the thinking block and the final move are
predicted; the history is context, not a target.
"""

import random
from pathlib import Path

import numpy as np
import torch

from . import tokens
from .annotate import Annotation
from .data import IGNORE
from .play import EFFORT


def encode(annotation: Annotation, effort: int, elo: int = 2600) -> tuple[list[int], list[int]]:
    variations, plies = EFFORT[effort]
    elo_token = tokens.elo_id(elo)
    ids = [tokens.BOS, elo_token, elo_token, tokens.effort_id(effort)]
    ids += [tokens.move_id(uci) for uci in annotation.history]
    targets = [IGNORE] * len(ids)

    thinking = []
    if variations:
        thinking.append(tokens.THINK)
        best = tokens.to_centipawns(*annotation.scores[0])
        for rank, (moves, score) in enumerate(
            zip(annotation.variations[:variations], annotation.scores[:variations], strict=False)
        ):
            thinking.append(tokens.PV)
            thinking += [tokens.move_id(uci) for uci in moves[:plies]]
            # The best line is scored absolutely; the rest by how much they lose.
            thinking.append(
                tokens.score_id(*score)
                if rank == 0
                else tokens.delta_id(best, tokens.to_centipawns(*score))
            )
            thinking.append(tokens.PV_END)
        thinking.append(tokens.THINK_END)

    best = tokens.move_id(annotation.variations[0][0])
    ids += [*thinking, best]
    targets += [*thinking, best]
    return ids[:-1], targets[1:]


def load(path: Path, efforts=tuple(EFFORT), elo: int = 2600, seed: int = 0):
    """Encode each annotation once, at a randomly drawn effort level."""
    rng = random.Random(seed)
    with path.open() as handle:
        for line in handle:
            annotation = Annotation.from_json(line)
            if not annotation.variations or not annotation.variations[0]:
                continue
            yield encode(annotation, rng.choice(efforts), elo)


def batches(samples, batch_size: int, block_size: int, device: str, shuffle=True, seed=0):
    """Length-bucketed batches; sorting by length keeps padding waste low."""
    samples = [s for s in samples if len(s[0]) <= block_size]
    samples.sort(key=lambda pair: len(pair[0]))
    groups = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]
    if shuffle:
        random.Random(seed).shuffle(groups)
    for group in groups:
        width = max(len(ids) for ids, _ in group)
        x = np.full((len(group), width), tokens.PAD, dtype=np.int64)
        y = np.full((len(group), width), IGNORE, dtype=np.int64)
        for row, (ids, targets) in enumerate(group):
            x[row, : len(ids)] = ids
            y[row, : len(targets)] = targets
        yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
