"""Stage 2: build training sequences from annotations at a chosen effort level.

An effort level is a truncation of the stored analysis, so all five levels come
from one pass of the engine. Only the thinking block and the final move are
predicted; the history is context, not a target.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch

from . import tokens
from .annotate import Annotation
from .data import IGNORE


def encode(
    annotation: Annotation,
    effort: int,
    elo: int = 2600,
    rng: random.Random | None = None,
    compact: bool = False,
) -> tuple[list[int], list[int]]:
    budget = tokens.EFFORT_BUDGET[effort]
    elo_token = tokens.elo_id(elo)
    ids = [tokens.BOS, elo_token, elo_token]
    ids += [tokens.move_id(uci) for uci in annotation.history]
    targets = [IGNORE] * len(ids)

    thinking = []
    if budget:
        ids.append(tokens.effort_id(effort))
        targets.append(IGNORE)
        thinking.append(tokens.THINK)
        best = tokens.to_centipawns(*annotation.scores[0])
        limit = budget if compact else (rng.randint(max(4, budget // 2), budget) if rng else budget)
        used = 0
        candidates = list(zip(annotation.variations, annotation.scores, strict=False))
        if compact:
            # Keep the best target represented; sample the other two candidates.
            rest = candidates[1:]
            if rng:
                rng.shuffle(rest)
            candidates = candidates[:1] + rest[:2]
        if rng:
            rng.shuffle(candidates)
        for moves, score in candidates:
            max_moves = min(len(moves), limit - used - 3, 3 if compact else budget)
            if max_moves < 1:
                break
            plies = max_moves if compact else (rng.randint(1, max_moves) if rng else max_moves)
            thinking.append(tokens.PV)
            thinking += [tokens.move_id(uci) for uci in moves[:plies]]
            # Candidate order is random; delta 0 identifies the best line without
            # leaking the answer through a fixed rank.
            delta = tokens.compact_delta_id if compact else tokens.delta_id
            thinking.append(delta(best, tokens.to_centipawns(*score)))
            thinking.append(tokens.PV_END)
            used += plies + 3
            if not compact and rng and rng.random() < 0.2:
                break
        thinking.append(tokens.THINK_END)

    best = tokens.move_id(annotation.variations[0][0])
    ids += [*thinking, best]
    targets += [*thinking, best]
    return ids[:-1], targets[1:]


def load(
    path: Path,
    efforts=(0, 0, 0, 0, 1, 2, 3, 4),
    elo: int = 2600,
    seed: int = 0,
    compact: bool = False,
):
    """Encode each annotation once, at a randomly drawn effort level."""
    rng = random.Random(seed)
    with path.open() as handle:
        for line in handle:
            annotation = Annotation.from_json(line)
            if not annotation.variations or not annotation.variations[0]:
                continue
            yield encode(annotation, rng.choice(efforts), elo, rng, compact=compact)


def load_onpolicy(path: Path, elo: int = 2600):
    """Keep the sampled trace verbatim; supervise only its expert final choice."""
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            policy = record["onpolicy"]
            if policy["version"] != 1:
                raise ValueError("unsupported on-policy trace version")
            trace = [tokens.ID[token] for token in policy["trace"]]
            if not trace or trace[0] != tokens.effort_id(2) or trace[-1] != tokens.THINK_END:
                raise ValueError("incomplete compact on-policy trace")
            target = tokens.move_id(policy["target"])
            roots = [trace[i + 1] for i, token in enumerate(trace[:-1]) if token == tokens.PV]
            if target not in roots:
                raise ValueError("expert choice must belong to sampled candidates")
            elo_token = tokens.elo_id(elo)
            context = [tokens.BOS, elo_token, elo_token]
            context += [tokens.move_id(move) for move in record["h"].split()]
            context += trace
            yield context, [IGNORE] * (len(context) - 1) + [target]


def decision_weights(targets, root_weight=0.5, decay=0.5, structure_weight=0.1):
    """Final move 1; PV moves root_weight * decay**ply; grammar/delta low weight."""
    present = targets != IGNORE
    starts = (targets == tokens.PV).long().cumsum(1)
    ends = (targets == tokens.PV_END).long().cumsum(1)
    in_pv = starts > ends
    move = (targets >= tokens.MOVE_START) & (targets < tokens.MOVE_START + len(tokens.MOVE_IDS))
    index = torch.arange(targets.shape[1], device=targets.device).expand_as(targets)
    last_start = torch.where(targets == tokens.PV, index, 0).cummax(1).values
    ply = (index - last_start - 1).clamp(min=0)
    weights = torch.full_like(targets, structure_weight, dtype=torch.float32)
    weights = torch.where(in_pv & move, root_weight * decay**ply, weights)
    weights = torch.where(move & ~in_pv, 1.0, weights)
    return weights * present


def weighted_mix(teacher, dagger, teacher_fraction: float, seed: int):
    """Mix finite streams without cycling either source."""
    if not 0 < teacher_fraction < 1:
        raise ValueError("teacher_fraction must be between 0 and 1")
    rng = random.Random(seed)
    teacher, dagger = iter(teacher), iter(dagger)
    teacher_done = dagger_done = False
    while not (teacher_done and dagger_done):
        choose_teacher = dagger_done or (not teacher_done and rng.random() < teacher_fraction)
        source = teacher if choose_teacher else dagger
        try:
            yield next(source)
        except StopIteration:
            if choose_teacher:
                teacher_done = True
            else:
                dagger_done = True


def _buffered_groups(samples, batch_size: int, block_size: int, buffer: int, rng):
    """Sort each buffer by length into batches, then shuffle the batch order."""
    held = []
    for sample in samples:
        if len(sample[0]) > block_size:
            continue
        held.append(sample)
        if len(held) < buffer:
            continue
        yield from _drain(held, batch_size, rng)
        held = []
    yield from _drain(held, batch_size, rng)


def _drain(held, batch_size: int, rng):
    held.sort(key=lambda pair: len(pair[0]))
    groups = [held[i : i + batch_size] for i in range(0, len(held), batch_size)]
    rng.shuffle(groups)
    return groups


def batches(samples, batch_size: int, block_size: int, device: str, seed=0, buffer=8192):
    """Length-bucketed batches over a bounded buffer, so the source can stream."""
    rng = random.Random(seed)
    for group in _buffered_groups(samples, batch_size, block_size, buffer, rng):
        width = max(len(ids) for ids, _ in group)
        x = np.full((len(group), width), tokens.PAD, dtype=np.int64)
        y = np.full((len(group), width), IGNORE, dtype=np.int64)
        for row, (ids, targets) in enumerate(group):
            x[row, : len(ids)] = ids
            y[row, : len(targets)] = targets
        yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
