"""Advisor-trust experiment: human moves conditioned on advice of varying quality.

Stream layout is identical to the Stockfish interleave (``A W A B ...``): before
each human move the model sees one advice token from the SF move namespace, or
the constant empty-note token ``NONE_ID`` (the historical "star" token 4080).
Sequence length and RoPE positions therefore never depend on the advisor.

One advisor is drawn per game (hidden from the model) so its reliability is a
latent, game-level property that can in principle be inferred in context: the
model sees each advice token *and* the human move that followed it.

Advisor names:

``none``            empty note at every ply
``sf30k``           the record's existing ``s`` field (SF 17.1, 30k nodes)
``sfd<N>``          SF at fixed depth N, stored in ``record["a"]["sfd<N>"]``
``sfn<N>``          SF at N nodes, stored in ``record["a"]["sfn<N>"]``
``random``          uniformly random legal move
``oracle<P>``       true human move with probability P/100, otherwise a wrong
                    move (the SF 30k move when it differs from the human move,
                    else a random legal non-human move). ``oracle100`` is pure
                    label leakage and must only be used for evaluation.
"""

from __future__ import annotations

import re
from hashlib import sha256

import chess
import numpy as np

from krasnal.data.interleave import human_target_mask
from krasnal.tokenization import reasoning as tokens

NONE_ID = tokens.SF_MOVE_IDS.stop
assert NONE_ID == 4080 and NONE_ID < tokens.INTERLEAVED_VOCAB_SIZE

_SF_LIMIT = re.compile(r"^sf(d|n)(\d+)$")
_ORACLE = re.compile(r"^oracle(\d+)$")


def sf_limit(name: str) -> dict | None:
    """Return python-chess Limit kwargs for an ``sfd<N>``/``sfn<N>`` advisor."""
    match = _SF_LIMIT.match(name)
    if not match:
        return None
    kind, value = match.groups()
    return {"depth": int(value)} if kind == "d" else {"nodes": int(value)}


def validate_advisor(name: str) -> str:
    if name in ("none", "sf30k", "random") or sf_limit(name) or _ORACLE.match(name):
        if (m := _ORACLE.match(name)) and not 0 <= int(m.group(1)) <= 100:
            raise ValueError(f"oracle probability out of range: {name}")
        return name
    raise ValueError(f"unknown advisor {name!r}")


def _rng(game: str, advisor: str, seed: int) -> np.random.Generator:
    digest = sha256(f"{seed}\t{advisor}\t{game}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def assign_advisor(game: str, mixture: dict[str, float], seed: int) -> str:
    """Deterministic per-game advisor draw (stable across reruns and shards)."""
    names = sorted(mixture)
    weights = np.asarray([mixture[name] for name in names], dtype=np.float64)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("mixture weights must be non-negative and not all zero")
    u = _rng(game, "__assign__", seed).random()
    return names[int(np.searchsorted(np.cumsum(weights / weights.sum()), u, side="right"))]


def _random_legal(board: chess.Board, rng: np.random.Generator, exclude: str | None) -> str:
    legal = sorted(move.uci() for move in board.legal_moves)
    if exclude is not None and len(legal) > 1:
        legal = [move for move in legal if move != exclude]
    return legal[int(rng.integers(len(legal)))]


def advice_for(record: dict, advisor: str, seed: int = 0) -> list[str | None]:
    """Return one advice move (or ``None`` for the empty note) per human ply."""
    validate_advisor(advisor)
    human = record["h"].split()
    if advisor == "none":
        return [None] * len(human)
    if advisor == "sf30k":
        advice = record["s"].split()
    elif sf_limit(advisor):
        stored = record.get("a", {}).get(advisor)
        if stored is None:
            raise KeyError(f"record {record['id']} has no {advisor} annotation")
        advice = stored.split()
    else:
        rng = _rng(record["id"], advisor, seed)
        board = chess.Board()
        advice = []
        oracle = _ORACLE.match(advisor)
        sf = record["s"].split() if oracle else None
        for ply, move in enumerate(human):
            if advisor == "random":
                advice.append(_random_legal(board, rng, None))
            else:
                correct = rng.random() < int(oracle.group(1)) / 100
                if correct:
                    advice.append(move)
                elif sf[ply] != move:
                    advice.append(sf[ply])
                else:
                    advice.append(_random_legal(board, rng, move))
            board.push_uci(move)
    if len(advice) != len(human):
        raise ValueError(f"{advisor} advice length differs from game length")
    return advice


def encode_advised_game(
    moves: list[str], advice: list[str | None], white_elo: int, black_elo: int
) -> tuple[list[int], np.ndarray]:
    """Encode ``A W A B ...``; returns ids and the human-label mask."""
    if len(moves) != len(advice):
        raise ValueError("human and advice move counts must match")
    ids = [tokens.BOS, tokens.elo_id(white_elo), tokens.elo_id(black_elo)]
    for hint, move in zip(advice, moves, strict=True):
        ids.extend((NONE_ID if hint is None else tokens.sf_move_id(hint), tokens.move_id(move)))
    ids.append(tokens.EOS)
    return ids, human_target_mask(len(ids), interleaved=True)
