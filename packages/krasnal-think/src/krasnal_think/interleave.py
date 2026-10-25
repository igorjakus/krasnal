"""Data primitives for the Stockfish-to-human move experiment.

The reusable on-disk record stores one original game and the Stockfish move
calculated before every human move.  The model stream is either the original
human game or the interleaved ``S W S B ...`` game.
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import torch

from . import tokens


def game_id(source: str, row: int, white_elo: int, black_elo: int, moves: list[str]) -> str:
    """Return a stable id for a source row without trusting row order globally."""
    payload = "\t".join(
        (source, str(row), str(white_elo), str(black_elo), " ".join(moves))
    ).encode()
    return sha256(payload).hexdigest()


def encode_human_game(moves: list[str], white_elo: int, black_elo: int) -> list[int]:
    """Encode the original human game in the pretraining-compatible format."""
    return [
        tokens.BOS,
        tokens.elo_id(white_elo),
        tokens.elo_id(black_elo),
        *(tokens.move_id(move) for move in moves),
        tokens.EOS,
    ]


def encode_interleaved_game(
    moves: list[str], sf_moves: list[str], white_elo: int, black_elo: int
) -> list[int]:
    """Encode ``S W S B ...`` with SF computed at each human-game prefix."""
    if len(moves) != len(sf_moves):
        raise ValueError("human and Stockfish move counts must match")
    encoded = [tokens.BOS, tokens.elo_id(white_elo), tokens.elo_id(black_elo)]
    for sf_move, human_move in zip(sf_moves, moves, strict=True):
        encoded.extend((tokens.sf_move_id(sf_move), tokens.move_id(human_move)))
    return [*encoded, tokens.EOS]


def human_target_mask(length: int, interleaved: bool) -> np.ndarray:
    """Mark token positions that are human-move labels in a full game stream.

    The mask is aligned with the token stream, so a batch crop uses
    ``mask[offset + 1: offset + 1 + block_size]`` alongside the next-token
    labels.  Conditioning and EOS are deliberately not marked.
    """
    mask = np.zeros(length, dtype=np.uint8)
    if interleaved:
        mask[4 : max(4, length - 1) : 2] = 1
    else:
        mask[3 : max(3, length - 1)] = 1
    return mask


def record_to_tokens(record: dict, interleaved: bool) -> tuple[list[int], np.ndarray]:
    """Validate a JSON record and return model tokens plus human-label mask."""
    moves = record["h"].split()
    white_elo = int(record["w"])
    black_elo = int(record["b"])
    if interleaved:
        sf_moves = record["s"].split()
        ids = encode_interleaved_game(moves, sf_moves, white_elo, black_elo)
    else:
        ids = encode_human_game(moves, white_elo, black_elo)
    return ids, human_target_mask(len(ids), interleaved)


def role_mask_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Exclude the other move namespace at known human/SF prediction slots.

    A move's role is fixed by the stream layout, not inferred from its UCI value.
    Other tokens (EOS, conditioning) retain the full original vocabulary.
    """
    if logits.shape[-1] != tokens.INTERLEAVED_VOCAB_SIZE:
        return logits
    masked = logits.clone()
    human = (targets >= tokens.MOVE_IDS.start) & (targets < tokens.MOVE_IDS.stop)
    sf = (targets >= tokens.SF_MOVE_IDS.start) & (targets < tokens.SF_MOVE_IDS.stop)
    masked[..., tokens.VOCAB_SIZE :].masked_fill_(
        human.unsqueeze(-1), float("-inf")
    )
    masked[..., tokens.MOVE_IDS.start : tokens.MOVE_IDS.stop].masked_fill_(
        sf.unsqueeze(-1), float("-inf")
    )
    return masked
