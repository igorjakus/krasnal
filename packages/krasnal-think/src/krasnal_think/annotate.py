"""Stockfish annotation of single positions.

Positions are analysed once at maximum effort (5 variations, 7 plies). Every
lower effort level is a truncation of that same record, so the whole effort
ladder costs one pass of the engine rather than one pass per level.
"""

import json
from dataclasses import dataclass

import chess
import chess.engine

MAX_VARIATIONS = 5
MAX_PLIES = 7


@dataclass
class Annotation:
    history: list[str]
    variations: list[list[str]]
    scores: list[tuple[int | None, int | None]]  # (centipawns, mate), side-to-move POV

    def to_json(self) -> str:
        return json.dumps(
            {
                "h": " ".join(self.history),
                "v": [" ".join(moves) for moves in self.variations],
                "s": self.scores,
            }
        )

    @classmethod
    def from_json(cls, line: str) -> "Annotation":
        record = json.loads(line)
        return cls(
            history=record["h"].split(),
            variations=[moves.split() for moves in record["v"]],
            scores=[tuple(score) for score in record["s"]],
        )

    def board(self) -> chess.Board:
        board = chess.Board()
        for uci in self.history:
            board.push(chess.Move.from_uci(uci))
        return board


def analyse(engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int) -> list:
    """Top variations for one position, best first."""
    infos = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=MAX_VARIATIONS,
    )
    variations, scores = [], []
    for info in infos:
        score = info["score"].relative
        variations.append([move.uci() for move in info.get("pv", [])[:MAX_PLIES]])
        scores.append((score.score(), score.mate()))
    return [variations, scores]
