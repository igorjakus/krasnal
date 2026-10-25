"""Play engines against each other and turn the results into an Elo delta.

Opponents implement `reset(board)`, `move() -> uci` and `push(uci)`, which
`Player` already satisfies.
"""

import math
import os
from dataclasses import dataclass

import chess
import chess.engine

STOCKFISH = os.environ.get("STOCKFISH_PATH", "stockfish")


class Stockfish:
    """Stockfish pinned to a UCI_Elo rung of the ladder."""

    def __init__(self, elo: int = 1500, nodes: int = 100_000):
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        self.limit = chess.engine.Limit(nodes=nodes)
        self.elo = elo
        self.board = chess.Board()

    def reset(self, board: chess.Board | None = None):
        self.board = board.copy() if board is not None else chess.Board()

    def move(self) -> str:
        uci = self.engine.play(self.board, self.limit).move.uci()
        self.push(uci)
        return uci

    def push(self, uci: str):
        self.board.push(chess.Move.from_uci(uci))

    def close(self):
        self.engine.quit()


def play_game(white, black, max_plies: int = 300, board=None) -> float:
    """Return white's score: 1.0, 0.5 or 0.0."""
    board = board.copy() if board is not None else chess.Board()
    white.reset(board)
    black.reset(board)
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        mover, waiter = (white, black) if board.turn == chess.WHITE else (black, white)
        uci = mover.move()
        board.push(chess.Move.from_uci(uci))
        waiter.push(uci)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


@dataclass
class Result:
    wins: float
    games: int

    @property
    def score(self) -> float:
        return self.wins / self.games

    @property
    def elo(self) -> float:
        """Elo difference implied by the score, clamped at a decisive result."""
        score = min(max(self.score, 0.5 / self.games), 1 - 0.5 / self.games)
        return -400 * math.log10(1 / score - 1)

    @property
    def margin(self) -> float:
        """Half-width of the 95% interval on the Elo difference."""
        score = min(max(self.score, 0.5 / self.games), 1 - 0.5 / self.games)
        error = 1.96 * math.sqrt(score * (1 - score) / self.games)
        low, high = (min(max(score + s * error, 1e-4), 1 - 1e-4) for s in (-1, 1))
        return 200 * math.log10((high / (1 - high)) / (low / (1 - low)))

    def __str__(self) -> str:
        return (
            f"{self.wins:g}/{self.games} ({self.score:.1%})  "
            f"Elo {self.elo:+.0f} +- {self.margin:.0f}"
        )


def match(first, second, games: int, max_plies: int = 300) -> Result:
    """Play `games` with alternating colours; score is from `first`'s side."""
    wins = 0.0
    for game in range(games):
        if game % 2 == 0:
            wins += play_game(first, second, max_plies)
        else:
            wins += 1 - play_game(second, first, max_plies)
    return Result(wins, games)
