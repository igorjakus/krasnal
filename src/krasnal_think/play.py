"""Constrained generation: think in legal principal variations, then move.

Every move sampled inside `<think>` is masked against a shadow board, so a
variation is legal by construction and the model cannot justify its choice with a
line that does not exist. Once the move is chosen the thinking is evicted from the
KV cache, leaving a moves-only prefix.
"""

from dataclasses import dataclass, field

import chess
import torch

from . import tokens
from .model import GPT, KVCache


def _ids(*prefixes):
    return [tokens.ID[name] for name in tokens.VOCAB if name.startswith(prefixes)]


ABSOLUTE = torch.tensor(_ids("<cp:", "<mate:"))
DELTA = torch.tensor(_ids("<delta:"))
SCORE_SET = set(_ids("<cp:", "<mate:", "<delta:"))
PREFIX = 4  # <bos> <elo> <elo> <effort>
# effort level -> (variations, plies per variation)
EFFORT = {0: (0, 0), 1: (2, 3), 2: (3, 5), 3: (4, 6), 4: (5, 7)}


@dataclass
class Player:
    model: GPT
    device: str = "cpu"
    effort: int = 0
    elo: int = 2600
    temperature: float = 0.4
    top_p: float = 0.95
    thinking: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.reset()

    def reset(self, board: chess.Board | None = None):
        self.board = board.copy() if board is not None else chess.Board()
        self.cache = KVCache(self.model.config, 1, self.device, torch.float32)
        self.thinking = []
        elo = tokens.elo_id(self.elo)
        self._feed([tokens.BOS, elo, elo, tokens.effort_id(self.effort)])

    def _feed(self, ids: list[int]):
        self.thinking.extend(tokens.VOCAB[i] for i in ids)
        idx = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            logits, _ = self.model(idx, cache=self.cache)
        self.logits = logits[0, -1]

    def _sample(self, allowed: torch.Tensor) -> int:
        masked = torch.full_like(self.logits, -torch.inf)
        masked[allowed] = self.logits[allowed]
        if self.temperature == 0:
            return int(masked.argmax())
        probs = torch.softmax(masked / self.temperature, dim=-1)
        order = probs.argsort(descending=True)
        keep = order[torch.cat([torch.tensor([True], device=probs.device),
                                probs[order].cumsum(0)[:-1] < self.top_p])]
        return int(keep[torch.multinomial(probs[keep], 1)])

    @staticmethod
    def _legal(board: chess.Board) -> torch.Tensor:
        return torch.tensor([tokens.ID[move.uci()] for move in board.legal_moves])

    def _variation(self, plies: int, first: bool):
        """Emit `<pv> move... <score> </pv>` against a shadow board."""
        shadow = self.board.copy()
        scores = ABSOLUTE if first else DELTA
        self._feed([tokens.PV])
        for ply in range(plies):
            if shadow.is_game_over():
                break
            allowed = self._legal(shadow)
            token = self._sample(torch.cat([allowed, scores]) if ply else allowed)
            if token in SCORE_SET:
                self._feed([token, tokens.PV_END])
                return
            shadow.push(chess.Move.from_uci(tokens.VOCAB[token]))
            self._feed([token])
        self._feed([self._sample(scores), tokens.PV_END])

    def _think(self):
        variations, plies = EFFORT[self.effort]
        if not variations:
            return
        self._feed([tokens.THINK])
        for i in range(variations):
            allowed = torch.tensor([tokens.PV, tokens.THINK_END] if i else [tokens.PV])
            if self._sample(allowed) == tokens.THINK_END:
                break
            self._variation(plies, first=i == 0)
        self._feed([tokens.THINK_END])

    def _ensure_room(self):
        """Rebuild the cache from recent history when the context fills up."""
        budget = max(EFFORT[self.effort][0] * (EFFORT[self.effort][1] + 4) + 4, 2)
        if self.cache.length + budget <= self.model.config.block_size:
            return
        keep = (self.model.config.block_size - budget - PREFIX) // 2
        history = [move.uci() for move in self.board.move_stack][-keep:]
        board = self.board.copy()
        self.reset()
        for uci in history:
            self._feed([tokens.ID[uci]])
        self.board = board

    def move(self) -> str:
        """Think, pick a legal move, then evict the thinking from the cache."""
        self._ensure_room()
        base = self.cache.length
        self.thinking = []
        self._think()
        uci = tokens.VOCAB[self._sample(self._legal(self.board))]
        self.cache.truncate(base)
        self.push(uci)
        return uci

    def push(self, uci: str):
        self._feed([tokens.ID[uci]])
        self.board.push(chess.Move.from_uci(uci))
