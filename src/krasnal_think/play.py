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


DELTA = torch.tensor(_ids("<delta:"))
SCORE_SET = set(_ids("<cp:", "<mate:", "<delta:"))
PREFIX = 3  # <bos> <elo> <elo>; per-move effort/thinking is temporary
EFFORT = tokens.EFFORT_BUDGET


@dataclass
class Player:
    model: GPT
    device: str = "cpu"
    effort: int = 0
    elo: int = 2600
    temperature: float = 0.4
    top_p: float = 0.95
    distinct_pv_roots: bool = False
    final_from_candidates: bool = False
    final_by_delta: bool = False
    compact: bool | None = None
    thinking: list[str] = field(default_factory=list)
    candidates: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        if self.compact is None:
            self.compact = getattr(self.model, "training_policy", {}).get("compact", False)
        if self.compact and self.effort:
            self.effort = 2
            self.distinct_pv_roots = True
        self.reset()

    def reset(self, board: chess.Board | None = None):
        self.board = board.copy() if board is not None else chess.Board()
        self.cache = KVCache(self.model.config, 1, self.device, torch.float32)
        self.thinking = []
        self.candidates = []
        elo = tokens.elo_id(self.elo)
        self._feed([tokens.BOS, elo, elo])
        history = [tokens.move_id(move.uci()) for move in self.board.move_stack]
        if history:
            budget = EFFORT[self.effort] + (3 if self.effort else 1)
            self._feed(history[-(self.model.config.block_size - PREFIX - budget) :])

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
        keep = order[
            torch.cat(
                [
                    torch.tensor([True], device=probs.device),
                    probs[order].cumsum(0)[:-1] < self.top_p,
                ]
            )
        ]
        return int(keep[torch.multinomial(probs[keep], 1)])

    @staticmethod
    def _legal(board: chess.Board) -> torch.Tensor:
        return torch.tensor([tokens.ID[move.uci()] for move in board.legal_moves])

    def _variation(self, budget: int) -> int:
        """Emit one variable-length PV within a content-token budget."""
        shadow = self.board.copy()
        scores = (
            torch.tensor([tokens.ID[f"<delta:{i}>"] for i in tokens.COMPACT_DELTA_BUCKETS])
            if self.compact
            else DELTA
        )
        self._feed([tokens.PV])
        used = 1
        moves = 0
        root = None
        while used + 2 < budget and not shadow.is_game_over():
            if self.compact and moves >= 3:
                break
            allowed = self._legal(shadow)
            if not moves and self.distinct_pv_roots and self.candidates:
                excluded = {candidate for candidate, _ in self.candidates}
                allowed = torch.tensor(
                    [token for token in allowed.tolist() if token not in excluded]
                )
                if not len(allowed):
                    break
            token = self._sample(torch.cat([allowed, scores]) if moves else allowed)
            if token in SCORE_SET:
                self._feed([token, tokens.PV_END])
                if root is not None:
                    self.candidates.append((root, token))
                return used + 2
            if root is None:
                root = token
            shadow.push(chess.Move.from_uci(tokens.VOCAB[token]))
            self._feed([token])
            used += 1
            moves += 1
        score = self._sample(scores)
        self._feed([score, tokens.PV_END])
        if root is not None:
            self.candidates.append((root, score))
        return used + 2

    def _think(self):
        budget = EFFORT[self.effort]
        if not budget:
            return
        self._feed([tokens.effort_id(self.effort), tokens.THINK])
        used = variations = 0
        while budget - used >= 4:
            if self.compact and variations >= 3:
                break
            if self.distinct_pv_roots and len(self.candidates) >= self.board.legal_moves.count():
                break
            allowed = torch.tensor([tokens.PV] if not variations else [tokens.PV, tokens.THINK_END])
            if self._sample(allowed) == tokens.THINK_END:
                break
            used += self._variation(budget - used)
            variations += 1
        self._feed([tokens.THINK_END])

    def _ensure_room(self):
        """Rebuild the cache from recent history when the context fills up."""
        budget = EFFORT[self.effort] + (3 if self.effort else 1)
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
        self.candidates = []
        self._think()
        allowed = self._legal(self.board)
        if self.candidates and (self.final_from_candidates or self.final_by_delta):
            if self.final_by_delta:
                token = min(self.candidates, key=lambda pair: pair[1])[0]
            else:
                token = self._sample(torch.tensor(sorted({move for move, _ in self.candidates})))
        else:
            token = self._sample(allowed)
        uci = tokens.VOCAB[token]
        self.cache.truncate(base)
        self.push(uci)
        return uci

    def push(self, uci: str):
        self._feed([tokens.ID[uci]])
        self.board.push(chess.Move.from_uci(uci))
