"""One-step grouped policy optimization against Stockfish."""

import random
from concurrent.futures import Executor

import chess
import chess.engine
import torch

from . import tokens
from .annotate import Annotation
from .play import Player


def reward(engine: chess.engine.SimpleEngine, board: chess.Board, move: str, nodes: int) -> float:
    side = board.turn
    after = board.copy(stack=False)
    after.push(chess.Move.from_uci(move))
    info = engine.analyse(after, chess.engine.Limit(nodes=nodes))
    score = info["score"].pov(side).score(mate_score=10_000)
    return max(-1000, min(1000, score)) / 1000


def sample(player: Player, annotation: Annotation) -> tuple[list[int], str]:
    player.reset(annotation.board())
    move = player.move()
    return [tokens.BOS, tokens.elo_id(player.elo), tokens.elo_id(player.elo),
            tokens.effort_id(player.effort)] + [tokens.move_id(uci) for uci in annotation.history] \
        + [tokens.ID[token] for token in player.thinking], move


def log_probability(model, sequence: list[int], generated_start: int, device: str) -> torch.Tensor:
    ids = torch.tensor(sequence, device=device).unsqueeze(0)
    logits, _ = model(ids[:, :-1], targets=ids[:, 1:])
    log_probs = logits.log_softmax(-1)
    generated = ids[:, generated_start:]
    return log_probs[:, generated_start - 1 :].gather(-1, generated.unsqueeze(-1)).squeeze(-1).sum()


def update(model, optimizer, annotations, engines, executor: Executor, args, device: str,
           rng: random.Random) -> tuple[float, float]:
    annotation = rng.choice(annotations)
    board = annotation.board()
    effort = rng.choice(args.efforts)
    players = [Player(model, device=device, effort=effort, temperature=args.temperature)
               for _ in range(args.group)]
    samples = [sample(player, annotation) for player in players]
    futures = [executor.submit(reward, engines[i % len(engines)], board, move, args.nodes)
               for i, (_, move) in enumerate(samples)]
    scores = torch.tensor([future.result() for future in futures], device=device)
    advantages = (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-6)
    generated_start = 4 + len(annotation.history)
    log_probs = torch.stack([
        log_probability(model, sequence, generated_start, device) for sequence, _ in samples
    ])
    loss = -(advantages.detach() * log_probs).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item(), scores.mean().item()
