"""Bounded full sampled-trace REINFORCE; legality-conditioned likelihood and KL."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import chess
import chess.engine
import torch

from krasnal.tokenization import reasoning as tokens
from krasnal.models.reasoning import GPT
from krasnal.inference.reasoning import Player
from krasnal.training.runtime import save


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


class RecordedPlayer(Player):
    def reset(self, board=None):
        self.sequence, self.actions = [], []
        super().reset(board)

    def _feed(self, ids):
        self.sequence.extend(ids)
        super()._feed(ids)

    def _sample(self, allowed):
        allowed = allowed.to(self.logits.device)
        token = super()._sample(allowed)
        self.actions.append((len(self.sequence), allowed.tolist(), token))
        return token


def likelihood(model, reference, player, device):
    sequence = torch.tensor([player.sequence], device=device)
    logits, _ = model(sequence[:, :-1], targets=sequence[:, 1:])
    with torch.no_grad():
        ref, _ = reference(sequence[:, :-1], targets=sequence[:, 1:])
    logps, kls, trace_logps = [], [], []
    for index, allowed, target in player.actions:
        assert player.sequence[index] == target, 'sample/feed mismatch'
        if len(allowed) == 1:
            continue
        mask = torch.tensor(allowed, device=device)
        lp = logits[0, index - 1, mask].float().log_softmax(-1)
        rp = ref[0, index - 1, mask].float().log_softmax(-1)
        value = lp[allowed.index(target)]
        logps.append(value)
        kls.append((lp.exp() * (lp - rp)).sum())
        if index < len(player.sequence) - 1 and len(allowed) > 1:
            trace_logps.append(value)
    assert trace_logps, 'no stochastic trace decisions'
    return torch.stack(logps).mean(), torch.stack(kls).mean(), torch.stack(trace_logps).mean()


def expectation(score, side, ply):
    return score.pov(side).wdl(model='sf', ply=ply).expectation()


def quality(engine, board, move=None):
    if board.is_game_over():
        result = board.outcome()
        return 0.5 if result.winner is None else float(result.winner == board.turn)
    if move is not None and chess.Move.from_uci(move) not in board.legal_moves:
        return 0.0
    info = engine.analyse(board, chess.engine.Limit(nodes=30000),
                          root_moves=None if move is None else [chess.Move.from_uci(move)])
    return expectation(info['score'], board.turn, board.ply())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha', required=True)
    p.add_argument('--budget-plan', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    assert sha(a.checkpoint) == a.checkpoint_sha
    assert sha(Path('runs/foundations/human-pretrain-100m-20260730/model.pt')) == 'b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81'
    global_deadline = json.loads(a.budget_plan.read_text())['deadline']
    assert time.time() < global_deadline
    torch.manual_seed(42)
    model = GPT.load(a.checkpoint, 'cuda').eval()
    reference = GPT.load(a.checkpoint, 'cuda').eval()
    reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0)
    gold = {tuple(json.loads(line)['h'].split()) for line in (a.source / 'eval-gold.jsonl').open()}
    positions, seen = [], set()
    with (a.source / 'data-000/round-03.jsonl').open() as handle:
        for line in handle:
            h = tuple(json.loads(line)['h'].split())
            if h in gold:
                continue
            board = chess.Board()
            for move in h:
                board.push_uci(move)
            key = board.fen()
            if board.is_game_over() or key in seen or len(h) > 800:
                continue
            positions.append(board)
            seen.add(key)
            if len(positions) == (1 if a.smoke else 256):
                break
    assert positions
    a.root.mkdir(parents=True, exist_ok=False)
    protocol = {'algorithm': 'REINFORCE group-centered reward/std floor0.05; mean stochastic-action logprob; exact masked KL',
                'checkpoint_sha256': a.checkpoint_sha, 'seed': 42, 'lr': 1e-6, 'kl_coefficient': 0.02,
                'gradient_clip': 1, 'temperature': 1, 'top_p': 1, 'group': 4, 'effort': 2, 'compact': True,
                'teacher': 'Stockfish30k deterministic Threads1 Hash64; root-constrained score at original board',
                'reward': '-clamp(best WDL expected outcome - selected WDL expected outcome,0,1); side-to-move',
                'positions': len(positions), 'positions_fen': [b.fen() for b in positions],
                'fixed_eval_excluded': True, 'global_deadline': global_deadline,
                'source_sha256': sha(a.source / 'data-000/round-03.jsonl'), 'script_sha256': sha(__file__),
                'forced_grammar': 'singleton/forced delimiters are not stochastic train decisions; sampled PV moves/delta/stop and final all included',
                'smoke': a.smoke, 'start': time.time()}
    (a.root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    engine = chess.engine.SimpleEngine.popen_uci('/usr/games/stockfish')
    engine.configure({'Threads': 1, 'Hash': 64})
    start = time.time()
    deadline = min(global_deadline - 600, start + 1770)
    completed, trace_gradient = 0, None
    assert expectation(chess.engine.PovScore(chess.engine.Mate(1), chess.WHITE), chess.WHITE, 0) == 1
    assert expectation(chess.engine.PovScore(chess.engine.Mate(1), chess.WHITE), chess.BLACK, 0) == 0
    assert expectation(chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE), chess.BLACK, 0) == 0.5
    assert max(0.0, min(1.0, 0.8 - 0.2)) > 0.59
    try:
        for board in positions:
            if time.time() >= deadline:
                break
            best = quality(engine, board)
            players, scores = [], []
            for _ in range(4):
                if time.time() >= deadline:
                    break
                player = RecordedPlayer(model, device='cuda', effort=2, temperature=1, top_p=1, compact=True)
                player.reset(board)
                move = player.move()
                assert chess.Move.from_uci(move) in board.legal_moves
                players.append(player)
                scores.append(-max(0.0, min(1.0, best - quality(engine, board, move))))
            if len(players) != 4:
                break
            reward = torch.tensor(scores, device='cuda')
            advantage = (reward - reward.mean()) / reward.std(unbiased=False).clamp_min(0.05)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i, player in enumerate(players):
                lp, kl, tlp = likelihood(model, reference, player, 'cuda')
                if a.smoke and i == 0:
                    assert abs(kl.item()) < 1e-5
                    gradients = torch.autograd.grad(tlp, model.parameters(), retain_graph=True, allow_unused=True)
                    trace_gradient = sum(g.abs().sum().item() for g in gradients if g is not None)
                    assert trace_gradient > 0 and torch.isfinite(tlp)
                loss = (-advantage[i].detach() * lp + 0.02 * kl) / 4
                loss.backward()
                losses.append(loss.item())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            assert torch.isfinite(norm)
            optimizer.step()
            completed += 1
            row = {'update': completed, 'rewards': scores, 'loss': sum(losses), 'gradient_norm': float(norm),
                   'trace_gradient_smoke': trace_gradient, 'seconds': time.time() - start,
                   'trace_lengths': [len(x.sequence) - (3 + len(board.move_stack)) for x in players]}
            with (a.root / 'updates.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
    finally:
        engine.quit()
    assert completed > 0
    model.training_policy = {'compact': True}
    save(a.root / 'model.pt', model, completed)
    (a.root / 'complete.json').write_text(json.dumps({'updates': completed, 'samples': completed * 4,
          'seconds': time.time() - start, 'deadline': global_deadline, 'checkpoint_sha256': sha(a.root / 'model.pt')}))


if __name__ == '__main__':
    main()
