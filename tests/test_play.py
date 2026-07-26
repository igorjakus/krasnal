import chess
import pytest
import torch

from krasnal_think.model import GPT, GPTConfig
from krasnal_think.play import EFFORT, Player
from krasnal_think.tokens import VOCAB_SIZE


def _player(effort):
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=256, n_layer=2, n_head=2, n_embd=32)
    return Player(GPT(config).eval(), effort=effort)


@pytest.mark.parametrize("effort", sorted(EFFORT))
def test_random_model_only_plays_legal_moves(effort):
    """An untrained model is pure noise, so legality must come from masking alone."""
    player = _player(effort)
    moves = []
    for _ in range(12):
        if player.board.is_game_over():
            break
        moves.append(player.move())

    replay = chess.Board()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        assert move in replay.legal_moves
        replay.push(move)
    assert replay.fen() == player.board.fen()


@pytest.mark.parametrize("effort", [0, 4])
def test_thinking_is_evicted_from_cache(effort):
    """Cache length must track the moves-only prefix, whatever the effort level."""
    player = _player(effort)
    prefix = player.cache.length
    for ply in range(1, 6):
        player.move()
        assert player.cache.length == prefix + ply


@pytest.mark.parametrize("effort", [1, 2, 3, 4])
def test_variations_are_legal_from_the_root_position(effort):
    player = _player(effort)
    for _ in range(6):
        root = player.board.copy()
        player.move()
        board = None
        for token in player.thinking:
            if token == "<pv>":
                board = root.copy()
            elif token == "</pv>":
                board = None
            elif board is not None and not token.startswith("<"):
                move = chess.Move.from_uci(token)
                assert move in board.legal_moves, f"illegal {token} in {board.fen()}"
                board.push(move)


def test_long_game_slides_the_context_window():
    """A game longer than block_size must slide, not overflow the cache."""
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=96, n_layer=2, n_head=2, n_embd=32)
    player = Player(GPT(config).eval(), effort=4)
    for _ in range(120):
        if player.board.is_game_over():
            break
        player.move()
        assert player.cache.length <= config.block_size


def test_reset_to_arbitrary_position():
    player = _player(2)
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    player.reset(board)
    assert chess.Move.from_uci(player.move()) in board.legal_moves
