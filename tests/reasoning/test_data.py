import numpy as np
import torch

from krasnal_think import tokens
from krasnal_think.data import (
    EVAL_IGNORE,
    IGNORE,
    EvalShardWriter,
    ShardWriter,
    encode_evals,
    encode_game,
    get_batch,
    load_eval_stream,
    load_stream,
    mtp_targets,
)


def test_encode_game_round_trip():
    ids = encode_game("e2e4 e7e5 g1f3", 1550, 2450)
    assert [tokens.VOCAB[i] for i in ids] == [
        "<bos>",
        "<elo:3>",
        "<elo:8>",
        "<effort:0>",
        "e2e4",
        "e7e5",
        "g1f3",
        "<eos>",
    ]


def test_get_batch_masks_conditioning_only(tmp_path):
    writer = ShardWriter(tmp_path / "s.bin")
    for _ in range(200):
        writer.write(encode_game("e2e4 e7e5 g1f3 b8c6", 1500, 1500))
    writer.close()

    stream = load_stream(tmp_path / "s.bin")
    x, y = get_batch(stream, 8, 16, "cpu", np.random.default_rng(0))
    assert x.shape == y.shape == (8, 16)

    masked = y == IGNORE
    assert masked.any(), "prefix tokens should be masked somewhere in the batch"
    shifted = np.asarray(x)[:, 1:]
    assert (
        np.asarray(y)[:, :-1][~np.asarray(masked)[:, :-1]] == shifted[~np.asarray(masked)[:, :-1]]
    ).all()
    kept = {tokens.VOCAB[int(t)] for t in np.asarray(y)[~np.asarray(masked)]}
    assert not any(name.startswith(("<elo:", "<effort:", "<bos>")) for name in kept)
    assert "<eos>" in kept and "e2e4" in kept


def test_eval_labels_align_with_moves(tmp_path):
    game = encode_game("e2e4 e7e5 g1f3", 1500, 1500)
    labels = encode_evals([20, -30], 3)
    assert len(labels) == len(game)
    assert labels[:4] == [EVAL_IGNORE] * 4
    assert labels[4:6] == [15, 12]
    assert labels[6:] == [EVAL_IGNORE, EVAL_IGNORE]

    token_writer = ShardWriter(tmp_path / "tokens.bin")
    eval_writer = EvalShardWriter(tmp_path / "evals.bin")
    for _ in range(100):
        token_writer.write(game)
        eval_writer.write(labels)
    token_writer.close()
    eval_writer.close()

    batch = get_batch(
        load_stream(tmp_path / "tokens.bin"),
        4,
        16,
        "cpu",
        np.random.default_rng(0),
        load_eval_stream(tmp_path / "evals.bin"),
    )
    assert len(batch) == 3
    assert batch[2].shape == (4, 16)


def test_mtp_targets_shift_and_mask_conditioning():
    eos = tokens.EOS
    m1, m2, m3 = (
        tokens.ID["e2e4"],
        tokens.ID["e7e5"],
        tokens.ID["g1f3"],
    )
    # Simulate get_batch output: conditioning *labels* already masked.
    targets = torch.tensor([[IGNORE, IGNORE, m1, m2, m3, eos]])
    out = mtp_targets(targets)[0]

    # position 0: x[i+2] is the elo conditioning token -> not a valid label
    assert out[0] == IGNORE
    # position 1: x[i+2] is e2e4 (realized t+1 was the elo token — still a legal input)
    assert out[1] == m1
    # position 2: x[i+2] is e7e5
    assert out[2] == m2
    # position 3: x[i+2] is g1f3
    assert out[3] == m3
    # position 5: no t+2 exists -> masked
    assert out[5] == IGNORE


def test_mtp_targets_match_get_batch_alignment(tmp_path):
    writer = ShardWriter(tmp_path / "s.bin")
    for _ in range(100):
        writer.write(encode_game("e2e4 e7e5 g1f3 b8c6", 1500, 1500))
    writer.close()
    stream = load_stream(tmp_path / "s.bin")
    x, targets = get_batch(stream, 4, 32, "cpu", np.random.default_rng(0))
    mtp = mtp_targets(targets)

    # Wherever the trunk label at i+1 is a real move, the MTP label at i must
    # equal the token two ahead of the input. The last two slots are excluded:
    # their t+2 sits at or past the crop edge, so x cannot verify them.
    valid = torch.nn.functional.pad(targets[..., 1:] != IGNORE, (0, 1), value=False)
    valid[..., -2:] = False
    two_ahead = torch.zeros_like(x)
    two_ahead[..., :-2] = x[..., 2:]
    assert (mtp[valid] == two_ahead[valid]).all()
