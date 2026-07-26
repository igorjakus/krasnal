import numpy as np

from krasnal_think import tokens
from krasnal_think.data import IGNORE, ShardWriter, encode_game, get_batch, load_stream


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
