import pytest

from krasnal_think import tokens
from krasnal_think.annotate import Annotation
from krasnal_think.data import IGNORE
from krasnal_think.play import EFFORT
from krasnal_think.sft import batches, encode

ANNOTATION = Annotation(
    history=["e2e4", "e7e5"],
    variations=[["g1f3", "b8c6", "f1b5"], ["f1c4", "g8f6"]],
    scores=[(30, None), (-15, None)],
)


def _names(ids):
    return [tokens.VOCAB[i] if i != IGNORE else None for i in ids]


def test_effort_zero_predicts_only_the_move():
    ids, targets = encode(ANNOTATION, effort=0)
    assert _names(ids) == ["<bos>", "<elo:8>", "<elo:8>", "<effort:0>", "e2e4", "e7e5"]
    assert _names(targets) == [None, None, None, None, None, "g1f3"]


def test_effort_two_emits_truncated_variations():
    ids, targets = encode(ANNOTATION, effort=2)
    assert _names(ids)[6:] == [
        "<think>",
        "<pv>",
        "g1f3",
        "b8c6",
        "f1b5",
        "<cp:16>",
        "</pv>",
        "<pv>",
        "f1c4",
        "g8f6",
        "<delta:5>",
        "</pv>",
        "</think>",
    ]
    assert _names(targets)[:6] == [None] * 5 + ["<think>"]
    assert _names(targets)[-1] == "g1f3", "the played move is the last target"


@pytest.mark.parametrize("effort", sorted(EFFORT))
def test_targets_are_the_inputs_shifted_by_one(effort):
    ids, targets = encode(ANNOTATION, effort=effort)
    assert len(ids) == len(targets)
    full = [*ids, tokens.move_id("g1f3")]
    for position, target in enumerate(targets):
        if target != IGNORE:
            assert target == full[position + 1]


def test_mate_scores_survive_the_delta_conversion():
    mate = Annotation(
        history=["e2e4"],
        variations=[["d7d5"], ["g8f6"]],
        scores=[(None, 2), (120, None)],
    )
    names = _names(encode(mate, effort=2)[0])
    assert "<mate:+2>" in names
    assert names[-3] == f"<delta:{len(tokens.DELTA_EDGES)}>", "a missed mate is a large loss"


def test_buffered_batching_keeps_every_sample_once():
    samples = [([1] * (n % 37 + 3), [1] * (n % 37 + 3)) for n in range(500)]
    lengths = sorted(len(ids) for ids, _ in samples)
    seen = []
    for x, y in batches(iter(samples), 8, 64, "cpu", buffer=64):
        assert x.shape == y.shape
        seen += [int((row != tokens.PAD).sum()) for row in x]
    assert sorted(seen) == lengths


def test_variations_are_capped_by_effort():
    for effort, (count, plies) in EFFORT.items():
        names = _names(encode(ANNOTATION, effort)[0])
        assert names.count("<pv>") == min(count, len(ANNOTATION.variations))
        for variation in " ".join(n for n in names if n).split("<pv>")[1:]:
            moves = [w for w in variation.split() if not w.startswith("<")]
            assert len(moves) <= plies
