import math

import pytest

from krasnal_think.arena import Result


def test_elo_from_score():
    assert Result(50, 100).elo == 0
    assert Result(75, 100).elo == pytest.approx(190.85, abs=0.01)
    assert Result(25, 100).elo == pytest.approx(-190.85, abs=0.01)


def test_clean_sweep_is_finite_and_bounded():
    for result in (Result(100, 100), Result(0, 100)):
        assert math.isfinite(result.elo) and math.isfinite(result.margin)
    assert Result(100, 100).elo > 0 > Result(0, 100).elo


def test_margin_shrinks_with_more_games():
    assert Result(60, 100).margin > Result(600, 1000).margin
