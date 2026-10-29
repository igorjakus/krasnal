"""Evaluate an interleaved checkpoint with the pretraining effort:0 prefix."""

from __future__ import annotations

import numpy as np

import evaluate_interleave as evaluator
from krasnal.tokenization import reasoning as tokens


_record_to_tokens = evaluator.record_to_tokens
_targets = evaluator._targets
_evaluate = evaluator._evaluate


def _record_with_effort0(record: dict, interleaved: bool):
    if not interleaved:
        raise ValueError("this evaluator requires augmented/interleaved mode")
    ids, mask = _record_to_tokens(record, interleaved)
    ids = list(ids)
    ids.insert(3, tokens.effort_id(0))
    return ids, np.insert(mask, 3, 0)


def _targets_with_effort0(ids: list[int], interleaved: bool):
    if not interleaved or len(ids) < 4 or ids[3] != tokens.effort_id(0):
        raise ValueError("expected effort:0 immediately before the first SF move")
    original = ids[:3] + ids[4:]
    human, human_positions, sf, sf_positions = _targets(original, interleaved)
    return (
        human,
        [position + 1 for position in human_positions],
        sf,
        [position + 1 for position in sf_positions],
    )


def _evaluate_with_effort0(args):
    if args.mode != "augmented" or args.device != "cpu":
        raise ValueError("run with --mode augmented --device cpu for float32 evaluation")
    result = _evaluate(args)
    result["effort0_conditioning"] = True
    result["precision"] = "float32"
    result["device"] = "cpu"
    return result


evaluator.record_to_tokens = _record_with_effort0
evaluator._targets = _targets_with_effort0
evaluator._evaluate = _evaluate_with_effort0


if __name__ == "__main__":
    evaluator.main()
