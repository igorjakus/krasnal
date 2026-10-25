"""Identical human evaluation, but every SF input is the constant empty note."""
import evaluate_interleave as evaluator
from krasnal_think import tokens

encode = evaluator.record_to_tokens
targets = evaluator._targets


def star_encode(record, interleaved):
    if not interleaved:
        raise ValueError("Star control must retain interleaved sequence lengths")
    ids, mask = encode(record, interleaved)
    return [tokens.SF_MOVE_IDS.stop if i in tokens.SF_MOVE_IDS else i for i in ids], mask


def human_targets(ids, interleaved):
    human, positions, _, _ = targets(ids, interleaved)
    return human, positions, [], []


original_evaluate = evaluator._evaluate


def evaluate(args):
    result = original_evaluate(args)
    result.update(teacher_forced_stockfish=False, stockfish_move_namespace="not_used",
                  conditioning="constant_star", star_id=tokens.SF_MOVE_IDS.stop)
    return result


if __name__ == "__main__":
    evaluator.record_to_tokens = star_encode
    evaluator._targets = human_targets
    evaluator._evaluate = evaluate
    evaluator.main()
