"""Evaluation interfaces for history and reasoning models."""

from importlib import import_module


def __getattr__(name):
    if name not in ("ChessEvaluator", "chess_evaluator_from_config"):
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.evaluator"), name)
    globals()[name] = value
    return value

__all__ = [
    "ChessEvaluator",
    "chess_evaluator_from_config",
]
