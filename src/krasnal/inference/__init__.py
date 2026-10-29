"""Inference interfaces; load architecture-specific dependencies on demand."""

from importlib import import_module

_EXPORTS = {
    "StatelessBatchInferenceSession": "batch",
    "NoLegalMovesError": "exceptions",
    "Game": "game",
    "sample_token": "sampling",
    "InferenceSession": "session",
    "load_model": "utils",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value

__all__ = [
    "Game",
    "InferenceSession",
    "NoLegalMovesError",
    "StatelessBatchInferenceSession",
    "load_model",
    "sample_token",
]
