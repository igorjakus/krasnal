"""Compatibility alias; use krasnal.tokenization.history in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.tokenization.history")
