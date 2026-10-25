"""Compatibility alias; use krasnal.tokenization.reasoning in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.tokenization.reasoning")
