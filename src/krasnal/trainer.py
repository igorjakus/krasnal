"""Compatibility alias; use krasnal.training.history in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.training.history")
