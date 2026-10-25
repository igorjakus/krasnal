"""Compatibility alias; use krasnal.training.rl in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.training.rl")
