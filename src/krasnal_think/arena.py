"""Compatibility alias; use krasnal.eval.arena in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.eval.arena")
