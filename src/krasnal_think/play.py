"""Compatibility alias; use krasnal.inference.reasoning in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.inference.reasoning")
