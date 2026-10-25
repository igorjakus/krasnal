"""Compatibility alias; use krasnal.training.sft in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.training.sft")
