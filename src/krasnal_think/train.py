"""Compatibility alias; use krasnal.training.runtime in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.training.runtime")
