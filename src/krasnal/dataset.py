"""Compatibility alias; use krasnal.data.history in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.data.history")
