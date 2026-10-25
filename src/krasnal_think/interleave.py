"""Compatibility alias; use krasnal.data.interleave in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.data.interleave")
