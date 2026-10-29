"""Compatibility alias; use krasnal.data.foundations in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.data.foundations")
