"""Compatibility alias; use krasnal.models.history in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.models.history")
