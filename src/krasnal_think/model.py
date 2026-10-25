"""Compatibility alias; use krasnal.models.reasoning in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.models.reasoning")
