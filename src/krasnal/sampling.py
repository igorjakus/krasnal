"""Compatibility alias; use krasnal.tokenization.sampling in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.tokenization.sampling")
