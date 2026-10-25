"""Compatibility alias; use krasnal.data.annotation in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.data.annotation")
