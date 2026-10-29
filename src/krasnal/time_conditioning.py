"""Compatibility alias; use krasnal.models.time_conditioning in new code."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("krasnal.models.time_conditioning")
