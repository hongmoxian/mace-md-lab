"""MACE dynamics and cross-framework conservative force benchmarks."""

from importlib.util import find_spec

if find_spec("e3nn") is not None:
    from . import _e3nn as _e3nn
