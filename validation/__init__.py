"""
Phase 2 — strategy validation harness.

Built on top of (never modifying) backtest/engine.py and backtest/metrics.py. Tests
every registered strategy across many symbols and out-of-sample through time, ranks
by robust cross-symbol consistency, corrects for multiple testing, and applies a
mechanical graduation rule. The out-of-sample set is sacred (see VALIDATION.md).
"""

from . import config, data, baselines, splits, multi_symbol, robustness, significance, report

__all__ = ["config", "data", "baselines", "splits", "multi_symbol",
           "robustness", "significance", "report"]
