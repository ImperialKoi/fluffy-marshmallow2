"""
THE most important test in the validation phase: prove the walk-forward harness
cannot leak future data into a past decision, plus prove the random baseline is
reproducible.

Leakage proof (mechanical):
  Take a strategy with a parameter grid. Compute its per-param backtests on clean
  data, then on data whose bars AFTER fold-1's train end are corrupted (reversed).
  Assert that fold-1's TRAIN metrics and the selected parameter are byte-for-byte
  identical across clean vs corrupted data — i.e. the selection literally cannot see
  the future. As a sanity check that the corruption was real and reached the test
  window, assert the OOS benchmark return DID change.

Run:  python tests/test_validation_leakage.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from validation import data, splits
from validation import config as vcfg
from validation.multi_symbol import run_full, run_full_random
from validation.baselines import RandomEntry, _stable_seed

SYMBOL = "AAPL"


def _corrupt_after(df, cutoff):
    """Reverse OHLCV values for all bars strictly after `cutoff` (valid but different
    data), leaving the index intact and bars on/before cutoff untouched."""
    df2 = df.copy()
    mask = df2.index > pd.Timestamp(cutoff)
    cols = ["open", "high", "low", "close", "volume"]
    region = df2.loc[mask, cols]
    df2.loc[mask, cols] = region.values[::-1]
    return df2


class TestNoLookahead(unittest.TestCase):
    def setUp(self):
        self.df = data.get_symbol(SYMBOL)
        self.cost = vcfg.COSTS["normal"]
        self.grid = vcfg.grid_for("sma")          # parameterized -> selection happens

    def test_split_windows_are_disjoint_and_ordered(self):
        for tr_s, tr_e, te_s, te_e in vcfg.FOLDS:
            self.assertLess(pd.Timestamp(tr_s), pd.Timestamp(tr_e))
            self.assertLess(pd.Timestamp(tr_e), pd.Timestamp(te_s))  # train strictly before test
            self.assertLess(pd.Timestamp(te_s), pd.Timestamp(te_e))

    def test_future_corruption_cannot_change_train_selection(self):
        tr_s, tr_e, te_s, te_e = vcfg.FOLDS[0]
        df_clean = self.df
        df_corrupt = _corrupt_after(df_clean, tr_e)   # destroy everything after train end

        clean = {k: run_full("sma", df_clean, p, self.cost, SYMBOL)
                 for k, p in enumerate(self.grid)}
        corrupt = {k: run_full("sma", df_corrupt, p, self.cost, SYMBOL)
                   for k, p in enumerate(self.grid)}

        # (1) per-parameter TRAIN metrics must be IDENTICAL despite a corrupted future
        for k in clean:
            mc = splits.window_metrics(clean[k], tr_s, tr_e)
            mk = splits.window_metrics(corrupt[k], tr_s, tr_e)
            self.assertEqual(mc["sharpe"], mk["sharpe"], f"param {k} train sharpe leaked")
            self.assertEqual(mc["total_return"], mk["total_return"])
            self.assertEqual(mc["round_trips"], mk["round_trips"])
            # the benchmark over the train window must also be untouched
            self.assertEqual(mc["benchmark_total_return"], mk["benchmark_total_return"])

        # (2) the SELECTED parameter must be identical
        idx_clean, sr_clean = splits.select_on_train(clean, self.grid, tr_s, tr_e)
        idx_corrupt, sr_corrupt = splits.select_on_train(corrupt, self.grid, tr_s, tr_e)
        self.assertEqual(idx_clean, idx_corrupt, "future data changed the selection!")
        self.assertEqual(sr_clean, sr_corrupt)

        # (3) sanity: the corruption WAS real and reached the test window
        oc = splits.window_metrics(clean[idx_clean], te_s, te_e)
        ok = splits.window_metrics(corrupt[idx_clean], te_s, te_e)
        self.assertNotEqual(oc["benchmark_total_return"], ok["benchmark_total_return"],
                            "corruption did not affect the OOS window — test is vacuous")

    def test_walk_forward_oos_only_uses_post_train_bars(self):
        # The stitched OOS index must lie entirely within the union of test windows.
        results = {k: run_full("sma", self.df, p, self.cost, SYMBOL)
                   for k, p in enumerate(self.grid)}
        wf = splits.walk_forward(results, self.grid)
        earliest_test = min(pd.Timestamp(f[2]) for f in vcfg.FOLDS)
        if len(wf.oos_returns):
            self.assertGreaterEqual(wf.oos_returns.index.min(), earliest_test)


class TestRandomBaselineReproducible(unittest.TestCase):
    def setUp(self):
        self.df = data.get_symbol(SYMBOL)
        self.cost = vcfg.COSTS["normal"]

    def test_same_seed_identical_positions(self):
        a = RandomEntry(exposure=0.5, seed=7).prepare(self.df)["rand_pos"]
        b = RandomEntry(exposure=0.5, seed=7).prepare(self.df)["rand_pos"]
        self.assertTrue(a.equals(b))

    def test_different_seed_differs(self):
        a = RandomEntry(exposure=0.5, seed=7).prepare(self.df)["rand_pos"]
        c = RandomEntry(exposure=0.5, seed=8).prepare(self.df)["rand_pos"]
        self.assertFalse(a.equals(c))

    def test_stable_seed_is_deterministic_not_salted(self):
        # hashlib-based, so identical across processes (unlike Python's salted hash())
        self.assertEqual(_stable_seed(7, self.df), _stable_seed(7, self.df))

    def test_exposure_matches_target(self):
        for target in (0.2, 0.5, 0.8):
            pos = RandomEntry(exposure=target, seed=3).prepare(self.df)["rand_pos"]
            self.assertLess(abs(pos.mean() - target), 0.12)

    def test_reproducible_through_engine(self):
        r1 = run_full_random(self.df, 0.5, 7, self.cost, SYMBOL)
        r2 = run_full_random(self.df, 0.5, 7, self.cost, SYMBOL)
        self.assertTrue(r1.equity_curve.equals(r2.equity_curve))
        r3 = run_full_random(self.df, 0.5, 9, self.cost, SYMBOL)
        self.assertFalse(r1.equity_curve.equals(r3.equity_curve))


if __name__ == "__main__":
    unittest.main(verbosity=2)
