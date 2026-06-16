"""
Statistical rigor: correct for the fact that testing many strategies on many
symbols inflates false winners.

Two complementary corrections:

  * Deflated Sharpe Ratio (DSR; Bailey & López de Prado, 2014). The PREFERRED
    correction. It adjusts an observed Sharpe for (a) the number of trials, (b)
    track-record length, and (c) non-normal returns (skew/kurtosis). It answers:
    "after accounting for how many strategies we tried, is this Sharpe still
    significantly > 0?" The deflation benchmark SR0 is the EXPECTED MAXIMUM Sharpe
    under the null given N trials and the cross-trial dispersion of Sharpes.

  * Bonferroni. The minimum bar: a one-sided t-test that a strategy's mean
    cross-symbol OOS Sharpe > 0, with alpha divided by the number of strategies.

Also here: the explicit baseline gates (beat buy-and-hold and a matched random
baseline on a majority of symbols) and the minimum-trade gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew as _skew, kurtosis as _kurtosis, ttest_1samp

from . import config as vcfg

EULER = 0.5772156649015329


def per_period_sharpe(returns: pd.Series) -> float:
    sd = returns.std()
    return float(returns.mean() / sd) if sd > 0 else 0.0


def prob_sharpe_ratio(sr: float, n: int, skew: float, kurt: float,
                      sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_star). `kurt` is non-excess."""
    var = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    denom = np.sqrt(max(var, 1e-12))
    z = (sr - sr_star) * np.sqrt(max(n - 1, 1)) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """E[max Sharpe] under the null across N independent trials (per-period units)."""
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_std * ((1.0 - EULER) * z1 + EULER * z2))


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_std: float) -> dict:
    """Deflated Sharpe Ratio for a (per-period) return series. Returns a dict with
    the DSR probability and the pieces that produced it."""
    n = len(returns)
    if n < 20:
        return {"dsr": float("nan"), "sr": float("nan"), "sr0": float("nan"),
                "skew": float("nan"), "kurt": float("nan"), "n_obs": int(n)}
    sr = per_period_sharpe(returns)
    sk = float(_skew(returns, bias=False))
    ku = float(_kurtosis(returns, fisher=False, bias=False))  # non-excess
    sr0 = expected_max_sharpe(n_trials, sr_std)
    dsr = prob_sharpe_ratio(sr, n, sk, ku, sr_star=sr0)
    return {"dsr": dsr, "sr": sr, "sr0": sr0, "skew": sk, "kurt": ku, "n_obs": int(n)}


def bonferroni_test(oos_sharpes, n_strategies: int, alpha: float = None) -> dict:
    """One-sided t-test (mean cross-symbol OOS Sharpe > 0) with Bonferroni alpha."""
    alpha = vcfg.BONFERRONI_ALPHA if alpha is None else alpha
    arr = np.array([s for s in oos_sharpes if s == s], dtype=float)
    if len(arr) < 3:
        return {"significant": False, "p_one_sided": float("nan"), "n": int(len(arr))}
    t, p_two = ttest_1samp(arr, 0.0)
    p_one = (p_two / 2.0) if t > 0 else (1.0 - p_two / 2.0)
    adj = alpha / max(1, n_strategies)
    return {"significant": bool(t > 0 and p_one < adj),
            "p_one_sided": float(p_one), "adj_alpha": float(adj),
            "t": float(t), "n": int(len(arr))}


def pooled_oos_returns(rows: list[dict]) -> pd.Series:
    """Equal-weight portfolio of a strategy's stitched OOS returns across symbols
    (mean across symbols by date). This is the strategy-level series we deflate."""
    series = []
    for r in rows:
        d = r.get("oos_returns") or {}
        if not d:
            continue
        s = pd.Series({pd.Timestamp(k): v for k, v in d.items()}).sort_index()
        if len(s) > 1:
            series.append(s)
    if not series:
        return pd.Series(dtype=float)
    frame = pd.concat(series, axis=1)
    return frame.mean(axis=1).dropna()


def cross_trial_sr_std(pooled_by_strategy: dict[str, pd.Series]) -> float:
    """Dispersion of per-period Sharpe across strategies — the DSR deflation input."""
    srs = [per_period_sharpe(s) for s in pooled_by_strategy.values() if len(s) > 1]
    srs = [x for x in srs if x == x]
    return float(np.std(srs, ddof=1)) if len(srs) > 1 else 0.0
