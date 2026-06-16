"""
Support / resistance ("floor" / "ceiling") detection with a breakout-probability
estimate. Everything here is strictly backward-looking, which is what lets the
backtest engine precompute these columns safely (see CLAUDE.md §3).

How levels are found
--------------------
1. Fractal swing pivots. A swing HIGH at bar j needs `left` lower highs before it
   and `right` lower highs after it; a swing LOW is the mirror. Critically, such a
   pivot can only be *known* once the `right` confirming bars have printed, i.e.
   at bar j+right. We therefore tag every pivot with a confirmation index and,
   when looking at bar i, only ever use pivots with confirmation index <= i. That
   single rule is what prevents lookahead: a pivot is invisible until the market
   has actually confirmed it.
2. Clustering. Confirmed pivot prices that sit within `tol` of each other are
   merged into one horizontal level; the number of pivots in the cluster is the
   "touch count" (how often the level has been respected).
3. For each bar we expose the nearest level *above* the close (resistance /
   ceiling) and the nearest *below* it (support / floor), plus each level's touch
   count.

Breakout probability  (transparent, documented, NOT curve-fit)
--------------------------------------------------------------
For the nearest resistance R above price P we estimate the probability that price
breaks UP through it within the near future, via a logistic score over five
backward-looking features (the canonical drivers cited in the S&R literature —
more tests weaken a level, rising volume fuels breaks, momentum/coiled
consolidation precede breaks, distance lowers imminence):

    z = b0
        + w_touch  * (touches - TOUCH_REF)      # more prior tests -> weaker level
        + w_vol    * (vol/vol_ma - 1)           # volume expansion into the level
        + w_mom    * mom                        # momentum pushing toward the level
        + w_consol * consolidation              # fraction of recent bars coiled at level
        - w_dist   * (distance / atr_frac)      # far away -> less imminent
    P(break) = 1 / (1 + e^-z)

The mirror score (negative momentum) gives P(break DOWN) through support. Weights
are illustrative heuristics chosen for sensible behaviour, NOT fitted to data —
the project validates edges out-of-sample in a later phase, so we deliberately
avoid overfitting these constants. References: StockCharts/Investopedia on S&R
strength; the "50%+ volume expansion confirms a breakout" rule of thumb.
"""

import numpy as np
import pandas as pd

from features import indicators as ind

# --- logistic weights (heuristic, documented, intentionally un-fitted) -------
B0 = -0.6
W_TOUCH = 0.30
TOUCH_REF = 2.0
W_VOL = 0.90
W_MOM = 6.0
W_CONSOL = 1.2
W_DIST = 1.0


def _pivots(high: np.ndarray, low: np.ndarray, left: int, right: int):
    """Return (pivot_highs, pivot_lows) as lists of (confirmation_index, price)."""
    n = len(high)
    ph, pl = [], []
    for j in range(left, n - right):
        wh = high[j - left:j + right + 1]
        wl = low[j - left:j + right + 1]
        if high[j] == wh.max() and wh.argmax() == left:
            ph.append((j + right, float(high[j])))
        if low[j] == wl.min() and wl.argmin() == left:
            pl.append((j + right, float(low[j])))
    return ph, pl


def _cluster(items, tol: float):
    """Merge (conf_idx, price) pairs whose prices sit within `tol` relative.
    Returns list of (level_price, touch_count)."""
    if not items:
        return []
    ordered = sorted(items, key=lambda x: x[1])
    levels, cur = [], [ordered[0]]
    for conf, price in ordered[1:]:
        if abs(price - cur[-1][1]) / cur[-1][1] <= tol:
            cur.append((conf, price))
        else:
            levels.append(cur)
            cur = [(conf, price)]
    levels.append(cur)
    return [(float(np.mean([p for _, p in g])), len(g)) for g in levels]


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def compute_levels(df: pd.DataFrame, left: int = 3, right: int = 3,
                   tol: float = 0.02, vol_window: int = 20,
                   mom_window: int = 10, consol_window: int = 20) -> pd.DataFrame:
    """
    Return a DataFrame (indexed like df) of backward-looking S/R columns:
        resistance, support             nearest level above / below the close
        res_touches, sup_touches        how many times each level was tested
        dist_resistance, dist_support   fractional distance to each level
        p_break_resistance              P(price breaks UP through resistance)
        p_break_support                 P(price breaks DOWN through support)
    """
    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    vol = df["volume"].to_numpy()
    vol_ma = df["volume"].rolling(vol_window, min_periods=1).mean().to_numpy()
    atrv = ind.atr(df["high"], df["low"], df["close"], 14).to_numpy()
    mom = (df["close"] / df["close"].shift(mom_window) - 1.0).to_numpy()

    ph, pl = _pivots(high, low, left, right)

    res = np.full(n, np.nan)
    sup = np.full(n, np.nan)
    res_t = np.zeros(n)
    sup_t = np.zeros(n)
    d_res = np.full(n, np.nan)
    d_sup = np.full(n, np.nan)
    p_res = np.full(n, np.nan)
    p_sup = np.full(n, np.nan)

    hi_i = lo_i = 0
    active_hi, active_lo = [], []  # lists of (conf_idx, price)
    for i in range(n):
        while hi_i < len(ph) and ph[hi_i][0] <= i:
            active_hi.append(ph[hi_i]); hi_i += 1
        while lo_i < len(pl) and pl[lo_i][0] <= i:
            active_lo.append(pl[lo_i]); lo_i += 1

        c = close[i]
        levels_hi = _cluster(active_hi, tol)
        levels_lo = _cluster(active_lo, tol)
        # support/resistance can come from either pivot type that sits above/below
        all_levels = levels_hi + levels_lo
        above = [(lp, t) for lp, t in all_levels if lp > c]
        below = [(lp, t) for lp, t in all_levels if lp < c]

        atr_frac = (atrv[i] / c) if (c > 0 and atrv[i] == atrv[i]) else np.nan

        if above:
            lp, t = min(above, key=lambda x: x[0] - c)
            res[i], res_t[i] = lp, t
            d_res[i] = (lp - c) / c
            p_res[i] = _break_prob(d_res[i], t, vol[i], vol_ma[i], mom[i],
                                   close, i, lp, tol, consol_window, atr_frac, up=True)
        if below:
            lp, t = min(below, key=lambda x: c - x[0])
            sup[i], sup_t[i] = lp, t
            d_sup[i] = (c - lp) / c
            p_sup[i] = _break_prob(d_sup[i], t, vol[i], vol_ma[i], mom[i],
                                   close, i, lp, tol, consol_window, atr_frac, up=False)

    return pd.DataFrame({
        "resistance": res, "support": sup,
        "res_touches": res_t, "sup_touches": sup_t,
        "dist_resistance": d_res, "dist_support": d_sup,
        "p_break_resistance": p_res, "p_break_support": p_sup,
    }, index=df.index)


def _break_prob(distance, touches, v, v_ma, mom, close, i, level, tol,
                consol_window, atr_frac, up: bool) -> float:
    vol_surge = (v / v_ma - 1.0) if v_ma and v_ma == v_ma else 0.0
    lo = max(0, i - consol_window + 1)
    window = close[lo:i + 1]
    consolidation = float(np.mean(np.abs(window - level) / level <= tol)) if len(window) else 0.0
    signed_mom = mom if mom == mom else 0.0
    if not up:
        signed_mom = -signed_mom
    dist_term = (distance / atr_frac) if (atr_frac and atr_frac == atr_frac) else distance * 50.0
    z = (B0
         + W_TOUCH * (touches - TOUCH_REF)
         + W_VOL * vol_surge
         + W_MOM * signed_mom
         + W_CONSOL * consolidation
         - W_DIST * dist_term)
    return _logistic(z)
