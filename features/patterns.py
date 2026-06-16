"""
Chart-pattern detection from swing points — all strictly backward-looking.

Every detector returns two boolean Series aligned to df: (entry, exit). They are
built by walking bars left-to-right and, at each bar i, consulting ONLY swing
pivots that have already been *confirmed* by bar i (a pivot at bar j with `right`
bars after it is confirmed at j+right). Because a pivot is invisible until the
market prints its confirming bars, no future information ever reaches bar i. The
strategy files turn (entry, exit) into a forward-filled long/flat position.

Patterns implemented: double bottom, inverse head-and-shoulders, ascending
triangle, cup-and-handle. Definitions follow Bulkowski / Investopedia. The
detectors are intentionally simple, transparent heuristics (not exhaustive
pattern recognisers) — the project validates real edges out-of-sample later.
"""

import numpy as np
import pandas as pd


def swing_pivots(df: pd.DataFrame, left: int = 4, right: int = 4):
    """Confirmed swing highs/lows: lists of (conf_idx, bar_idx, price)."""
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    n = len(h)
    highs, lows = [], []
    for j in range(left, n - right):
        wh = h[j - left:j + right + 1]
        wl = l[j - left:j + right + 1]
        if h[j] == wh.max() and wh.argmax() == left:
            highs.append((j + right, j, float(h[j])))
        if l[j] == wl.min() and wl.argmin() == left:
            lows.append((j + right, j, float(l[j])))
    return highs, lows


def _empty(df):
    z = pd.Series(False, index=df.index)
    return z.copy(), z.copy()


def double_bottom(df, left=4, right=4, tol=0.04, lookback=120):
    """Two similar swing lows separated by a peak; buy the break above that peak."""
    highs, lows = swing_pivots(df, left, right)
    close = df["close"].to_numpy()
    n = len(close)
    entry = np.zeros(n, bool)
    exit_ = np.zeros(n, bool)
    hi_i = lo_i = 0
    act_hi, act_lo = [], []
    for i in range(n):
        while hi_i < len(highs) and highs[hi_i][0] <= i:
            act_hi.append(highs[hi_i]); hi_i += 1
        while lo_i < len(lows) and lows[lo_i][0] <= i:
            act_lo.append(lows[lo_i]); lo_i += 1
        recent_lo = [p for p in act_lo if i - p[1] <= lookback]
        if len(recent_lo) >= 2:
            (_, b1, p1), (_, b2, p2) = recent_lo[-2], recent_lo[-1]
            if b2 > b1 and abs(p1 - p2) / p1 <= tol:
                peaks = [p for p in act_hi if b1 < p[1] < b2]
                if peaks:
                    neckline = max(pk[2] for pk in peaks)
                    if close[i] > neckline:
                        entry[i] = True
                    if close[i] < min(p1, p2):
                        exit_[i] = True
    return pd.Series(entry, index=df.index), pd.Series(exit_, index=df.index)


def inverse_head_shoulders(df, left=4, right=4, tol=0.05, lookback=160):
    """Three lows (shoulder-head-shoulder, head lowest, shoulders similar); buy the
    break above the neckline (the highs between them)."""
    highs, lows = swing_pivots(df, left, right)
    close = df["close"].to_numpy()
    n = len(close)
    entry = np.zeros(n, bool)
    exit_ = np.zeros(n, bool)
    hi_i = lo_i = 0
    act_hi, act_lo = [], []
    for i in range(n):
        while hi_i < len(highs) and highs[hi_i][0] <= i:
            act_hi.append(highs[hi_i]); hi_i += 1
        while lo_i < len(lows) and lows[lo_i][0] <= i:
            act_lo.append(lows[lo_i]); lo_i += 1
        recent_lo = [p for p in act_lo if i - p[1] <= lookback]
        if len(recent_lo) >= 3:
            (_, bL, pL), (_, bH, pH), (_, bR, pR) = recent_lo[-3], recent_lo[-2], recent_lo[-1]
            head_lowest = pH < pL and pH < pR
            shoulders_similar = abs(pL - pR) / pL <= tol
            if bL < bH < bR and head_lowest and shoulders_similar:
                necks = [pk[2] for pk in act_hi if bL < pk[1] < bR]
                if necks:
                    neckline = max(necks)
                    if close[i] > neckline:
                        entry[i] = True
                    if close[i] < pR:
                        exit_[i] = True
    return pd.Series(entry, index=df.index), pd.Series(exit_, index=df.index)


def ascending_triangle(df, left=4, right=4, tol=0.03, lookback=120):
    """Flat resistance (similar highs) + rising lows; buy the break above resistance."""
    highs, lows = swing_pivots(df, left, right)
    close = df["close"].to_numpy()
    n = len(close)
    entry = np.zeros(n, bool)
    exit_ = np.zeros(n, bool)
    hi_i = lo_i = 0
    act_hi, act_lo = [], []
    for i in range(n):
        while hi_i < len(highs) and highs[hi_i][0] <= i:
            act_hi.append(highs[hi_i]); hi_i += 1
        while lo_i < len(lows) and lows[lo_i][0] <= i:
            act_lo.append(lows[lo_i]); lo_i += 1
        rh = [p for p in act_hi if i - p[1] <= lookback]
        rl = [p for p in act_lo if i - p[1] <= lookback]
        if len(rh) >= 2 and len(rl) >= 2:
            (_, _, h1), (_, _, h2) = rh[-2], rh[-1]
            (_, _, l1), (_, _, l2) = rl[-2], rl[-1]
            flat_top = abs(h1 - h2) / h1 <= tol
            rising_lows = l2 > l1
            if flat_top and rising_lows:
                resistance = (h1 + h2) / 2.0
                if close[i] > resistance:
                    entry[i] = True
                if close[i] < l2:
                    exit_[i] = True
    return pd.Series(entry, index=df.index), pd.Series(exit_, index=df.index)


def cup_and_handle(df, left=4, right=4, rim_tol=0.05, min_depth=0.12,
                   max_depth=0.45, lookback=200):
    """Rounded base back to a prior rim, a shallow handle, then break above the rim.
    Heuristic: a confirmed high (rim), a cup low `min_depth`..`max_depth` below it,
    price recovered to near the rim, then a close breaks above the rim."""
    highs, lows = swing_pivots(df, left, right)
    close = df["close"].to_numpy()
    n = len(close)
    entry = np.zeros(n, bool)
    exit_ = np.zeros(n, bool)
    hi_i = lo_i = 0
    act_hi, act_lo = [], []
    for i in range(n):
        while hi_i < len(highs) and highs[hi_i][0] <= i:
            act_hi.append(highs[hi_i]); hi_i += 1
        while lo_i < len(lows) and lows[lo_i][0] <= i:
            act_lo.append(lows[lo_i]); lo_i += 1
        rims = [p for p in act_hi if 20 <= i - p[1] <= lookback]
        if rims:
            _, b_rim, p_rim = max(rims, key=lambda x: x[2])
            cup_lows = [p for p in act_lo if b_rim < p[1] < i]
            if cup_lows:
                _, _, p_low = min(cup_lows, key=lambda x: x[2])
                depth = (p_rim - p_low) / p_rim
                if min_depth <= depth <= max_depth and close[i] > p_rim:
                    entry[i] = True
                if close[i] < p_low:
                    exit_[i] = True
    return pd.Series(entry, index=df.index), pd.Series(exit_, index=df.index)


def to_position(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Forward-fill a long/flat position from entry/exit triggers (backward-looking)."""
    raw = pd.Series(np.where(entry, 1.0, np.where(exit_, 0.0, np.nan)), index=entry.index)
    return raw.ffill().fillna(0.0)
