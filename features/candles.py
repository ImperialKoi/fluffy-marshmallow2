"""
Candlestick primitives + a tiny helper used by every candlestick strategy.

All functions are vectorised over OHLC columns and look only at the bar(s)
referenced (bar t and a few prior bars via .shift(+k)), so they are strictly
backward-looking. The pattern definitions follow the standard descriptions in
Steve Nison's "Japanese Candlestick Charting Techniques" / Investopedia.
"""

import numpy as np
import pandas as pd


def body(o, c):
    return (c - o).abs()


def candle_range(h, l):
    return (h - l).replace(0.0, np.nan)


def upper_shadow(o, h, c):
    return h - pd.concat([o, c], axis=1).max(axis=1)


def lower_shadow(o, l, c):
    return pd.concat([o, c], axis=1).min(axis=1) - l


def is_bullish(o, c):
    return c > o


def is_bearish(o, c):
    return c < o


def trend_down(close, window=5):
    """Simple prior-downtrend filter: close below its SMA `window` bars back."""
    return close.shift(1) < close.rolling(window, min_periods=window).mean().shift(1)


def trend_up(close, window=5):
    return close.shift(1) > close.rolling(window, min_periods=window).mean().shift(1)


def hold_position(signal_bool: pd.Series, hold: int) -> pd.Series:
    """
    Turn a one-shot pattern trigger into a held long position: be long for `hold`
    bars after any trigger. `signal_bool[t]` True means the pattern completed on
    the close of bar t. Using rolling().max() over the trailing window keeps this
    backward-looking (only bars <= t are consulted).
    """
    pos = signal_bool.fillna(False).astype(float).rolling(hold, min_periods=1).max()
    return pos.fillna(0.0)
