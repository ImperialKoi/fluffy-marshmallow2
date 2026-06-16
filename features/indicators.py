"""
Feature engine: technical indicators.

IMPORTANT — no lookahead bias: every function here is strictly *backward looking*.
The value at row t uses only data from rows <= t. That is what makes it safe for
the backtest engine to precompute these on the full series; the engine guarantees
the rest by only acting on the *next* bar's open (see backtest/engine.py).

These are written from scratch with pandas/numpy so the demo has no heavy
dependencies. On your own machine you can swap in `pandas_ta` or TA-Lib for the
150+ indicators they provide; the strategy interface won't change.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing). Ranges 0-100."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range — a volatility measure used for sizing/stops."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Returns (middle, upper, lower) Bollinger Bands."""
    mid = sma(series, window)
    sd = series.rolling(window=window, min_periods=window).std()
    return mid, mid + num_std * sd, mid - num_std * sd


# ===========================================================================
# Extended indicator library (all strictly backward-looking).
#
# Every function below produces, for row t, a value that depends only on rows
# <= t. Building blocks used: rolling windows (look back), ewm with positive
# spans/alphas (look back), and series.shift(k) with k > 0 (look back). We NEVER
# use center=True, .shift(-k), or whole-series statistics, because those would
# leak future information and break the no-lookahead contract in CLAUDE.md §3.
# ===========================================================================


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).max()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).min()


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's RMA smoothing (used by RSI/ADX/ATR), as an EWM with alpha=1/n."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def stochastic(high, low, close, k_window: int = 14, d_window: int = 3):
    """Stochastic oscillator. Returns (%K, %D), both 0-100. (George Lane)."""
    hh = rolling_max(high, k_window)
    ll = rolling_min(low, k_window)
    percent_k = 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)
    percent_d = percent_k.rolling(d_window, min_periods=d_window).mean()
    return percent_k, percent_d


def williams_r(high, low, close, window: int = 14) -> pd.Series:
    """Williams %R, ranges -100..0 (Larry Williams)."""
    hh = rolling_max(high, window)
    ll = rolling_min(low, window)
    return -100.0 * (hh - close) / (hh - ll).replace(0.0, np.nan)


def cci(high, low, close, window: int = 20) -> pd.Series:
    """Commodity Channel Index (Donald Lambert)."""
    tp = (high + low + close) / 3.0
    ma = sma(tp, window)
    md = (tp - ma).abs().rolling(window, min_periods=window).mean()
    return (tp - ma) / (0.015 * md).replace(0.0, np.nan)


def roc(series: pd.Series, window: int = 12) -> pd.Series:
    """Rate of change, percent."""
    return (series / series.shift(window) - 1.0) * 100.0


def adx(high, low, close, window: int = 14):
    """Average Directional Index. Returns (+DI, -DI, ADX). (Welles Wilder)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = true_range(high, low, close)
    atr_ = wilder_smooth(tr, window)
    plus_di = 100.0 * wilder_smooth(plus_dm, window) / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * wilder_smooth(minus_dm, window) / atr_.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_ = wilder_smooth(dx, window)
    return plus_di, minus_di, adx_


def mfi(high, low, close, volume, window: int = 14) -> pd.Series:
    """Money Flow Index — a volume-weighted RSI, 0-100 (Gene Quong)."""
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    pos = raw_mf.where(tp > tp.shift(1), 0.0)
    neg = raw_mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(window, min_periods=window).sum()
    neg_sum = neg.rolling(window, min_periods=window).sum()
    mr = pos_sum / neg_sum.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + mr)


def obv(close, volume) -> pd.Series:
    """On-Balance Volume (Joe Granville). Cumulative -> uses only past bars."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def ad_line(high, low, close, volume) -> pd.Series:
    """Accumulation/Distribution line (Marc Chaikin)."""
    rng = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfm = mfm.fillna(0.0)
    return (mfm * volume).cumsum()


def vwap_rolling(high, low, close, volume, window: int = 20) -> pd.Series:
    """Rolling VWAP over `window` bars (anchored windows avoid future leakage)."""
    tp = (high + low + close) / 3.0
    pv = (tp * volume).rolling(window, min_periods=window).sum()
    vv = volume.rolling(window, min_periods=window).sum()
    return pv / vv.replace(0.0, np.nan)


def keltner(high, low, close, ema_window: int = 20, atr_window: int = 10,
            mult: float = 2.0):
    """Keltner Channels (Chester Keltner). Returns (middle, upper, lower)."""
    mid = ema(close, ema_window)
    a = atr(high, low, close, atr_window)
    return mid, mid + mult * a, mid - mult * a


def donchian(high, low, window: int = 20):
    """Donchian channel (Richard Donchian). Returns (upper, lower, middle)."""
    upper = rolling_max(high, window)
    lower = rolling_min(low, window)
    return upper, lower, (upper + lower) / 2.0


def supertrend(high, low, close, window: int = 10, mult: float = 3.0):
    """
    Supertrend (Olivier Seban). Returns (line, direction) where direction is
    +1 (uptrend) / -1 (downtrend). Recursive but strictly backward-looking: bar
    i depends only on bars <= i.
    """
    a = atr(high, low, close, window)
    hl2 = (high + low) / 2.0
    upper = (hl2 + mult * a).to_numpy()
    lower = (hl2 - mult * a).to_numpy()
    c = close.to_numpy()
    n = len(c)
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    st = np.full(n, np.nan)
    direction = np.ones(n)
    for i in range(n):
        if i == 0 or np.isnan(upper[i]) or np.isnan(fu[i - 1]):
            fu[i] = upper[i]
            fl[i] = lower[i]
            st[i] = lower[i]
            direction[i] = 1.0
            continue
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if c[i] > fu[i - 1]:
            direction[i] = 1.0
        elif c[i] < fl[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
        st[i] = fl[i] if direction[i] > 0 else fu[i]
    return pd.Series(st, index=close.index), pd.Series(direction, index=close.index)


def parabolic_sar(high, low, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Parabolic SAR (Welles Wilder). Recursive, backward-looking."""
    h = high.to_numpy()
    l = low.to_numpy()
    n = len(h)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=high.index)
    bull = True
    af = af_step
    ep = h[0]
    sar[0] = l[0]
    for i in range(1, n):
        prev = sar[i - 1]
        if bull:
            cur = prev + af * (ep - prev)
            cur = min(cur, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if h[i] > ep:
                ep = h[i]
                af = min(af + af_step, af_max)
            if l[i] < cur:
                bull = False
                cur = ep
                ep = l[i]
                af = af_step
        else:
            cur = prev + af * (ep - prev)
            cur = max(cur, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if l[i] < ep:
                ep = l[i]
                af = min(af + af_step, af_max)
            if h[i] > cur:
                bull = True
                cur = ep
                ep = h[i]
                af = af_step
        sar[i] = cur
    return pd.Series(sar, index=high.index)


def ichimoku(high, low, close, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    """
    Ichimoku Kinko Hyo (Goichi Hosoda). Returns
    (conversion, base, senkou_span_a, senkou_span_b). The two spans are shifted
    FORWARD by `kijun` (shift(+kijun)), so the cloud value at bar i is derived
    from data at bar i-kijun -> backward-looking. The chikou (lagging) span is
    intentionally omitted because using it for signals would require future data.
    """
    conv = (rolling_max(high, tenkan) + rolling_min(low, tenkan)) / 2.0
    base = (rolling_max(high, kijun) + rolling_min(low, kijun)) / 2.0
    span_a = ((conv + base) / 2.0).shift(kijun)
    span_b = ((rolling_max(high, senkou_b) + rolling_min(low, senkou_b)) / 2.0).shift(kijun)
    return conv, base, span_a, span_b


def heikin_ashi(open_, high, low, close):
    """Heikin-Ashi candles. Returns (ha_open, ha_high, ha_low, ha_close)."""
    ha_close = (open_ + high + low + close) / 4.0
    c = ha_close.to_numpy()
    o = open_.to_numpy()
    n = len(c)
    ha_open = np.empty(n)
    ha_open[0] = (o[0] + close.iloc[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + c[i - 1]) / 2.0
    ha_open = pd.Series(ha_open, index=close.index)
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close


def streak(close: pd.Series) -> pd.Series:
    """Signed consecutive up/down close streak (+n up days, -n down days)."""
    d = close.diff().to_numpy()
    s = np.zeros(len(d))
    for i in range(1, len(d)):
        if d[i] > 0:
            s[i] = s[i - 1] + 1 if s[i - 1] > 0 else 1
        elif d[i] < 0:
            s[i] = s[i - 1] - 1 if s[i - 1] < 0 else -1
        else:
            s[i] = 0
    return pd.Series(s, index=close.index)


def percent_rank(series: pd.Series, window: int = 100) -> pd.Series:
    """Rolling percent-rank: % of the prior `window` values below the current one."""
    def _pr(x):
        return float((x[:-1] < x[-1]).mean() * 100.0) if len(x) > 1 else 50.0
    return series.rolling(window, min_periods=window).apply(_pr, raw=True)


def renko_trend(close: pd.Series, brick: pd.Series) -> pd.Series:
    """
    Renko brick direction (+1 up / -1 down) sampled at each bar's close. `brick`
    is the per-bar brick size (e.g. an ATR series). A new up-brick forms when the
    close rises a full brick above the last brick top; a down-brick mirrors it.
    Recursive but strictly backward-looking (bar i uses only bars <= i).
    """
    c = close.to_numpy()
    b = brick.to_numpy()
    n = len(c)
    direction = np.zeros(n)
    base = c[0]
    cur_dir = 0
    for i in range(n):
        size = b[i]
        if not (size and size == size) or size <= 0:
            direction[i] = cur_dir
            continue
        moved = False
        while c[i] >= base + size:
            base += size
            cur_dir = 1
            moved = True
        while c[i] <= base - size:
            base -= size
            cur_dir = -1
            moved = True
        direction[i] = cur_dir
        if not moved:
            direction[i] = cur_dir
    return pd.Series(direction, index=close.index)


def connors_rsi(close, rsi_window: int = 3, streak_window: int = 2,
                rank_window: int = 100) -> pd.Series:
    """
    ConnorsRSI (Larry Connors): average of (1) a short RSI of price, (2) an RSI
    of the up/down streak, and (3) the percent-rank of the 1-day ROC.
    Source: StockCharts ChartSchool, ConnorsRSI. 0-100, backward-looking.
    """
    r_price = rsi(close, rsi_window)
    r_streak = rsi(streak(close), streak_window)
    roc1 = close.pct_change() * 100.0
    r_rank = percent_rank(roc1, rank_window)
    return (r_price + r_streak + r_rank) / 3.0
