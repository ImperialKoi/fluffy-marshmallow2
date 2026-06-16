"""
Data layer.

The backtest engine doesn't care *where* bars come from, only that it receives a
clean OHLCV DataFrame indexed by date with columns:
    open, high, low, close, volume

This file gives you three sources behind one interface:

  * load_csv(...)      -> the cached real S&P 500 dataset (works offline / in the
                          sandbox). This is what the demo uses.
  * load_yfinance(...) -> live historical data from Yahoo Finance. Use this on
                          your own machine; needs `pip install yfinance` and
                          internet. Great for backtesting any ticker, any range.
  * load_alpaca(...)   -> historical bars from your Alpaca account. Use this so
                          your backtest data matches the broker you'll trade on.

All three return the same shape, so swapping sources is a one-line change.
"""

import pandas as pd


_REQUIRED = ["open", "high", "low", "close", "volume"]


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns, sort by date, drop bad rows. Shared by every loader."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")
    df = df[_REQUIRED]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    # Forward-fill the occasional gap, then drop anything still empty.
    df[_REQUIRED] = df[_REQUIRED].ffill()
    df = df.dropna(subset=_REQUIRED)
    df = df[df["close"] > 0]
    return df


def load_csv(path: str, symbol: str, start=None, end=None) -> pd.DataFrame:
    """
    Load one symbol from the cached multi-symbol dataset.

    The bundled file has columns: date, open, high, low, close, volume, Name
    """
    raw = pd.read_csv(path, parse_dates=["date"])
    raw = raw[raw["Name"] == symbol]
    if raw.empty:
        available = ", ".join(sorted(pd.read_csv(path, usecols=["Name"])["Name"].unique())[:15])
        raise ValueError(f"Symbol '{symbol}' not in dataset. Some available: {available} ...")
    raw = raw.set_index("date")
    df = _validate(raw)
    return _slice(df, start, end)


def load_yfinance(symbol: str, start=None, end=None, interval="1d") -> pd.DataFrame:
    """
    Live historical data via Yahoo Finance. Run this on your own machine.

        df = load_yfinance("TSLA", start="2018-01-01")
    """
    import yfinance as yf  # imported lazily so the sandbox demo doesn't need it
    raw = yf.download(symbol, start=start, end=end, interval=interval,
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for '{symbol}'. Check the ticker / dates.")
    # yfinance may return a MultiIndex column when one ticker is requested.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return _validate(raw)


def load_alpaca(symbol: str, start, end, timeframe="1Day",
                api_key=None, secret_key=None, feed="iex") -> pd.DataFrame:
    """
    Historical bars from your Alpaca account so backtest data matches your broker.
    Requires `pip install alpaca-py` and your API keys (paper keys are fine for
    pulling data). Run on your own machine.

    feed: "iex" (free, ~2.5% of volume) or "sip" (paid, full market). Free
    accounts must use "iex" or the request will be rejected.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    client = StockHistoricalDataClient(api_key, secret_key)
    tf = {"1Day": TimeFrame.Day, "1Hour": TimeFrame.Hour, "1Min": TimeFrame.Minute}[timeframe]
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end,
                           feed=DataFeed.IEX if feed == "iex" else DataFeed.SIP)
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise ValueError(f"Alpaca returned no bars for '{symbol}'.")
    bars = bars.reset_index()
    bars = bars[bars["symbol"] == symbol].set_index("timestamp")
    bars.index = pd.to_datetime(bars.index).tz_localize(None)
    return _validate(bars)


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df
