"""
Universe loader for validation.

The project's data/loader.load_csv re-reads the whole multi-symbol CSV on every
call — fine for one backtest, ruinous for thousands. Here we read the CSV ONCE,
group by symbol, and reuse data/loader._validate so each per-symbol frame is
normalized identically to the rest of the project (no reimplementation).

Universes:
  * "sp500"  -> every symbol in the bundled dataset (~505)
  * "sample" -> a curated sector-stratified subset of large caps (fast, broad)
  * an int N -> N evenly-spaced symbols from the sorted list (reproducible)

NOTE: the bundled CSV has no GICS sector column, so the "sample" universe is a
hand-curated cross-section across the 11 GICS sectors (documented proxy). See the
survivorship-bias caveat in VALIDATION.md.
"""

from __future__ import annotations

import functools

import pandas as pd

import config as proj
from data import loader

# A hand-curated sector-stratified cross-section (GICS sector -> tickers), chosen
# from large caps present in the 2013-2018 S&P 500 dataset. ~3 per sector.
SECTOR_SAMPLE = {
    "Information Technology": ["AAPL", "MSFT", "INTC", "CSCO"],
    "Health Care": ["JNJ", "PFE", "UNH", "MRK"],
    "Financials": ["JPM", "BAC", "WFC", "GS"],
    "Consumer Discretionary": ["AMZN", "HD", "MCD", "NKE"],
    "Consumer Staples": ["KO", "PG", "WMT", "PEP"],
    "Energy": ["XOM", "CVX", "SLB"],
    "Industrials": ["BA", "GE", "HON", "UPS"],
    "Materials": ["DD", "DOW", "NEM"],
    "Utilities": ["DUK", "SO", "NEE"],
    "Real Estate": ["AMT", "SPG", "PLD"],
    "Communication Services": ["T", "VZ", "DIS"],
}


@functools.lru_cache(maxsize=1)
def _raw():
    return pd.read_csv(proj.CACHE_CSV, parse_dates=["date"])


@functools.lru_cache(maxsize=1)
def _grouped() -> dict:
    """symbol -> validated OHLCV DataFrame (computed once)."""
    raw = _raw()
    out = {}
    for name, g in raw.groupby("Name"):
        try:
            out[name] = loader._validate(g.set_index("date"))
        except Exception:
            continue
    return out


def all_symbols() -> list[str]:
    return sorted(_grouped().keys())


def get_symbol(symbol: str) -> pd.DataFrame:
    return _grouped()[symbol]


def sector_map() -> dict[str, str]:
    """ticker -> sector (only for the curated sample)."""
    m = {}
    for sector, tickers in SECTOR_SAMPLE.items():
        for t in tickers:
            m[t] = sector
    return m


def resolve_universe(universe, max_symbols: int | None = None) -> list[str]:
    """Turn a universe spec into a concrete, existing, de-duplicated symbol list."""
    available = set(all_symbols())
    if universe == "sp500":
        syms = all_symbols()
    elif universe == "sample":
        syms = [t for ts in SECTOR_SAMPLE.values() for t in ts if t in available]
    else:
        # an integer N -> evenly spaced across the sorted universe
        n = int(universe)
        full = all_symbols()
        if n >= len(full):
            syms = full
        else:
            step = len(full) / n
            syms = [full[int(i * step)] for i in range(n)]
    # keep only symbols actually present, preserve order, dedupe
    seen, out = set(), []
    for s in syms:
        if s in available and s not in seen:
            seen.add(s); out.append(s)
    if max_symbols:
        out = out[:max_symbols]
    return out
