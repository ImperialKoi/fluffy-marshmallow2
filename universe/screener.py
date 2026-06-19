"""
Deterministic universe screener — the ONLY source of discovery candidates.

CRITICAL DESIGN RULE: the LLM never invents tickers. Discovery candidates come from
THIS module, computed from real Alpaca market data, so every candidate is provably
tradable on Alpaca and clears a hard liquidity floor before any model sees it.

Pipeline
--------
1. Start from Alpaca's ACTIVE, tradable US-equity asset list (broker.list_assets).
   Restrict to supported exchanges. This is the tradability guarantee — many
   penny/OTC names are NOT on Alpaca and are auto-excluded here.
2. Pull recent daily bars (chunked multi-symbol requests) and compute, per symbol:
     * last price            -> penny / high-risk filter (price <= SCREEN_PRICE_CAP)
     * realized volatility   -> stdev of daily returns (want HIGH: high risk/reward)
     * relative volume       -> today's volume vs trailing avg (want HIGH: "high volume")
     * avg daily $ volume    -> HARD LIQUIDITY FLOOR (enter AND exit; applies to pennies)
3. Best-effort float (yfinance) on the survivors only -> "low supply" tilt. Missing
   float never drops a name (unless SCREEN_REQUIRE_FLOAT); it just can't earn the
   low-float bonus.
4. Fold in the hype tracker's most-hyped names as an extra discovery feed (still
   subjected to the same tradability + liquidity gates).
5. Rank by a composite (volatility + relative volume + low-float + hype + cheapness)
   and return the top SCREEN_DAILY_CANDIDATES.

Everything network-facing is injectable (assets_fn, bars_fn, float_fn, hype_names_fn)
so the screens are unit-tested with no live calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import config

log = logging.getLogger("universe.screener")


@dataclass
class ScreenConfig:
    price_cap: float = 5.0
    min_dollar_volume: float = 1_000_000.0     # HARD liquidity floor
    min_volatility: float = 0.04
    rvol_min: float = 1.5
    max_float: float = 75_000_000
    require_float: bool = False
    lookback_days: int = 30
    top_n: int = 20
    max_assets_scanned: int = 3000
    bars_chunk: int = 200
    include_hype: bool = True
    exchanges: tuple = ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")

    @classmethod
    def from_config(cls) -> "ScreenConfig":
        return cls(
            price_cap=config.SCREEN_PRICE_CAP,
            min_dollar_volume=config.SCREEN_MIN_DOLLAR_VOLUME,
            min_volatility=config.SCREEN_MIN_VOLATILITY,
            rvol_min=config.SCREEN_RVOL_MIN,
            max_float=config.SCREEN_MAX_FLOAT,
            require_float=config.SCREEN_REQUIRE_FLOAT,
            lookback_days=config.SCREEN_LOOKBACK_DAYS,
            top_n=config.SCREEN_DAILY_CANDIDATES,
            max_assets_scanned=config.SCREEN_MAX_ASSETS_SCANNED,
            bars_chunk=config.SCREEN_BARS_CHUNK,
            include_hype=config.SCREEN_INCLUDE_HYPE,
            exchanges=tuple(config.SCREEN_EXCHANGES),
        )


@dataclass
class Candidate:
    symbol: str
    price: float
    dollar_volume: float          # avg daily $ volume over the window (liquidity)
    volatility: float             # stdev of daily returns
    rel_volume: float             # today vs trailing avg
    float_shares: float | None = None
    hype: bool = False            # surfaced by the hype feed
    score: float = 0.0            # composite ranking score
    reasons: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {"symbol": self.symbol, "price": round(self.price, 4),
                "dollar_volume": round(self.dollar_volume, 0),
                "volatility": round(self.volatility, 4),
                "rel_volume": round(self.rel_volume, 3),
                "float_shares": (None if self.float_shares is None
                                 else int(self.float_shares)),
                "hype": self.hype, "score": round(self.score, 4),
                "reasons": "|".join(self.reasons)}


# --------------------------------------------------------------------------- #
# per-symbol metrics (pure; unit-tested directly)
# --------------------------------------------------------------------------- #
def compute_metrics(df, cfg: ScreenConfig) -> dict | None:
    """Daily-bar metrics for one symbol, or None if there isn't enough data."""
    if df is None or "close" not in getattr(df, "columns", []) or len(df) < 5:
        return None
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    win = min(cfg.lookback_days, len(close))
    c = close.iloc[-win:]
    v = vol.iloc[-win:]
    price = float(c.iloc[-1])
    rets = c.pct_change().dropna()
    if len(rets) < 2:
        return None
    volatility = float(rets.std())
    dollar_volume = float((c * v).mean())
    today_vol = float(v.iloc[-1])
    base_vol = float(v.iloc[:-1].mean()) if len(v) > 1 else 0.0
    rel_volume = (today_vol / base_vol) if base_vol > 0 else 0.0
    return {"price": price, "volatility": volatility,
            "dollar_volume": dollar_volume, "rel_volume": rel_volume}


def passes_hard_gates(m: dict, cfg: ScreenConfig) -> tuple[bool, list[str]]:
    """The non-negotiable filters (price cap + LIQUIDITY FLOOR + a high-risk signal).

    Returns (passed, reasons). The liquidity floor is a HARD gate that applies even to
    penny picks: never propose a name so thin its own order moves the price."""
    reasons = []
    if m["price"] <= 0 or m["price"] > cfg.price_cap:
        return False, ["price>cap"]
    reasons.append("penny")
    # HARD liquidity floor — can we actually enter AND exit?
    if m["dollar_volume"] < cfg.min_dollar_volume:
        return False, ["illiquid"]
    reasons.append("liquid")
    # must show at least one high-risk/high-reward signal (vol OR relative volume)
    hot_vol = m["volatility"] >= cfg.min_volatility
    hot_rvol = m["rel_volume"] >= cfg.rvol_min
    if not (hot_vol or hot_rvol):
        return False, ["not_hot"]
    if hot_vol:
        reasons.append("high_vol")
    if hot_rvol:
        reasons.append("high_rvol")
    return True, reasons


def _composite_score(m: dict, float_shares, cfg: ScreenConfig, hype: bool) -> float:
    """Rank toward asymmetric upside: reward volatility, relative volume, low float,
    hype, and cheapness. Each term is saturated to [0,1)-ish so no single axis dominates."""
    def sat(x):
        x = max(0.0, x)
        return x / (x + 1.0)
    vol_term = sat(m["volatility"] / max(cfg.min_volatility, 1e-9) - 1.0)
    rvol_term = sat(m["rel_volume"] / max(cfg.rvol_min, 1e-9) - 1.0)
    cheap_term = max(0.0, 1.0 - m["price"] / max(cfg.price_cap, 1e-9))
    float_term = 0.0
    if float_shares and float_shares > 0:
        # lower float -> higher term (log scale); at/above the ceiling -> ~0
        float_term = max(0.0, min(1.0, math.log10(cfg.max_float / float_shares) / 2.0))
    hype_term = 1.0 if hype else 0.0
    return (0.30 * vol_term + 0.30 * rvol_term + 0.20 * float_term
            + 0.10 * cheap_term + 0.10 * hype_term)


# --------------------------------------------------------------------------- #
# default (live) data providers — injectable for tests
# --------------------------------------------------------------------------- #
def _tradable_symbols(assets: list, cfg: ScreenConfig) -> list[str]:
    ex = {e.upper() for e in cfg.exchanges}
    out = []
    for a in assets:
        if not a.get("tradable") or str(a.get("status", "")).upper() not in ("ACTIVE", "ASSETSTATUS.ACTIVE"):
            continue
        if a.get("exchange", "").upper().replace("ASSETEXCHANGE.", "") not in ex:
            continue
        out.append(a["symbol"].upper())
    return out


def _default_float_fn(symbol: str):
    """Best-effort float via yfinance. Returns shares or None (never raises out)."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        for k in ("floatShares", "sharesOutstanding"):
            v = info.get(k)
            if v:
                return float(v)
    except Exception as e:  # noqa: BLE001 — float data is optional, fail soft
        log.debug("float lookup failed for %s: %s", symbol, e)
    return None


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def screen(broker=None, hype=None, cfg: ScreenConfig = None, *,
           assets_fn=None, bars_fn=None, float_fn=None, hype_names_fn=None) -> list[Candidate]:
    """Return a ranked list of real, tradable, liquid high-risk/high-reward candidates.

    Injection points (all default to live Alpaca / yfinance / hype tracker):
      assets_fn()              -> list of asset dicts (tradable universe)
      bars_fn(list[str])       -> {symbol: daily OHLCV DataFrame}
      float_fn(symbol)         -> float shares or None
      hype_names_fn()          -> list of extra symbols to fold in (most-hyped)
    """
    cfg = cfg or ScreenConfig.from_config()
    assets_fn = assets_fn or (broker.list_assets if broker is not None else (lambda: []))
    bars_fn = bars_fn or (broker.daily_bars if broker is not None else (lambda s: {}))
    float_fn = float_fn or _default_float_fn

    # 1. tradable asset universe (the tradability guarantee)
    try:
        assets = assets_fn() or []
    except Exception as e:  # noqa: BLE001
        log.warning("asset list fetch failed: %s", e)
        assets = []
    tradable = _tradable_symbols(assets, cfg)
    tradable_set = set(tradable)
    scan = tradable[:cfg.max_assets_scanned] if cfg.max_assets_scanned else []

    # fold in hype names (only those that are actually tradable on Alpaca)
    hype_names = set()
    if cfg.include_hype:
        try:
            names = (hype_names_fn() if hype_names_fn else _hype_names(hype))
            hype_names = {s.upper() for s in (names or [])} & (tradable_set or set(names or []))
        except Exception as e:  # noqa: BLE001
            log.debug("hype feed unavailable: %s", e)
    scan = list(dict.fromkeys(scan + sorted(hype_names)))
    if not scan:
        log.warning("screener has no symbols to scan (no assets/hype feed)")
        return []

    # 2. daily bars (chunked) -> per-symbol metrics -> hard gates
    survivors: list[tuple[str, dict]] = []
    for i in range(0, len(scan), cfg.bars_chunk):
        chunk = scan[i:i + cfg.bars_chunk]
        try:
            frames = bars_fn(chunk) or {}
        except Exception as e:  # noqa: BLE001
            log.warning("bar fetch failed for a chunk: %s", e)
            continue
        for sym in chunk:
            m = compute_metrics(frames.get(sym), cfg)
            if m is None:
                continue
            ok, reasons = passes_hard_gates(m, cfg)
            if not ok:
                continue
            m["_reasons"] = reasons
            m["_hype"] = sym in hype_names
            survivors.append((sym, m))

    # 3. best-effort float on survivors only (limits yfinance calls)
    cands: list[Candidate] = []
    for sym, m in survivors:
        fl = None
        try:
            fl = float_fn(sym)
        except Exception as e:  # noqa: BLE001
            log.debug("float fn raised for %s: %s", sym, e)
        reasons = list(m["_reasons"])
        if fl is not None and fl <= cfg.max_float:
            reasons.append("low_float")
        elif fl is None and cfg.require_float:
            continue                      # require_float drops names lacking data
        if m["_hype"]:
            reasons.append("hype")
        c = Candidate(symbol=sym, price=m["price"], dollar_volume=m["dollar_volume"],
                      volatility=m["volatility"], rel_volume=m["rel_volume"],
                      float_shares=fl, hype=m["_hype"], reasons=reasons)
        c.score = _composite_score(m, fl, cfg, m["_hype"])
        cands.append(c)

    # 4. rank + cap
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands[:cfg.top_n]


def _hype_names(hype) -> list[str]:
    if hype is None:
        return []
    ranked = hype.rank()          # most-hyped first over the hype watchlist
    return [r["symbol"] for r in ranked
            if r.get("score") == r.get("score")]   # drop NaN scores


# --------------------------------------------------------------------------- #
# CLI: print a sample day's real candidates so they can be eyeballed
# --------------------------------------------------------------------------- #
def _print_table(cands: list[Candidate]):
    print(f"\n=== SCREENER CANDIDATES ({len(cands)}) ===")
    print(f"  {'sym':<7}{'price':>9}{'$vol(M)':>10}{'vol':>8}{'rvol':>7}"
          f"{'float(M)':>10}{'score':>8}  reasons")
    for c in cands:
        fl = "" if c.float_shares is None else f"{c.float_shares/1e6:.1f}"
        print(f"  {c.symbol:<7}{c.price:>9.2f}{c.dollar_volume/1e6:>10.2f}"
              f"{c.volatility:>8.3f}{c.rel_volume:>7.2f}{fl:>10}{c.score:>8.3f}"
              f"  {'|'.join(c.reasons)}")


def main():
    import argparse
    import logging as _logging
    p = argparse.ArgumentParser(description="Print a sample day's screener candidates.")
    p.add_argument("--max-assets", type=int, default=None,
                   help="override SCREEN_MAX_ASSETS_SCANNED (smaller = faster sample)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    _logging.basicConfig(level=_logging.WARNING if args.quiet else _logging.INFO,
                         format="%(levelname)s %(name)s: %(message)s")
    from service.secrets import maybe_load_ssm
    maybe_load_ssm()
    from broker.alpaca_broker import AlpacaBroker
    from signals.hype import HypeTracker
    cfg = ScreenConfig.from_config()
    if args.max_assets is not None:
        cfg.max_assets_scanned = args.max_assets
    broker = AlpacaBroker(paper=True)
    cands = screen(broker=broker, hype=HypeTracker(), cfg=cfg)
    _print_table(cands)


if __name__ == "__main__":
    main()
