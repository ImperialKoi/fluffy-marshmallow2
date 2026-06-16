"""
yfinance analyst adapter (free, keyless) — analyst ratings & price targets.

Pulls three things from yfinance and normalizes them into `analyst` Items:
  * upgrades_downgrades : dated rating changes (firm, from/to grade, action)
  * analyst_price_targets : current mean/median/high/low target (a snapshot, no date)
  * recommendations : current strongBuy/buy/hold/sell/strongSell distribution (snapshot)

CAVEAT (documented and important): this is SCRAPED Yahoo Finance data, not an
official API. It is best-effort — expect gaps, schema drift, and occasional
breakage when Yahoo changes their site. Every Item here is flagged
`extra["scraped"] = True`. Each access is wrapped so a failure in one field does
not lose the others, and a total failure returns [] rather than raising.

Snapshot items (price targets / recommendation distribution) have
published_utc=None because Yahoo does not date them; the retriever sorts them after
timestamped items but still returns them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .base import Source
from ..schema import Item, ANALYST, make_id, to_utc

log = logging.getLogger("info_tool.yfinance_analyst")


class YFinanceAnalyst(Source):
    name = "yfinance_analyst"
    item_types = (ANALYST,)
    default_on = True

    def available(self) -> bool:
        try:
            import yfinance  # noqa: F401
            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        return "yfinance not installed"

    def fetch(self, ticker: str, limit: int = 50) -> list[Item]:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        items: list[Item] = []
        items += self._upgrades_downgrades(t, ticker, limit)
        items += self._price_targets(t, ticker)
        items += self._recommendation_summary(t, ticker)
        return items[:limit] if limit else items

    # -- dated rating changes --------------------------------------------- #
    def _upgrades_downgrades(self, t, ticker, limit) -> list[Item]:
        try:
            df = t.upgrades_downgrades
        except Exception as e:  # noqa: BLE001
            log.warning("yfinance upgrades_downgrades failed for %s: %s", ticker, e)
            return []
        if df is None or len(df) == 0:
            return []
        out = []
        df = df.reset_index()  # GradeDate becomes a column
        # newest first; keep it bounded
        try:
            df = df.sort_values(df.columns[0], ascending=False)
        except Exception:  # noqa: BLE001
            pass
        for _, row in df.head(max(1, limit)).iterrows():
            grade_date = row.get("GradeDate")
            firm = _s(row.get("Firm"))
            action = _s(row.get("Action"))
            to_grade = _s(row.get("ToGrade"))
            from_grade = _s(row.get("FromGrade"))
            published = to_utc(grade_date.to_pydatetime() if hasattr(grade_date, "to_pydatetime")
                               else grade_date)
            headline = f"{ticker.upper()}: {firm} {action} — {from_grade or '?'} → {to_grade or '?'}"
            out.append(Item(
                id=make_id(self.name, headline=f"{published}|{firm}|{to_grade}"),
                tickers=[ticker.upper()],
                published_utc=published,
                source="yfinance",
                item_type=ANALYST,
                headline=headline,
                summary=f"{firm} {action} rating: {from_grade} -> {to_grade}.",
                url=f"https://finance.yahoo.com/quote/{ticker.upper()}/analysis",
                extra={"scraped": True, "caveat": "Yahoo-scraped, best-effort",
                       "firm": firm, "action": action,
                       "from_grade": from_grade, "to_grade": to_grade},
            ))
        return out

    # -- current price-target snapshot ------------------------------------ #
    def _price_targets(self, t, ticker) -> list[Item]:
        try:
            pt = t.analyst_price_targets
        except Exception as e:  # noqa: BLE001
            log.warning("yfinance analyst_price_targets failed for %s: %s", ticker, e)
            return []
        if not pt:
            return []
        # pt is typically a dict: {current, high, low, mean, median}
        try:
            d = dict(pt)
        except (TypeError, ValueError):
            return []
        parts = ", ".join(f"{k}={d[k]}" for k in ("mean", "median", "high", "low", "current")
                          if k in d and d[k] is not None)
        if not parts:
            return []
        headline = f"{ticker.upper()}: analyst price targets ({parts})"
        return [Item(
            id=make_id(self.name, headline=f"price_targets|{ticker.upper()}|{parts}"),
            tickers=[ticker.upper()],
            published_utc=None,  # Yahoo gives no timestamp for the snapshot
            source="yfinance",
            item_type=ANALYST,
            headline=headline,
            summary="Current consensus analyst price targets (snapshot, undated).",
            url=f"https://finance.yahoo.com/quote/{ticker.upper()}/analysis",
            extra={"scraped": True, "caveat": "Yahoo-scraped, best-effort",
                   "snapshot": True, "price_target": d,
                   "fetched_utc": datetime.now(timezone.utc).isoformat()},
        )]

    # -- current recommendation distribution ------------------------------ #
    def _recommendation_summary(self, t, ticker) -> list[Item]:
        try:
            df = t.recommendations
        except Exception as e:  # noqa: BLE001
            log.warning("yfinance recommendations failed for %s: %s", ticker, e)
            return []
        if df is None or len(df) == 0:
            return []
        try:
            row = df.iloc[0].to_dict()  # most recent period row
        except Exception:  # noqa: BLE001
            return []
        cols = ("strongBuy", "buy", "hold", "sell", "strongSell")
        dist = {c: int(row[c]) for c in cols if c in row and row[c] == row[c]}
        if not dist:
            return []
        parts = ", ".join(f"{k}={v}" for k, v in dist.items())
        headline = f"{ticker.upper()}: analyst recommendations ({parts})"
        return [Item(
            id=make_id(self.name, headline=f"recs|{ticker.upper()}|{parts}"),
            tickers=[ticker.upper()],
            published_utc=None,
            source="yfinance",
            item_type=ANALYST,
            headline=headline,
            summary="Current analyst recommendation distribution (snapshot, undated).",
            url=f"https://finance.yahoo.com/quote/{ticker.upper()}/analysis",
            extra={"scraped": True, "caveat": "Yahoo-scraped, best-effort",
                   "snapshot": True, "recommendation_distribution": dist,
                   "period": _s(row.get("period"))},
        )]


def _s(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s
