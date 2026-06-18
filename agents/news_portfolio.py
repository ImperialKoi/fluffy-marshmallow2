"""
News-driven cross-sectional AI strategy (Phase 3, forward-test only).

Per rebalance, for each symbol in the universe:
  1. pull recent news / filings / analyst data via get_info(symbol, as_of=now)
     (as_of=now guarantees no future leakage),
  2. summarize it and ask the LLM for STRUCTURED JSON
     {ticker, score in [-1,1], confidence in [0,1], rationale},
  3. validate the schema, numeric ranges, and ticker membership,
  4. audit the full prompt + raw response + parsed result to a dated JSONL file.

Guardrails (critical): malformed/empty/timeout/refusal/missing-news all default that
symbol to score 0 (no change). An out-of-universe ticker is ignored. A bad LLM
response never raises. The LLM is ADVISORY — the deterministic constructor + risk
layer decide what actually trades.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import config
from strategies.portfolio_base import PortfolioStrategy, PortfolioSignal, SymbolSignal
from agents.llm import LLM, LLMResult

log = logging.getLogger("agents.news_portfolio")

PER_SYMBOL_INSTRUCTIONS = (
    "You are a sell-side equity analyst. Read the material below for ONE ticker and "
    "judge the NET directional impact on the stock over the next few weeks. Respond "
    "with STRICT JSON only, no prose, exactly these keys:\n"
    '{"ticker": "<SYMBOL>", "score": <float -1..1>, "confidence": <float 0..1>, '
    '"rationale": "<one or two sentences>"}\n'
    "score: +1 very bullish, 0 neutral/no signal, -1 very bearish. "
    "confidence: how much the material supports your score (0 if little/no news). "
    "Only judge the ticker given; do not mention other tickers."
)


class NewsPortfolioStrategy(PortfolioStrategy):
    name = "news_portfolio"

    def __init__(self, llm: LLM, universe: list[str] = None, get_info_fn=None,
                 news_limit: int = None, audit_dir: str = None,
                 exposure_pass: bool = None, call_sleep: float = None):
        self.llm = llm
        self.universe = [s.upper() for s in (universe or config.AI_UNIVERSE)]
        self.news_limit = news_limit or config.AI_NEWS_LIMIT
        self.audit_dir = audit_dir or config.AI_AUDIT_DIR
        self.exposure_pass = (config.AI_EXPOSURE_PASS if exposure_pass is None
                              else exposure_pass)
        self.call_sleep = (getattr(config, "AI_LLM_SLEEP", 0.0)
                           if call_sleep is None else call_sleep)
        self.free_trade = getattr(config, "AI_FREE_TRADE", False)
        self.discovery_count = getattr(config, "AI_DISCOVERY_COUNT", 0)
        if get_info_fn is not None:
            self.get_info_fn = get_info_fn
        else:
            from tools.info import get_info
            self.get_info_fn = get_info

    # -- main entry -------------------------------------------------------- #
    def evaluate(self, universe: list[str] = None, as_of: datetime = None) -> PortfolioSignal:
        universe = [s.upper() for s in (universe or self.universe)]
        as_of = as_of or datetime.now(timezone.utc)
        audit_path = self._audit_path(as_of)

        # complete-freedom: let the AI propose NEW tickers to consider, beyond the
        # base universe (and beyond current holdings, which the caller already includes).
        if self.free_trade and self.discovery_count > 0:
            discovered = self._discover(as_of, self.discovery_count, audit_path)
            if discovered:
                universe = list(dict.fromkeys(universe + discovered))  # dedupe, keep order

        signals: dict[str, SymbolSignal] = {}
        summaries: dict[str, str] = {}
        for sym in universe:
            material, n_items = self._gather(sym, as_of)
            summaries[sym] = material
            prompt = self._prompt(sym, material)
            if n_items == 0:
                # no news -> default neutral, don't even call the model
                sig = SymbolSignal(sym, 0.0, 0.0, "no recent material", ok=False,
                                   error="no_news")
                self._audit(audit_path, sym, prompt, LLMResult(None, "", "no_news"), sig)
            else:
                res = self.llm.complete_json(prompt)
                sig = self._validate(res, sym, universe)
                self._audit(audit_path, sym, prompt, res, sig)
                if self.call_sleep:
                    import time
                    time.sleep(self.call_sleep)   # throttle to respect free-tier RPM
            signals[sym] = sig

        exposure = self._exposure_multiplier(signals, summaries, as_of, audit_path)
        return PortfolioSignal(signals=signals, exposure_multiplier=exposure,
                               as_of=as_of,
                               meta={"llm": self.llm.name, "audit": audit_path,
                                     "universe": universe})

    # -- free-trade discovery: ask the LLM for candidate tickers ----------- #
    def _discover(self, as_of, n, audit_path) -> list[str]:
        prompt = (
            "You are a portfolio manager with complete discretion over US equities. "
            f"Name up to {n} liquid US common stocks (NYSE/Nasdaq tickers) that are "
            "most worth analyzing RIGHT NOW for a multi-day directional trade — names "
            "with notable current news, catalysts, or momentum. Avoid ETFs, OTC, and "
            "illiquid micro-caps. Respond STRICT JSON only: "
            '{"tickers": ["AAA", "BBB", ...]}.')
        res = self.llm.complete_json(prompt)
        tickers = []
        if res and res.parsed:
            for t in (res.parsed.get("tickers") or [])[:n]:
                t = str(t).upper().strip()
                if t.isalpha() and 1 <= len(t) <= 5:   # basic sanity; tradability checked at execution
                    tickers.append(t)
        self._audit(audit_path, "__DISCOVERY__", prompt, res,
                    SymbolSignal("__DISCOVERY__", 0.0, 0.0, ",".join(tickers) or "(none)"))
        log.info("free-trade discovery proposed %d tickers: %s", len(tickers), tickers)
        return tickers

    # -- per-symbol validation (guardrails) -------------------------------- #
    def _validate(self, res: LLMResult, sym: str, universe: list[str]) -> SymbolSignal:
        if res is None or res.parsed is None:
            return SymbolSignal(sym, 0.0, 0.0, "", ok=False,
                                error=(res.error if res else "no_result") or "no_parse")
        p = res.parsed
        # ticker sanity: if the model named a DIFFERENT, out-of-context ticker, distrust it
        rt = str(p.get("ticker", "") or "").upper().strip()
        if rt and rt != sym:
            if rt not in universe:
                return SymbolSignal(sym, 0.0, 0.0, "", ok=False,
                                    error=f"out_of_universe_ticker:{rt}")
            return SymbolSignal(sym, 0.0, 0.0, "", ok=False, error=f"ticker_mismatch:{rt}")
        score = _num(p.get("score"))
        conf = _num(p.get("confidence"))
        if score is None or conf is None:
            return SymbolSignal(sym, 0.0, 0.0, "", ok=False, error="missing_numeric")
        rationale = str(p.get("rationale", ""))[:500]
        return SymbolSignal(sym, score, conf, rationale, ok=True).clamped()

    # -- optional whole-market exposure pass ------------------------------- #
    def _exposure_multiplier(self, signals, summaries, as_of, audit_path) -> float:
        if not self.exposure_pass:
            return 1.0
        lines = [f"{t}: score={s.score:+.2f} conf={s.confidence:.2f} {s.rationale[:120]}"
                 for t, s in signals.items()]
        prompt = (
            "Given these per-stock assessments, judge overall market risk appetite. "
            'Respond STRICT JSON only: {"exposure": <float 0..1>, "rationale": "<text>"}. '
            "1.0 = deploy full risk budget, 0.0 = stay in cash.\n\n" + "\n".join(lines))
        res = self.llm.complete_json(prompt)
        exposure = 1.0
        if res and res.parsed is not None:
            v = _num(res.parsed.get("exposure"))
            if v is not None:
                exposure = float(max(0.0, min(1.0, v)))
        self._audit(audit_path, "__EXPOSURE__", prompt, res,
                    SymbolSignal("__EXPOSURE__", exposure, 1.0, "exposure pass"))
        return exposure

    # -- material gathering / formatting ----------------------------------- #
    def _gather(self, sym: str, as_of) -> tuple[str, int]:
        try:
            items = self.get_info_fn(sym, as_of=as_of, limit=self.news_limit) or []
        except Exception as e:  # noqa: BLE001 — a failing feed must not crash the run
            log.warning("get_info failed for %s: %s", sym, e)
            return f"(no data retrieved for {sym})", 0
        if not items:
            return f"(no recent material for {sym})", 0
        by_type = {"news": [], "filing": [], "analyst": []}
        for it in items:
            by_type.setdefault(_attr(it, "item_type"), []).append(it)
        parts = []
        for label, key in (("NEWS", "news"), ("SEC FILINGS", "filing"),
                           ("ANALYST", "analyst")):
            rows = by_type.get(key) or []
            if not rows:
                continue
            parts.append(f"== {label} ==")
            for it in rows[:self.news_limit]:
                ts = _attr(it, "published_utc")
                date = ts.date().isoformat() if ts else "n/a"
                head = _attr(it, "headline") or ""
                summ = (_attr(it, "summary") or "")[:240]
                src = _attr(it, "source") or ""
                parts.append(f"- [{date}] ({src}) {head}. {summ}".strip())
        return "\n".join(parts), len(items)

    def _prompt(self, sym: str, material: str) -> str:
        return (f"{PER_SYMBOL_INSTRUCTIONS}\n\nTICKER: {sym}\n\nMATERIAL:\n{material}\n")

    # -- audit ------------------------------------------------------------- #
    def _audit_path(self, as_of) -> str:
        os.makedirs(self.audit_dir, exist_ok=True)
        return os.path.join(self.audit_dir, f"{as_of.date().isoformat()}.jsonl")

    def _audit(self, path, sym, prompt, res: LLMResult, sig: SymbolSignal):
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": sym, "llm": self.llm.name, "prompt": prompt,
            "raw_response": (res.raw if res else ""),
            "parsed": (res.parsed if res else None),
            "result": {"score": sig.score, "confidence": sig.confidence,
                       "rationale": sig.rationale, "ok": sig.ok, "error": sig.error},
        }
        try:
            with open(path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("audit write failed: %s", e)


def _num(x):
    try:
        if x is None or isinstance(x, bool):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _attr(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
