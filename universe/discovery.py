"""
Daily universe discovery: screener -> LLM evaluation -> deterministic gate -> store.

Flow (once per day, separate from the 60s scan and the hourly rebalance):

  1. SCREEN — universe/screener.screen() returns real, tradable, liquid high-risk
     candidates from Alpaca market data. The LLM never sees a ticker the screener
     didn't produce.
  2. EVALUATE — the existing news scorer (reused) scores the candidates for inclusion
     with news context, nudged toward asymmetric-upside setups. ADVISORY only.
  3. GATE (deterministic, FINAL SAY) — iterate the SCREENER's candidates (never the
     LLM's output: a hallucinated/off-list ticker is structurally ignored), and for
     each, re-confirm Alpaca tradable/active + the hard liquidity floor + a minimum LLM
     conviction, respect the managed wall-off, enforce the universe-size cap (evicting
     the weakest/stalest dynamic name if needed), tag risk tier / added_date / rationale
     into the inventory metadata, and admit it to the persisted universe.

Every decision (admitted or rejected, with the reason) is logged to a CSV. The LLM is
advisory; the gate is deterministic and has the final say — the same design as the rest
of the system.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone

import config
from agents.news_portfolio import NewsPortfolioStrategy
from universe.screener import screen, ScreenConfig
from universe.store import UniverseStore, SPECULATIVE

log = logging.getLogger("universe.discovery")


# --------------------------------------------------------------------------- #
# LLM evaluator — reuse the news scorer, with an inclusion/asymmetric-upside prompt
# --------------------------------------------------------------------------- #
class DiscoveryScorer(NewsPortfolioStrategy):
    """Reuses the news scorer (get_info gathering, batched JSON call, audit logging,
    out-of-set ticker rejection) but frames the question as *inclusion*: among these
    REAL candidates, which have the best asymmetric (high-risk/high-reward) upside.

    It never proposes tickers — it only scores the candidate list it is handed."""

    name = "discovery_scorer"

    def __init__(self, llm, audit_dir=None, news_limit=None, call_sleep=None):
        super().__init__(llm=llm, universe=[], get_info_fn=None,
                         news_limit=news_limit, audit_dir=audit_dir,
                         exposure_pass=False,
                         call_sleep=(config.DISCOVERY_LLM_SLEEP if call_sleep is None
                                     else call_sleep))
        self.free_trade = False          # discovery never lets the LLM invent tickers
        self.discovery_count = 0

    def _batch_prompt(self, symbols, summaries):
        blocks = "\n\n".join(f"=== {s} ===\n{summaries.get(s, '')}" for s in symbols)
        return (
            "You are a high-risk/high-reward equity analyst screening SMALL, VOLATILE "
            "US stocks that a deterministic screener already flagged (low-priced, high "
            "volatility / relative volume, often low float). For EACH ticker below, "
            "judge the NET asymmetric UPSIDE over the next few days/weeks from its "
            "material — favor genuine catalysts with outsized upside, penalize names "
            "that look like pure pump/dilution risk. Respond STRICT JSON ONLY, no prose:\n"
            '{"scores": [{"ticker": "AAA", "score": <float -1..1>, '
            '"confidence": <float 0..1>, "rationale": "<one sentence>"}, ...]}\n'
            "score: +1 strong asymmetric upside, 0 neutral/no signal, -1 avoid. "
            "Include EVERY ticker listed exactly once. Do NOT include any ticker not "
            "listed below.\n\n"
            f"TICKERS: {', '.join(symbols)}\n\n{blocks}\n")


# --------------------------------------------------------------------------- #
# deterministic gate (final say)
# --------------------------------------------------------------------------- #
def _is_excluded(inventory, symbol: str) -> bool:
    """Respect a manual wall-off: a symbol whose metadata explicitly sets managed=False
    is excluded from auto-discovery (we never silently flip a hand-excluded name)."""
    if inventory is None:
        return False
    try:
        meta = inventory.meta.get(symbol)
    except Exception:  # noqa: BLE001
        return False
    return meta.get("managed") is False


def gate(candidates, signals, *, broker=None, store: UniverseStore, inventory=None,
         cfg: ScreenConfig = None, min_conviction: float = None,
         max_size: int = None, mode: str = "paper") -> dict:
    """Deterministic admission. Iterates the SCREENER's candidates only.

    Returns {"admitted": [...], "rejected": [(symbol, reason), ...], "evicted": [...]}.
    Mutates `store` (and inventory metadata) unless mode == 'dry'.
    """
    cfg = cfg or ScreenConfig.from_config()
    min_conviction = (config.DISCOVERY_MIN_CONVICTION if min_conviction is None
                      else min_conviction)
    max_size = max_size or config.UNIVERSE_MAX_SIZE

    # protect open positions from eviction (don't orphan something we hold)
    protected = set()
    if inventory is not None:
        try:
            protected = {p["symbol"].upper() for p in inventory.broker.list_positions()}
        except Exception:  # noqa: BLE001
            protected = set()

    admitted, rejected, evicted = [], [], []
    for c in candidates:                                   # SCREENER order (ranked)
        sym = c.symbol.upper()
        sig = signals.get(sym)                             # may be None / off-list-ignored

        # already in the universe? refresh its freshness and move on.
        if sym in store:
            conv = (sig.score * sig.confidence) if sig else None
            if mode != "dry":
                store.touch(sym, conviction=conv)
            rejected.append((sym, "already_in_universe"))
            continue
        if _is_excluded(inventory, sym):
            rejected.append((sym, "manually_excluded"))
            continue

        # 1. re-confirm tradability at admission time (delist/halt since the screen)
        if broker is not None:
            asset = broker.get_asset(sym)
            if asset is None or not asset.get("tradable"):
                rejected.append((sym, "not_tradable_now"))
                continue

        # 2. re-confirm the HARD liquidity floor + data availability
        if c.price <= 0 or c.dollar_volume < cfg.min_dollar_volume:
            rejected.append((sym, "below_liquidity_floor"))
            continue

        # 3. LLM conviction (advisory inclusion vote)
        conviction = (sig.score * sig.confidence) if (sig and sig.ok) else 0.0
        if conviction < min_conviction:
            rejected.append((sym, f"low_conviction:{conviction:.2f}"))
            continue

        # 4. universe-size cap + churn/eviction (never evict pinned/held)
        if store.is_full(max_size):
            freed = store.make_room(1, protected=protected, max_size=max_size) \
                if mode != "dry" else _dry_make_room(store, protected, max_size)
            if not freed and store.is_full(max_size):
                rejected.append((sym, "universe_full"))
                continue
            evicted.extend(freed)

        # 5. admit: tag tier / added_date / rationale into inventory metadata + store
        tier = classify_tier(c, cfg)
        rationale = (sig.rationale if sig else "") or "|".join(c.reasons)
        if mode != "dry":
            store.add(sym, tier=tier, rationale=rationale, conviction=conviction,
                      screen_meta=c.as_row())
            _stamp_inventory(inventory, sym, tier, rationale)
        admitted.append({"symbol": sym, "tier": tier, "conviction": round(conviction, 3),
                         "price": c.price, "reasons": c.reasons, "rationale": rationale})

    if mode != "dry":
        store.save()
    return {"admitted": admitted, "rejected": rejected, "evicted": evicted}


def _dry_make_room(store, protected, max_size):
    """Eviction PREVIEW for dry mode (no mutation): which names WOULD be evicted."""
    cands = store.eviction_candidates(protected)
    return cands[:1] if cands else []


def classify_tier(candidate, cfg: ScreenConfig) -> str:
    """Discovered names are high-risk by construction (penny / high-vol / low-float),
    so they enter the SPECULATIVE sleeve. Core tier is reserved for the pinned universe."""
    return SPECULATIVE


def _stamp_inventory(inventory, symbol, tier, rationale):
    if inventory is None:
        return
    try:
        inventory.meta.set(symbol, risk_tier=tier,
                           added_date=datetime.now(timezone.utc).date().isoformat(),
                           rationale=rationale[:500], managed=True)
    except Exception as e:  # noqa: BLE001
        log.warning("inventory tag failed for %s: %s", symbol, e)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_discovery(*, broker, store: UniverseStore, llm, hype=None, inventory=None,
                  cfg: ScreenConfig = None, mode: str = "paper", as_of=None,
                  scorer=None, screen_fn=None) -> dict:
    """One daily discovery pass. Returns a summary dict (also logged)."""
    cfg = cfg or ScreenConfig.from_config()
    as_of = as_of or datetime.now(timezone.utc)

    # 1. SCREEN (deterministic, real data)
    if screen_fn is not None:
        candidates = screen_fn()
    else:
        candidates = screen(broker=broker, hype=hype, cfg=cfg)
    log.info("[DISCOVERY] screener produced %d candidates: %s",
             len(candidates), [c.symbol for c in candidates])

    signals = {}
    if candidates:
        # 2. EVALUATE (LLM, advisory) — reuse the news scorer over the candidate set
        scorer = scorer or DiscoveryScorer(llm=llm)
        syms = [c.symbol for c in candidates]
        try:
            ps = scorer.evaluate(syms, as_of=as_of)
            signals = ps.signals
        except Exception as e:  # noqa: BLE001 — a scorer failure must not crash discovery
            log.warning("[DISCOVERY] LLM eval failed (%s) -> no admissions this pass", e)
            signals = {}

    # 3. GATE (deterministic, final say)
    result = gate(candidates, signals, broker=broker, store=store, inventory=inventory,
                  cfg=cfg, mode=mode)

    _log_decisions(as_of, mode, candidates, signals, result)
    log.info("[DISCOVERY %s] admitted=%d rejected=%d evicted=%d universe_size=%d",
             mode, len(result["admitted"]), len(result["rejected"]),
             len(result["evicted"]), len(store.symbols()))
    return {"candidates": candidates, "signals": signals, **result,
            "universe_size": len(store.symbols())}


def _log_decisions(as_of, mode, candidates, signals, result):
    path = config.UNIVERSE_DISCOVERY_LOG
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ts = as_of.isoformat()
    admitted = {a["symbol"] for a in result["admitted"]}
    rej = dict(result["rejected"])
    cols = ["ts", "mode", "symbol", "decision", "reason", "tier", "price",
            "dollar_volume", "volatility", "rel_volume", "float_shares",
            "llm_score", "llm_conf", "conviction", "reasons", "rationale"]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for c in candidates:
            sym = c.symbol.upper()
            sig = signals.get(sym)
            conv = (sig.score * sig.confidence) if (sig and sig.ok) else 0.0
            decision = "admitted" if sym in admitted else "rejected"
            adm = next((a for a in result["admitted"] if a["symbol"] == sym), None)
            w.writerow({
                "ts": ts, "mode": mode, "symbol": sym, "decision": decision,
                "reason": ("" if decision == "admitted" else rej.get(sym, "")),
                "tier": (adm["tier"] if adm else ""),
                "price": round(c.price, 4), "dollar_volume": round(c.dollar_volume, 0),
                "volatility": round(c.volatility, 4), "rel_volume": round(c.rel_volume, 3),
                "float_shares": ("" if c.float_shares is None else int(c.float_shares)),
                "llm_score": (round(sig.score, 4) if sig else ""),
                "llm_conf": (round(sig.confidence, 4) if sig else ""),
                "conviction": round(conv, 4),
                "reasons": "|".join(c.reasons),
                "rationale": (sig.rationale[:200] if sig else ""),
            })
