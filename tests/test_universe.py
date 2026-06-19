"""
Dynamic-universe DISCOVERY tests — fully mocked, NO live network.

Covers the contract the feature must hold:
  * screener threshold filters (price cap, the HARD liquidity floor, hot-signal gate),
  * the gate rejects non-Alpaca / illiquid names (tradability + liquidity, final say),
  * the anti-hallucination contract: a ticker the LLM returns that the SCREENER did not
    produce is structurally ignored (the gate iterates candidates, never LLM output),
  * universe size cap + churn/eviction (weakest/stalest first, never pinned/held),
  * the speculative-sleeve cap + tighter per-name cap enforced by the risk layer,
  * tier-aware TIGHT stops/take-profits for speculative (penny) names,
  * store persistence + tier map.

Run:  python tests/test_universe.py
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from universe.screener import (ScreenConfig, screen, compute_metrics,
                               passes_hard_gates, Candidate)
from universe.store import UniverseStore, CORE, SPECULATIVE
from universe import discovery
from universe.discovery import gate
from portfolio.risk import (SpeculativeLimits, enforce_speculative_sleeve)
from strategies.portfolio_base import SymbolSignal


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_bars(prices, volumes=None):
    n = len(prices)
    volumes = volumes if volumes is not None else [1_000_000] * n
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": prices,
                         "high": [p * 1.03 for p in prices],
                         "low": [p * 0.97 for p in prices],
                         "close": prices, "volume": volumes}, index=idx)


def oscillating(base, amp, n=30):
    """High-volatility price path (alternating +/- amp around base)."""
    return [round(base * (1 + (amp if i % 2 else -amp)), 4) for i in range(n)]


def flat(base, n=30):
    return [round(base * (1 + 0.001 * i), 4) for i in range(n)]   # ~flat drift


def asset(sym, exchange="NASDAQ", tradable=True, status="ACTIVE"):
    return {"symbol": sym, "name": sym, "exchange": exchange, "tradable": tradable,
            "status": status, "fractionable": True, "shortable": False,
            "marginable": False, "easy_to_borrow": False}


CFG = ScreenConfig(price_cap=5.0, min_dollar_volume=1_000_000.0, min_volatility=0.04,
                   rvol_min=1.5, max_float=75_000_000, lookback_days=30, top_n=20,
                   include_hype=False)


# --------------------------------------------------------------------------- #
# screener metrics + threshold filters
# --------------------------------------------------------------------------- #
class TestScreenerFilters(unittest.TestCase):
    def test_metrics_basic(self):
        m = compute_metrics(make_bars(oscillating(2.5, 0.10), [2_000_000] * 30), CFG)
        self.assertAlmostEqual(m["price"], 2.5 * (1 + 0.10), places=3)  # last bar is +amp
        self.assertGreater(m["volatility"], 0.04)
        self.assertGreater(m["dollar_volume"], 1_000_000)

    def test_penny_highvol_liquid_passes(self):
        m = compute_metrics(make_bars(oscillating(2.5, 0.10), [2_000_000] * 30), CFG)
        ok, reasons = passes_hard_gates(m, CFG)
        self.assertTrue(ok)
        self.assertIn("penny", reasons)
        self.assertIn("liquid", reasons)
        self.assertIn("high_vol", reasons)

    def test_price_above_cap_rejected(self):
        m = compute_metrics(make_bars(oscillating(50.0, 0.10), [2_000_000] * 30), CFG)
        ok, reasons = passes_hard_gates(m, CFG)
        self.assertFalse(ok)
        self.assertEqual(reasons, ["price>cap"])

    def test_liquidity_floor_rejects_thin_penny(self):
        # a penny with huge volatility but TINY dollar volume must be rejected — the
        # liquidity floor applies even to pennies (can't enter/exit a name this thin).
        m = compute_metrics(make_bars(oscillating(2.5, 0.10), [100] * 30), CFG)
        self.assertLess(m["dollar_volume"], CFG.min_dollar_volume)
        ok, reasons = passes_hard_gates(m, CFG)
        self.assertFalse(ok)
        self.assertEqual(reasons, ["illiquid"])

    def test_flat_quiet_name_rejected_not_hot(self):
        m = compute_metrics(make_bars(flat(2.5), [1_000_000] * 30), CFG)
        ok, reasons = passes_hard_gates(m, CFG)
        self.assertFalse(ok)
        self.assertEqual(reasons, ["not_hot"])

    def test_screen_end_to_end_filters_and_ranks(self):
        assets = [asset("PENNY"), asset("BIGP"), asset("THIN"), asset("FLAT"),
                  asset("OTCX", exchange="OTC")]      # OTC -> excluded by exchange
        bars = {
            "PENNY": make_bars(oscillating(2.5, 0.12), [3_000_000] * 30),
            "BIGP": make_bars(oscillating(50.0, 0.12), [3_000_000] * 30),   # price>cap
            "THIN": make_bars(oscillating(2.5, 0.12), [50] * 30),           # illiquid
            "FLAT": make_bars(flat(2.5), [3_000_000] * 30),                 # not hot
            "OTCX": make_bars(oscillating(2.0, 0.15), [3_000_000] * 30),    # not tradable exch
        }
        cands = screen(assets_fn=lambda: assets, bars_fn=lambda syms: bars,
                       float_fn=lambda s: None, hype_names_fn=lambda: [], cfg=CFG)
        syms = [c.symbol for c in cands]
        self.assertEqual(syms, ["PENNY"])     # only the liquid penny high-vol name

    def test_low_float_earns_reason_and_rank(self):
        assets = [asset("LOWF"), asset("HIF")]
        bars = {"LOWF": make_bars(oscillating(2.5, 0.12), [3_000_000] * 30),
                "HIF": make_bars(oscillating(2.5, 0.12), [3_000_000] * 30)}
        floats = {"LOWF": 5_000_000, "HIF": 500_000_000}
        cands = screen(assets_fn=lambda: assets, bars_fn=lambda syms: bars,
                       float_fn=lambda s: floats[s], hype_names_fn=lambda: [], cfg=CFG)
        by = {c.symbol: c for c in cands}
        self.assertIn("low_float", by["LOWF"].reasons)
        self.assertNotIn("low_float", by["HIF"].reasons)
        self.assertGreater(by["LOWF"].score, by["HIF"].score)   # low float ranks higher


# --------------------------------------------------------------------------- #
# gate (deterministic, final say)
# --------------------------------------------------------------------------- #
class FakeBroker:
    def __init__(self, tradable=None, positions=None):
        self.tradable = set(tradable or [])
        self._positions = positions or []
    def get_asset(self, sym):
        return asset(sym) if sym.upper() in self.tradable else None
    def list_positions(self):
        return list(self._positions)


def cand(sym, price=2.5, dv=5_000_000, vol=0.12, rvol=2.0, reasons=("penny", "liquid")):
    return Candidate(symbol=sym, price=price, dollar_volume=dv, volatility=vol,
                     rel_volume=rvol, reasons=list(reasons))


def sig(sym, score=0.8, conf=0.9, ok=True):
    return SymbolSignal(sym, score=score, confidence=conf, ok=ok)


def _store(tmp, pinned=("AAPL",)):
    return UniverseStore(path=os.path.join(tmp, "u.json"), pinned=list(pinned))


class TestGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # keep discovery's CSV log out of the repo during tests
        import config
        self._log = config.UNIVERSE_DISCOVERY_LOG
        config.UNIVERSE_DISCOVERY_LOG = os.path.join(self.tmp, "disc.csv")

    def tearDown(self):
        import config
        config.UNIVERSE_DISCOVERY_LOG = self._log

    def test_admits_tradable_liquid_high_conviction(self):
        store = _store(self.tmp)
        broker = FakeBroker(tradable=["ABCD"])
        res = gate([cand("ABCD")], {"ABCD": sig("ABCD")}, broker=broker, store=store,
                   cfg=CFG, mode="paper")
        self.assertEqual([a["symbol"] for a in res["admitted"]], ["ABCD"])
        self.assertEqual(store.tier("ABCD"), SPECULATIVE)
        self.assertIn("ABCD", store)

    def test_rejects_non_alpaca_name(self):
        store = _store(self.tmp)
        broker = FakeBroker(tradable=[])           # Alpaca has no such tradable asset
        res = gate([cand("ZZZZ")], {"ZZZZ": sig("ZZZZ")}, broker=broker, store=store,
                   cfg=CFG, mode="paper")
        self.assertEqual(res["admitted"], [])
        self.assertIn(("ZZZZ", "not_tradable_now"), res["rejected"])
        self.assertNotIn("ZZZZ", store)

    def test_rejects_illiquid_below_floor(self):
        store = _store(self.tmp)
        broker = FakeBroker(tradable=["THIN"])
        res = gate([cand("THIN", dv=10_000)], {"THIN": sig("THIN")}, broker=broker,
                   store=store, cfg=CFG, mode="paper")
        self.assertIn(("THIN", "below_liquidity_floor"), res["rejected"])
        self.assertNotIn("THIN", store)

    def test_low_conviction_rejected(self):
        store = _store(self.tmp)
        broker = FakeBroker(tradable=["MEH"])
        res = gate([cand("MEH")], {"MEH": sig("MEH", score=0.05, conf=0.2)},
                   broker=broker, store=store, cfg=CFG, mode="paper",
                   min_conviction=0.10)
        self.assertTrue(any(r[0] == "MEH" and r[1].startswith("low_conviction")
                            for r in res["rejected"]))

    def test_off_list_llm_ticker_is_ignored(self):
        # ANTI-HALLUCINATION CONTRACT: the LLM returns a great score for FAKEZ, which the
        # SCREENER never produced. The gate iterates CANDIDATES only, so FAKEZ can never
        # enter the universe — only the real candidate REAL is admitted.
        store = _store(self.tmp)
        broker = FakeBroker(tradable=["REAL", "FAKEZ"])   # even if it were "tradable"
        signals = {"REAL": sig("REAL"), "FAKEZ": sig("FAKEZ", score=1.0, conf=1.0)}
        res = gate([cand("REAL")], signals, broker=broker, store=store, cfg=CFG,
                   mode="paper")
        admitted = [a["symbol"] for a in res["admitted"]]
        self.assertEqual(admitted, ["REAL"])
        self.assertNotIn("FAKEZ", store)
        self.assertNotIn("FAKEZ", admitted)

    def test_respects_manual_exclusion(self):
        store = _store(self.tmp)
        broker = FakeBroker(tradable=["EXC"])

        class Inv:
            class meta:
                @staticmethod
                def get(s): return {"managed": False}   # explicitly walled off
                @staticmethod
                def set(s, **k): raise AssertionError("must not tag an excluded name")
            broker = FakeBroker(tradable=["EXC"])
        res = gate([cand("EXC")], {"EXC": sig("EXC")}, broker=broker, store=store,
                   inventory=Inv(), cfg=CFG, mode="paper")
        self.assertIn(("EXC", "manually_excluded"), res["rejected"])
        self.assertNotIn("EXC", store)

    def test_universe_cap_and_eviction(self):
        # max_size 3, pinned AAPL (never evicted) + 2 dynamic. A stronger candidate
        # forces eviction of the WEAKEST dynamic name (lowest conviction).
        store = _store(self.tmp, pinned=("AAPL",))
        store.add("WEAK", tier=SPECULATIVE, conviction=0.11)
        store.add("MID", tier=SPECULATIVE, conviction=0.40)
        self.assertEqual(len(store.symbols()), 3)        # full at max_size=3
        broker = FakeBroker(tradable=["STRONG"])
        res = gate([cand("STRONG")], {"STRONG": sig("STRONG", 0.9, 0.9)},
                   broker=broker, store=store, cfg=CFG, mode="paper", max_size=3)
        self.assertIn("STRONG", store)
        self.assertIn("WEAK", res["evicted"])            # weakest dynamic evicted
        self.assertNotIn("WEAK", store)
        self.assertIn("MID", store)                      # stronger one survives
        self.assertIn("AAPL", store.symbols())           # pinned never evicted
        self.assertEqual(len(store.symbols()), 3)        # cap respected

    def test_eviction_never_touches_held_or_pinned(self):
        store = _store(self.tmp, pinned=("AAPL",))
        store.add("HELD", tier=SPECULATIVE, conviction=0.0)   # weakest, but we hold it
        store.add("OTHER", tier=SPECULATIVE, conviction=0.5)
        # inventory reports a position in HELD -> protected from eviction
        broker = FakeBroker(tradable=["STRONG"], positions=[{"symbol": "HELD", "qty": 10}])

        class Inv:
            def __init__(self, b): self.broker = b
            class meta:
                @staticmethod
                def get(s): return {}
                @staticmethod
                def set(s, **k): pass
        res = gate([cand("STRONG")], {"STRONG": sig("STRONG", 0.9, 0.9)},
                   broker=broker, store=store, inventory=Inv(broker), cfg=CFG,
                   mode="paper", max_size=3)
        self.assertIn("HELD", store)                     # held name not evicted
        self.assertIn("OTHER", res["evicted"])           # the next-weakest went instead

    def test_dry_mode_writes_nothing(self):
        store = _store(self.tmp)
        broker = FakeBroker(tradable=["ABCD"])
        res = gate([cand("ABCD")], {"ABCD": sig("ABCD")}, broker=broker, store=store,
                   cfg=CFG, mode="dry")
        self.assertEqual([a["symbol"] for a in res["admitted"]], ["ABCD"])  # previewed
        self.assertNotIn("ABCD", store)                  # but NOT persisted


# --------------------------------------------------------------------------- #
# DiscoveryScorer anti-hallucination at the scorer layer (reuses batch parser)
# --------------------------------------------------------------------------- #
class TestDiscoveryScorer(unittest.TestCase):
    def test_scorer_ignores_off_list_ticker(self):
        import json
        from agents.llm import LLM, LLMResult

        class WildLLM(LLM):
            name = "wild"
            def complete_json(self, prompt):
                arr = [{"ticker": "REAL", "score": 0.7, "confidence": 0.8, "rationale": "ok"},
                       {"ticker": "FAKEZ", "score": 1.0, "confidence": 1.0, "rationale": "hallucinated"}]
                return LLMResult(parsed={"scores": arr}, raw=json.dumps({"scores": arr}))

        def fake_info(sym, as_of=None, limit=12):
            return [{"item_type": "news", "headline": f"{sym} news", "summary": "x",
                     "source": "t", "published_utc": None}]
        s = discovery.DiscoveryScorer(llm=WildLLM(), audit_dir=tempfile.mkdtemp())
        s.get_info_fn = fake_info
        ps = s.evaluate(["REAL"])                  # only REAL was screened
        self.assertIn("REAL", ps.signals)
        self.assertNotIn("FAKEZ", ps.signals)      # off-list hallucination dropped


# --------------------------------------------------------------------------- #
# speculative sleeve (risk layer)
# --------------------------------------------------------------------------- #
class TestSpeculativeSleeve(unittest.TestCase):
    SPEC = SpeculativeLimits(enabled=True, sleeve_pct=0.15, max_weight=0.05,
                             stop_pct=0.04, take_profit_pct=0.08)

    def test_combined_sleeve_capped(self):
        # three speculative names at 0.05 each = 0.15 sleeve OK; push them higher and the
        # combined sleeve is scaled back to 0.15.
        weights = {"S1": 0.05, "S2": 0.05, "S3": 0.05, "CORE": 0.20}
        tiers = {"S1": "speculative", "S2": "speculative", "S3": "speculative", "CORE": "core"}
        # bump each to exceed per-name first
        weights = {"S1": 0.10, "S2": 0.10, "S3": 0.10, "CORE": 0.20}
        w, notes = enforce_speculative_sleeve(weights, tiers, self.SPEC)
        # per-name cap 0.05 each -> 0.15 combined == sleeve cap, core untouched
        self.assertAlmostEqual(w["S1"], 0.05, places=6)
        self.assertAlmostEqual(w["CORE"], 0.20, places=6)
        spec_total = w["S1"] + w["S2"] + w["S3"]
        self.assertLessEqual(spec_total, 0.15 + 1e-9)

    def test_sleeve_scales_when_over(self):
        # two speculative names each within the 0.05 per-name cap but combined 0.10... set
        # the sleeve to 0.06 so the combined must scale down.
        spec = SpeculativeLimits(enabled=True, sleeve_pct=0.06, max_weight=0.05)
        weights = {"S1": 0.05, "S2": 0.05, "CORE": 0.30}
        tiers = {"S1": "speculative", "S2": "speculative", "CORE": "core"}
        w, notes = enforce_speculative_sleeve(weights, tiers, spec)
        self.assertAlmostEqual(w["S1"] + w["S2"], 0.06, places=6)
        self.assertAlmostEqual(w["CORE"], 0.30, places=6)        # core never touched
        self.assertTrue(any("sleeve" in n for n in notes))

    def test_core_only_is_noop(self):
        weights = {"A": 0.20, "B": 0.20}
        tiers = {"A": "core", "B": "core"}
        w, notes = enforce_speculative_sleeve(weights, tiers, self.SPEC)
        self.assertEqual(w, weights)
        self.assertEqual(notes, [])

    def test_disabled_is_noop(self):
        spec = SpeculativeLimits(enabled=False)
        weights = {"S1": 0.50}
        w, _ = enforce_speculative_sleeve(weights, {"S1": "speculative"}, spec)
        self.assertEqual(w, weights)


# --------------------------------------------------------------------------- #
# tier-aware tight stops/take-profits
# --------------------------------------------------------------------------- #
class TestTierAwareStops(unittest.TestCase):
    def test_spec_exit_settings_are_tighter(self):
        from service.risk_exits import ExitSettings, settings_for_tier
        core = ExitSettings.from_config()
        spec = ExitSettings.spec_from_config()
        self.assertLess(spec.stop_pct, core.stop_pct)
        self.assertLess(spec.take_profit_pct, core.take_profit_pct)
        self.assertIs(settings_for_tier("speculative", core, spec), spec)
        self.assertIs(settings_for_tier("core", core, spec), core)

    def test_spec_exit_triggers_sooner(self):
        from service.risk_exits import evaluate_exit, ExitSettings
        core = ExitSettings(stop_pct=0.08, take_profit_pct=0.20, use_sr=False)
        spec = ExitSettings(stop_pct=0.04, take_profit_pct=0.08, use_sr=False)
        # down 5% from a $2.00 entry: core holds (>-8%), speculative stops out (<=-4%)
        self.assertFalse(evaluate_exit(2.00, 1.90, None, core)["exit"])
        self.assertTrue(evaluate_exit(2.00, 1.90, None, spec)["exit"])
        # up 10%: core holds (<+20%), speculative takes profit (>=+8%)
        self.assertFalse(evaluate_exit(2.00, 2.20, None, core)["exit"])
        self.assertEqual(evaluate_exit(2.00, 2.20, None, spec)["reason"], "take_profit")

    def test_protective_uses_tight_settings_for_speculative(self):
        from service.protective import ProtectiveOrderManager

        class Inv:
            class meta:
                @staticmethod
                def get(s):
                    return {"risk_tier": "speculative"} if s == "PENNY" else {"risk_tier": "core"}
        pom = ProtectiveOrderManager()
        s_spec = pom._settings_for(Inv(), "PENNY")
        s_core = pom._settings_for(Inv(), "BIGCO")
        self.assertLess(s_spec.stop_pct, s_core.stop_pct)
        # tight stop floor sits closer to entry for the speculative name
        self.assertGreater(pom.stop_price(10.0, s_spec), pom.stop_price(10.0, s_core))


# --------------------------------------------------------------------------- #
# store persistence + tiers
# --------------------------------------------------------------------------- #
class TestUniverseStore(unittest.TestCase):
    def test_persist_and_reload(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "u.json")
        s = UniverseStore(path=path, pinned=["AAPL", "MSFT"])
        self.assertTrue(s.add("SNDL", tier=SPECULATIVE, rationale="penny squeeze",
                              conviction=0.3))
        self.assertFalse(s.add("AAPL"))            # pinned not stored as dynamic
        s.save()
        s2 = UniverseStore(path=path, pinned=["AAPL", "MSFT"])
        self.assertIn("SNDL", s2.dynamic_symbols())
        self.assertEqual(s2.tier("SNDL"), SPECULATIVE)
        self.assertEqual(s2.tier("AAPL"), CORE)
        self.assertTrue(s2.entry("SNDL")["added_date"])

    def test_tier_map_covers_pinned_and_dynamic(self):
        tmp = tempfile.mkdtemp()
        s = UniverseStore(path=os.path.join(tmp, "u.json"), pinned=["AAPL"])
        s.add("XYZ", tier=SPECULATIVE)
        tm = s.tier_map()
        self.assertEqual(tm["AAPL"], CORE)
        self.assertEqual(tm["XYZ"], SPECULATIVE)
        self.assertEqual(set(s.symbols()), {"AAPL", "XYZ"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
