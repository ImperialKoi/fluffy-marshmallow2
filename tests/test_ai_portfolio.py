"""
Phase 3 AI portfolio tests — mocked LLM + mocked broker, NO live network.

Covers: LLM JSON parsing/validation, graceful handling of malformed/empty/
out-of-universe LLM output, constructor weighting + limit enforcement, multi-symbol
reconciliation (incl. the unmanaged-position wall-off and new-entry flag), and the
benchmark-tracking math.

Run:  python tests/test_ai_portfolio.py
"""

import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm import LLM, LLMResult, extract_json, StubLLM
from agents.news_portfolio import NewsPortfolioStrategy
from strategies.portfolio_base import SymbolSignal
from portfolio.constructor import construct_targets
from portfolio.risk import PortfolioLimits
from live_portfolio import (compute_orders, equal_weight_benchmark, spy_benchmark,
                            portfolio_turnover)

UTC = timezone.utc


# --------------------------------------------------------------------------- #
# mocks
# --------------------------------------------------------------------------- #
class MockLLM(LLM):
    """Returns a preset score per ticker (parsed from the prompt), or a forced
    response for malformed/edge-case tests."""
    name = "mock"

    def __init__(self, mapping=None, force=None):
        self.mapping = mapping or {}
        self.force = force

    def complete_json(self, prompt: str) -> LLMResult:
        if self.force is not None:
            return self.force(prompt)
        m = re.search(r"TICKER:\s*(\w+)", prompt)
        sym = m.group(1) if m else ""
        if sym in self.mapping:
            score, conf = self.mapping[sym]
            parsed = {"ticker": sym, "score": score, "confidence": conf,
                      "rationale": "mock rationale"}
            return LLMResult(parsed=parsed, raw=json.dumps(parsed))
        return LLMResult(parsed=None, raw="", error="no mapping")


def fake_get_info(n=3):
    def _fn(sym, as_of=None, limit=12):
        return [{"item_type": "news", "published_utc": datetime(2026, 6, 15, tzinfo=UTC),
                 "headline": f"{sym} headline {i}", "summary": "details",
                 "source": "rss/test"} for i in range(n)]
    return _fn


class MockBroker:
    def __init__(self, positions, equity):
        self._pos = positions          # symbol -> qty
        self._equity = equity
        self.orders = []

    def equity(self):
        return self._equity

    def list_positions(self):
        return [{"symbol": s, "qty": q} for s, q in self._pos.items() if q]


# --------------------------------------------------------------------------- #
# LLM parsing / validation
# --------------------------------------------------------------------------- #
class TestLLMParsing(unittest.TestCase):
    def test_extract_json_plain_and_fenced(self):
        self.assertEqual(extract_json('{"a": 1}')["a"], 1)
        self.assertEqual(extract_json('```json\n{"a": 2}\n```')["a"], 2)
        self.assertEqual(extract_json("prefix {\"a\": 3} suffix")["a"], 3)
        self.assertIsNone(extract_json("not json at all"))
        self.assertIsNone(extract_json(""))

    def test_stub_llm_is_deterministic_structured(self):
        stub = StubLLM()
        r1 = stub.complete_json("TICKER: AAPL\nAAPL beats record upgrade surge")
        r2 = stub.complete_json("TICKER: AAPL\nAAPL beats record upgrade surge")
        self.assertEqual(r1.parsed, r2.parsed)
        self.assertGreater(r1.parsed["score"], 0)         # bullish words -> positive
        bear = stub.complete_json("TICKER: X\nlawsuit downgrade plunge miss recall")
        self.assertLess(bear.parsed["score"], 0)


class TestFallbackLLM(unittest.TestCase):
    from agents.llm import LLMResult as _R

    def _llm(self, result):
        from agents.llm import LLM, LLMResult
        class _One(LLM):
            name = "x"
            def __init__(self, r): self.r = r; self.calls = 0
            def complete_json(self, prompt):
                self.calls += 1
                return self.r
        return _One(result)

    def test_falls_back_only_on_503(self):
        from agents.llm import FallbackLLM, LLMResult
        primary = self._llm(LLMResult(parsed=None, error="ServerError: 503 UNAVAILABLE"))
        fb = self._llm(LLMResult(parsed={"ticker": "AAPL", "score": 0.5, "confidence": 0.8}))
        fl = FallbackLLM(primary, fb)
        out = fl.complete_json("TICKER: AAPL")
        self.assertEqual(out.parsed["score"], 0.5)        # used OpenAI
        self.assertEqual(fb.calls, 1)

    def test_no_fallback_on_success(self):
        from agents.llm import FallbackLLM, LLMResult
        primary = self._llm(LLMResult(parsed={"ticker": "AAPL", "score": 0.9, "confidence": 1.0}))
        fb = self._llm(LLMResult(parsed={"score": 0.0}))
        fl = FallbackLLM(primary, fb)
        out = fl.complete_json("TICKER: AAPL")
        self.assertEqual(out.parsed["score"], 0.9)        # primary's answer
        self.assertEqual(fb.calls, 0)                     # fallback NOT called

    def test_no_fallback_on_nonoutage_error(self):
        from agents.llm import FallbackLLM, LLMResult
        primary = self._llm(LLMResult(parsed=None, error="ValueError: unparseable JSON"))
        fb = self._llm(LLMResult(parsed={"score": 0.0}))
        fl = FallbackLLM(primary, fb)
        out = fl.complete_json("TICKER: AAPL")
        self.assertIsNone(out.parsed)                     # passed through, no fallback
        self.assertEqual(fb.calls, 0)

    def test_three_tier_gemini_cohere_openai(self):
        # Gemini 503 -> Cohere 503 -> OpenAI succeeds (3rd choice)
        from agents.llm import FallbackLLM, LLMResult
        gem = self._llm(LLMResult(parsed=None, error="ServerError: 503 UNAVAILABLE"))
        coh = self._llm(LLMResult(parsed=None, error="503 service unavailable"))
        oai = self._llm(LLMResult(parsed={"ticker": "AAPL", "score": 0.3, "confidence": 0.7}))
        fl = FallbackLLM(gem, coh, oai)
        out = fl.complete_json("TICKER: AAPL")
        self.assertEqual(out.parsed["score"], 0.3)        # used the 3rd tier
        self.assertEqual((gem.calls, coh.calls, oai.calls), (1, 1, 1))

    def test_gemini_circuit_breaker_then_failover(self):
        # Gemini 429s -> after 3 tries the breaker opens; the NEXT call skips Gemini
        # entirely (no _call) and the chain rolls to Cohere.
        from agents.llm import GeminiLLM, FallbackLLM, LLMResult
        gem = GeminiLLM.__new__(GeminiLLM)        # bypass SDK/key init
        gem.retries, gem.timeout, gem.temperature = 3, 5, 0.0
        gem.name = "gemini:test"; gem._cooldown_until = 0.0; gem.retry_backoff = 0.0
        calls = {"n": 0}

        def boom(prompt):
            calls["n"] += 1
            raise RuntimeError("ClientError: 429 RESOURCE_EXHAUSTED ... PerDay ... limit: 20")
        gem._call = boom

        r1 = gem.complete_json("TICKER: AAPL")
        self.assertEqual(calls["n"], 3)           # exactly 3 tries, then give up
        self.assertIn("429", r1.error)
        r2 = gem.complete_json("TICKER: MSFT")    # breaker open -> no _call at all
        self.assertEqual(calls["n"], 3)           # unchanged: Gemini was skipped
        self.assertIn("429", r2.error)            # 429-flavored -> chain advances

        coh = self._llm(LLMResult(parsed={"ticker": "MSFT", "score": 0.4, "confidence": 0.7}))
        chain = FallbackLLM(gem, coh)
        out = chain.complete_json("TICKER: MSFT")
        self.assertEqual(out.parsed["score"], 0.4)   # served by Cohere
        self.assertEqual(calls["n"], 3)              # Gemini still skipped (cooldown)

    def test_404_model_error_rolls_over(self):
        # Cohere returns a 404 (retired model) -> must advance to OpenAI, not dead-end.
        from agents.llm import FallbackLLM, LLMResult
        gem = self._llm(LLMResult(parsed=None, error="429 RESOURCE_EXHAUSTED"))
        coh = self._llm(LLMResult(parsed=None, error="NotFoundError: 404 model removed"))
        oai = self._llm(LLMResult(parsed={"ticker": "AAPL", "score": 0.5, "confidence": 0.8}))
        out = FallbackLLM(gem, coh, oai).complete_json("TICKER: AAPL")
        self.assertEqual(out.parsed["score"], 0.5)
        self.assertEqual(oai.calls, 1)

    def test_stops_at_cohere_when_it_answers(self):
        # Gemini 503 -> Cohere answers -> OpenAI never called (2nd choice wins)
        from agents.llm import FallbackLLM, LLMResult
        gem = self._llm(LLMResult(parsed=None, error="503 UNAVAILABLE"))
        coh = self._llm(LLMResult(parsed={"ticker": "AAPL", "score": 0.6, "confidence": 0.8}))
        oai = self._llm(LLMResult(parsed={"score": 0.0}))
        fl = FallbackLLM(gem, coh, oai)
        out = fl.complete_json("TICKER: AAPL")
        self.assertEqual(out.parsed["score"], 0.6)
        self.assertEqual(oai.calls, 0)                    # 3rd tier untouched


class TestStrategyValidation(unittest.TestCase):
    UNIVERSE = ["AAPL", "MSFT", "TSLA"]

    def _strategy(self, llm):
        s = NewsPortfolioStrategy(llm=llm, universe=self.UNIVERSE,
                                  get_info_fn=fake_get_info(), audit_dir=tempfile.mkdtemp())
        s.batch_scoring = False        # these exercise the per-symbol path explicitly
        return s

    def test_valid_scores_parsed(self):
        llm = MockLLM(mapping={"AAPL": (0.8, 0.9), "MSFT": (-0.5, 0.6), "TSLA": (0.1, 0.3)})
        ps = self._strategy(llm).evaluate(self.UNIVERSE, as_of=datetime(2026, 6, 15, tzinfo=UTC))
        self.assertAlmostEqual(ps.signals["AAPL"].score, 0.8)
        self.assertTrue(ps.signals["AAPL"].ok)
        self.assertAlmostEqual(ps.signals["MSFT"].score, -0.5)

    def test_out_of_range_is_clamped(self):
        llm = MockLLM(mapping={"AAPL": (5.0, 9.0), "MSFT": (-3.0, -1.0), "TSLA": (0.0, 0.0)})
        ps = self._strategy(llm).evaluate(self.UNIVERSE)
        self.assertEqual(ps.signals["AAPL"].score, 1.0)        # clamped to [-1,1]
        self.assertEqual(ps.signals["AAPL"].confidence, 1.0)   # clamped to [0,1]
        self.assertEqual(ps.signals["MSFT"].score, -1.0)
        self.assertEqual(ps.signals["MSFT"].confidence, 0.0)

    def test_malformed_defaults_to_zero(self):
        llm = MockLLM(force=lambda p: LLMResult(parsed=None, raw="garbage", error="bad"))
        ps = self._strategy(llm).evaluate(self.UNIVERSE)
        for sym in self.UNIVERSE:
            self.assertEqual(ps.signals[sym].score, 0.0)
            self.assertFalse(ps.signals[sym].ok)

    def test_out_of_universe_ticker_ignored(self):
        # model returns a different, out-of-universe ticker -> distrust, default 0
        llm = MockLLM(force=lambda p: LLMResult(
            parsed={"ticker": "ZZZZ", "score": 0.9, "confidence": 0.9, "rationale": "x"},
            raw="{}"))
        ps = self._strategy(llm).evaluate(self.UNIVERSE)
        for sym in self.UNIVERSE:
            self.assertEqual(ps.signals[sym].score, 0.0)
            self.assertIn("out_of_universe_ticker", ps.signals[sym].error)

    def test_missing_news_defaults_zero_without_calling_llm(self):
        calls = {"n": 0}

        class CountingLLM(MockLLM):
            def complete_json(self, prompt):
                calls["n"] += 1
                return super().complete_json(prompt)

        strat = NewsPortfolioStrategy(llm=CountingLLM(mapping={"AAPL": (0.5, 0.5)}),
                                      universe=["AAPL"], get_info_fn=fake_get_info(n=0),
                                      audit_dir=tempfile.mkdtemp())
        strat.discovery_count = 0                       # isolate the per-symbol no-news path
        ps = strat.evaluate(["AAPL"])
        self.assertEqual(ps.signals["AAPL"].score, 0.0)
        self.assertEqual(ps.signals["AAPL"].error, "no_news")
        self.assertEqual(calls["n"], 0)                 # no per-symbol LLM call when no news

    def test_strategy_never_raises_on_get_info_failure(self):
        def boom(sym, as_of=None, limit=12):
            raise RuntimeError("feed down")
        strat = NewsPortfolioStrategy(llm=MockLLM(), universe=["AAPL"], get_info_fn=boom,
                                      audit_dir=tempfile.mkdtemp())
        ps = strat.evaluate(["AAPL"])                   # must not raise
        self.assertEqual(ps.signals["AAPL"].score, 0.0)


# --------------------------------------------------------------------------- #
# constructor
# --------------------------------------------------------------------------- #
def _sigs(d):
    return {t: SymbolSignal(t, score=s, confidence=c) for t, (s, c) in d.items()}


class TestBatchScoring(unittest.TestCase):
    UNIVERSE = ["AAPL", "MSFT", "TSLA"]

    class BatchLLM(LLM):
        """Returns one JSON batch for whatever tickers appear in the prompt; counts calls."""
        name = "batch-mock"
        def __init__(self, scores, drop=()):  # scores: sym->(score,conf); drop: omit these
            self.scores = scores; self.drop = set(drop); self.calls = 0
        def complete_json(self, prompt):
            self.calls += 1
            syms = re.findall(r"=== (\w+) ===", prompt)
            arr = [{"ticker": s, "score": self.scores[s][0], "confidence": self.scores[s][1],
                    "rationale": "batch"} for s in syms
                   if s in self.scores and s not in self.drop]
            return LLMResult(parsed={"scores": arr}, raw=json.dumps({"scores": arr}))

    def _strat(self, llm):
        s = NewsPortfolioStrategy(llm=llm, universe=self.UNIVERSE,
                                  get_info_fn=fake_get_info(), audit_dir=tempfile.mkdtemp())
        s.batch_scoring = True
        s.discovery_count = 0          # isolate batch scoring
        return s

    def test_one_call_scores_whole_basket(self):
        llm = self.BatchLLM({"AAPL": (0.8, 0.9), "MSFT": (-0.3, 0.6), "TSLA": (0.1, 0.5)})
        ps = self._strat(llm).evaluate(self.UNIVERSE)
        self.assertEqual(llm.calls, 1)                       # ONE call for all 3 symbols
        self.assertAlmostEqual(ps.signals["AAPL"].score, 0.8)
        self.assertAlmostEqual(ps.signals["MSFT"].score, -0.3)
        self.assertTrue(all(ps.signals[s].ok for s in self.UNIVERSE))

    def test_missing_symbol_defaults(self):
        llm = self.BatchLLM({"AAPL": (0.8, 0.9), "MSFT": (0.5, 0.7), "TSLA": (0.2, 0.5)},
                            drop=("TSLA",))                   # model omits TSLA
        ps = self._strat(llm).evaluate(self.UNIVERSE)
        self.assertTrue(ps.signals["AAPL"].ok)
        self.assertFalse(ps.signals["TSLA"].ok)
        self.assertEqual(ps.signals["TSLA"].error, "missing_from_batch")
        self.assertEqual(ps.signals["TSLA"].score, 0.0)

    def test_out_of_range_clamped_and_oos_ignored(self):
        class Wild(LLM):
            name = "w"
            def complete_json(self, prompt):
                return LLMResult(parsed={"scores": [
                    {"ticker": "AAPL", "score": 5.0, "confidence": 9.0, "rationale": "x"},
                    {"ticker": "ZZZZ", "score": 0.9, "confidence": 0.9},   # not requested
                    {"ticker": "MSFT", "score": -0.4, "confidence": 0.8},
                    {"ticker": "TSLA", "score": 0.0, "confidence": 0.0},
                ]}, raw="{}")
        ps = self._strat(Wild()).evaluate(self.UNIVERSE)
        self.assertEqual(ps.signals["AAPL"].score, 1.0)      # clamped
        self.assertEqual(ps.signals["AAPL"].confidence, 1.0)
        self.assertNotIn("ZZZZ", ps.signals)                 # out-of-set ignored

    def test_chunking_splits_calls(self):
        llm = self.BatchLLM({s: (0.5, 0.7) for s in self.UNIVERSE})
        strat = self._strat(llm); strat.batch_chunk = 1      # force 3 chunks
        ps = strat.evaluate(self.UNIVERSE)
        self.assertEqual(llm.calls, 3)
        self.assertTrue(all(ps.signals[s].ok for s in self.UNIVERSE))

    def test_no_news_symbols_excluded_from_batch_call(self):
        # all symbols have no news -> NO llm call at all
        llm = self.BatchLLM({s: (0.5, 0.7) for s in self.UNIVERSE})
        strat = NewsPortfolioStrategy(llm=llm, universe=self.UNIVERSE,
                                      get_info_fn=fake_get_info(n=0), audit_dir=tempfile.mkdtemp())
        strat.batch_scoring = True; strat.discovery_count = 0
        ps = strat.evaluate(self.UNIVERSE)
        self.assertEqual(llm.calls, 0)
        self.assertTrue(all(ps.signals[s].error == "no_news" for s in self.UNIVERSE))


class TestConstructor(unittest.TestCase):
    def test_long_only_selects_positives_above_min_score(self):
        limits = PortfolioLimits(max_weight=0.5, max_gross=1.0, min_cash=0.0,
                                 min_positions=1, max_positions=10, min_score=0.05,
                                 weighting="equal")
        sigs = _sigs({"A": (0.6, 0.9), "B": (-0.2, 0.9), "C": (0.02, 0.9)})
        res = construct_targets(sigs, limits)
        self.assertIn("A", res.weights)
        self.assertNotIn("B", res.weights)       # negative -> excluded (long only)
        self.assertNotIn("C", res.weights)       # below min_score

    def test_max_weight_cap_pushes_excess_to_cash(self):
        limits = PortfolioLimits(max_weight=0.10, max_gross=0.95, min_cash=0.05,
                                 min_positions=1, max_positions=10, weighting="equal",
                                 min_score=0.0)
        sigs = _sigs({"A": (0.9, 1.0), "B": (0.8, 1.0)})
        res = construct_targets(sigs, limits)
        self.assertLessEqual(max(res.weights.values()), 0.10 + 1e-9)
        self.assertGreater(res.cash_weight, 0.5)  # most is cash because of the tight cap

    def test_max_positions_trims(self):
        limits = PortfolioLimits(max_weight=1.0, max_gross=1.0, min_cash=0.0,
                                 min_positions=1, max_positions=2, weighting="equal",
                                 min_score=0.0)
        sigs = _sigs({"A": (0.9, 1), "B": (0.8, 1), "C": (0.7, 1), "D": (0.6, 1)})
        res = construct_targets(sigs, limits)
        self.assertEqual(len(res.weights), 2)
        self.assertEqual(set(res.weights), {"A", "B"})   # top-2 by score

    def test_min_positions_scales_budget_down(self):
        limits = PortfolioLimits(max_weight=1.0, max_gross=1.0, min_cash=0.0,
                                 min_positions=4, max_positions=10, weighting="equal",
                                 min_score=0.0)
        sigs = _sigs({"A": (0.9, 1), "B": (0.8, 1)})     # only 2 of 4 wanted
        res = construct_targets(sigs, limits)
        invested = sum(res.weights.values())
        self.assertAlmostEqual(invested, 0.5, places=6)  # 2/4 of full budget
        self.assertAlmostEqual(res.cash_weight, 0.5, places=6)

    def test_gross_and_cash_respected(self):
        limits = PortfolioLimits(max_weight=0.5, max_gross=0.8, min_cash=0.2,
                                 min_positions=1, max_positions=10, weighting="confidence",
                                 min_score=0.0)
        sigs = _sigs({"A": (0.9, 0.9), "B": (0.7, 0.8), "C": (0.5, 0.5)})
        res = construct_targets(sigs, limits)
        self.assertLessEqual(sum(res.weights.values()), 0.8 + 1e-9)
        self.assertGreaterEqual(res.cash_weight, 0.2 - 1e-9)

    def test_exposure_multiplier_scales(self):
        limits = PortfolioLimits(max_weight=1.0, max_gross=1.0, min_cash=0.0,
                                 min_positions=1, max_positions=10, weighting="equal",
                                 min_score=0.0)
        sigs = _sigs({"A": (0.9, 1), "B": (0.8, 1)})
        full = construct_targets(sigs, limits, exposure_multiplier=1.0)
        half = construct_targets(sigs, limits, exposure_multiplier=0.5)
        self.assertAlmostEqual(sum(half.weights.values()),
                               0.5 * sum(full.weights.values()), places=6)

    def test_no_candidates_all_cash(self):
        limits = PortfolioLimits(min_score=0.05)
        res = construct_targets(_sigs({"A": (-0.1, 0.9), "B": (0.0, 0.9)}), limits)
        self.assertEqual(res.weights, {})
        self.assertEqual(res.cash_weight, 1.0)

    # ---- conviction gating + neutral dead-band (the exit-logic fix) ---- #
    def test_weak_held_signal_trims_to_cap_not_exit(self):
        # AAPL: score -0.10 x conf 0.50 = -0.05 conviction -> NEUTRAL (|.05| <= band .10).
        # Held at 95% -> must trim toward the 20% cap, NOT liquidate to 0%.
        limits = PortfolioLimits(max_weight=0.20, neutral_band=0.10, min_score=0.05)
        res = construct_targets(_sigs({"AAPL": (-0.10, 0.50)}), limits,
                                current_weights={"AAPL": 0.95})
        self.assertAlmostEqual(res.weights["AAPL"], 0.20, places=6)   # trimmed to cap
        self.assertNotIn("AAPL", res.selected)                        # not a fresh buy

    def test_strong_confident_negative_held_exits(self):
        # XOM-like: -0.80 x 0.90 = -0.72 conviction -> decisive exit to 0.
        limits = PortfolioLimits(neutral_band=0.10)
        res = construct_targets(_sigs({"XOM": (-0.80, 0.90)}), limits,
                                current_weights={"XOM": 0.20})
        self.assertNotIn("XOM", res.weights)
        self.assertAlmostEqual(res.cash_weight, 1.0)

    def test_gating_is_conviction_not_raw_score(self):
        # High score but low confidence -> low conviction -> NOT a candidate;
        # modest score with high confidence -> a candidate.
        limits = PortfolioLimits(neutral_band=0.10, min_score=0.05, max_weight=1.0,
                                 max_gross=1.0, min_cash=0.0, min_positions=1)
        res = construct_targets(_sigs({"A": (0.9, 0.05), "B": (0.3, 0.9)}), limits)
        self.assertEqual(res.selected, ["B"])
        self.assertNotIn("A", res.weights)

    def test_unheld_neutral_stays_flat(self):
        limits = PortfolioLimits(neutral_band=0.10)
        res = construct_targets(_sigs({"A": (0.1, 0.5)}), limits)   # conv 0.05, not held
        self.assertEqual(res.weights, {})

    def test_turnover_cap_limits_daily_move(self):
        # AAPL neutral, held 95%, would trim to 20% (move 0.75). With a 0.10 cap the
        # move is scaled so AAPL stays well above the cap this rebalance.
        limits = PortfolioLimits(max_weight=0.20, neutral_band=0.10, turnover_cap=0.10)
        res = construct_targets(_sigs({"AAPL": (-0.10, 0.50)}), limits,
                                current_weights={"AAPL": 0.95})
        # move scaled by 0.10/0.75 -> AAPL = 0.95 - 0.75*(0.10/0.75) = 0.85
        self.assertAlmostEqual(res.weights["AAPL"], 0.85, places=2)
        total_move = abs(res.weights["AAPL"] - 0.95)
        self.assertLessEqual(total_move, 0.10 + 1e-6)


# --------------------------------------------------------------------------- #
# reconciliation
# --------------------------------------------------------------------------- #
class TestReconciliation(unittest.TestCase):
    def test_orders_buy_sell_and_new_entry(self):
        targets = {"AAPL": 0.30, "MSFT": 0.20}
        prices = {"AAPL": 100.0, "MSFT": 50.0, "TSLA": 25.0}
        current = {"AAPL": 100, "TSLA": 40}     # hold AAPL + TSLA
        managed = {"AAPL": True, "MSFT": True, "TSLA": True}
        universe = ["AAPL", "MSFT", "TSLA"]
        orders, skipped = compute_orders(targets, 100_000, prices, current, managed, universe)
        by = {o.symbol: o for o in orders}
        # AAPL target = 0.30*100000/100 = 300 shares; have 100 -> BUY 200
        self.assertEqual(by["AAPL"].delta, 200)
        self.assertFalse(by["AAPL"].new_entry)
        # MSFT target = 0.20*100000/50 = 400; have 0 -> BUY 400, NEW entry
        self.assertEqual(by["MSFT"].delta, 400)
        self.assertTrue(by["MSFT"].new_entry)
        # TSLA target 0 (not in targets) but held+managed -> SELL all 40
        self.assertEqual(by["TSLA"].delta, -40)
        self.assertEqual(by["TSLA"].side, "SELL")

    def test_unmanaged_held_position_is_walled_off(self):
        targets = {"AAPL": 0.30}
        prices = {"AAPL": 100.0, "GME": 20.0}
        current = {"GME": 500}                  # discovered, unmanaged
        managed = {"AAPL": True, "GME": False}
        orders, skipped = compute_orders(targets, 100_000, prices, current,
                                         managed, ["AAPL"])
        syms = {o.symbol for o in orders}
        self.assertNotIn("GME", syms)           # never touched
        self.assertIn(("GME", "unmanaged_position"), skipped)

    def test_missing_price_skips(self):
        orders, skipped = compute_orders({"AAPL": 0.3}, 100_000, {"AAPL": None},
                                         {}, {"AAPL": True}, ["AAPL"])
        self.assertEqual(orders, [])
        self.assertIn(("AAPL", "no_price"), skipped)

    def test_free_trade_sells_any_held_position(self):
        # free-trade: a discovered/seed position with managed=True (Inventory.is_managed
        # returns True for all in free mode) is fully tradeable -> can be sold to target 0.
        targets = {}                                   # nothing wanted
        prices = {"GME": 20.0}
        current = {"GME": 500}
        managed = {"GME": True}                        # free-trade -> all managed
        orders, skipped = compute_orders(targets, 100_000, prices, current, managed, ["GME"])
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, "SELL")
        self.assertEqual(orders[0].delta, -500)


# --------------------------------------------------------------------------- #
# benchmarks
# --------------------------------------------------------------------------- #
class TestDegradedRunGuard(unittest.TestCase):
    """When most symbols fail (e.g. a Gemini 503 outage -> score 0), run_once must
    HOLD current positions, not rebalance on near-zero information."""

    def _run(self, ok_map, mode="dry"):
        import tempfile
        import live_portfolio as lp
        from strategies.portfolio_base import PortfolioStrategy, PortfolioSignal, SymbolSignal
        from portfolio.risk import PortfolioLimits, KillSwitch
        universe = list(ok_map)

        class StubStrat(PortfolioStrategy):
            name = "stub"
            llm = type("L", (), {"name": "stub"})()
            def evaluate(self, uni, as_of=None):
                sigs = {}
                for s, ok in ok_map.items():
                    sigs[s] = SymbolSignal(s, score=(0.8 if ok else 0.0),
                                           confidence=(0.9 if ok else 0.0), ok=ok)
                return PortfolioSignal(signals=sigs, exposure_multiplier=1.0)

        class MockBroker:
            def equity(self): return 100_000.0
            def list_positions(self):
                return [{"symbol": "AAPL", "qty": 100, "avg_entry_price": 200.0,
                         "current_price": 210.0}]
            def latest_price(self, s): return 210.0 if s == "AAPL" else 300.0
            def submit_market_order(self, *a, **k):
                raise AssertionError("must not trade in a degraded run")

        class MockInv:
            def is_managed(self, s): return True
            def get(self, s): return {"qty": 100 if s == "AAPL" else 0}

        tmp = tempfile.mkdtemp()
        ks = KillSwitch(max_drawdown=0.99,
                        state_file=tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)
        with mock.patch.object(lp.config, "AI_DECISIONS_LOG", f"{tmp}/d.csv"), \
             mock.patch.object(lp.config, "AI_EQUITY_LOG", f"{tmp}/e.csv"), \
             mock.patch.object(lp.config, "AI_BENCH_STATE", f"{tmp}/b.json"):
            return lp.run_once(MockBroker(), StubStrat(), PortfolioLimits.from_config(),
                               ks, universe, mode, inventory=MockInv())

    def test_degraded_holds_no_orders(self):
        # only 1 of 4 symbols usable -> 25% < 50% -> degraded -> hold
        res = self._run({"AAPL": True, "MSFT": False, "NVDA": False, "GOOGL": False})
        self.assertEqual(res["orders"], [])                       # no rebalance
        self.assertAlmostEqual(res["targets"].get("AAPL", 0), 100 * 210 / 100_000, places=4)

    def test_healthy_run_rebalances(self):
        # 3 of 4 usable -> 75% >= 50% -> normal path produces a real target book
        res = self._run({"AAPL": True, "MSFT": True, "NVDA": True, "GOOGL": False})
        self.assertTrue(len(res["targets"]) >= 1)


class TestBenchmarks(unittest.TestCase):
    def test_equal_weight_benchmark(self):
        start = {"A": 100.0, "B": 50.0}
        cur = {"A": 110.0, "B": 55.0}           # both +10%
        self.assertAlmostEqual(equal_weight_benchmark(100_000, start, cur), 110_000.0)
        cur2 = {"A": 120.0, "B": 50.0}          # +20% and 0% -> +10% avg
        self.assertAlmostEqual(equal_weight_benchmark(100_000, start, cur2), 110_000.0)

    def test_spy_benchmark(self):
        self.assertAlmostEqual(spy_benchmark(100_000, 400.0, 440.0), 110_000.0)
        self.assertEqual(spy_benchmark(100_000, None, 440.0), 100_000.0)  # no start -> flat

    def test_turnover(self):
        from live_portfolio import Order
        orders = [Order("A", 300, 100, 200, "BUY", 0.3),
                  Order("B", 0, 40, -40, "SELL", 0.0)]
        prices = {"A": 100.0, "B": 50.0}
        # traded notional = 200*100 + 40*50 = 22000; / 100000 = 0.22
        self.assertAlmostEqual(portfolio_turnover(orders, 100_000, prices), 0.22)


if __name__ == "__main__":
    unittest.main(verbosity=2)
