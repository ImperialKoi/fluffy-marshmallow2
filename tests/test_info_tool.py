"""
Tests for the real-time information tool (tools/info/).

Unit tests use NO live network: source HTTP is mocked, and retriever-level tests
inject stub sources. Run them with:

    python tests/test_info_tool.py
    # or: python -m unittest tests.test_info_tool

A single LIVE smoke test fetches real AAPL data across all three types and is
gated behind an env flag so it never runs in CI by accident:

    RUN_LIVE_INFO_TESTS=1 python tests/test_info_tool.py
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.info.schema import Item, NEWS, FILING, ANALYST
from tools.info.store import Cache
from tools.info import retriever
from tools.info.sources.base import Source


def _dt(*args):
    return datetime(*args, tzinfo=timezone.utc)


class StubSource(Source):
    """A source whose output is fixed up front (or which raises on demand)."""
    item_types = (NEWS, FILING, ANALYST)

    def __init__(self, name, items=None, raises=False):
        self.name = name
        self._items = items or []
        self._raises = raises

    def fetch(self, ticker, limit=50):
        if self._raises:
            raise RuntimeError(f"{self.name} boom")
        return list(self._items)


def _fresh_cache():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return Cache(path=tmp.name, ttl_seconds=600)


def _patch_sources(stubs):
    """Patch retriever's source builder to return the given stubs."""
    return mock.patch.object(retriever.sources_mod, "build",
                             side_effect=lambda names=None: list(stubs))


class TestDedup(unittest.TestCase):
    def test_dedupe_by_url_and_fuzzy_headline(self):
        items = [
            Item("a", ["AAPL"], _dt(2024, 1, 3), "rss/yahoo", NEWS,
                 "Apple unveils new iPhone today", url="https://ex.com/x?utm=1"),
            # exact-ish URL dup (query string differs) -> dropped
            Item("b", ["AAPL"], _dt(2024, 1, 2), "rss/google", NEWS,
                 "Totally different headline", url="https://ex.com/x?utm=2"),
            # fuzzy headline near-dup, different URL -> dropped
            Item("c", ["AAPL"], _dt(2024, 1, 1), "alpaca/benzinga", NEWS,
                 "Apple unveils a new iPhone today!", url="https://other.com/y"),
            # genuinely distinct -> kept
            Item("d", ["AAPL"], _dt(2024, 1, 4), "rss/nasdaq", NEWS,
                 "Microsoft raises dividend", url="https://ex.com/z"),
        ]
        stubs = [StubSource("s", items)]
        with _patch_sources(stubs):
            out = retriever.get_info("AAPL", cache=_fresh_cache())
        headlines = [it.headline for it in out]
        # 4 in -> 2 out (the Apple cluster collapses to one; MSFT distinct)
        self.assertEqual(len(out), 2, headlines)
        self.assertIn("Microsoft raises dividend", headlines)
        # surviving Apple item is the newest of its cluster (the url-dup one at 1/3)
        apple = [it for it in out if "Apple" in it.headline][0]
        self.assertEqual(apple.published_utc, _dt(2024, 1, 3))

    def test_sorted_newest_first(self):
        items = [
            Item("1", ["AAPL"], _dt(2024, 1, 1), "s", NEWS, "old", url="u1"),
            Item("2", ["AAPL"], _dt(2024, 3, 1), "s", NEWS, "new", url="u2"),
            Item("3", ["AAPL"], None, "s", ANALYST, "undated snapshot", url="u3"),
        ]
        with _patch_sources([StubSource("s", items)]):
            out = retriever.get_info("AAPL", cache=_fresh_cache())
        self.assertEqual([it.headline for it in out], ["new", "old", "undated snapshot"])


class TestAsOf(unittest.TestCase):
    def test_as_of_excludes_future_keeps_undated(self):
        items = [
            Item("past", ["AAPL"], _dt(2024, 1, 1), "s", NEWS, "past", url="u1"),
            Item("future", ["AAPL"], _dt(2024, 6, 1), "s", NEWS, "future", url="u2"),
            Item("undated", ["AAPL"], None, "s", ANALYST, "undated", url="u3"),
        ]
        as_of = _dt(2024, 3, 1)
        with _patch_sources([StubSource("s", items)]):
            out = retriever.get_info("AAPL", as_of=as_of, cache=_fresh_cache())
        names = {it.headline for it in out}
        self.assertIn("past", names)
        self.assertIn("undated", names)        # can't prove it's in the future -> kept
        self.assertNotIn("future", names)      # published after as_of -> dropped

    def test_as_of_accepts_iso_string(self):
        items = [Item("f", ["AAPL"], _dt(2024, 6, 1), "s", NEWS, "future", url="u")]
        with _patch_sources([StubSource("s", items)]):
            out = retriever.get_info("AAPL", as_of="2024-03-01T00:00:00Z",
                                     cache=_fresh_cache())
        self.assertEqual(out, [])


class TestGracefulFailure(unittest.TestCase):
    def test_one_source_raises_others_survive(self):
        good = StubSource("good", [
            Item("g", ["AAPL"], _dt(2024, 1, 1), "good", NEWS, "good item", url="u1")])
        bad = StubSource("bad", raises=True)
        with _patch_sources([bad, good]):
            out = retriever.get_info("AAPL", cache=_fresh_cache())  # must NOT raise
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].headline, "good item")

    def test_all_sources_failing_returns_empty(self):
        with _patch_sources([StubSource("b1", raises=True), StubSource("b2", raises=True)]):
            out = retriever.get_info("AAPL", cache=_fresh_cache())
        self.assertEqual(out, [])


class TestTypeFilter(unittest.TestCase):
    def test_types_filter(self):
        items = [
            Item("n", ["AAPL"], _dt(2024, 1, 1), "s", NEWS, "news", url="u1"),
            Item("f", ["AAPL"], _dt(2024, 1, 2), "s", FILING, "filing", url="u2"),
            Item("a", ["AAPL"], _dt(2024, 1, 3), "s", ANALYST, "analyst", url="u3"),
        ]
        with _patch_sources([StubSource("s", items)]):
            out = retriever.get_info("AAPL", types=["filing"], cache=_fresh_cache())
        self.assertEqual([it.item_type for it in out], [FILING])

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            retriever.get_info("AAPL", types=["bogus"], cache=_fresh_cache())


class TestCache(unittest.TestCase):
    def test_put_get_and_ttl_expiry(self):
        cache = _fresh_cache()
        items = [Item("x", ["AAPL"], _dt(2024, 1, 1), "s", NEWS, "h", url="u")]
        cache.put("s", "AAPL", items)
        got = cache.get("s", "AAPL")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].headline, "h")
        # force expiry
        cache.ttl = -1
        self.assertIsNone(cache.get("s", "AAPL"))

    def test_clear(self):
        cache = _fresh_cache()
        cache.put("s", "AAPL", [Item("x", ["AAPL"], None, "s", NEWS, "h", url="u")])
        self.assertEqual(cache.clear(), 1)
        self.assertIsNone(cache.get("s", "AAPL"))

    def test_cache_round_trip_in_retriever(self):
        cache = _fresh_cache()
        calls = {"n": 0}

        class Counting(StubSource):
            def fetch(self, ticker, limit=50):
                calls["n"] += 1
                return [Item("x", ["AAPL"], _dt(2024, 1, 1), "c", NEWS, "h", url="u")]

        stub = Counting("c")
        with _patch_sources([stub]):
            retriever.get_info("AAPL", cache=cache)            # fetches
            retriever.get_info("AAPL", cache=cache)            # served from cache
            retriever.get_info("AAPL", cache=cache, use_cache=False)  # forced refetch
        self.assertEqual(calls["n"], 2)


class TestRSSAdapterMocked(unittest.TestCase):
    SAMPLE = """<?xml version="1.0"?>
    <rss version="2.0"><channel>
      <item>
        <title>Apple hits record high</title>
        <link>https://news.example.com/apple-record</link>
        <description>Shares jumped.</description>
        <guid>guid-1</guid>
        <pubDate>Mon, 15 Jan 2024 14:30:00 GMT</pubDate>
      </item>
    </channel></rss>"""

    def test_rss_parses_mocked_http(self):
        from tools.info.sources.rss_news import RSSNews
        src = RSSNews(feeds={"yahoo": "https://feeds.example/{t}"})
        with mock.patch("tools.info.sources.rss_news.http.get_text",
                        return_value=self.SAMPLE):
            items = src.fetch("AAPL", limit=10)
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertEqual(it.item_type, NEWS)
        self.assertEqual(it.headline, "Apple hits record high")
        self.assertEqual(it.url, "https://news.example.com/apple-record")
        self.assertEqual(it.published_utc, _dt(2024, 1, 15, 14, 30, 0))

    def test_rss_feed_failure_is_isolated(self):
        from tools.info.sources.rss_news import RSSNews
        src = RSSNews(feeds={"a": "https://x/{t}", "b": "https://y/{t}"})

        def fake_get_text(url, **kw):
            if "x/" in url:
                raise RuntimeError("feed a down")
            return self.SAMPLE

        with mock.patch("tools.info.sources.rss_news.http.get_text",
                        side_effect=fake_get_text):
            items = src.fetch("AAPL")
        self.assertEqual(len(items), 1)  # only feed b survives


class TestEdgarAdapterMocked(unittest.TestCase):
    TICKER_MAP = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    SUBMISSIONS = {
        "filings": {"recent": {
            "form": ["8-K", "10-Q", "DEF 14A", "4"],
            "accessionNumber": ["0000320193-24-000001", "0000320193-24-000002",
                                "0000320193-24-000003", "0000320193-24-000004"],
            "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.xml"],
            "primaryDocDescription": ["8-K", "10-Q", "Proxy", "Form 4"],
            "filingDate": ["2024-01-10", "2024-01-05", "2024-01-04", "2024-01-03"],
            "acceptanceDateTime": ["2024-01-10T16:30:00.000Z", "2024-01-05T16:00:00.000Z",
                                   "2024-01-04T16:00:00.000Z", "2024-01-03T18:00:00.000Z"],
            "items": ["2.02,9.01", "", "", ""],
        }}
    }

    def test_edgar_parses_and_filters_forms(self):
        from tools.info.sources import edgar_filings
        edgar_filings._TICKER_CIK = None  # reset process cache

        def fake_get_json(url, **kw):
            if "company_tickers" in url:
                return self.TICKER_MAP
            if "submissions" in url:
                return self.SUBMISSIONS
            raise AssertionError(f"unexpected url {url}")

        src = edgar_filings.EdgarFilings()
        with mock.patch("tools.info.sources.edgar_filings.http.get_json",
                        side_effect=fake_get_json):
            items = src.fetch("AAPL", limit=50)

        forms = [it.extra["form"] for it in items]
        self.assertEqual(forms, ["8-K", "10-Q", "4"])      # DEF 14A excluded
        eightk = items[0]
        self.assertEqual(eightk.item_type, FILING)
        self.assertEqual(eightk.extra["cik"], 320193)
        self.assertEqual(eightk.extra["accession"], "0000320193-24-000001")
        self.assertIn("0000320193", eightk.url)            # CIK in archive URL
        self.assertEqual(eightk.published_utc, _dt(2024, 1, 10, 16, 30, 0))
        self.assertTrue(items[2].extra["is_insider"])      # Form 4 flagged

    def test_edgar_unknown_ticker_returns_empty(self):
        from tools.info.sources import edgar_filings
        edgar_filings._TICKER_CIK = None
        with mock.patch("tools.info.sources.edgar_filings.http.get_json",
                        return_value=self.TICKER_MAP):
            items = edgar_filings.EdgarFilings().fetch("ZZZZ")
        self.assertEqual(items, [])


@unittest.skipUnless(os.environ.get("RUN_LIVE_INFO_TESTS"),
                     "set RUN_LIVE_INFO_TESTS=1 to run the live network smoke test")
class TestLiveSmoke(unittest.TestCase):
    def test_live_aapl_all_types(self):
        from tools.info import get_info
        items = get_info("AAPL", types=["news", "filing", "analyst"], limit=40,
                         use_cache=False)
        self.assertIsInstance(items, list)
        by_type = {}
        for it in items:
            by_type.setdefault(it.item_type, []).append(it)
        print("\n--- LIVE AAPL sample ---")
        for t in ("news", "filing", "analyst"):
            sample = by_type.get(t, [])
            print(f"\n[{t}] {len(sample)} items")
            for it in sample[:3]:
                print(f"  {it.published_utc}  {it.source}: {it.headline[:90]}")
        # At minimum the call must succeed and return a list; we don't hard-assert
        # per-type counts because free sources can be flaky / key-gated.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
