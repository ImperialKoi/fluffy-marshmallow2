"""
SEC EDGAR filings adapter (free, official, keyless).

Uses the official EDGAR JSON APIs (no key required):
  * ticker -> CIK map : https://www.sec.gov/files/company_tickers.json   (cached in-process)
  * recent filings    : https://data.sec.gov/submissions/CIK##########.json

Returns recent 8-K, 10-K, 10-Q and Form 4 (insider) filings by default, newest first.

IMPORTANT — User-Agent: the SEC BLOCKS requests without a descriptive User-Agent
that includes a contact email (format: "Sample Company AdminContact@example.com").
Set it via the SEC_EDGAR_USER_AGENT env var; if it's missing we fall back to a
generic UA and log a warning (the SEC may rate-limit/deny generic UAs).

Limits/licensing: max ~10 requests/second per IP (we stay well under and cache);
EDGAR data is public domain. Ref: https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import logging
import os
from datetime import timezone

from .base import Source
from .. import http
from ..schema import Item, FILING, make_id, to_utc

log = logging.getLogger("info_tool.edgar")

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
DEFAULT_FORMS = ("8-K", "10-K", "10-Q", "4")

_GENERIC_UA = "trading-bot-info/1.0 (set SEC_EDGAR_USER_AGENT with your contact email)"

# process-wide ticker->CIK cache (the map is ~10k entries, fetched once)
_TICKER_CIK: dict | None = None


class EdgarFilings(Source):
    name = "edgar_filings"
    item_types = (FILING,)
    default_on = True

    def __init__(self, forms=DEFAULT_FORMS):
        self.forms = tuple(forms)

    def _ua(self) -> str:
        ua = os.environ.get("SEC_EDGAR_USER_AGENT")
        if not ua:
            log.warning("SEC_EDGAR_USER_AGENT not set; using a generic UA. EDGAR may "
                        "rate-limit or block. Set it to 'Your Name your@email.com'.")
            return _GENERIC_UA
        return ua

    def _headers(self) -> dict:
        return {"User-Agent": self._ua(), "Accept-Encoding": "gzip, deflate"}

    def _cik_for(self, ticker: str):
        global _TICKER_CIK
        if _TICKER_CIK is None:
            data = http.get_json(TICKER_MAP_URL, headers=self._headers())
            # data is {"0": {"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."}, ...}
            _TICKER_CIK = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
        return _TICKER_CIK.get(ticker.upper())

    def fetch(self, ticker: str, limit: int = 50) -> list[Item]:
        cik = self._cik_for(ticker)
        if cik is None:
            log.info("EDGAR: no CIK for ticker %s", ticker)
            return []
        cik10 = f"{cik:010d}"
        data = http.get_json(SUBMISSIONS_URL.format(cik10=cik10), headers=self._headers())

        recent = (data.get("filings") or {}).get("recent") or {}
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        primary_desc = recent.get("primaryDocDescription", [])
        filing_dates = recent.get("filingDate", [])
        accept_dts = recent.get("acceptanceDateTime", [])
        items_field = recent.get("items", [])

        want = set(self.forms)
        out: list[Item] = []
        for i in range(len(forms)):
            form = forms[i]
            base = form.split("/")[0]  # treat "10-K/A" like "10-K" for matching
            if not (form in want or base in want):
                continue
            accession = accessions[i] if i < len(accessions) else ""
            acc_nodash = accession.replace("-", "")
            doc = primary_docs[i] if i < len(primary_docs) else ""
            url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
                   if acc_nodash and doc else
                   f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik10}")
            published = to_utc(accept_dts[i]) if i < len(accept_dts) and accept_dts[i] \
                else to_utc(filing_dates[i] if i < len(filing_dates) else None)
            desc = primary_desc[i] if i < len(primary_desc) else ""
            ev = items_field[i] if i < len(items_field) else ""
            headline = f"{ticker.upper()} {form}" + (f" — {desc}" if desc else "")
            out.append(Item(
                id=make_id(self.name, native_id=accession or None, url=url),
                tickers=[ticker.upper()],
                published_utc=published,
                source="sec_edgar",
                item_type=FILING,
                headline=headline,
                summary=(f"Form {form} filed with the SEC."
                         + (f" Items: {ev}." if ev else "")),
                url=url,
                extra={
                    "form": form,
                    "cik": cik,
                    "accession": accession,
                    "filing_date": filing_dates[i] if i < len(filing_dates) else None,
                    "primary_doc_description": desc,
                    "event_items": ev,
                    "is_insider": base == "4",
                },
            ))
            if len(out) >= limit:
                break
        return out
