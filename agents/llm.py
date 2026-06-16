"""
Thin LLM provider wrapper (Gemini), behind a small interface so the provider can be
swapped without touching the strategy.

Design notes / verified SDK usage
---------------------------------
Google ships TWO Python SDKs:
  * NEW (recommended): `google-genai` -> `from google import genai`;
       client = genai.Client(api_key=...);
       client.models.generate_content(model=..., contents=..., config=...)
       structured JSON via config response_mime_type="application/json".
  * LEGACY: `google-generativeai` -> `import google.generativeai as genai`;
       genai.configure(api_key=...); GenerativeModel(model).generate_content(...)
       structured JSON via generation_config={"response_mime_type":"application/json"}.

GeminiLLM prefers the new SDK and transparently falls back to the legacy one if
that's what's installed. The API key is read from GEMINI_API_KEY (never hardcoded).

`StubLLM` is an OFFLINE, deterministic fallback (a tiny lexical sentiment model). It
is clearly NOT the real model — it exists so the pipeline and tests run without a key
or network. The harness uses it only when no GEMINI_API_KEY is set, and labels it.

Every call returns an `LLMResult(parsed, raw, error)`; the caller audits raw+parsed.
A failed/garbled call yields error set and parsed=None — it never raises to the run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("agents.llm")


@dataclass
class LLMResult:
    parsed: Optional[dict]
    raw: str = ""
    error: str = ""


def extract_json(text: str) -> Optional[dict]:
    """Best-effort: parse a JSON object from model text (handles ```json fences)."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # fall back to the first {...} block
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class LLM(ABC):
    name: str = "llm"

    @abstractmethod
    def complete_json(self, prompt: str) -> LLMResult:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
class GeminiLLM(LLM):
    def __init__(self, model: str = "gemini-3.5-flash", api_key: str = None,
                 retries: int = 3, timeout: int = 30, temperature: float = 0.0):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GEMINI_API_KEY not set (needed for GeminiLLM).")
        self.retries = retries
        self.timeout = timeout
        self.temperature = temperature
        self.name = f"gemini:{model}"
        self._sdk, self._handle = self._init_sdk()

    def _init_sdk(self):
        # prefer the new unified SDK
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            return "genai", client
        except Exception:  # noqa: BLE001
            pass
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            return "legacy", genai.GenerativeModel(self.model)
        except Exception as e:  # noqa: BLE001
            raise EnvironmentError(
                "No usable Gemini SDK. Install `google-genai` (recommended) or "
                f"`google-generativeai`. Underlying error: {e}")

    def _call(self, prompt: str) -> str:
        if self._sdk == "genai":
            from google.genai import types
            cfg = types.GenerateContentConfig(
                response_mime_type="application/json", temperature=self.temperature)
            resp = self._handle.models.generate_content(
                model=self.model, contents=prompt, config=cfg)
            return resp.text or ""
        else:  # legacy
            resp = self._handle.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json",
                                   "temperature": self.temperature},
                request_options={"timeout": self.timeout})
            return resp.text or ""

    def complete_json(self, prompt: str) -> LLMResult:
        last_err = ""
        for attempt in range(self.retries):
            try:
                raw = self._call(prompt)
                parsed = extract_json(raw)
                if parsed is None:
                    last_err = "unparseable JSON"
                    raise ValueError(last_err)
                return LLMResult(parsed=parsed, raw=raw)
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                log.warning("Gemini attempt %d/%d failed: %s",
                            attempt + 1, self.retries, last_err)
                if attempt < self.retries - 1:
                    # rate-limit (429 / RESOURCE_EXHAUSTED) needs a much longer wait
                    # than a transient error; the per-minute window is ~60s.
                    rate_limited = ("429" in last_err or "RESOURCE_EXHAUSTED" in last_err)
                    time.sleep(min(self.timeout, 20.0) if rate_limited else 1.5 ** attempt)
        return LLMResult(parsed=None, raw="", error=last_err)


# --------------------------------------------------------------------------- #
# Offline stub (deterministic; NOT the real model)
# --------------------------------------------------------------------------- #
_BULL = ("beat", "beats", "surge", "surges", "record", "upgrade", "upgraded",
         "outperform", "buy", "raises", "raised", "jumps", "soars", "strong",
         "growth", "wins", "approval", "rally", "tops", "bullish", "expands")
_BEAR = ("miss", "misses", "downgrade", "downgraded", "cut", "cuts", "plunge",
         "plunges", "lawsuit", "probe", "recall", "warning", "weak", "falls",
         "drops", "slumps", "sell", "underperform", "bearish", "halts", "decline")


class StubLLM(LLM):
    """Offline lexical sentiment over the prompt text. Deterministic. For tests and
    for running the pipeline without a GEMINI_API_KEY. NOT a real LLM."""

    name = "stub:lexical"

    def complete_json(self, prompt: str) -> LLMResult:
        low = prompt.lower()
        ticker = ""
        m = re.search(r"ticker[\"']?\s*[:=]\s*[\"']?([A-Z]{1,5})", prompt)
        if m:
            ticker = m.group(1)
        bull = sum(low.count(w) for w in _BULL)
        bear = sum(low.count(w) for w in _BEAR)
        total = bull + bear
        if total == 0:
            score, conf = 0.0, 0.1
        else:
            score = (bull - bear) / total
            conf = min(1.0, 0.2 + 0.1 * total)
        parsed = {"ticker": ticker, "score": round(score, 3),
                  "confidence": round(conf, 3),
                  "rationale": f"stub lexical: {bull} bullish / {bear} bearish cues"}
        return LLMResult(parsed=parsed, raw=json.dumps(parsed))


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def build_llm(provider: str = "gemini", model: str = "gemini-3.5-flash",
              allow_stub_fallback: bool = True, **kwargs) -> LLM:
    """Build an LLM. If provider='gemini' but no key/SDK is available and
    allow_stub_fallback is True, return StubLLM with a loud warning."""
    if provider == "stub":
        return StubLLM()
    if provider == "gemini":
        try:
            return GeminiLLM(model=model, **kwargs)
        except Exception as e:  # noqa: BLE001
            if not allow_stub_fallback:
                raise
            log.warning("Gemini unavailable (%s) -> falling back to OFFLINE StubLLM "
                        "(NOT the real model). Set GEMINI_API_KEY for real scores.", e)
            return StubLLM()
    raise ValueError(f"Unknown LLM provider '{provider}'")
