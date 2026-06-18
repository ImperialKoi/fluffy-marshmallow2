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
        # circuit breaker: once Gemini rate-limits/overloads, skip it until this time
        # so the rest of the basket fails over to the chain instead of re-hammering it.
        self._cooldown_until = 0.0
        self.retry_backoff = 1.0   # short in-call backoff -> hand off to the chain fast

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
        # circuit breaker: if Gemini recently rate-limited/overloaded, don't even call it
        # — return a 429-flavored error so the FallbackLLM advances straight to Cohere.
        remaining = self._cooldown_until - time.monotonic()
        if remaining > 0:
            return LLMResult(parsed=None, raw="",
                             error=f"429 RESOURCE_EXHAUSTED: Gemini in cooldown "
                                   f"~{int(remaining)}s (skipping, using fallback)")

        last_err, el, capacity, rate_limited = "", "", False, False
        for attempt in range(self.retries):              # at most `retries` (3) tries
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
                el = last_err.upper()
                rate_limited = "429" in el or "RESOURCE_EXHAUSTED" in el
                capacity = rate_limited or any(k in el for k in (
                    "503", "500", "UNAVAILABLE", "OVERLOADED", "SERVERERROR"))
                if attempt < self.retries - 1:
                    time.sleep(self.retry_backoff)        # short -> 3 tries finish fast
        # After 3 tries: if it was a quota/overload (429 OR 503), open the circuit breaker
        # so the rest of the basket fails over to Cohere instead of re-hammering Gemini.
        if capacity:
            self._open_cooldown(el, rate_limited)
        return LLMResult(parsed=None, raw="", error=last_err)

    def _open_cooldown(self, err_upper: str, rate_limited: bool):
        """Skip Gemini for a while after a quota/capacity failure (daily quota = long)."""
        if any(k in err_upper for k in ("PERDAY", "PER_DAY", "REQUESTSPERDAY")):
            secs = 1800          # free-tier DAILY quota exhausted -> back off 30 min
        elif rate_limited:
            secs = 120           # per-minute rate limit
        else:
            secs = 60            # 5xx capacity spike
        self._cooldown_until = time.monotonic() + secs
        log.warning("Gemini circuit breaker OPEN for %ds -> failing over to the chain "
                    "(Cohere/OpenAI) for subsequent calls", secs)


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
# OpenAI (secondary / fallback)
# --------------------------------------------------------------------------- #
class OpenAILLM(LLM):
    """ChatGPT via the OpenAI SDK, structured JSON output. Key from OPENAI_API_KEY."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None,
                 retries: int = 3, timeout: int = 30, temperature: float = 0.0):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY not set (needed for OpenAILLM).")
        self.retries = retries
        self.timeout = timeout
        self.temperature = temperature
        self.name = f"openai:{model}"
        from openai import OpenAI  # lazy import
        self._client = OpenAI(api_key=self.api_key, timeout=timeout)

    def complete_json(self, prompt: str) -> LLMResult:
        last_err = ""
        for attempt in range(self.retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}])
                raw = resp.choices[0].message.content or ""
                parsed = extract_json(raw)
                if parsed is None:
                    last_err = "unparseable JSON"
                    raise ValueError(last_err)
                return LLMResult(parsed=parsed, raw=raw)
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                log.warning("OpenAI attempt %d/%d failed: %s", attempt + 1, self.retries, last_err)
                if attempt < self.retries - 1:
                    time.sleep(1.5 ** attempt)
        return LLMResult(parsed=None, raw="", error=last_err)


# --------------------------------------------------------------------------- #
# Cohere (secondary)
# --------------------------------------------------------------------------- #
class CohereLLM(LLM):
    """Cohere Command via the cohere SDK. Key from COHERE_API_KEY. We rely on the
    prompt's 'STRICT JSON only' instruction + extract_json (rather than a
    version-specific response_format) for maximum SDK compatibility (v2 and v1)."""

    def __init__(self, model: str = "command-r", api_key: str = None,
                 retries: int = 3, timeout: int = 30, temperature: float = 0.0):
        self.model = model
        self.api_key = (api_key or os.environ.get("COHERE_API_KEY")
                        or os.environ.get("CO_API_KEY"))
        if not self.api_key:
            raise EnvironmentError("COHERE_API_KEY not set (needed for CohereLLM).")
        self.retries = retries
        self.timeout = timeout
        self.temperature = temperature
        self.name = f"cohere:{model}"
        import cohere  # lazy import
        self._v2 = hasattr(cohere, "ClientV2")
        self._client = cohere.ClientV2(self.api_key) if self._v2 else cohere.Client(self.api_key)

    def _call(self, prompt: str) -> str:
        if self._v2:
            resp = self._client.chat(
                model=self.model, temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}])
            msg = getattr(resp, "message", None)
            content = getattr(msg, "content", None) if msg else None
            if content:
                return "".join(getattr(p, "text", "") or "" for p in content)
            return getattr(resp, "text", "") or ""
        resp = self._client.chat(message=prompt, model=self.model,
                                 temperature=self.temperature)
        return getattr(resp, "text", "") or ""

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
                log.warning("Cohere attempt %d/%d failed: %s", attempt + 1, self.retries, last_err)
                if attempt < self.retries - 1:
                    el = last_err.upper()
                    transient = any(k in el for k in (
                        "429", "503", "500", "UNAVAILABLE", "OVERLOADED", "TIMEOUT"))
                    time.sleep(min(self.timeout, 5.0 * (2 ** attempt)) if transient else 1.5 ** attempt)
        return LLMResult(parsed=None, raw="", error=last_err)


# --------------------------------------------------------------------------- #
# Fallback chain
# --------------------------------------------------------------------------- #
# Errors meaning "couldn't reach / use this provider right now" -> advance to the next.
# Includes 404/NOTFOUND so a misconfigured or retired model on one provider rolls over
# to the next instead of dead-ending the chain.
_FALLBACK_TRIGGERS = ("503", "500", "UNAVAILABLE", "OVERLOADED", "SERVERERROR",
                      "429", "RESOURCE_EXHAUSTED", "TIMEOUT", "CONNECTION",
                      "APICONNECTION", "TEMPORARILY", "404", "NOTFOUND")


class FallbackLLM(LLM):
    """Try a CHAIN of providers in order. If one returns a usable result, use it. If one
    is unreachable/overloaded (e.g. Gemini 503 after its 3 internal retries), advance to
    the next provider; a non-capacity failure (e.g. unparseable output) stops the chain.

    Default chain here is Gemini -> Cohere -> OpenAI."""

    def __init__(self, *llms: LLM):
        self.llms = [l for l in llms if l is not None]
        self.name = " -> ".join(l.name for l in self.llms)

    def complete_json(self, prompt: str) -> LLMResult:
        last = LLMResult(parsed=None, error="no LLMs configured")
        for i, llm in enumerate(self.llms):
            last = llm.complete_json(prompt)
            if last.parsed is not None:
                return last
            err = (last.error or "").upper()
            unreachable = any(t in err for t in _FALLBACK_TRIGGERS)
            if not unreachable:
                return last       # content/parse failure -> don't burn the next provider
            if i < len(self.llms) - 1:
                log.warning("LLM '%s' unavailable (%s) -> trying '%s'",
                            llm.name, last.error, self.llms[i + 1].name)
        return last


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def build_llm(provider: str = "gemini", model: str = "gemini-3.5-flash",
              allow_stub_fallback: bool = True, **kwargs) -> LLM:
    """Build the LLM. For provider='gemini', assemble a fallback CHAIN ordered
    Gemini -> Cohere -> OpenAI, including only the providers whose key/SDK are
    available. If a provider is unreachable/overloaded at call time, the chain rolls
    over to the next. If none are available, use the offline StubLLM."""
    def _cfg(name, default):
        try:
            import config
            return getattr(config, name, default)
        except Exception:  # noqa: BLE001
            return default

    if provider == "stub":
        return StubLLM()
    if provider == "openai":
        return OpenAILLM(model=_cfg("AI_OPENAI_MODEL", "gpt-4o-mini"), **kwargs)
    if provider == "cohere":
        return CohereLLM(model=_cfg("AI_COHERE_MODEL", "command-r"), **kwargs)
    if provider == "gemini":
        chain = []
        # 1st choice: Gemini
        try:
            chain.append(GeminiLLM(model=model, **kwargs))
        except Exception as e:  # noqa: BLE001
            log.warning("Gemini unavailable (%s); will use fallbacks if configured", e)
        # 2nd choice: Cohere (if a key is present)
        if os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY"):
            try:
                chain.append(CohereLLM(model=_cfg("AI_COHERE_MODEL", "command-r")))
            except Exception as e:  # noqa: BLE001
                log.warning("Cohere unavailable (%s); skipping it in the chain", e)
        # 3rd choice: OpenAI (if a key is present)
        if os.environ.get("OPENAI_API_KEY"):
            try:
                chain.append(OpenAILLM(model=_cfg("AI_OPENAI_MODEL", "gpt-4o-mini")))
            except Exception as e:  # noqa: BLE001
                log.warning("OpenAI unavailable (%s); skipping it in the chain", e)

        if not chain:
            if not allow_stub_fallback:
                raise EnvironmentError("No LLM available (set GEMINI/COHERE/OPENAI key).")
            log.warning("No real LLM available -> OFFLINE StubLLM (NOT a real model). "
                        "Set GEMINI_API_KEY (and optionally COHERE_API_KEY/OPENAI_API_KEY).")
            return StubLLM()
        if len(chain) == 1:
            return chain[0]
        chained = FallbackLLM(*chain)
        log.info("LLM fallback chain: %s", chained.name)
        return chained
    raise ValueError(f"Unknown LLM provider '{provider}'")
