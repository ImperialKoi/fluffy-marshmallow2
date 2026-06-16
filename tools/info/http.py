"""
Shared HTTP helper: one pooled session, sane timeouts, and exponential backoff
with retry on transient errors (timeouts, connection errors, 429, 5xx).

Adapters use this instead of calling requests directly so that resilience policy
(timeouts/backoff) lives in one place.
"""

from __future__ import annotations

import logging
import random
import time

log = logging.getLogger("info_tool.http")

DEFAULT_TIMEOUT = 15
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.6

_session = None


class TransientHTTPError(Exception):
    """A retryable HTTP response (429 / 5xx)."""


def _get_session():
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
        _session.headers.update({"Accept": "*/*"})
    return _session


def get(url, *, headers=None, params=None, timeout=DEFAULT_TIMEOUT,
        retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF):
    """GET with retries/backoff. Returns a requests.Response or raises the last error."""
    import requests

    session = _get_session()
    last_err = None
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                raise TransientHTTPError(f"{resp.status_code} for {url}")
            resp.raise_for_status()
            return resp
        except (requests.RequestException, TransientHTTPError) as e:
            last_err = e
            if attempt == retries - 1:
                break
            sleep = backoff ** attempt + random.uniform(0, 0.4)
            log.warning("HTTP attempt %d/%d failed for %s (%s); retrying in %.1fs",
                        attempt + 1, retries, url, e, sleep)
            time.sleep(sleep)
    raise last_err


def get_json(url, **kwargs):
    return get(url, **kwargs).json()


def get_text(url, **kwargs):
    return get(url, **kwargs).text
