"""
Secret loading from AWS SSM Parameter Store (for the EC2 deployment).

On the box, secrets live in SSM as SecureString parameters under a path prefix
(default `/tradingbot/`) and are pulled into the process environment at startup via
the EC2 instance's **IAM role** (boto3 picks up role credentials automatically) —
there are NO static AWS keys and NO plaintext .env on the instance.

Locally this is a NO-OP unless explicitly enabled, so development never needs AWS:
`maybe_load_ssm()` only calls SSM when `TRADINGBOT_USE_SSM` is truthy or
`TRADINGBOT_SSM_PREFIX` is set. boto3 is imported lazily (only needed on the box).

Parameter naming: the leaf name of each parameter maps to an env var. The canonical
set is below; any other `/tradingbot/<NAME>` parameter is loaded as env var `<NAME>`.
By default we do NOT overwrite a variable already present in the environment.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("service.secrets")

DEFAULT_PREFIX = "/tradingbot/"

# SSM leaf name -> environment variable (identity by default; listed for documentation)
SSM_TO_ENV = {
    "ALPACA_KEY": "ALPACA_KEY",
    "ALPACA_SECRET": "ALPACA_SECRET",
    "ALPACA_LIVE_KEY": "ALPACA_LIVE_KEY",
    "ALPACA_LIVE_SECRET": "ALPACA_LIVE_SECRET",
    "GEMINI_API_KEY": "GEMINI_API_KEY",
    "SEC_EDGAR_USER_AGENT": "SEC_EDGAR_USER_AGENT",
}


def load_secrets_from_ssm(prefix: str = None, region: str = None, client=None,
                          overwrite: bool = False) -> list[str]:
    """Load every SecureString under `prefix` into os.environ. Returns the env var
    names that were set. Raises on AWS/permission errors (caller decides whether fatal)."""
    prefix = prefix or os.environ.get("TRADINGBOT_SSM_PREFIX", DEFAULT_PREFIX)
    if not prefix.endswith("/"):
        prefix += "/"
    region = (region or os.environ.get("AWS_REGION")
              or os.environ.get("AWS_DEFAULT_REGION"))

    if client is None:
        import boto3  # lazy: only present/needed on the EC2 box
        client = boto3.client("ssm", region_name=region)

    loaded = []
    paginator = client.get_paginator("get_parameters_by_path")
    for page in paginator.paginate(Path=prefix, WithDecryption=True, Recursive=True):
        for p in page.get("Parameters", []):
            leaf = p["Name"].rsplit("/", 1)[-1]
            env_var = SSM_TO_ENV.get(leaf, leaf)
            if overwrite or not os.environ.get(env_var):
                os.environ[env_var] = p["Value"]
                loaded.append(env_var)
    return loaded


def maybe_load_ssm() -> list[str]:
    """Load SSM secrets IF configured (env-gated); otherwise a quiet no-op. Never raises
    — a failure here is logged and the app proceeds (and will fail clearly later if a
    required key is genuinely missing)."""
    enabled = (os.environ.get("TRADINGBOT_USE_SSM", "").lower() in ("1", "true", "yes")
               or bool(os.environ.get("TRADINGBOT_SSM_PREFIX")))
    if not enabled:
        return []
    try:
        loaded = load_secrets_from_ssm()
        log.info("loaded %d secret(s) from SSM Parameter Store: %s", len(loaded), loaded)
        return loaded
    except Exception as e:  # noqa: BLE001
        log.warning("SSM secret load failed (%s); continuing with existing environment", e)
        return []
