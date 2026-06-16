"""
SSM secret-loader tests — fully mocked, NO AWS / no network.

Verifies the deployment secret path: SecureStrings under /tradingbot/* are pulled into
the environment, the env-gating keeps it a no-op locally, and existing env vars are not
clobbered unless overwrite is requested.

Run:  python tests/test_secrets.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service import secrets


class FakeSSM:
    """Minimal stand-in for boto3 ssm client: a paginator over get_parameters_by_path."""
    def __init__(self, params, expect_decryption=True):
        self._params = params            # {full_name: value}
        self.expect_decryption = expect_decryption
        self.calls = []

    def get_paginator(self, op):
        assert op == "get_parameters_by_path"
        client = self

        class _Pag:
            def paginate(self, Path, WithDecryption, Recursive):
                client.calls.append((Path, WithDecryption, Recursive))
                assert WithDecryption is True          # SecureString must be decrypted
                items = [{"Name": k, "Value": v} for k, v in client._params.items()
                         if k.startswith(Path)]
                # split across two pages to exercise pagination
                yield {"Parameters": items[:1]}
                yield {"Parameters": items[1:]}
        return _Pag()


class TestLoadFromSSM(unittest.TestCase):
    def setUp(self):
        for v in ("ALPACA_KEY", "ALPACA_SECRET", "GEMINI_API_KEY",
                  "SEC_EDGAR_USER_AGENT", "TRADINGBOT_USE_SSM", "TRADINGBOT_SSM_PREFIX"):
            os.environ.pop(v, None)

    def test_loads_params_into_env(self):
        ssm = FakeSSM({
            "/tradingbot/ALPACA_KEY": "PKTEST",
            "/tradingbot/ALPACA_SECRET": "secret123",
            "/tradingbot/GEMINI_API_KEY": "gem123",
            "/tradingbot/SEC_EDGAR_USER_AGENT": "bot you@example.com",
        })
        loaded = secrets.load_secrets_from_ssm(prefix="/tradingbot/", client=ssm)
        self.assertEqual(set(loaded),
                         {"ALPACA_KEY", "ALPACA_SECRET", "GEMINI_API_KEY", "SEC_EDGAR_USER_AGENT"})
        self.assertEqual(os.environ["ALPACA_KEY"], "PKTEST")
        self.assertEqual(os.environ["SEC_EDGAR_USER_AGENT"], "bot you@example.com")
        self.assertTrue(ssm.calls and ssm.calls[0][1] is True)   # WithDecryption used

    def test_does_not_overwrite_existing_by_default(self):
        os.environ["ALPACA_KEY"] = "ALREADY_SET"
        ssm = FakeSSM({"/tradingbot/ALPACA_KEY": "FROM_SSM"})
        loaded = secrets.load_secrets_from_ssm(prefix="/tradingbot/", client=ssm)
        self.assertEqual(os.environ["ALPACA_KEY"], "ALREADY_SET")
        self.assertNotIn("ALPACA_KEY", loaded)

    def test_overwrite_when_requested(self):
        os.environ["ALPACA_KEY"] = "ALREADY_SET"
        ssm = FakeSSM({"/tradingbot/ALPACA_KEY": "FROM_SSM"})
        secrets.load_secrets_from_ssm(prefix="/tradingbot/", client=ssm, overwrite=True)
        self.assertEqual(os.environ["ALPACA_KEY"], "FROM_SSM")

    def test_unknown_leaf_maps_to_its_own_name(self):
        ssm = FakeSSM({"/tradingbot/SOME_FUTURE_KEY": "v"})
        loaded = secrets.load_secrets_from_ssm(prefix="/tradingbot/", client=ssm)
        self.assertIn("SOME_FUTURE_KEY", loaded)
        self.assertEqual(os.environ["SOME_FUTURE_KEY"], "v")
        os.environ.pop("SOME_FUTURE_KEY", None)


class TestMaybeLoad(unittest.TestCase):
    def setUp(self):
        os.environ.pop("TRADINGBOT_USE_SSM", None)
        os.environ.pop("TRADINGBOT_SSM_PREFIX", None)

    def test_noop_when_not_enabled(self):
        # not gated on -> must not even import boto3 / hit AWS
        self.assertEqual(secrets.maybe_load_ssm(), [])

    def test_failure_is_swallowed(self):
        os.environ["TRADINGBOT_USE_SSM"] = "1"
        # boto3 will try real creds/region and fail in this sandbox; must not raise
        try:
            result = secrets.maybe_load_ssm()
        finally:
            os.environ.pop("TRADINGBOT_USE_SSM", None)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
