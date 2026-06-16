"""
Baseline strategies for honest comparison.

A strategy that cannot beat a coin-flip with the same market exposure has no
signal. `RandomEntry` is that coin flip: seeded (reproducible), and parameterized
by a target `exposure` so it can be matched to the time-in-market of whatever
strategy it's being compared against — otherwise the comparison is unfair (a
rarely-invested strategy vs. an always-invested random one tells you nothing).

Reproducibility: positions are drawn from a hashlib-seeded RNG keyed by
(seed, symbol date-range), NOT Python's salted hash(), so the exact same bars are
chosen on every run and every machine. The reproducibility test depends on this.

It is, of course, strictly backward-looking — random entries use no data at all,
let alone future data.

Buy & hold is already in the project (strategies.registry "buyhold"); we reuse it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from strategies.base import Strategy


def _stable_seed(seed: int, df: pd.DataFrame) -> int:
    key = f"{seed}|{df.index[0].isoformat()}|{df.index[-1].isoformat()}|{len(df)}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class RandomEntry(Strategy):
    """Long with probability `exposure`, drawn in blocks to keep turnover sane."""

    def __init__(self, exposure: float = 0.5, seed: int = 1, block: int = 5):
        self.exposure = float(min(max(exposure, 0.0), 1.0))
        self.seed = int(seed)
        self.block = max(1, int(block))
        self.name = f"RandomEntry(exp={self.exposure:.2f}, seed={seed})"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        rng = np.random.default_rng(_stable_seed(self.seed, df))
        n_blocks = (n + self.block - 1) // self.block
        block_long = rng.random(n_blocks) < self.exposure
        pos = np.repeat(block_long, self.block)[:n].astype(float)
        df["rand_pos"] = pos
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        return int(df["rand_pos"].iloc[i])
