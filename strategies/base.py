"""
The decision layer.

Every strategy is a subclass of `Strategy`. This is the single seam in the whole
system where the "brain" lives. Today it's a rule (moving averages, RSI). Later it
can be an ML model, an LLM agent, or a hybrid — and *nothing else in the system
has to change*, because the engine only ever asks a strategy one question:

    "Given everything known as of the close of bar `i`, what position do you want
     for the next bar?"  ->  +1 long, 0 flat, -1 short.

That order then fills at the NEXT bar's open. Deciding on close[i] and filling at
open[i+1] is what keeps the backtest honest (no acting on a price you couldn't
have traded at).

To add an AI strategy later, you'd write something like:

    class MLStrategy(Strategy):
        def prepare(self, df):
            df = super().prepare(df)
            df["pred"] = self.model.predict(make_features(df))  # backward-looking!
            return df
        def signal(self, df, i):
            return 1 if df["pred"].iloc[i] > 0.5 else 0

    class LLMAgentStrategy(Strategy):
        def signal(self, df, i):
            # summarise df.iloc[:i+1], ask an LLM for buy/hold/sell, parse it.
            ...
"""

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    name = "base"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Precompute any indicator columns the strategy needs. Called once before
        the backtest loop. Must only use backward-looking features.
        """
        return df

    @abstractmethod
    def signal(self, df: pd.DataFrame, i: int) -> int:
        """
        Return the desired position for the *next* bar, using only df rows 0..i.
        +1 = long, 0 = flat, -1 = short. Read df.iloc[i] and earlier ONLY.
        """
        raise NotImplementedError
