"""
Strategy registry.

A single mapping from a short CLI name -> Strategy class, so `run_backtest.py` and
`live_trader.py` can offer every strategy via `--strategy <name>` without either of
them importing 40+ modules by hand. Add a new strategy by writing its file and
adding one line here.

Use:
    from strategies.registry import build, names
    strat = build("supertrend")          # default params
    strat = build("sma", fast=10, slow=40)
"""

from strategies.concrete import SMACrossover, RSIMeanReversion, BuyAndHold

# trend / moving averages
from strategies.ema_crossover import EMACrossover
from strategies.triple_ma import TripleMA
from strategies.macd_cross import MACDCross
from strategies.adx_trend import ADXTrend
from strategies.ichimoku_cloud import IchimokuCloud
from strategies.supertrend import SupertrendStrategy
from strategies.parabolic_sar import ParabolicSAR
from strategies.donchian_breakout import DonchianBreakout
from strategies.keltner_channel import KeltnerChannel

# momentum / oscillators
from strategies.stochastic import StochasticOscillator
from strategies.williams_r import WilliamsR
from strategies.cci_strategy import CCIStrategy
from strategies.roc_momentum import ROCMomentum
from strategies.connors_rsi import ConnorsRSIStrategy
from strategies.money_flow_index import MoneyFlowIndex

# mean reversion / volatility
from strategies.bollinger_reversion import BollingerReversion
from strategies.bollinger_breakout import BollingerBreakout
from strategies.zscore_reversion import ZScoreReversion
from strategies.chandelier_exit import ChandelierExit
from strategies.vwap_reversion import VWAPReversion

# breakout / channel
from strategies.opening_range_breakout import OpeningRangeBreakout
from strategies.high_52week import High52Week
from strategies.nr7_breakout import NR7Breakout
from strategies.gap_and_go import GapAndGo

# volume
from strategies.obv_trend import OBVTrend
from strategies.accumulation_distribution import AccumulationDistribution
from strategies.volume_weighted_momentum import VolumeWeightedMomentum

# candlestick patterns
from strategies.engulfing import Engulfing
from strategies.hammer import Hammer
from strategies.doji_reversal import DojiReversal
from strategies.harami import Harami
from strategies.morning_evening_star import MorningStar
from strategies.piercing_dark_cloud import PiercingLine
from strategies.three_soldiers_crows import ThreeWhiteSoldiers
from strategies.marubozu import Marubozu

# chart patterns
from strategies.double_top_bottom import DoubleBottom
from strategies.head_shoulders import InverseHeadShoulders
from strategies.triangle_breakout import TriangleBreakout
from strategies.cup_and_handle import CupAndHandle

# obscure / niche
from strategies.heikin_ashi import HeikinAshi
from strategies.renko import Renko
from strategies.pivot_points import PivotPoints
from strategies.fibonacci_retracement import FibonacciRetracement
from strategies.seasonality import Seasonality
from strategies.consecutive_down_days import ConsecutiveDownDays

# support / resistance with breakout probability
from strategies.sr_breakout import SRBreakout
from strategies.sr_reversion import SRReversion


# short name -> class
REGISTRY = {
    # baseline (already in the repo)
    "sma": SMACrossover,
    "rsi": RSIMeanReversion,
    "buyhold": BuyAndHold,
    # trend
    "ema": EMACrossover,
    "triple_ma": TripleMA,
    "macd": MACDCross,
    "adx": ADXTrend,
    "ichimoku": IchimokuCloud,
    "supertrend": SupertrendStrategy,
    "psar": ParabolicSAR,
    "donchian": DonchianBreakout,
    "keltner": KeltnerChannel,
    # momentum
    "stochastic": StochasticOscillator,
    "williams_r": WilliamsR,
    "cci": CCIStrategy,
    "roc": ROCMomentum,
    "connors_rsi": ConnorsRSIStrategy,
    "mfi": MoneyFlowIndex,
    # mean reversion / volatility
    "bollinger_reversion": BollingerReversion,
    "bollinger_breakout": BollingerBreakout,
    "zscore": ZScoreReversion,
    "chandelier": ChandelierExit,
    "vwap_reversion": VWAPReversion,
    # breakout / channel
    "orb": OpeningRangeBreakout,
    "high_52w": High52Week,
    "nr7": NR7Breakout,
    "gap_and_go": GapAndGo,
    # volume
    "obv": OBVTrend,
    "ad_line": AccumulationDistribution,
    "vol_momentum": VolumeWeightedMomentum,
    # candlestick
    "engulfing": Engulfing,
    "hammer": Hammer,
    "doji": DojiReversal,
    "harami": Harami,
    "morning_star": MorningStar,
    "piercing": PiercingLine,
    "three_soldiers": ThreeWhiteSoldiers,
    "marubozu": Marubozu,
    # chart patterns
    "double_bottom": DoubleBottom,
    "inv_head_shoulders": InverseHeadShoulders,
    "triangle": TriangleBreakout,
    "cup_handle": CupAndHandle,
    # obscure
    "heikin_ashi": HeikinAshi,
    "renko": Renko,
    "pivot_points": PivotPoints,
    "fibonacci": FibonacciRetracement,
    "seasonality": Seasonality,
    "down_days": ConsecutiveDownDays,
    # support / resistance
    "sr_breakout": SRBreakout,
    "sr_reversion": SRReversion,
}


def names():
    """All registered short names (sorted)."""
    return sorted(REGISTRY)


def build(name: str, **kwargs):
    """Instantiate a strategy by short name. Extra kwargs go to its constructor."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Choices: {', '.join(names())}")
    return REGISTRY[name](**kwargs)
