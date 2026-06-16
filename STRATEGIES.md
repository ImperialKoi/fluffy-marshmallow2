# Strategy Library

A catalog of every strategy that plugs into the existing backtest engine. Each one
is its own file in `strategies/`, subclasses the `Strategy` ABC (`prepare` +
`signal`), is **strictly backward-looking** (a decision at bar `i` uses only bars
`0..i`; the engine fills it at the open of bar `i+1`), and is selectable by its CLI
short name:

```bash
python run_backtest.py --symbol AAPL --strategy supertrend
python run_backtest.py --list-strategies          # print every name
python live_trader.py  --symbol AAPL --strategy supertrend --mode paper
python tests/test_strategies.py                   # no-lookahead + end-to-end for all
```

**Defaults are honest, not curve-fit.** Almost all are long/flat; the few that can
short take `allow_short=True` and are noted below. Most lose to buy-and-hold over
the bundled 2013–2018 window — that is expected. The value here is a correct,
broad library; real edges (if any) come from out-of-sample validation later.

Shared building blocks live in `features/`:
`indicators.py` (all the technical indicators), `candles.py` (candlestick
primitives), `patterns.py` (swing-point chart-pattern detectors), and `levels.py`
(support/resistance + breakout probability).

---

## Trend / moving averages

| CLI name | Strategy | Description | Key params |
|---|---|---|---|
| `sma` | SMA Crossover | Long when fast SMA > slow SMA (baseline). | `fast=20, slow=50` |
| `ema` | EMA Crossover | EMA version of the crossover; reacts faster. | `fast=12, slow=26, allow_short=False` |
| `triple_ma` | Triple MA | Long only when fast>medium>slow aligned. | `fast=10, medium=20, slow=50` |
| `macd` | MACD Crossover | Long when MACD line > signal (histogram>0). | `fast=12, slow=26, signal=9, allow_short=False` |
| `adx` | ADX Trend | Trade only when ADX shows a real trend and +DI>-DI. | `window=14, adx_threshold=25` |
| `ichimoku` | Ichimoku Cloud | Long above the cloud with Tenkan>Kijun. | `tenkan=9, kijun=26, senkou_b=52` |
| `supertrend` | Supertrend | ATR band trend flip; line doubles as trailing stop. | `window=10, mult=3.0, allow_short=False` |
| `psar` | Parabolic SAR | Long while close > SAR dots. | `af_step=0.02, af_max=0.20, allow_short=False` |
| `donchian` | Donchian Breakout | Turtle 20-high entry / 10-low exit. | `entry=20, exit_window=10, allow_short=False` |
| `keltner` | Keltner Channel | Long on close above the ATR-based upper band. | `ema_window=20, atr_window=10, mult=2.0` |

## Momentum / oscillators

| CLI name | Strategy | Description | Key params |
|---|---|---|---|
| `rsi` | RSI Mean-Reversion | Buy oversold, exit on recovery (baseline). | `window=14, oversold=30, exit_level=55` |
| `stochastic` | Stochastic | Buy %K/%D cross up from oversold; exit overbought. | `k_window=14, d_window=3, oversold=20, overbought=80` |
| `williams_r` | Williams %R | Buy as %R leaves oversold (-80); exit at -20. | `window=14` |
| `cci` | CCI | Buy CCI crossing up through -100; exit through +100. | `window=20` |
| `roc` | ROC Momentum | Long when N-bar rate-of-change is positive (time-series momentum). | `window=90, threshold=0` |
| `connors_rsi` | ConnorsRSI | Short-term oversold pullback (price/streak/rank blend) above a trend MA. | `oversold=10, exit_level=50, trend_ma=200` |
| `mfi` | Money Flow Index | Volume-weighted RSI; buy out of oversold. | `window=14, oversold=20, overbought=80` |

## Mean reversion / volatility

| CLI name | Strategy | Description | Key params |
|---|---|---|---|
| `bollinger_reversion` | Bollinger Reversion | Buy a close under the lower band; exit at the middle. | `window=20, num_std=2.0` |
| `bollinger_breakout` | Bollinger Breakout | Buy a close above the upper band; exit at the middle. | `window=20, num_std=2.0` |
| `zscore` | Z-Score Reversion | Buy when price z-score < -2; exit at 0. | `window=20, entry_z=2.0, exit_z=0.0` |
| `chandelier` | Chandelier Exit | ATR trailing-stop trend (highest-high − 3·ATR). | `window=22, mult=3.0` |
| `vwap_reversion` | VWAP Reversion | Buy a stretch below rolling VWAP; exit back at VWAP. | `window=20, band=0.02` |

## Breakout / channel

| CLI name | Strategy | Description | Key params |
|---|---|---|---|
| `orb` | Opening-Range Breakout | Daily analogue: break prior bar's high (intraday ORB isn't reproducible on daily bars). | `lookback=1` |
| `high_52w` | 52-Week High | Buy new 52-week highs; exit on a give-back. | `window=252, give_back=0.10` |
| `nr7` | NR7 Breakout | After the narrowest range in 7 bars, trade the break. | `window=7` |
| `gap_and_go` | Gap-and-Go | Buy an up-gap that holds, on above-average volume. | `min_gap=0.0, vol_mult=1.0` |

## Volume

| CLI name | Strategy | Description | Key params |
|---|---|---|---|
| `obv` | OBV Trend | Long when On-Balance Volume > its SMA. | `ma_window=20` |
| `ad_line` | Accumulation/Distribution | Long when the A/D line > its SMA. | `ma_window=20` |
| `vol_momentum` | Volume-Weighted Momentum | Positive ROC confirmed by a volume expansion. | `window=20, vol_mult=1.2` |

## Candlestick patterns

All are reversal/continuation patterns; entry holds for `hold` bars (plus the
shared risk stop). Sources: Steve Nison, *Japanese Candlestick Charting Techniques*.

| CLI name | Strategy | Description | Key params |
|---|---|---|---|
| `engulfing` | Bullish Engulfing | Bullish body engulfs prior bearish body after a downtrend. | `hold=5` |
| `hammer` | Hammer | Long lower shadow / small body after a downtrend. | `hold=5, shadow_ratio=2.0` |
| `doji` | Doji Reversal | Indecision doji after a downtrend. | `hold=3, body_frac=0.1` |
| `harami` | Bullish Harami | Small bullish inside-bar after a bearish bar. | `hold=5` |
| `morning_star` | Morning Star | Three-bar bullish bottom (bear, star, strong bull). | `hold=5, star_frac=0.5` |
| `piercing` | Piercing Line | Bull bar closes back above the midpoint of a prior bear bar. | `hold=5` |
| `three_soldiers` | Three White Soldiers | Three strong rising bullish bars. | `hold=5, body_frac=0.6` |
| `marubozu` | Bullish Marubozu | Near-shadowless strong bullish bar. | `hold=3, shadow_frac=0.05` |

## Chart patterns (swing-point based)

Detected only from swing pivots **confirmed** by bar `i` (see `features/patterns.py`),
so no future bars leak in. Sources: Bulkowski, *Encyclopedia of Chart Patterns*;
O'Neil for cup-and-handle. Detectors are transparent heuristics, not exhaustive recognisers.

| CLI name | Strategy | Description | Key params |
|---|---|---|---|
| `double_bottom` | Double Bottom | Two similar lows; buy the break above the interim peak. | `left=4, right=4, tol=0.04, lookback=120` |
| `inv_head_shoulders` | Inverse Head & Shoulders | Shoulder-head-shoulder lows; buy the neckline break. | `left=4, right=4, tol=0.05, lookback=160` |
| `triangle` | Ascending Triangle | Flat highs + rising lows; buy the break above resistance. | `left=4, right=4, tol=0.03, lookback=120` |
| `cup_handle` | Cup and Handle | Rounded base back to the rim, then breakout. | `min_depth=0.12, max_depth=0.45, lookback=200` |

## Obscure / niche

| CLI name | Strategy | Description | Key params |
|---|---|---|---|
| `heikin_ashi` | Heikin-Ashi Trend | Long on consecutive green HA candles; exit on first red. | — |
| `renko` | Renko Trend | ATR-brick direction; long while the last brick is up. | `atr_window=14, brick_mult=1.0` |
| `pivot_points` | Floor Pivots | Long above the prior-day pivot P; exit below S1. | — |
| `fibonacci` | Fibonacci Retracement | Buy a 38.2–61.8% pullback bounce within an uptrend. | `swing_window=60, trend_ma=100` |
| `seasonality` | Turn-of-Month | Long only on the first/last calendar days of a month. | `first_days=3, last_days=2, weekday=None` |
| `down_days` | Consecutive Down Days | Buy after N down closes in an uptrend; exit on an up day. | `n_down=3, max_hold=5, trend_ma=200` |

## Support / resistance with breakout probability

`features/levels.py` finds horizontal floors/ceilings from clustered, confirmed
swing pivots and estimates a **probability of breaking through** each nearest level
from backward-looking features (touch count, volume expansion into the level,
momentum, consolidation tightness, distance) via a transparent, documented logistic
score. Both strategies expose `p_break_resistance` / `p_break_support` columns on
the prepared frame for analysis.

| CLI name | Strategy | Description | Key params |
|---|---|---|---|
| `sr_breakout` | S/R Breakout | Buy a break through resistance **only when P(break) is high**. | `p_threshold=0.55, left=3, right=3, tol=0.02` |
| `sr_reversion` | S/R Reversion | Fade a floor **when P(break) is low** (expect a bounce). | `p_threshold=0.40, near=0.03` |

## Baseline

| CLI name | Strategy | Description |
|---|---|---|
| `buyhold` | Buy & Hold | The benchmark every strategy is measured against. |

---

### Quick read on the bundled 2013–2018 data (AAPL / MSFT)

A full sweep is in the project notes. Headline: buy-and-hold returned **+135.6%
(AAPL)** and **+227.6% (MSFT)** over the window, and **most strategies trail it** —
exactly what the literature predicts for simple rules. A handful posted respectable
risk-adjusted numbers on one symbol (e.g. `inv_head_shoulders`, `cci`, `macd` on
AAPL; `vwap_reversion`, `stochastic`, `hammer`, `bollinger_reversion` on MSFT) but
none should be trusted without out-of-sample / walk-forward validation. **Do not
read any of these as "beating the market."**
