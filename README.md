# AI Trading Bot — Backtest Engine (Phase 1 + 2)

A modular, **backtest-first** trading system that runs on **real historical
market data**. The architecture is built so the "decision layer" can be swapped
from a simple rule today to an ML model or LLM agent later, with nothing else in
the system changing. It runs entirely offline on a bundled real dataset, and the
same code can pull live data from Yahoo Finance or your Alpaca account.

> **Not investment advice.** This is engineering for research. Backtest results
> do not predict live results. Trade real money only after paper trading, and
> only with money you can afford to lose.

## Why backtest-first

The hard part of a trading bot isn't the code, it's not fooling yourself. This
engine is deliberately conservative so that good-looking results are trustworthy:

- **No lookahead** — a signal is computed from the close of bar *i* and fills at
  the **open of bar *i+1***, the next price you could actually have traded.
- **Costs are real** — slippage and commission always work against you.
- **Risk is separate** — position sizing, stop-loss, take-profit, and a
  portfolio drawdown kill switch live in their own module with veto power.
- **A benchmark you must beat** — every run is compared to buy & hold. Most
  simple strategies lose to it; the engine will tell you so honestly.

## Architecture

```
data/loader.py        Data layer    -> cached CSV | yfinance | Alpaca (one interface)
features/indicators.py Feature engine -> SMA, EMA, RSI, MACD, ATR, Bollinger (backward-looking)
strategies/base.py    Decision layer -> Strategy ABC  *** this is where AI plugs in ***
strategies/concrete.py               -> SMA crossover, RSI mean-reversion, Buy & Hold
risk/manager.py       Risk module   -> sizing, stops, drawdown kill switch
backtest/engine.py    Engine        -> event-driven, next-open fills, no lookahead
backtest/metrics.py   Metrics       -> return, Sharpe, Sortino, drawdown, win rate...
run_backtest.py       Entry point   -> CLI, prints report, saves chart + trade log
broker/alpaca_broker.py Broker       -> Alpaca wrapper, paper-default, live gated
check_alpaca.py       Connectivity  -> verify your keys / account
live_trader.py        Live runner   -> same strategy+risk vs Alpaca (Phase 4)
tests/test_engine.py  Tests         -> proves next-open fills, slippage, stops
```

Identical strategy + risk code runs in backtest, paper, and live. Only the data
source and execution target change. That is what makes results transfer.

## Quick start

```bash
pip install -r requirements.txt          # core: pandas numpy matplotlib

python run_backtest.py                            # AAPL, SMA crossover (offline)
python run_backtest.py --symbol MSFT --strategy rsi
python run_backtest.py --symbol JPM --strategy sma --fast 10 --slow 40
python tests/test_engine.py                       # correctness checks
```

The bundled dataset (`data_cache/sp500_5yr.csv`) is real daily OHLCV for ~505
S&P 500 stocks, Feb 2013 – Feb 2018. Any ticker in it works.

### Using your own / live data

```bash
# Yahoo Finance (any ticker, any range — needs internet)
python run_backtest.py --symbol TSLA --source yfinance --start 2019-01-01

# Your Alpaca account (so backtest data matches your broker)
export ALPACA_KEY=...   ALPACA_SECRET=...
python run_backtest.py --symbol NVDA --source alpaca --start 2021-01-01 --end 2024-01-01
```

## Live + paper trading with Alpaca (Phase 4)

The same `Strategy` and `RiskManager` used in the backtest run unchanged against
Alpaca. **Paper trading (simulated money) is the default**; live trading is
intentionally gated behind several deliberate steps.

```bash
pip install alpaca-py                      # plus the core deps

cp .env.example .env                       # fill in your PAPER keys, then:
set -a && source .env && set +a            # (bash) load them into the shell

python check_alpaca.py                     # 1. confirm the connection works
python live_trader.py --symbol AAPL --strategy sma --mode dry    # 2. decide, no orders
python live_trader.py --symbol AAPL --strategy sma --mode paper  # 3. paper trade
```

Modes, safest first:
- `--mode dry` computes and logs the decision, places **no** orders.
- `--mode paper` trades the Alpaca paper account (simulated money). **Default.**
- `--mode live` real money, and only after: `--mode live` **and**
  `ALPACA_ALLOW_LIVE=yes` **and** typing a confirmation phrase. The risk manager
  and drawdown kill switch stay active in every mode.

Run it once per trading day on a schedule (cron / Task Scheduler):
```bash
python live_trader.py --symbol AAPL --strategy sma --mode paper --once
```
or loop: `--loop 86400`. Every decision is appended to `results/live_log.csv`,
and a persistent high-water mark in `results/live_state.json` powers the
drawdown kill switch across runs.

> Timing caveat: the backtest decides on a bar's close and fills at the next
> open. Live, your market order fills at the current price, so results will
> differ. Measuring that gap on paper, before risking real money, is the point.

## Roadmap: adding the AI (Phase 3)

The whole point of the `Strategy` base class is that the brain is swappable. To
add AI, subclass it — the engine, risk, costs, and metrics stay exactly as they
are, so an AI strategy is held to the same honest standard as the simple rules.

1. **Classical ML** — engineer backward-looking features, train an XGBoost/LSTM
   model to predict next-day direction, and return `1` when its probability
   clears a threshold. *Critical:* train and tune on one period, then validate on
   a later period the model never saw (walk-forward). In-sample results lie.
2. **LLM agent** — summarise recent price action / news as of bar *i* and ask a
   model for buy/hold/sell with a rationale, then parse it. Watch for
   hallucinated indicators and never let it see future data.
3. **Hybrid** — ML produces a signal, an LLM (or a rule) confirms or vetoes.

After Phase 3: **Phase 4** runs it on Alpaca *paper* trading for weeks and
compares live fills against the backtest, then **Phase 5** goes to small real
capital with the kill switch armed.

## Scheduling (optional — nothing is installed by default)

The bot does **not** install any scheduler or autostart job. Trigger runs yourself,
e.g. a single AI-portfolio dry pass (compute + log, no orders):

```bash
python live_portfolio.py --mode dry --once
```

If you want to automate a daily dry run, two opt-in helpers are provided for you to
install **manually** (see `AI_STRATEGY.md` → *Scheduling*):

- `scripts/daily_dry_run.sh` — wrapper that loads `.env` and runs one dry pass.
- `scripts/com.tradingbot.dailydryrun.plist` — a macOS **launchd** agent (weekdays
  09:45 local). Install/uninstall commands are in the plist's header comment:
  ```bash
  cp scripts/com.tradingbot.dailydryrun.plist ~/Library/LaunchAgents/
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tradingbot.dailydryrun.plist
  # uninstall:
  launchctl bootout gui/$(id -u)/com.tradingbot.dailydryrun
  rm ~/Library/LaunchAgents/com.tradingbot.dailydryrun.plist
  ```
  A cron one-liner alternative is documented in `scripts/daily_dry_run.sh`.

## Known limitations (by design, for now)

- Single asset per run (the engine is built for clarity first; portfolio support
  is the natural next extension).
- Daily bars only in the demo (intraday works via yfinance/Alpaca timeframes).
- The bundled dataset ends in 2018 — fine for engine validation, but pull recent
  data via yfinance/Alpaca for current research.
- Survivorship bias: the S&P 500 dataset only contains companies that were in the
  index, which flatters long-only results. Be aware when interpreting.
```
