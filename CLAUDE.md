# AI Stock Trading Bot — Project Context & Handoff

> Drop this in the repo root as `CLAUDE.md` (Claude Code reads it automatically)
> or paste it into a session to bring the agent up to speed. It captures the
> goal, the research behind the choices, the build process so far, the current
> verified state, and what to do next.

---

## 1. Goal

Build a stock trading bot that makes AI-driven buy/sell decisions and can trade
live. Deliberately sequenced **backtest-first → paper → live** so we validate on
real data before risking money. Current status: a correct, tested backtest
engine plus an Alpaca paper-trading runner are built. The AI decision layer is
the next phase (not yet built).

**Not investment advice. Backtest results do not predict live results.**

---

## 2. Research summary (what informed the design)

**Broker / execution.** Alpaca was chosen as the broker: commission-free US
stocks/ETFs, API-first, $0 minimum, fractional shares, and crucially *unlimited
paper trading* with a sandbox that closely mirrors production. Trade-off: it
only covers US stocks/ETFs/options/crypto and leans on community support.
Interactive Brokers is the heavier alternative for global markets / more order
types. Current SDK is `alpaca-py`: `TradingClient(key, secret, paper=True)` for
orders, `StockHistoricalDataClient` for data, `MarketOrderRequest` +
`submit_order`. Free data uses the IEX feed (~2.5% of volume); SIP (full market)
is paid.

**AI approaches (for the upcoming Phase 3).** Two dominant patterns, often
combined:
- *Classical ML* — XGBoost / LSTM models predicting next-period direction from
  engineered indicator features.
- *LLM agents* — multi-agent setups (e.g. the open-source `ai-hedge-fund`
  project: valuation / sentiment / fundamentals / technicals agents + a risk
  manager + a portfolio manager). Note that reference project simulates only, no
  live execution.

**Universal warnings (these shaped every guardrail in the code).**
- Lookahead, survivorship, and selection bias will fool you if you are careless.
- LLMs hallucinate plausible-but-wrong indicators/thresholds — validate everything.
- A great backtest still fails live from microstructure, latency, regime change.
- Most simple retail strategies do **not** beat buy-and-hold. Always benchmark.
- LLMs are assistants, not oracles; gate everything behind backtests + risk limits.

---

## 3. Architecture & design invariants

Modular pipeline; each stage is independently testable:

```
data → features → strategy (decision) → risk → engine/execution → metrics/monitoring
```

```
config.py                 tunable settings (symbol, costs, sizing, stops, kill switch)
data/loader.py            pluggable data: cached CSV | yfinance | Alpaca (one interface)
features/indicators.py    SMA, EMA, RSI, MACD, ATR, Bollinger — all backward-looking
strategies/base.py        Strategy ABC  *** the single seam where AI plugs in ***
strategies/concrete.py    SMACrossover, RSIMeanReversion, BuyAndHold (benchmark)
risk/manager.py           position sizing, stop-loss/take-profit, drawdown kill switch
backtest/engine.py        event-driven, next-open fills, no lookahead, costs+slippage
backtest/metrics.py       return, CAGR, Sharpe, Sortino, max drawdown, win rate, etc.
run_backtest.py           CLI: run a backtest, print report, save chart + trade log
broker/alpaca_broker.py   Alpaca wrapper, paper-default, live triple-gated
check_alpaca.py           verify keys / account connectivity
live_trader.py            runs the SAME strategy+risk vs Alpaca (paper default)
tests/test_engine.py      proves next-open fills, slippage, stop triggers
data_cache/sp500_5yr.csv  real data: ~505 S&P500 stocks, daily, 2013-02 to 2018-02
```

**Invariants that MUST be preserved by any future change:**
1. **No lookahead.** A signal is computed from the *close* of bar `i` and fills
   at the *open* of bar `i+1`. Indicators must only use data ≤ current bar.
2. **Same code in backtest, paper, and live.** Only the data source and
   execution target change. This is why `Strategy` and `RiskManager` are reused
   verbatim by `live_trader.py`.
3. **Costs always work against the trader** (slippage on entry and exit,
   commission on notional).
4. **Risk module has veto power** and runs in every mode, including the
   portfolio drawdown kill switch.
5. **Paper is the default; live is intentionally hard** (see §6).

The `Strategy` interface (what an AI subclass must implement):
```python
class Strategy(ABC):
    def prepare(self, df) -> df:        # precompute backward-looking features (once)
    def signal(self, df, i) -> int:     # desired NEXT-bar position: +1 long / 0 flat / -1 short
                                        # may read df rows 0..i ONLY
```

---

## 4. Step-by-step development process (what was done, in order)

1. **Research.** Surveyed 2026 broker APIs and AI-trading approaches; picked
   Alpaca + a backtest-first plan; catalogued the bias/overfitting warnings.
2. **Data sourcing.** Yahoo/Alpaca were unreachable in the build sandbox, so a
   real dataset (S&P 500 5-year daily OHLCV, ~619k rows, 505 symbols) was pulled
   from a GitHub mirror and cached. Verified: 1259 bars/symbol, ~clean.
3. **Scaffold.** Created the modular package (`data`, `features`, `strategies`,
   `risk`, `backtest`, `broker`, `tests`).
4. **Feature engine.** Wrote backward-looking indicators from scratch (no heavy
   deps); documented that `pandas_ta`/TA-Lib can swap in later.
5. **Decision layer.** Wrote `Strategy` ABC + three concrete strategies
   (SMA crossover, RSI mean-reversion, buy-and-hold benchmark).
6. **Risk module.** Position sizing, %-based stop-loss/take-profit, intraday
   stop checks (worst-case tie-break), persistent drawdown kill switch.
7. **Backtest engine.** Event-driven loop: decide on close → fill next open →
   apply slippage+commission → check stops intraday → mark to market → kill
   switch → next signal.
8. **Metrics + CLI.** Sharpe/Sortino/CAGR/drawdown/win-rate/Calmar, vs benchmark;
   `run_backtest.py` prints a report and saves an equity+drawdown chart and a
   trade-log CSV.
9. **Verification.** Ran AAPL/MSFT/KO; confirmed honest results (SMA returned
   ~21.6% vs buy-and-hold ~135.6% on AAPL — expected). Buy-and-hold through the
   engine matched the independent benchmark within entry slippage → engine sane.
10. **Correctness tests.** `tests/test_engine.py` proves next-open fills (no
    lookahead) on a hand-computed toy series, that slippage worsens fills, and
    that stop-loss triggers on an intraday breach. All pass.
11. **Alpaca integration.** Verified `alpaca-py` API against the installed SDK.
    Built `AlpacaBroker` (paper-default, env-var keys, triple-gated live),
    `check_alpaca.py`, and `live_trader.py` (reuses Strategy+RiskManager; dry /
    paper / live modes; persistent kill-switch state file; CSV decision log).
12. **Verification (no network).** Confirmed all imports/signatures, that the
    safety guards raise correctly, and — via a mock broker on real cached data —
    that the signal→target-shares→reconcile→order logic and the dry-run path
    behave correctly.

---

## 5. How to run

```bash
pip install -r requirements.txt          # pandas numpy matplotlib (+ alpaca-py, yfinance)

# Backtest on the bundled real dataset (offline)
python run_backtest.py --symbol AAPL --strategy sma
python run_backtest.py --symbol MSFT --strategy rsi
python tests/test_engine.py              # correctness checks

# Backtest on live/recent data (your machine, needs internet)
python run_backtest.py --symbol TSLA --source yfinance --start 2019-01-01
```

---

## 6. Alpaca paper/live (Phase 4 — built, needs your keys to run live)

```bash
cp .env.example .env                      # paste PAPER keys from app.alpaca.markets
set -a && source .env && set +a           # load into shell (bash)

python check_alpaca.py                                            # verify connection
python live_trader.py --symbol AAPL --strategy sma --mode dry     # decide, NO orders
python live_trader.py --symbol AAPL --strategy sma --mode paper   # paper trade (default)
python live_trader.py --symbol AAPL --strategy sma --mode paper --once   # one cycle (for cron)
```

- Env vars: `ALPACA_KEY` / `ALPACA_SECRET` (paper). For live:
  `ALPACA_LIVE_KEY` / `ALPACA_LIVE_SECRET` **and** `ALPACA_ALLOW_LIVE=yes` **and**
  a typed confirmation phrase. Keys are never hardcoded.
- Decisions log to `results/live_log.csv`; kill-switch high-water mark persists
  in `results/live_state.json` (`--reset-state` to clear).
- Timing caveat: backtest fills at next open; live market orders fill now.
  Schedule `--once` at a consistent daily time and measure the gap on paper.

---

## 7. Next steps (priority order)

1. **Phase 3 — AI decision layer.** Add an AI strategy by subclassing `Strategy`
   so it inherits the same engine + risk + honest benchmarking. Either:
   - **ML strategy:** engineer backward-looking features → train XGBoost/LSTM →
     `signal()` returns long when predicted-up probability clears a threshold.
     **Must** train/tune on one period and validate on a later unseen period
     (walk-forward). In-sample metrics are meaningless.
   - **LLM-agent strategy:** summarize price/indicator state (and optionally
     news) as of bar `i`, ask a model for buy/hold/sell + rationale, parse it.
     Guard against hallucinated signals; never expose future data.
2. **Paper-trade a stretch** to confirm the live loop and measure live-vs-backtest slippage.
3. **Portfolio support** (multi-symbol; engine is single-asset for clarity now).
4. **Walk-forward / out-of-sample harness** and richer indicators (pandas_ta/TA-Lib).

---

## 8. Known limitations / gotchas

- Single asset per backtest run (by design, for now).
- Bundled dataset ends 2018 (fine for engine validation; pull recent data via
  yfinance/Alpaca for current research).
- **Survivorship bias** in the S&P 500 dataset flatters long-only results.
- Demo strategies are plumbing, not edges — they lose to buy-and-hold. The point
  so far is a *correct* system; the edge (if any) comes from Phase 3, validated
  out-of-sample.
- Free Alpaca data = IEX feed (partial volume); upgrade to SIP for full coverage.
