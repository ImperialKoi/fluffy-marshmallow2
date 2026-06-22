"""
Live (paper) trader.

This is Phase 4: take the EXACT strategy and risk code that was backtested and
run it against Alpaca. Nothing about the decision logic changes, which is the
whole point: live behaviour should match what you tested.

What one run does
-----------------
  1. Pull recent daily bars for the symbol from Alpaca.
  2. Run strategy.prepare() + strategy.signal() on the latest bar -> target -1/0/1.
  3. Ask the RiskManager for a target share count (and check the drawdown kill
     switch against your real account equity, persisted across runs).
  4. Compare to your current Alpaca position and submit a market order for the
     difference.
  5. Log the decision and any order.

Modes (default = paper, simulated money)
  --mode dry    compute and log the decision, place NO orders (safest)
  --mode paper  trade Alpaca paper account (simulated money)        [default]
  --mode live   real money. Requires --mode live AND env ALPACA_ALLOW_LIVE=yes
                AND typing the confirmation phrase when prompted.

Scheduling
  --once          run a single decision and exit (pair with cron/Task Scheduler)
  --loop SECONDS  keep running, sleeping SECONDS between decisions

Timing note: the backtest assumes you decide on a bar's close and fill at the
next open. To mirror that live, schedule --once to run once per trading day at a
consistent time (e.g. shortly after the open, acting on yesterday's close).
Intraday market-order fills will still differ from the backtest; that gap is
exactly what paper trading is for measuring.
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

import config
from broker.alpaca_broker import AlpacaBroker
from risk.manager import RiskManager
from strategies.registry import build, names
from alpaca.trading.enums import OrderSide

STATE_FILE = "results/live_state.json"
LOG_FILE = "results/live_log.csv"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def build_strategy(name, fast, slow):
    # Same registry the backtester uses, so live runs the IDENTICAL strategy code.
    # `sma` keeps its window flags for backwards compatibility; others use defaults.
    if name == "sma":
        return build("sma", fast=fast, slow=slow)
    return build(name)


def fetch_recent_bars(symbol, lookback_days=400, feed="iex"):
    """Daily OHLCV from Alpaca, shaped like the backtest's loader output."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    key = os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_LIVE_KEY")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_LIVE_SECRET")
    dc = StockHistoricalDataClient(key, secret)
    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start,
        feed=DataFeed.IEX if feed == "iex" else DataFeed.SIP,
    )
    bars = dc.get_stock_bars(req).df
    if bars.empty:
        raise ValueError(f"No bars returned for {symbol}.")
    bars = bars.reset_index()
    bars = bars[bars["symbol"] == symbol].set_index("timestamp")
    bars.index = pd.to_datetime(bars.index).tz_localize(None)
    return bars[["open", "high", "low", "close", "volume"]].sort_index()


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"peak_equity": 0.0, "halted": False}


def save_state(state):
    os.makedirs("results", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def log_decision(row):
    os.makedirs("results", exist_ok=True)
    header = not os.path.exists(LOG_FILE)
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=header, index=False)


# --------------------------------------------------------------------------- #
# one decision cycle
# --------------------------------------------------------------------------- #
def run_once(broker, strategy, risk, symbol, mode, allow_short):
    df = fetch_recent_bars(symbol)
    df = strategy.prepare(df)
    i = len(df) - 1
    target_signal = strategy.signal(df, i)
    last_close = float(df["close"].iloc[i])
    last_date = df.index[i].date().isoformat()

    # Drawdown kill switch against REAL account equity, persisted across runs.
    state = load_state()
    equity = broker.equity()
    state["peak_equity"] = max(state.get("peak_equity", 0.0), equity)
    if (config.MAX_DRAWDOWN_KILL is not None and state["peak_equity"] > 0
            and (state["peak_equity"] - equity) / state["peak_equity"] >= config.MAX_DRAWDOWN_KILL):
        state["halted"] = True
    save_state(state)

    if state.get("halted"):
        target_signal = 0  # kill switch forces flat and blocks new entries

    if target_signal < 0 and not allow_short:
        target_signal = 0  # long/flat only unless explicitly allowed

    desired = risk.target_shares(target_signal, equity, last_close)
    current = broker.position_shares(symbol)
    delta = desired - current

    decision = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": symbol, "bar_date": last_date, "close": round(last_close, 4),
        "signal": target_signal, "equity": round(equity, 2),
        "current_shares": current, "desired_shares": desired, "delta": delta,
        "mode": mode, "halted": state.get("halted", False), "order_id": "",
    }

    print(f"[{decision['ts']}] {symbol}  close={last_close:.2f}  signal={target_signal}  "
          f"have {current} -> want {desired}  (delta {delta})  mode={mode}"
          + ("  [HALTED]" if state.get("halted") else ""))

    if delta != 0 and mode != "dry":
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        order = broker.submit_market_order(symbol, abs(delta), side)
        if order is not None:
            decision["order_id"] = str(order.id)
            print(f"    submitted {side.value} {abs(delta)} {symbol}  order={order.id}")
    elif delta != 0:
        print(f"    DRY RUN: would {('BUY' if delta>0 else 'SELL')} {abs(delta)} {symbol}")
    else:
        print("    no change needed")

    log_decision(decision)
    return decision


def confirm_live():
    print("\n*** LIVE MODE — THIS TRADES REAL MONEY ***")
    phrase = "trade real money"
    typed = input(f'Type "{phrase}" to proceed: ').strip().lower()
    if typed != phrase:
        raise SystemExit("Confirmation failed. Aborting.")


def main():
    p = argparse.ArgumentParser(description="Run a strategy live against Alpaca (paper by default).")
    p.add_argument("--symbol", default=config.SYMBOL)
    p.add_argument("--strategy", default="sma", choices=names(),
                   metavar="NAME", help="strategy short name (see STRATEGIES.md)")
    p.add_argument("--mode", default="paper", choices=["dry", "paper", "live"])
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--allow-short", action="store_true", help="permit short positions")
    p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    p.add_argument("--loop", type=int, metavar="SECONDS", help="loop, sleeping N seconds")
    p.add_argument("--reset-state", action="store_true", help="clear the kill-switch state file")
    args = p.parse_args()

    if args.reset_state and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("Kill-switch state reset.")

    if args.mode == "live":
        confirm_live()

    broker = AlpacaBroker(paper=(args.mode != "live"))
    acct = broker.account_summary()
    print(f"Account: {acct['mode']}  equity=${acct['equity']:,.2f}  "
          f"buying_power=${acct['buying_power']:,.2f}")
    if acct["blocked"]:
        raise SystemExit("Trading is blocked on this account.")

    strategy = build_strategy(args.strategy, args.fast, args.slow)
    risk = RiskManager(
        position_fraction=config.POSITION_FRACTION,
        stop_loss_pct=config.STOP_LOSS_PCT,
        take_profit_pct=config.TAKE_PROFIT_PCT,
        max_drawdown_kill=config.MAX_DRAWDOWN_KILL,
        risk_multiplier=config.RISK_MULTIPLIER,
    )

    if args.loop:
        print(f"Looping every {args.loop}s. Ctrl-C to stop.")
        try:
            while True:
                run_once(broker, strategy, risk, args.symbol, args.mode, args.allow_short)
                time.sleep(args.loop)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        run_once(broker, strategy, risk, args.symbol, args.mode, args.allow_short)


if __name__ == "__main__":
    main()
