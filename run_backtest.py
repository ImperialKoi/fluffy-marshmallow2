"""
Entry point. Run a backtest on real market data, print a report, and save an
equity-curve chart plus a trade log.

Examples
--------
    python run_backtest.py                          # AAPL, SMA crossover
    python run_backtest.py --symbol MSFT --strategy rsi
    python run_backtest.py --symbol JPM --strategy sma --fast 10 --slow 40
    python run_backtest.py --symbol AMZN --source yfinance --start 2019-01-01

Data sources:
    --source cache     bundled S&P 500 2013-2018 dataset (default, offline)
    --source yfinance  live history from Yahoo (needs internet + `pip install yfinance`)
    --source alpaca    history from your Alpaca account (set ALPACA_KEY / ALPACA_SECRET)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from data import loader
from risk.manager import RiskManager
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics, format_report
from strategies.registry import build, names


def build_strategy(args):
    # `sma` is the one strategy whose windows are exposed as top-level CLI flags
    # (kept for backwards compatibility); every other strategy uses its defaults.
    if args.strategy == "sma":
        return build("sma", fast=args.fast, slow=args.slow)
    return build(args.strategy)


def load_data(args):
    if args.source == "cache":
        return loader.load_csv(config.CACHE_CSV, args.symbol, args.start, args.end)
    if args.source == "yfinance":
        return loader.load_yfinance(args.symbol, args.start, args.end)
    if args.source == "alpaca":
        return loader.load_alpaca(args.symbol, args.start, args.end,
                                  api_key=os.environ.get("ALPACA_KEY"),
                                  secret_key=os.environ.get("ALPACA_SECRET"))
    raise ValueError(f"Unknown source '{args.source}'")


def save_chart(result, metrics, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1],
                                   sharex=True)
    ax1.plot(result.equity_curve.index, result.equity_curve.values,
             label=result.strategy_name, linewidth=1.6)
    ax1.plot(result.benchmark_curve.index, result.benchmark_curve.values,
             label="Buy & Hold", linewidth=1.2, alpha=0.7, linestyle="--")
    ax1.set_title(f"{result.symbol}: {result.strategy_name} vs Buy & Hold")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.25)

    # Drawdown panel
    curve = result.equity_curve
    dd = (curve - curve.cummax()) / curve.cummax() * 100
    ax2.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.35)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_trade_log(result, path):
    rows = [{"date": t.date.date().isoformat(), "side": t.side, "shares": t.shares,
             "price": round(t.price, 4), "reason": t.reason, "fee": round(t.cost, 2)}
            for t in result.trades]
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description="Backtest a trading strategy on real data.")
    p.add_argument("--symbol", default=config.SYMBOL)
    p.add_argument("--strategy", default="sma", choices=names(),
                   metavar="NAME", help="strategy short name (see STRATEGIES.md)")
    p.add_argument("--list-strategies", action="store_true",
                   help="print all available strategy names and exit")
    p.add_argument("--source", default="cache", choices=["cache", "yfinance", "alpaca"])
    p.add_argument("--fast", type=int, default=20, help="fast SMA window")
    p.add_argument("--slow", type=int, default=50, help="slow SMA window")
    p.add_argument("--start", default=config.START)
    p.add_argument("--end", default=config.END)
    args = p.parse_args()

    if args.list_strategies:
        print("Available strategies (--strategy <name>):")
        for nm in names():
            print(f"  {nm}")
        return

    print(f"\nLoading {args.symbol} from '{args.source}' ...")
    df = load_data(args)
    print(f"Loaded {len(df)} bars: {df.index[0].date()} -> {df.index[-1].date()}")

    strategy = build_strategy(args)
    risk = RiskManager(
        position_fraction=config.POSITION_FRACTION,
        stop_loss_pct=config.STOP_LOSS_PCT,
        take_profit_pct=config.TAKE_PROFIT_PCT,
        max_drawdown_kill=config.MAX_DRAWDOWN_KILL,
    )

    result = run_backtest(
        df, strategy, risk,
        initial_cash=config.INITIAL_CASH,
        commission_bps=config.COMMISSION_BPS,
        slippage_bps=config.SLIPPAGE_BPS,
        symbol=args.symbol,
    )

    metrics = compute_metrics(result, config.TRADING_DAYS_PER_YEAR, config.RISK_FREE_RATE)
    print("\n" + "=" * 56)
    print("  BACKTEST RESULT")
    print("=" * 56)
    print(format_report(metrics))
    print("=" * 56)

    os.makedirs("results", exist_ok=True)
    tag = f"{args.symbol}_{args.strategy}"
    chart_path = f"results/{tag}_equity.png"
    log_path = f"results/{tag}_trades.csv"
    save_chart(result, metrics, chart_path)
    save_trade_log(result, log_path)
    print(f"\nSaved chart  -> {chart_path}")
    print(f"Saved trades -> {log_path}\n")


if __name__ == "__main__":
    main()
