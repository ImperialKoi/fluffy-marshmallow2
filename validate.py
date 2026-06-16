"""
Phase 2 validation CLI.

Examples
--------
    # Fast: all strategies on the sector-stratified sample, walk-forward, both costs
    python validate.py --strategy all --universe sample --mode walkforward

    # One strategy, full S&P 500 universe (slow), normal costs only
    python validate.py --strategy supertrend --universe sp500 --cost normal

    # Evenly-spaced 60-symbol universe, limited parallelism
    python validate.py --universe 60 --jobs 4

Outputs: results/validation/leaderboard_<cost>.csv, charts (cross-symbol Sharpe
distribution, stitched OOS equity, IS-vs-OOS overfit scatter, parameter heatmaps),
and VALIDATION.md (methodology + leaderboard + plain-English verdict).

The out-of-sample test set is never used to select or tune anything (see VALIDATION.md).
"""

import argparse
import logging
import os
import time

from strategies.registry import names as all_strategy_names
from validation import config as vcfg
from validation import data, multi_symbol, report


def main():
    p = argparse.ArgumentParser(description="Validate trading strategies out-of-sample.")
    p.add_argument("--strategy", default="all",
                   help="'all' or a comma-separated list of registry names")
    p.add_argument("--universe", default="sample",
                   help="'sample' (sector-stratified), 'sp500' (all ~505), or an integer N")
    p.add_argument("--mode", default="walkforward",
                   choices=["walkforward", "split", "all"],
                   help="reported analyses (walk-forward is the headline; both run regardless)")
    p.add_argument("--cost", default="both", choices=["normal", "stress", "both"])
    p.add_argument("--max-symbols", type=int, default=None, help="cap the universe size")
    p.add_argument("--jobs", type=int, default=None, help="parallel workers (default cpu-1)")
    p.add_argument("--out", default=report.OUT_DIR)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    # strategies
    if args.strategy == "all":
        strategies = all_strategy_names()
    else:
        strategies = [s.strip() for s in args.strategy.split(",") if s.strip()]
        unknown = [s for s in strategies if s not in all_strategy_names()]
        if unknown:
            raise SystemExit(f"Unknown strategies: {unknown}")

    symbols = data.resolve_universe(args.universe, args.max_symbols)
    if not symbols:
        raise SystemExit("No symbols resolved for the requested universe.")
    costs = ["normal", "stress"] if args.cost == "both" else [args.cost]

    os.makedirs(args.out, exist_ok=True)
    print(f"\nValidating {len(strategies)} strategies x {len(symbols)} symbols "
          f"= {len(strategies) * len(symbols)} backtests/cost  | costs={costs}")
    print(f"Universe[{args.universe}]: {', '.join(symbols[:12])}"
          f"{' ...' if len(symbols) > 12 else ''}\n")

    results_by_cost, meta_by_cost = {}, {}
    for cost in costs:
        t0 = time.time()
        print(f"=== cost scenario: {cost} ===")
        rows = multi_symbol.run_universe(strategies, symbols, cost_name=cost,
                                         jobs=args.jobs, progress=not args.quiet)
        df, meta = report.build_leaderboard(rows, cost_name=cost)
        path = report.save_leaderboard_csv(df, args.out, cost)
        report.save_charts(df, meta, args.out)
        results_by_cost[cost] = df
        meta_by_cost[cost] = meta
        grads = list(df[df["graduates"]]["strategy"])
        print(f"  done in {time.time() - t0:.0f}s -> {path}")
        print(f"  graduated [{cost}]: {grads if grads else 'NONE'}\n")

    md = report.write_validation_md(results_by_cost, meta_by_cost, args)
    print(f"Wrote {md}")

    # console summary (primary = normal if present)
    primary = "normal" if "normal" in results_by_cost else costs[0]
    df = results_by_cost[primary]
    print(f"\nTop 12 by median OOS Sharpe [{primary} costs]:")
    show = ["strategy", "symbols_sufficient", "median_oos_sharpe", "pct_beat_bh",
            "pct_beat_random", "dsr", "graduates"]
    with_pd_opts(df, show)


def with_pd_opts(df, cols):
    import pandas as pd
    with pd.option_context("display.max_rows", 60, "display.width", 160):
        print(df[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
