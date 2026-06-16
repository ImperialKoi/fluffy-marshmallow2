# Phase 2 — Strategy Validation

> **Not investment advice.** Backtests, even out-of-sample, do not predict live results. This phase exists to *disprove* edges, not to find them.

## The cardinal rule

Out-of-sample (test) data is **sacred**: it never selects a strategy, tunes a parameter, or influences any train-period decision. The test set is scored exactly once. This is enforced structurally — the engine has no lookahead, so a metric over a train window depends only on bars up to that window's end; parameter selection reads train-window metrics only. `tests/test_validation_leakage.py` proves this mechanically: it corrupts every bar after a fold's train end and asserts the fold's train metrics and parameter choice are byte-for-byte identical (while OOS results change).

## Methodology

- **Universe:** `sample` (37 symbols). The bundled S&P 500 2013–2018 dataset.
- **Train/test split:** train 2013-01-01..2016-12-31, test 2017-01-01..2018-12-31 — reported separately so overfitting (strong IS, weak OOS) is visible.
- **Walk-forward:** rolling 2y train → next 1y test, 3 folds. Parameters are chosen on each train window, the next unseen window is scored, and OOS segments are stitched into one continuous curve.
- **Multi-symbol ranking by consistency:** strategies are ranked by the *median* walk-forward OOS Sharpe across symbols, not the best single name. A real edge shows up broadly; noise shows up on one ticker.
- **Parameter robustness:** small grids swept on TRAIN data only; a result on an isolated spike (vs a stable plateau) is flagged as overfit.
- **Baselines:** every strategy is compared to buy-and-hold AND a seeded, matched-exposure random-entry strategy. Beating random is the floor — a strategy that can't has no signal.
- **Costs:** slippage + commission stay on throughout; a `stress` run (5.0 bps commission / 25.0 bps slippage) checks which edges survive friction.

## Multiple-testing correction

- **Trials:** 1850 independent (strategy × symbol) backtests across 50 strategies. Testing this many configurations guarantees some will look good by chance.
- **Deflated Sharpe Ratio (preferred):** each strategy's pooled equal-weight OOS Sharpe is deflated for the trial count and for non-normal returns (skew/kurtosis). The deflation benchmark SR0 is the expected maximum Sharpe under the null given the trial count and the cross-trial Sharpe dispersion. Graduation requires DSR > 0.95.
- **Bonferroni (minimum bar):** a one-sided t-test that the mean cross-symbol OOS Sharpe > 0, with α divided by the number of strategies.
- **Minimum trades:** strategies with < 30 OOS round trips on a symbol are not credited there; with < 10 adequately-traded symbols a strategy is marked *insufficient evidence* rather than ranked.

## Graduation rule (applied mechanically)

A strategy graduates to the Phase 3 AI layer **only if**, out-of-sample, ALL hold: it beats buy-and-hold **and** the random baseline on a majority of symbols; has adequate OOS trade counts on enough symbols; shows parameter robustness (a plateau, not a spike); and its Deflated Sharpe survives the trial count (plus Bonferroni significance).

## Verdict

**No strategy graduated.** Under honest out-of-sample, multi-symbol, cost-aware, multiple-testing-corrected validation, none of the library's rule-based strategies showed a durable edge over buy-and-hold and a matched random baseline. **This is the expected and correct result** — simple technical rules rarely beat the market once you stop fooling yourself. It is reported plainly; the criteria were not loosened to manufacture a winner.

## Leaderboard — `normal` costs (ranked by median OOS Sharpe)

| strategy | symbols_sufficient | median_oos_sharpe | pct_beat_bh | pct_beat_random | total_oos_trades | overfit_gap | robust_kind | dsr | graduates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| psar | 22 | 0.39 | 32% | 77% | 944 | -0.27 | n/a (single config) | 0.00 | — |
| renko | 18 | 0.34 | 22% | 78% | 976 | -0.21 | n/a (single config) | 0.00 | — |
| nr7 | 29 | 0.22 | 17% | 66% | 1615 | -0.25 | n/a (single config) | 0.00 | — |
| sr_reversion | 16 | 0.20 | 0% | 44% | 1122 | -0.05 | plateau | 0.00 | — |
| roc | 14 | 0.19 | 14% | 71% | 946 | 0.07 | plateau | 0.01 | — |
| chandelier | 19 | 0.19 | 37% | 74% | 1036 | -0.12 | spike | 0.00 | — |
| harami | 10 | 0.19 | 50% | 50% | 927 | 0.11 | n/a (single config) | 0.01 | — |
| heikin_ashi | 28 | 0.17 | 14% | 64% | 2402 | -0.41 | n/a (single config) | 0.00 | — |
| ad_line | 18 | 0.16 | 22% | 56% | 971 | -0.34 | n/a (single config) | 0.00 | — |
| doji | 18 | 0.11 | 33% | 56% | 1100 | 0.09 | n/a (single config) | 0.00 | — |
| sr_breakout | 16 | 0.11 | 19% | 50% | 1038 | -0.06 | plateau | 0.00 | — |
| macd | 23 | 0.10 | 17% | 61% | 1175 | -0.11 | plateau | 0.01 | — |
| seasonality | 35 | 0.09 | 26% | 31% | 1368 | 0.36 | n/a (single config) | 0.00 | — |
| down_days | 11 | 0.03 | 18% | 18% | 886 | -0.15 | spike | 0.00 | — |
| pivot_points | 27 | -0.02 | 7% | 33% | 2652 | -0.34 | n/a (single config) | 0.00 | — |
| obv | 28 | -0.04 | 18% | 68% | 1657 | -0.11 | n/a (single config) | 0.00 | — |
| orb | 27 | -0.29 | 15% | 33% | 1963 | -0.21 | n/a (single config) | 0.00 | — |
| gap_and_go | 3 | 0.81 | 0% | 67% | 860 | 0.07 | n/a (single config) | 0.00 | — |
| cci | 0 | 0.37 | 0% | 0% | 464 | -0.18 | plateau | 0.02 | — |
| triangle | 0 | 0.35 | 0% | 0% | 213 | -0.18 | n/a (single config) | 0.04 | — |
| bollinger_breakout | 2 | 0.34 | 50% | 50% | 697 | -0.19 | plateau | 0.00 | — |
| williams_r | 8 | 0.28 | 50% | 75% | 844 | 0.02 | plateau | 0.00 | — |
| rsi | 0 | 0.27 | 0% | 0% | 283 | 0.13 | plateau | 0.00 | — |
| triple_ma | 0 | 0.26 | 0% | 0% | 622 | -0.02 | plateau | 0.02 | — |
| supertrend | 0 | 0.25 | 0% | 0% | 540 | -0.13 | plateau | 0.01 | — |

Full table: `results/validation/leaderboard_normal.csv`. Charts: cross-symbol Sharpe distribution, stitched OOS equity, in-sample-vs-OOS overfit scatter, and per-strategy parameter heatmaps in `results/validation/`.

## Cost stress test

Under stress costs, no strategy graduated. See `results/validation/leaderboard_stress.csv`.

## Caveat: survivorship bias

The bundled S&P 500 dataset contains only companies that **remained in the index** through 2013–2018 — failed/delisted names are absent. This flatters long-only strategies *even out-of-sample*, because the universe is conditioned on survival. Any strategy that looks good here must be re-validated on a broader, **point-in-time** universe (including delisted names) before it can be trusted. Survivorship bias makes these results an optimistic upper bound, not a guarantee.
