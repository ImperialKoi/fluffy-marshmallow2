"""
Reporting: turn raw (strategy x symbol) rows into a leaderboard ranked by
CONSISTENCY, apply the graduation rule mechanically, render charts, and write
VALIDATION.md.

Leaderboard ranking is by robust central tendency (median walk-forward OOS Sharpe
across symbols) among strategies with sufficient evidence — NOT by best single
symbol. Graduation is a hard AND of: beats buy-and-hold AND a matched random
baseline on a majority of symbols, adequate OOS trades, parameter robustness (not a
spike), and a Deflated Sharpe that survives the trial count (plus Bonferroni).
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as vcfg
from . import robustness
from . import significance as sig

OUT_DIR = "results/validation"


# --------------------------------------------------------------------------- #
# aggregation + graduation
# --------------------------------------------------------------------------- #
def build_leaderboard(rows: list[dict], cost_name: str) -> tuple[pd.DataFrame, dict]:
    by_strat = defaultdict(list)
    for r in rows:
        if r.get("cost", cost_name) == cost_name:
            by_strat[r["strategy"]].append(r)
    strategies = sorted(by_strat)

    # pooled OOS portfolio returns per strategy + per-pair sharpes for DSR inputs
    pooled = {s: sig.pooled_oos_returns(by_strat[s]) for s in strategies}
    pair_sharpes = []
    for s in strategies:
        for r in by_strat[s]:
            d = r.get("oos_returns") or {}
            if len(d) > 1:
                ser = pd.Series({pd.Timestamp(k): v for k, v in d.items()})
                pair_sharpes.append(sig.per_period_sharpe(ser))
    pair_sharpes = [x for x in pair_sharpes if x == x]
    n_trials = max(2, len(pair_sharpes))           # trials = strategy x symbol backtests
    sr_std_all = float(np.std(pair_sharpes, ddof=1)) if len(pair_sharpes) > 1 else 0.0
    n_strategies = len(strategies)

    records = []
    for s in strategies:
        srows = by_strat[s]
        evaluated = [r for r in srows if "wf_oos_sharpe" in r and not r.get("error")]
        sufficient = [r for r in evaluated if r.get("wf_oos_rt", 0) >= vcfg.MIN_ROUND_TRIPS]
        judged = sufficient if sufficient else evaluated

        wf_sharpes = [r["wf_oos_sharpe"] for r in judged if r["wf_oos_sharpe"] == r["wf_oos_sharpe"]]
        beat_bh = [bool(r.get("wf_beats_bh")) for r in sufficient]
        beat_rnd = [bool(r.get("wf_beats_random")) for r in sufficient]

        rob = robustness.summarize(srows)
        ds = sig.deflated_sharpe(pooled[s], n_trials=n_trials, sr_std=sr_std_all)
        bon = sig.bonferroni_test(wf_sharpes, n_strategies)

        n_suff = len(sufficient)
        pct_bh = float(np.mean(beat_bh)) if beat_bh else 0.0
        pct_rnd = float(np.mean(beat_rnd)) if beat_rnd else 0.0
        med_sharpe = float(np.median(wf_sharpes)) if wf_sharpes else float("nan")
        med_ret = float(np.median([r["wf_oos_return"] for r in judged
                                   if r.get("wf_oos_return") == r.get("wf_oos_return")])) \
            if judged else float("nan")
        total_rt = int(sum(r.get("wf_oos_rt", 0) for r in evaluated))
        is_sh = np.nanmean([r.get("is_sharpe", np.nan) for r in evaluated]) if evaluated else np.nan
        oos_sh = np.nanmean([r.get("split_oos_sharpe", np.nan) for r in evaluated]) if evaluated else np.nan

        # ---- graduation rule (mechanical) ----
        insufficient = n_suff < vcfg.MIN_SYMBOLS_EVALUATED
        g_beats = (pct_bh > vcfg.MAJORITY) and (pct_rnd > vcfg.MAJORITY)
        g_robust = bool(rob["robust"])
        g_dsr = (ds["dsr"] == ds["dsr"]) and (ds["dsr"] > vcfg.DSR_THRESHOLD)
        g_bonf = bool(bon["significant"])
        graduates = bool((not insufficient) and g_beats and g_robust and g_dsr and g_bonf)

        reasons = []
        if insufficient:
            reasons.append(f"insufficient evidence (<{vcfg.MIN_SYMBOLS_EVALUATED} symbols w/ >={vcfg.MIN_ROUND_TRIPS} trades)")
        if not g_beats:
            reasons.append(f"beats BH {pct_bh:.0%} / random {pct_rnd:.0%} (need >50% both)")
        if not g_robust:
            reasons.append(f"param {rob['kind']}")
        if not g_dsr:
            reasons.append(f"DSR {ds['dsr']:.2f}<={vcfg.DSR_THRESHOLD}")
        if not g_bonf:
            reasons.append("not Bonferroni-significant")

        records.append({
            "strategy": s,
            "symbols_eval": len(evaluated),
            "symbols_sufficient": n_suff,
            "median_oos_sharpe": med_sharpe,
            "median_oos_return": med_ret,
            "pct_beat_bh": pct_bh,
            "pct_beat_random": pct_rnd,
            "total_oos_trades": total_rt,
            "is_sharpe_mean": float(is_sh) if is_sh == is_sh else float("nan"),
            "oos_sharpe_mean": float(oos_sh) if oos_sh == oos_sh else float("nan"),
            "overfit_gap": float(is_sh - oos_sh) if (is_sh == is_sh and oos_sh == oos_sh) else float("nan"),
            "param_robust": g_robust, "robust_kind": rob["kind"],
            "plateau_frac": rob["plateau_frac"],
            "dsr": ds["dsr"], "dsr_sr": ds["sr"], "dsr_sr0": ds["sr0"],
            "bonferroni_sig": g_bonf, "bonferroni_p": bon["p_one_sided"],
            "insufficient_evidence": insufficient,
            "graduates": graduates,
            "fail_reasons": "; ".join(reasons) if reasons else "",
        })

    df = pd.DataFrame.from_records(records)
    # rank by consistency: graduates first, then sufficient, then median OOS Sharpe
    df = df.sort_values(
        by=["graduates", "insufficient_evidence", "median_oos_sharpe"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    meta = {"n_trials": n_trials, "n_strategies": n_strategies,
            "sr_std_all": sr_std_all, "pooled": pooled,
            "by_strat": by_strat, "cost": cost_name}
    return df, meta


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def save_charts(df: pd.DataFrame, meta: dict, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    cost = meta["cost"]
    _chart_sharpe_distribution(meta["by_strat"], out_dir, cost)
    _chart_oos_equity(df, meta, out_dir, cost)
    _chart_overfit(df, out_dir, cost)
    _chart_heatmaps(meta["by_strat"], out_dir, cost)


def _chart_sharpe_distribution(by_strat, out_dir, cost):
    data, labels = [], []
    order = sorted(by_strat,
                   key=lambda s: np.nanmedian([r.get("wf_oos_sharpe", np.nan)
                                               for r in by_strat[s]] or [np.nan]))
    for s in order:
        vals = [r.get("wf_oos_sharpe") for r in by_strat[s]
                if r.get("wf_oos_sharpe") == r.get("wf_oos_sharpe")]
        if vals:
            data.append(vals); labels.append(s)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.28)))
    ax.boxplot(data, vert=False, labels=labels, showfliers=False)
    ax.axvline(0, color="crimson", lw=1, ls="--")
    ax.set_title(f"Cross-symbol walk-forward OOS Sharpe by strategy [{cost} costs]")
    ax.set_xlabel("OOS Sharpe (per symbol)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/cross_symbol_sharpe_{cost}.png", dpi=120)
    plt.close(fig)


def _chart_oos_equity(df, meta, out_dir, cost):
    pooled = meta["pooled"]
    top = list(df["strategy"].head(6))
    for ref in ("buyhold",):
        if ref in pooled and ref not in top:
            top.append(ref)
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = 0
    for s in top:
        ser = pooled.get(s)
        if ser is None or len(ser) < 2:
            continue
        curve = (1.0 + ser.sort_index()).cumprod()
        ax.plot(curve.index, curve.values, lw=1.6 if s != "buyhold" else 1.2,
                ls="--" if s == "buyhold" else "-", label=s)
        plotted += 1
    if not plotted:
        plt.close(fig); return
    ax.set_title(f"Stitched walk-forward OOS equity (equal-weight across symbols) [{cost} costs]")
    ax.set_ylabel("Growth of $1 (OOS only)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/oos_equity_top_{cost}.png", dpi=120)
    plt.close(fig)


def _chart_overfit(df, out_dir, cost):
    sub = df.dropna(subset=["is_sharpe_mean", "oos_sharpe_mean"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sub["is_sharpe_mean"], sub["oos_sharpe_mean"], alpha=0.7)
    lim = [min(sub["is_sharpe_mean"].min(), sub["oos_sharpe_mean"].min(), -0.5),
           max(sub["is_sharpe_mean"].max(), sub["oos_sharpe_mean"].max(), 0.5)]
    ax.plot(lim, lim, color="grey", ls=":", label="IS = OOS (no degradation)")
    ax.axhline(0, color="crimson", lw=0.8, ls="--")
    ax.set_xlabel("In-sample Sharpe (train 2013-2016)")
    ax.set_ylabel("Out-of-sample Sharpe (test 2017-2018)")
    ax.set_title(f"Overfitting check: in-sample vs out-of-sample [{cost} costs]")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/overfit_is_vs_oos_{cost}.png", dpi=120)
    plt.close(fig)


def _chart_heatmaps(by_strat, out_dir, cost):
    made = 0
    for s in sorted(by_strat):
        if made >= 12:
            break
        surface = robustness.aggregate_surface(by_strat[s])
        hm = robustness.heatmap_matrix(surface)
        if hm is None:
            continue
        xs, ys, xk, yk, Z = hm
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
        ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys)
        ax.set_xlabel(xk); ax.set_ylabel(yk)
        ax.set_title(f"{s}: mean TRAIN Sharpe over param grid [{cost}]")
        for yi in range(len(ys)):
            for xi in range(len(xs)):
                if Z[yi, xi] == Z[yi, xi]:
                    ax.text(xi, yi, f"{Z[yi, xi]:.2f}", ha="center", va="center",
                            color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="train Sharpe")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/heatmap_{s}_{cost}.png", dpi=110)
        plt.close(fig)
        made += 1


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def save_leaderboard_csv(df: pd.DataFrame, out_dir: str, cost: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/leaderboard_{cost}.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    return path


def write_validation_md(results_by_cost: dict, meta_by_cost: dict, args,
                        path: str = "VALIDATION.md") -> str:
    """Compose VALIDATION.md: methodology, leaderboard(s), and a plain verdict."""
    primary_cost = "normal" if "normal" in results_by_cost else next(iter(results_by_cost))
    df = results_by_cost[primary_cost]
    meta = meta_by_cost[primary_cost]
    grads = list(df[df["graduates"]]["strategy"])

    L = []
    L.append("# Phase 2 — Strategy Validation\n")
    L.append("> **Not investment advice.** Backtests, even out-of-sample, do not "
             "predict live results. This phase exists to *disprove* edges, not to "
             "find them.\n")

    L.append("## The cardinal rule\n")
    L.append("Out-of-sample (test) data is **sacred**: it never selects a strategy, "
             "tunes a parameter, or influences any train-period decision. The test "
             "set is scored exactly once. This is enforced structurally — the engine "
             "has no lookahead, so a metric over a train window depends only on bars "
             "up to that window's end; parameter selection reads train-window metrics "
             "only. `tests/test_validation_leakage.py` proves this mechanically: it "
             "corrupts every bar after a fold's train end and asserts the fold's train "
             "metrics and parameter choice are byte-for-byte identical (while OOS "
             "results change).\n")

    L.append("## Methodology\n")
    L.append(f"- **Universe:** `{args.universe}` "
             f"({_universe_count(meta)} symbols). The bundled "
             "S&P 500 2013–2018 dataset.\n"
             f"- **Train/test split:** train {vcfg.SPLIT_TRAIN[0]}..{vcfg.SPLIT_TRAIN[1]}, "
             f"test {vcfg.SPLIT_TEST[0]}..{vcfg.SPLIT_TEST[1]} — reported separately so "
             "overfitting (strong IS, weak OOS) is visible.\n"
             f"- **Walk-forward:** rolling 2y train → next 1y test, {len(vcfg.FOLDS)} "
             "folds. Parameters are chosen on each train window, the next unseen window "
             "is scored, and OOS segments are stitched into one continuous curve.\n"
             "- **Multi-symbol ranking by consistency:** strategies are ranked by the "
             "*median* walk-forward OOS Sharpe across symbols, not the best single name. "
             "A real edge shows up broadly; noise shows up on one ticker.\n"
             "- **Parameter robustness:** small grids swept on TRAIN data only; a result "
             "on an isolated spike (vs a stable plateau) is flagged as overfit.\n"
             "- **Baselines:** every strategy is compared to buy-and-hold AND a seeded, "
             "matched-exposure random-entry strategy. Beating random is the floor — a "
             "strategy that can't has no signal.\n"
             f"- **Costs:** slippage + commission stay on throughout; a `stress` run "
             "({} bps commission / {} bps slippage) checks which edges survive friction.\n"
             .format(vcfg.COSTS['stress'].commission_bps, vcfg.COSTS['stress'].slippage_bps))

    L.append("## Multiple-testing correction\n")
    L.append(f"- **Trials:** {meta['n_trials']} independent (strategy × symbol) "
             f"backtests across {meta['n_strategies']} strategies. Testing this many "
             "configurations guarantees some will look good by chance.\n"
             "- **Deflated Sharpe Ratio (preferred):** each strategy's pooled "
             "equal-weight OOS Sharpe is deflated for the trial count and for "
             "non-normal returns (skew/kurtosis). The deflation benchmark SR0 is the "
             "expected maximum Sharpe under the null given the trial count and the "
             "cross-trial Sharpe dispersion. Graduation requires DSR > "
             f"{vcfg.DSR_THRESHOLD}.\n"
             "- **Bonferroni (minimum bar):** a one-sided t-test that the mean "
             "cross-symbol OOS Sharpe > 0, with α divided by the number of strategies.\n"
             f"- **Minimum trades:** strategies with < {vcfg.MIN_ROUND_TRIPS} OOS round "
             "trips on a symbol are not credited there; with < "
             f"{vcfg.MIN_SYMBOLS_EVALUATED} adequately-traded symbols a strategy is "
             "marked *insufficient evidence* rather than ranked.\n")

    L.append("## Graduation rule (applied mechanically)\n")
    L.append("A strategy graduates to the Phase 3 AI layer **only if**, out-of-sample, "
             "ALL hold: it beats buy-and-hold **and** the random baseline on a majority "
             "of symbols; has adequate OOS trade counts on enough symbols; shows "
             "parameter robustness (a plateau, not a spike); and its Deflated Sharpe "
             "survives the trial count (plus Bonferroni significance).\n")

    # verdict
    L.append("## Verdict\n")
    if grads:
        L.append(f"**{len(grads)} strategy(ies) graduated:** "
                 + ", ".join(f"`{g}`" for g in grads) + ".\n")
        L.append("These survived every gate. Treat that as *promising, not proven* — "
                 "re-validate on a broader, point-in-time universe before trusting them "
                 "(see the survivorship caveat).\n")
    else:
        L.append("**No strategy graduated.** Under honest out-of-sample, multi-symbol, "
                 "cost-aware, multiple-testing-corrected validation, none of the "
                 "library's rule-based strategies showed a durable edge over buy-and-hold "
                 "and a matched random baseline. **This is the expected and correct "
                 "result** — simple technical rules rarely beat the market once you stop "
                 "fooling yourself. It is reported plainly; the criteria were not loosened "
                 "to manufacture a winner.\n")

    L.append(f"## Leaderboard — `{primary_cost}` costs (ranked by median OOS Sharpe)\n")
    L.append(leaderboard_markdown(df, top=25) + "\n")
    L.append(f"Full table: `{OUT_DIR}/leaderboard_{primary_cost}.csv`. "
             "Charts: cross-symbol Sharpe distribution, stitched OOS equity, "
             "in-sample-vs-OOS overfit scatter, and per-strategy parameter heatmaps "
             f"in `{OUT_DIR}/`.\n")

    if "stress" in results_by_cost:
        sdf = results_by_cost["stress"]
        sgr = list(sdf[sdf["graduates"]]["strategy"])
        L.append("## Cost stress test\n")
        L.append(f"Under stress costs, {('these graduated: ' + ', '.join(sgr)) if sgr else 'no strategy graduated'}. "
                 f"See `{OUT_DIR}/leaderboard_stress.csv`.\n")

    L.append("## Caveat: survivorship bias\n")
    L.append("The bundled S&P 500 dataset contains only companies that **remained in "
             "the index** through 2013–2018 — failed/delisted names are absent. This "
             "flatters long-only strategies *even out-of-sample*, because the universe "
             "is conditioned on survival. Any strategy that looks good here must be "
             "re-validated on a broader, **point-in-time** universe (including delisted "
             "names) before it can be trusted. Survivorship bias makes these results an "
             "optimistic upper bound, not a guarantee.\n")

    text = "\n".join(L)
    with open(path, "w") as f:
        f.write(text)
    return path


def _universe_count(meta) -> int:
    return len({r["symbol"] for rows in meta["by_strat"].values() for r in rows})


def leaderboard_markdown(df: pd.DataFrame, top: int = 20) -> str:
    cols = ["strategy", "symbols_sufficient", "median_oos_sharpe", "pct_beat_bh",
            "pct_beat_random", "total_oos_trades", "overfit_gap", "robust_kind",
            "dsr", "graduates"]
    sub = df[cols].head(top).copy()
    sub["median_oos_sharpe"] = sub["median_oos_sharpe"].map(lambda x: f"{x:.2f}")
    sub["pct_beat_bh"] = sub["pct_beat_bh"].map(lambda x: f"{x:.0%}")
    sub["pct_beat_random"] = sub["pct_beat_random"].map(lambda x: f"{x:.0%}")
    sub["overfit_gap"] = sub["overfit_gap"].map(lambda x: f"{x:.2f}" if x == x else "-")
    sub["dsr"] = sub["dsr"].map(lambda x: f"{x:.2f}" if x == x else "-")
    sub["graduates"] = sub["graduates"].map(lambda b: "✅" if b else "—")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, r in sub.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)
