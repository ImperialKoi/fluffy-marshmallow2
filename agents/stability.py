"""
Score-stability gate.

The AI's per-symbol score can swing run-to-run (e.g. AAPL scored -0.10 on one pass
and +0.20 on the next). Acting decisively on a signal that keeps flipping sign is how
you churn the book on noise. This module looks at the last N runs' scores recorded in
the decisions log and flags symbols whose sign is unstable, so the go/no-go is
MECHANICAL rather than by eye.

It is ADVISORY: the service logs a warning listing unstable names each rebalance, and
`python live_portfolio.py stability` prints a verdict table. It does not hard-block
trades (the deterministic constructor + risk layer remain the trade authority); use it
as a human gate, or wire the flagged set into your own policy if you want.

Verdicts per symbol (over the last `runs` runs):
  * insufficient — fewer than `min_runs` scored runs (not enough evidence)
  * unstable     — >= `flip_threshold` sign flips among nonzero scores in the window
  * flipped      — stable-ish but the latest score flipped sign vs the prior one
  * stable       — consistent sign
"""

from __future__ import annotations

import statistics

import config


def _verdict(scores, flip_threshold, min_runs):
    nz = [s for s in scores if abs(s) > 1e-9]
    signs = [1 if s > 0 else -1 for s in nz]
    flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    if len(scores) < min_runs:
        v = "insufficient"
    elif flips >= flip_threshold:
        v = "unstable"
    elif len(signs) >= 2 and signs[-1] != signs[-2]:
        v = "flipped"
    else:
        v = "stable"
    return v, flips


def analyze(scores_by_symbol: dict, flip_threshold: int = None,
            min_runs: int = None) -> dict:
    """`scores_by_symbol` maps symbol -> chronological list of scores (oldest..newest).
    Returns per-symbol {n, scores, sign_flips, mean, latest, verdict}."""
    flip_threshold = flip_threshold or config.AI_STABILITY_FLIP_THRESHOLD
    min_runs = min_runs or config.AI_STABILITY_MIN_RUNS
    out = {}
    for sym, scores in scores_by_symbol.items():
        scores = [float(s) for s in scores]
        v, flips = _verdict(scores, flip_threshold, min_runs)
        out[sym] = {
            "n": len(scores), "scores": scores, "sign_flips": flips,
            "mean": statistics.mean(scores) if scores else 0.0,
            "latest": scores[-1] if scores else None, "verdict": v,
        }
    return out


def load_recent_scores(decisions_csv: str = None, runs: int = None) -> tuple[dict, list]:
    """Read the decisions log and return (scores_by_symbol, run_timestamps) for the last
    `runs` distinct runs. Empty/blank scores are skipped (e.g. halted runs)."""
    import pandas as pd

    decisions_csv = decisions_csv or config.AI_DECISIONS_LOG
    runs = runs or config.AI_STABILITY_RUNS
    df = pd.read_csv(decisions_csv)
    df = df[pd.to_numeric(df["score"], errors="coerce").notna()].copy()
    df["score"] = df["score"].astype(float)
    df["ts"] = pd.to_datetime(df["ts"])
    recent_ts = sorted(df["ts"].unique())[-runs:]
    df = df[df["ts"].isin(recent_ts)]
    by_symbol = {}
    for sym, g in df.groupby("symbol"):
        by_symbol[sym] = list(g.sort_values("ts")["score"])
    return by_symbol, list(recent_ts)


def unstable_symbols(results: dict) -> list[str]:
    return sorted(s for s, r in results.items() if r["verdict"] in ("unstable", "flipped"))


def format_table(results: dict) -> str:
    order = {"unstable": 0, "flipped": 1, "insufficient": 2, "stable": 3}
    rows = sorted(results.items(), key=lambda kv: (order.get(kv[1]["verdict"], 9), kv[0]))
    lines = [f"  {'symbol':<8}{'runs':>5}{'flips':>6}{'latest':>8}{'mean':>8}  verdict",
             "  " + "-" * 52]
    for sym, r in rows:
        latest = "  n/a" if r["latest"] is None else f"{r['latest']:+.2f}"
        flag = "" if r["verdict"] == "stable" else "  <--"
        lines.append(f"  {sym:<8}{r['n']:>5}{r['sign_flips']:>6}{latest:>8}"
                     f"{r['mean']:>8.2f}  {r['verdict']}{flag}")
    return "\n".join(lines)
