"""
Parameter robustness: does good TRAIN performance sit on a stable plateau, or is it
a single fragile spike? A spike means the parameters were (implicitly) overfit and
won't survive deployment.

We aggregate each strategy's per-symbol TRAIN-window grid surfaces (Sharpe per
parameter set, averaged across symbols), then:
  * plateau_frac = fraction of grid points whose mean train Sharpe is within
    ROBUST_TOL of the best point.
  * spike        = (>= 3 grid points) AND (plateau_frac < ROBUST_PLATEAU_FRAC).
  * robust       = NOT spike. Single-config strategies (no grid) are "n/a" and
                   treated as robust — there is no tunable knob to overfit.

All of this uses TRAIN data only; the test set is never consulted for tuning.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import config as vcfg


def aggregate_surface(rows: list[dict]) -> list[dict]:
    """Mean TRAIN Sharpe per grid param-set across this strategy's symbols."""
    acc = defaultdict(list)
    order = []
    for r in rows:
        for cell in r.get("grid_surface", []):
            key = _param_key(cell["params"])
            if key not in acc:
                order.append((key, cell["params"]))
            s = cell["train_sharpe"]
            if s == s:  # not NaN
                acc[key].append(s)
    surface = []
    for key, params in order:
        vals = acc[key]
        surface.append({"params": params,
                        "mean_train_sharpe": float(np.mean(vals)) if vals else float("nan"),
                        "n": len(vals)})
    return surface


def summarize(rows: list[dict]) -> dict:
    surface = aggregate_surface(rows)
    finite = [c["mean_train_sharpe"] for c in surface if c["mean_train_sharpe"] == c["mean_train_sharpe"]]
    n_params = len(surface)

    if n_params <= 1:
        return {"robust": True, "kind": "n/a (single config)", "n_params": n_params,
                "plateau_frac": float("nan"), "best_params": surface[0]["params"] if surface else {},
                "best_train_sharpe": finite[0] if finite else float("nan"),
                "surface": surface}
    if not finite:
        return {"robust": False, "kind": "no valid grid points", "n_params": n_params,
                "plateau_frac": 0.0, "best_params": {}, "best_train_sharpe": float("nan"),
                "surface": surface}

    best = max(finite)
    tol = vcfg.ROBUST_TOL * abs(best) if best != 0 else vcfg.ROBUST_TOL
    near = [v for v in finite if abs(v - best) <= tol]
    plateau_frac = len(near) / len(finite)
    spike = (len(finite) >= 3) and (plateau_frac < vcfg.ROBUST_PLATEAU_FRAC)
    best_cell = max((c for c in surface if c["mean_train_sharpe"] == c["mean_train_sharpe"]),
                    key=lambda c: c["mean_train_sharpe"])
    return {"robust": (not spike), "kind": ("spike" if spike else "plateau"),
            "n_params": n_params, "plateau_frac": float(plateau_frac),
            "best_params": best_cell["params"], "best_train_sharpe": float(best),
            "surface": surface}


def heatmap_matrix(surface: list[dict]):
    """For a 2-parameter grid, return (xs, ys, x_name, y_name, Z) for a heatmap.
    Returns None if the grid isn't 2-D."""
    if not surface:
        return None
    keys = list(surface[0]["params"].keys())
    if len(keys) != 2:
        return None
    xk, yk = keys
    xs = sorted({c["params"][xk] for c in surface})
    ys = sorted({c["params"][yk] for c in surface})
    Z = np.full((len(ys), len(xs)), np.nan)
    for c in surface:
        xi = xs.index(c["params"][xk])
        yi = ys.index(c["params"][yk])
        Z[yi, xi] = c["mean_train_sharpe"]
    return xs, ys, xk, yk, Z


def _param_key(params: dict) -> str:
    return ",".join(f"{k}={params[k]}" for k in sorted(params)) if params else "default"
