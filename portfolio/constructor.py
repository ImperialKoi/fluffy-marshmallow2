"""
Deterministic portfolio constructor.

Turns the LLM's per-symbol views into concrete target weights. This is NOT the LLM —
it is plain, auditable arithmetic, and together with portfolio/risk.py it has the
final say over what trades. Long-only by default; whatever isn't deployed is cash.

Conviction gating (the key safety property)
-------------------------------------------
Actions are gated on **conviction = score × confidence**, never raw score, so a weak
or unsure signal can't trigger a big trade. With `band = limits.neutral_band`:

  * conviction >  +band : eligible BUY candidate (ranked/weighted by conviction).
  * |conviction| ≤ band : NEUTRAL. A currently-held name is trimmed toward the
                          max-weight cap (de-concentrate) but NOT exited; an unheld
                          name simply stays flat. This is why a held 95% position on a
                          -0.05 conviction becomes "trim toward ~cap", not "sell all".
  * conviction <  -band : decisive EXIT to 0 (strong, confident negatives still leave).

Then: rank candidates, keep top `max_positions`, weight equal- or conviction-weighted
within an invested budget of min(max_gross, 1−min_cash) × exposure (reserving weight
for neutral holds), cap each at `max_weight` (excess → cash), scale down if fewer than
`min_positions`, and finally apply an optional per-rebalance turnover cap so the book
can't fully flip on one day's read (caps are then approached over several rebalances).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConstructorResult:
    weights: dict[str, float]          # symbol -> target weight (fraction of equity)
    cash_weight: float
    selected: list[str]                # the active BUY book
    notes: list[str]


def construct_targets(signals: dict, limits, exposure_multiplier: float = 1.0,
                      weighting: str = None, current_weights: dict = None,
                      turnover_cap: float = None) -> ConstructorResult:
    """`signals` maps symbol -> object with .score and .confidence (SymbolSignal).
    `current_weights` maps symbol -> current fraction of equity (for hold/trim + turnover)."""
    weighting = weighting or limits.weighting
    band = getattr(limits, "neutral_band", 0.10)
    if turnover_cap is None:
        turnover_cap = getattr(limits, "turnover_cap", None)
    cur = {t.upper(): float(w) for t, w in (current_weights or {}).items()}
    exposure_multiplier = float(max(0.0, min(1.0, exposure_multiplier)))
    notes = []

    conviction = {t: s.score * max(s.confidence, 0.0) for t, s in signals.items()}

    # 1. classify by conviction
    longs = [(t, conviction[t]) for t, s in signals.items()
             if conviction[t] > band and s.score > limits.min_score]
    longs.sort(key=lambda x: x[1], reverse=True)
    if len(longs) > limits.max_positions:
        notes.append(f"trimmed {len(longs)}->{limits.max_positions} by max_positions")
        longs = longs[:limits.max_positions]
    active_set = {t for t, _ in longs}

    # 2. neutral HELD names -> trim toward the cap (de-concentrate), do NOT exit.
    #    decisive negatives (conviction < -band) and unheld neutrals -> target 0.
    held_holds = {}
    for t in signals:
        if t in active_set:
            continue
        held_w = cur.get(t.upper(), 0.0)
        if held_w > 1e-9 and abs(conviction[t]) <= band:
            held_holds[t] = min(held_w, limits.max_weight)
            if held_w > limits.max_weight + 1e-9:
                notes.append(f"{t} neutral hold -> trim {held_w:.2%}->{limits.max_weight:.2%}")
    reserved = sum(held_holds.values())

    # 3. budget for the active book (after reserving neutral holds)
    budget = min(limits.max_gross, 1.0 - limits.min_cash) * exposure_multiplier
    if longs and len(longs) < limits.min_positions:
        scale = len(longs) / float(limits.min_positions)
        budget *= scale
        notes.append(f"only {len(longs)} candidates < min_positions "
                     f"{limits.min_positions} -> budget x{scale:.2f}")
    active_budget = max(0.0, budget - reserved)

    # 4. weight the active book
    if longs:
        if weighting == "equal":
            raw = {t: 1.0 for t, _ in longs}
        else:
            raw = {t: max(c, 1e-9) for t, c in longs}
        wsum = sum(raw.values())
        active = {t: active_budget * raw[t] / wsum for t in raw}
        for t in list(active):
            if active[t] > limits.max_weight:
                notes.append(f"{t} capped {active[t]:.2%}->{limits.max_weight:.2%}")
                active[t] = limits.max_weight
    else:
        active = {}

    targets = {**held_holds, **active}

    # 5. never exceed max_gross overall
    invested = sum(targets.values())
    if invested > limits.max_gross + 1e-9:
        scale = limits.max_gross / invested
        targets = {t: w * scale for t, w in targets.items()}
        notes.append(f"scaled to respect max_gross {limits.max_gross:.2%}")

    # 6. optional turnover cap: scale all moves toward current so the book can't flip
    if turnover_cap is not None:
        allt = set(targets) | set(cur)
        total_change = sum(abs(targets.get(t, 0.0) - cur.get(t, 0.0)) for t in allt)
        if total_change > turnover_cap > 0 and total_change > 1e-12:
            f = turnover_cap / total_change
            moved = {}
            for t in allt:
                c0 = cur.get(t, 0.0)
                w = c0 + (targets.get(t, 0.0) - c0) * f
                if abs(w) > 1e-6:
                    moved[t] = w
            targets = moved
            notes.append(f"turnover capped {total_change:.2f}->{turnover_cap:.2f} "
                         f"(moves x{f:.2f}); caps approached over multiple rebalances")

    cash_weight = max(0.0, 1.0 - sum(targets.values()))
    return ConstructorResult(weights=targets, cash_weight=cash_weight,
                             selected=[t for t, _ in longs], notes=notes)
