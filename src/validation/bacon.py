"""
Goodman-Bacon decomposition of a staggered two-way fixed effects estimate.

Why this exists
---------------
`reports/phase7_external_validity.md` raised the problem and stopped there: under
staggered adoption with heterogeneous effects, two-way fixed effects does not
estimate a single average treatment effect. It estimates a weighted average of
every 2x2 difference-in-differences comparison the panel admits, and some of
those comparisons use **already-treated** units as controls. When the effect
changes over time, the already-treated control is itself still moving, and that
movement enters the estimate with a negative sign.

Goodman-Bacon (2021) shows the decomposition is exact. Every staggered TWFE
coefficient is

    beta_twfe = sum_over_comparisons ( weight * beta_2x2 )

with weights that depend only on group sizes and treatment timing, never on the
outcome. So the question "how much of this estimate rests on bad comparisons" has
an arithmetic answer rather than a judgement.

Three comparison types
----------------------
* **treated vs never-treated** - clean. The control never moves.
* **earlier vs later, later as control** - clean. The control is not yet treated
  during the window used.
* **later vs earlier, earlier as control** - the bad one. The control is already
  treated, so its own dynamic response is subtracted from the treated group's.

What this does and does not tell you
------------------------------------
A small bad-comparison weight means TWFE is close to a well-defined average of
clean comparisons. It does **not** mean the design is identified: parallel trends
can fail for every comparison in the table, good and bad alike. On the Rossmann
Promo2 design it does fail, which Phase 7 established with an event study before
this decomposition was written. The decomposition then answers a narrower
question - whether the headline number is at least the thing it claims to be -
and the answer can be "yes, and it is still not usable".

Implementation notes
--------------------
Units treated before the panel opens are dropped. They have no pre-period, so
they cannot appear in any 2x2, and TWFE silently uses them as pure controls.
Never-treated units are kept as the U group.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("northstar.bacon")

TREATED_VS_NEVER = "treated vs never-treated"
EARLY_VS_LATE = "earlier vs later (later as control)"
LATE_VS_EARLY = "later vs earlier (already-treated control)"

BAD_COMPARISON = LATE_VS_EARLY


def _balanced_grid(
    panel: pd.DataFrame, unit: str, time: str, outcome: str, treatment: str
) -> pd.DataFrame:
    """
    Collapse to one row per unit x period and require a complete rectangle.

    The decomposition's weights are functions of group sizes and treatment
    shares, and both are only meaningful on a balanced panel. An unbalanced one
    does not fail loudly here, it just makes the weights describe a panel that
    does not exist.
    """
    grid = (
        panel.groupby([unit, time], as_index=False)
        .agg(**{outcome: (outcome, "mean"), treatment: (treatment, "max")})
    )
    units, periods = grid[unit].nunique(), grid[time].nunique()
    if len(grid) != units * periods:
        counts = grid.groupby(unit).size()
        keep = counts[counts == periods].index
        LOGGER.warning(
            "Unbalanced panel: dropping %d of %d units with incomplete coverage",
            len(counts) - len(keep), len(counts),
        )
        grid = grid[grid[unit].isin(keep)]
    return grid.sort_values([unit, time]).reset_index(drop=True)


def timing_groups(grid: pd.DataFrame, unit: str, time: str, treatment: str) -> pd.Series:
    """
    First treated period per unit; NaT-equivalent (None) for never-treated.

    Units already treated in the first period are returned as `-1`, which the
    caller drops. They have no pre-period, so no 2x2 comparison can include
    them, but nothing in a TWFE regression says so.
    """
    periods = sorted(grid[time].unique())
    first_period = periods[0]

    treated_rows = grid[grid[treatment] == 1]
    first_treated = treated_rows.groupby(unit)[time].min()

    groups = pd.Series(index=sorted(grid[unit].unique()), dtype=object)
    groups[:] = None
    for unit_id, period in first_treated.items():
        groups[unit_id] = -1 if period == first_period else period
    return groups


def _two_by_two(
    wide: pd.DataFrame, treated_units: List, control_units: List, pre: List, post: List
) -> float:
    """One 2x2 DiD: change in the treated group minus change in the control group."""
    treated_pre = wide.loc[treated_units, pre].to_numpy().mean()
    treated_post = wide.loc[treated_units, post].to_numpy().mean()
    control_pre = wide.loc[control_units, pre].to_numpy().mean()
    control_post = wide.loc[control_units, post].to_numpy().mean()
    return float((treated_post - treated_pre) - (control_post - control_pre))


def decompose(
    panel: pd.DataFrame,
    unit: str = "store_id",
    time: str = "period",
    outcome: str = "log_units",
    treatment: str = "treated",
) -> pd.DataFrame:
    """
    Every 2x2 comparison inside a staggered TWFE estimate, with its weight.

    Returns one row per comparison: the two groups involved, the comparison
    type, the 2x2 estimate and its weight. Weights sum to 1, so
    `(weight * estimate).sum()` reproduces the TWFE coefficient - which
    `tests/test_validation.py` asserts rather than takes on trust.
    """
    n_units_supplied = panel[unit].nunique()
    grid = _balanced_grid(panel, unit, time, outcome, treatment)
    n_unbalanced = n_units_supplied - grid[unit].nunique()
    groups = timing_groups(grid, unit, time, treatment)

    always_treated = [u for u, g in groups.items() if g == -1]
    if always_treated:
        LOGGER.info("Dropping %d units treated before the panel opens", len(always_treated))
        grid = grid[~grid[unit].isin(always_treated)]
        groups = groups.drop(always_treated)

    periods = sorted(grid[time].unique())
    n_periods = len(periods)
    wide = grid.pivot(index=unit, columns=time, values=outcome)

    never = [u for u, g in groups.items() if g is None]
    timed = sorted({g for g in groups.values if g is not None})
    if not timed:
        raise ValueError("no unit is treated inside the panel window")

    members = {g: [u for u, gg in groups.items() if gg == g] for g in timed}
    n_total = len(groups)

    # Share of periods each group spends treated. The never-treated group's
    # share is zero by construction, which is what makes it a valid control
    # for every timing group.
    treated_share = {
        g: float(sum(1 for p in periods if p >= g) / n_periods) for g in timed
    }
    size = {g: len(members[g]) / n_total for g in timed}
    size_never = len(never) / n_total if never else 0.0

    rows: List[Dict[str, object]] = []

    for g in timed:
        pre = [p for p in periods if p < g]
        post = [p for p in periods if p >= g]
        if never and pre and post:
            share = size[g] / (size[g] + size_never)
            weight = (
                (size[g] + size_never) ** 2
                * share * (1 - share)
                * treated_share[g] * (1 - treated_share[g])
            )
            rows.append({
                "treated_group": g, "control_group": "never",
                "comparison": TREATED_VS_NEVER,
                "estimate": _two_by_two(wide, members[g], never, pre, post),
                "raw_weight": weight,
                "n_treated": len(members[g]), "n_control": len(never),
            })

    for i, k in enumerate(timed):
        for later in timed[i + 1:]:
            share = size[k] / (size[k] + size[later])
            d_k, d_l = treated_share[k], treated_share[later]

            # k is treated, `later` is still untreated: use only the periods
            # before `later` adopts.
            pre_k = [p for p in periods if p < k]
            mid = [p for p in periods if k <= p < later]
            if pre_k and mid and d_l < 1:
                weight = (
                    ((size[k] + size[later]) * (1 - d_l)) ** 2
                    * share * (1 - share)
                    * ((d_k - d_l) / (1 - d_l)) * ((1 - d_k) / (1 - d_l))
                )
                rows.append({
                    "treated_group": k, "control_group": later,
                    "comparison": EARLY_VS_LATE,
                    "estimate": _two_by_two(wide, members[k], members[later], pre_k, mid),
                    "raw_weight": weight,
                    "n_treated": len(members[k]), "n_control": len(members[later]),
                })

            # `later` is treated, k is *already* treated and used as a control.
            post_l = [p for p in periods if p >= later]
            if mid and post_l and d_k > 0:
                weight = (
                    ((size[k] + size[later]) * d_k) ** 2
                    * share * (1 - share)
                    * (d_l / d_k) * ((d_k - d_l) / d_k)
                )
                rows.append({
                    "treated_group": later, "control_group": k,
                    "comparison": LATE_VS_EARLY,
                    "estimate": _two_by_two(wide, members[later], members[k], mid, post_l),
                    "raw_weight": weight,
                    "n_treated": len(members[later]), "n_control": len(members[k]),
                })

    table = pd.DataFrame(rows)
    total = table["raw_weight"].sum()
    if total <= 0:
        raise ValueError("decomposition weights sum to zero; check treatment timing")
    table["weight"] = table["raw_weight"] / total
    table["contribution"] = table["weight"] * table["estimate"]
    table = table.sort_values("weight", ascending=False).reset_index(drop=True)

    # What was excluded, so the report quotes the counts this decomposition
    # actually used rather than recomputing them from the unfiltered panel and
    # quietly disagreeing with itself.
    table.attrs.update({
        "n_units_supplied": int(n_units_supplied),
        "n_dropped_unbalanced": int(n_unbalanced),
        "n_dropped_always_treated": int(len(always_treated)),
        "n_never_treated": len(never),
        "n_timing_groups": len(timed),
        "n_periods": n_periods,
    })
    return table


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Weight and weighted-average effect per comparison type."""
    grouped = table.groupby("comparison").apply(
        lambda part: pd.Series({
            "weight": part["weight"].sum(),
            "average_effect": np.average(part["estimate"], weights=part["weight"]),
            "contribution": part["contribution"].sum(),
        }),
        include_groups=False,
    )
    counts = table.groupby("comparison").size().rename("comparisons")
    grouped = grouped.join(counts).reset_index()
    grouped["comparisons"] = grouped["comparisons"].astype(int)
    return grouped[
        ["comparison", "comparisons", "weight", "average_effect", "contribution"]
    ].sort_values("weight", ascending=False)


def twfe_from_decomposition(table: pd.DataFrame) -> float:
    """The implied TWFE coefficient. Equals the regression estimate, by theorem."""
    return float(table["contribution"].sum())
