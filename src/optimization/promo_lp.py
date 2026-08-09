"""
Promotion budget allocation as an integer program.

Decision: for each store x SKU, run no promotion or exactly one, at one of the
six discount depths, subject to a budget and execution capacity.

The accounting that most portfolios get wrong
---------------------------------------------
It is tempting to value a promotion as *incremental units x margin*. That is
wrong, and generously so. Discounting cuts the margin on the volume that would
have sold anyway, and that sacrifice is usually the largest single term:

    profit with promotion    = (baseline + incremental) x discounted margin
                               - promotion cost - cannibalisation
    profit without promotion = baseline x full margin

    incremental profit       = the difference

For a shallow discount on a low-uplift SKU that difference is **negative** - the
retailer pays for volume it already had. The optimiser is only interesting
because it can decline, and the report shows how many candidates it declines.

Cannibalisation
---------------
Phase 3 measured it directly: promoting one SKU depresses its non-promoted
category neighbours by roughly 6% for the first concurrent promotion, deepening
to 16% by the fourth before saturating. That is genuinely quadratic - the loss
depends on how many *other* promotions run in the same store and category - which
an integer program cannot express directly.

Two linear devices stand in for it, and both are stated as approximations:

1. every selected promotion is charged the first-promotion marginal loss against
   its category's untreated baseline in that store;
2. a cap on promotions per store x category keeps the plan inside the range where
   that linear charge is a reasonable approximation, rather than out where the
   effect has saturated and the charge would overstate it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOGGER = logging.getLogger("promopulse.optimization.promo")

DISCOUNT_DEPTHS = [5, 10, 15, 20, 25, 30]

# Phase 3, measured on untreated rows with pair and date effects absorbed: the
# marginal depression from the first concurrent promotion in a store x category.
CANNIBALISATION_FIRST_PROMO = 0.061

# Retailer-funded share of promotional spend, from the generated promotion file.
DEFAULT_VENDOR_FUNDED = 0.40


def build_candidates(
    pairs: pd.DataFrame,
    effect_curve: Dict[str, Dict[int, float]],
    category_baseline: pd.DataFrame,
    duration_days: int = 8,
    vendor_funded: float = DEFAULT_VENDOR_FUNDED,
    cannibalisation_rate: float = CANNIBALISATION_FIRST_PROMO,
) -> pd.DataFrame:
    """
    Enumerate every (store, SKU, depth) option with its incremental profit.

    `effect_curve` maps a price-elasticity segment to {discount depth -> log
    uplift}. Passing a different curve is how the report compares optimising on
    a naive estimate against optimising on the causal one.

    `cannibalisation_rate` is the single most consequential assumption in this
    model. Setting it to zero - which is what an optimiser that ignores
    cross-SKU effects implicitly does - changes the recommendation by two orders
    of magnitude, so the report sweeps it rather than fixing it.
    """
    records: List[Dict[str, object]] = []
    category_lookup = category_baseline.set_index(
        ["store_id", "category"]
    )["baseline_units_per_day"]

    for pair in pairs.itertuples(index=False):
        segment_curve = effect_curve.get(pair.price_elasticity_segment)
        if segment_curve is None:
            continue

        baseline_units = float(pair.mean_daily_units) * duration_days
        full_margin = float(pair.regular_unit_price_gbp) - float(pair.unit_cost_gbp)
        if baseline_units <= 0 or full_margin <= 0:
            continue

        # Category volume this promotion can cannibalise: everything else in the
        # same store and category over the same window.
        category_units = float(
            category_lookup.get((pair.store_id, pair.category), 0.0)
        ) * duration_days
        other_units = max(category_units - baseline_units, 0.0)
        cannibalisation_loss = other_units * cannibalisation_rate * full_margin

        for depth in DISCOUNT_DEPTHS:
            log_uplift = segment_curve.get(depth)
            if log_uplift is None:
                continue
            uplift = float(np.expm1(log_uplift))

            promo_price = float(pair.regular_unit_price_gbp) * (1 - depth / 100)
            promo_margin = promo_price - float(pair.unit_cost_gbp)
            promoted_units = baseline_units * (1 + uplift)
            incremental_units = baseline_units * uplift

            # Budget consumption: the retailer-funded share of the discount given
            # away. This is what the promotional budget is measured in.
            promotion_cost = (
                promoted_units
                * (float(pair.regular_unit_price_gbp) - promo_price)
                * (1 - vendor_funded)
            )

            # Profit does NOT subtract promotion_cost again. The discount is
            # already reflected in promo_margin (promo price less unit cost);
            # deducting the give-away a second time would double-count it and
            # make every candidate look loss-making. The budget constrains the
            # spend, the objective measures the P&L outcome.
            profit_with = promoted_units * promo_margin - cannibalisation_loss
            profit_without = baseline_units * full_margin

            records.append({
                "store_id": pair.store_id,
                "sku_id": pair.sku_id,
                "category": pair.category,
                "price_elasticity_segment": pair.price_elasticity_segment,
                "discount_pct": depth,
                "baseline_units": baseline_units,
                "incremental_units": incremental_units,
                "promoted_units": promoted_units,
                "full_margin": full_margin,
                "promo_margin": promo_margin,
                "margin_sacrificed": baseline_units * (full_margin - promo_margin),
                "promotion_cost": promotion_cost,
                "cannibalisation_loss": cannibalisation_loss,
                "incremental_profit": profit_with - profit_without,
            })
    return pd.DataFrame(records)


def solve(
    candidates: pd.DataFrame,
    budget: float,
    max_per_store: Optional[int] = None,
    max_per_store_category: Optional[int] = 3,
    time_limit_seconds: int = 120,
) -> Dict[str, object]:
    """
    Maximise total incremental profit subject to budget and capacity.

    Only candidates with positive incremental profit enter the program - a
    loss-making option can never improve the objective, and dropping them shrinks
    the model substantially.
    """
    viable = candidates[candidates["incremental_profit"] > 0].reset_index(drop=True)
    if viable.empty:
        return {
            "status": "no viable candidates", "plan": viable,
            "total_profit": 0.0, "total_cost": 0.0, "n_selected": 0,
            "n_candidates": int(len(candidates)), "n_viable": 0,
        }

    problem = pulp.LpProblem("promotion_budget", pulp.LpMaximize)
    choose = [
        pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(len(viable))
    ]

    problem += pulp.lpSum(
        choose[i] * float(viable.at[i, "incremental_profit"]) for i in range(len(viable))
    )
    problem += (
        pulp.lpSum(choose[i] * float(viable.at[i, "promotion_cost"]) for i in range(len(viable)))
        <= budget
    ), "budget"

    # At most one depth per store x SKU.
    for _, group in viable.groupby(["store_id", "sku_id"], sort=False):
        problem += pulp.lpSum(choose[i] for i in group.index) <= 1

    if max_per_store is not None:
        for _, group in viable.groupby("store_id", sort=False):
            problem += pulp.lpSum(choose[i] for i in group.index) <= max_per_store

    if max_per_store_category is not None:
        for _, group in viable.groupby(["store_id", "category"], sort=False):
            problem += pulp.lpSum(choose[i] for i in group.index) <= max_per_store_category

    LOGGER.info("Solving ILP: %d binaries, budget £%.0f", len(viable), budget)
    problem.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_seconds))

    selected = [i for i in range(len(viable)) if choose[i].value() and choose[i].value() > 0.5]
    plan = viable.loc[selected].reset_index(drop=True)

    return {
        "status": pulp.LpStatus[problem.status],
        "plan": plan,
        "total_profit": float(plan["incremental_profit"].sum()),
        "total_cost": float(plan["promotion_cost"].sum()),
        "budget": budget,
        "budget_used": float(plan["promotion_cost"].sum()) / max(budget, 1e-9),
        "n_selected": int(len(plan)),
        "n_candidates": int(len(candidates)),
        "n_viable": int(len(viable)),
    }


def evaluate_plan_under(
    plan: pd.DataFrame,
    pairs: pd.DataFrame,
    true_curve: Dict[str, Dict[int, float]],
    category_baseline: pd.DataFrame,
    duration_days: int = 8,
    vendor_funded: float = DEFAULT_VENDOR_FUNDED,
) -> Dict[str, float]:
    """
    Score an already-chosen plan against a different effect curve.

    This is what makes the estimate-quality experiment possible: build a plan
    using whatever the analyst believed, then compute what it would actually have
    delivered under the true promotional response.
    """
    if plan.empty:
        return {
            "realised_profit": 0.0, "n_promotions": 0, "n_loss_making": 0,
            "loss_from_bad_picks": 0.0, "spend": 0.0,
        }

    keys = plan[["store_id", "sku_id", "discount_pct"]]
    truth = build_candidates(pairs, true_curve, category_baseline, duration_days, vendor_funded)
    merged = keys.merge(truth, on=["store_id", "sku_id", "discount_pct"], how="left")

    realised = merged["incremental_profit"].fillna(0.0)
    return {
        "realised_profit": float(realised.sum()),
        "n_promotions": int(len(merged)),
        "n_loss_making": int((realised < 0).sum()),
        "loss_from_bad_picks": float(realised[realised < 0].sum()),
        "spend": float(merged["promotion_cost"].fillna(0.0).sum()),
    }


def curve_from_dose_response(
    dose_response: pd.DataFrame, segment_column: str = "price_elasticity_segment"
) -> Dict[str, Dict[int, float]]:
    """Turn a Phase 3 dose-response table into the mapping the optimiser wants."""
    curve: Dict[str, Dict[int, float]] = {}
    for segment, group in dose_response.groupby(segment_column, observed=True):
        curve[str(segment)] = {
            int(row.discount_pct): float(row.estimate) for row in group.itertuples(index=False)
        }
    return curve


def scale_curve(curve: Dict[str, Dict[int, float]], factor: float) -> Dict[str, Dict[int, float]]:
    """Scale every effect, for sensitivity to a systematically biased estimate."""
    return {
        segment: {depth: value * factor for depth, value in depths.items()}
        for segment, depths in curve.items()
    }


def plan_summary(plan: pd.DataFrame) -> pd.DataFrame:
    """Where the recommended budget goes."""
    if plan.empty:
        return pd.DataFrame()
    by_depth = plan.groupby("discount_pct").agg(
        promotions=("sku_id", "size"),
        spend=("promotion_cost", "sum"),
        profit=("incremental_profit", "sum"),
        incremental_units=("incremental_units", "sum"),
    ).reset_index()
    by_depth["profit_per_pound"] = by_depth["profit"] / by_depth["spend"].clip(lower=1e-9)
    return by_depth
