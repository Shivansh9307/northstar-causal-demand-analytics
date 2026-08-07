"""
Monte Carlo evaluation of a promotion plan.

A single profit number for a plan is a fiction. The plan rests on three estimated
quantities, each with its own uncertainty, and Phase 4 established that one of
them is biased in a known direction:

* **Promotional response.** Phase 4's best DiD estimate came in above the
  simulated truth (0.675 against 0.593 log points) and the report concluded it
  should be read as an **upper bound**, because the pre-period leads drift
  upward. The prior here is therefore deliberately asymmetric - centred below the
  point estimate, with more mass on the downside.
* **Baseline demand.** Phase 5's held-out forecast error gives the spread.
* **Cannibalisation.** Phase 3 measured 6% for the first concurrent promotion,
  but with a confidence interval and a saturating shape.

Sampling all three produces a profit *distribution*, which is what
PROJECT_ARCHITECTURE.md §6 Phase 6 asks for. The P10 is the number worth quoting
to a finance director; the mean is the number that gets a plan approved and then
missed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOGGER = logging.getLogger("promopulse.optimization.montecarlo")


def simulate_plan(
    plan: pd.DataFrame,
    n_draws: int = 4000,
    effect_bias: float = 0.88,
    effect_sd: float = 0.18,
    demand_sd: float = 0.30,
    cannibalisation_sd: float = 0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Draw `n_draws` profit outcomes for a fixed plan.

    Parameters
    ----------
    effect_bias
        Multiplier applied to the estimated promotional uplift. The default 0.88
        reflects Phase 4's finding that the DiD estimate overshot the simulated
        truth by roughly 12% on the log scale, so the central case assumes the
        estimate is optimistic rather than unbiased.
    effect_sd, demand_sd, cannibalisation_sd
        Lognormal spreads for the three uncertain inputs.
    """
    if plan.empty:
        return pd.DataFrame({"profit": []})

    rng = np.random.default_rng(seed)
    n = len(plan)

    baseline = plan["baseline_units"].to_numpy()
    incremental = plan["incremental_units"].to_numpy()
    promo_margin = plan["promo_margin"].to_numpy()
    full_margin = plan["full_margin"].to_numpy()
    cannibalisation = plan["cannibalisation_loss"].to_numpy()

    profits = np.empty(n_draws)
    for draw in range(n_draws):
        # Lognormal keeps every multiplier positive and right-skewed.
        effect_shock = effect_bias * rng.lognormal(-0.5 * effect_sd**2, effect_sd, n)
        demand_shock = rng.lognormal(-0.5 * demand_sd**2, demand_sd, n)
        cannibal_shock = rng.lognormal(-0.5 * cannibalisation_sd**2, cannibalisation_sd, n)

        realised_baseline = baseline * demand_shock
        realised_incremental = incremental * effect_shock * demand_shock
        realised_promoted = realised_baseline + realised_incremental

        # Mirrors promo_lp.build_candidates: the discount is already in
        # promo_margin, so the give-away is not deducted a second time.
        profit_with = realised_promoted * promo_margin - cannibalisation * cannibal_shock
        profit_without = realised_baseline * full_margin
        profits[draw] = float((profit_with - profit_without).sum())

    return pd.DataFrame({"profit": profits})


def summarise(draws: pd.DataFrame, deterministic_profit: float) -> Dict[str, float]:
    """Percentiles and downside risk, rather than a point estimate."""
    if draws.empty:
        return {}
    profit = draws["profit"].to_numpy()
    return {
        "deterministic": float(deterministic_profit),
        "mean": float(profit.mean()),
        "p10": float(np.percentile(profit, 10)),
        "p50": float(np.percentile(profit, 50)),
        "p90": float(np.percentile(profit, 90)),
        "std": float(profit.std()),
        "probability_of_loss": float((profit < 0).mean()),
        "probability_below_deterministic": float((profit < deterministic_profit).mean()),
        "shortfall_vs_deterministic": float(deterministic_profit - np.percentile(profit, 50)),
    }


def sensitivity(
    plan: pd.DataFrame,
    deterministic_profit: float,
    effect_biases: List[float],
    n_draws: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    How the recommendation holds up if the promotional effect estimate is wrong
    by more or less than Phase 4 suggested.
    """
    rows = []
    for bias in effect_biases:
        draws = simulate_plan(plan, n_draws=n_draws, effect_bias=bias, seed=seed)
        summary = summarise(draws, deterministic_profit)
        rows.append({
            "effect_multiplier": bias,
            "mean_profit": summary["mean"],
            "p10": summary["p10"],
            "p90": summary["p90"],
            "probability_of_loss": summary["probability_of_loss"],
        })
    return pd.DataFrame(rows)
