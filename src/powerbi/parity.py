"""
DAX / Python parity harness.

PROJECT_ARCHITECTURE.md §6 Phase 8 asks that "DAX measures should reproduce the
Python-calculated figures as a parity check". A parity check nobody can run is
just a promise, so this computes what every measure in `powerbi/measures.dax`
*should* return, straight from the exported CSVs, and writes the answers to
`powerbi/powerbi_data/dax_parity.csv`.

That gives two things:

* a test (`tests/test_powerbi.py`) that fails if the exported data drifts away
  from the figures the reports quote;
* a reference table a reviewer can hold beside Power BI Desktop and tick off,
  because the DAX is written against the same tables with the same filters.

What this does **not** do is execute DAX. Nothing here proves the DAX itself is
correct - only that the numbers it is supposed to produce are pinned down and
reproducible. Verifying the expressions needs Power BI Desktop, and the Phase 8
report says so plainly rather than implying otherwise.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerbi.export import output_dir  # noqa: E402

LOGGER = logging.getLogger("northstar.powerbi.parity")


def _load(name: str) -> pd.DataFrame:
    path = output_dir() / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python src/powerbi/export.py` first."
        )
    return pd.read_csv(path)


def compute_expected() -> pd.DataFrame:
    """Expected value of every DAX measure, computed from the exported tables."""
    fact = _load("fact_daily_category")
    causal = _load("causal_estimates")
    dose = _load("dose_response")
    spillover = _load("spillover")
    service = _load("service_levels")
    plan = _load("promo_plan")
    uncertainty = _load("promo_plan_uncertainty")
    economics = _load("promo_economics")

    rows: List[Dict[str, object]] = []

    def record(page: str, measure: str, value: float, unit: str, note: str) -> None:
        rows.append({
            "page": page, "measure": measure, "expected_value": float(value),
            "unit": unit, "note": note,
        })

    # ---- Executive summary ------------------------------------------------
    revenue = fact["revenue_gbp"].sum()
    profit = fact["gross_profit_gbp"].sum()
    units = fact["units_sold"].sum()
    sku_days = fact["sku_days"].sum()

    record("Executive Summary", "Total Revenue", revenue, "GBP", "SUM of revenue_gbp")
    record("Executive Summary", "Total Gross Profit", profit, "GBP", "SUM of gross_profit_gbp")
    record("Executive Summary", "Gross Margin %", profit / revenue * 100, "percent",
           "Gross profit over revenue")
    record("Executive Summary", "Units Sold", units, "units", "SUM of units_sold")
    record("Executive Summary", "Promotion Rate %",
           fact["promoted_sku_days"].sum() / sku_days * 100, "percent",
           "Promoted SKU-days over all SKU-days")
    record("Executive Summary", "Stockout Rate %",
           fact["stockout_sku_days"].sum() / sku_days * 100, "percent",
           "Stockout SKU-days over all SKU-days")
    record("Executive Summary", "Promoted Revenue Share %",
           fact["promoted_revenue_gbp"].sum() / revenue * 100, "percent",
           "Share of revenue earned on promotion")

    # ---- Promotion ROI ----------------------------------------------------
    naive = causal.loc[causal["method"] == "promoted rows vs all others"].iloc[0]
    best_did = causal.loc[
        causal["family"].eq("Difference-in-differences")
        & causal["method"].eq("DiD: uncannibalised + seasonal effects")
    ].iloc[0]

    record("Promotion ROI", "Naive Promo Lift %", naive["effect_pct"], "percent",
           "Uncorrected promoted-vs-not comparison")
    record("Promotion ROI", "Causal Promo Lift %", best_did["effect_pct"], "percent",
           "Best difference-in-differences specification")
    record("Promotion ROI", "True Promo Lift %", naive["true_effect_pct"], "percent",
           "Simulated truth - available only because the data is synthetic")
    record("Promotion ROI", "Naive Bias pp", naive["error_pp"], "pp",
           "Naive estimate less the truth")
    record("Promotion ROI", "Causal Bias pp", best_did["error_pp"], "pp",
           "Causal estimate less the truth")
    record("Promotion ROI", "Bias Removed %",
           (1 - abs(best_did["error_log"]) / abs(naive["error_log"])) * 100, "percent",
           "Share of naive bias the causal correction removes")

    # ---- Elasticity explorer ----------------------------------------------
    record("Elasticity Explorer", "Dose Response Points", len(dose), "count",
           "Segment x depth combinations estimated")
    record("Elasticity Explorer", "CI Coverage %",
           dose["ci_covers_truth"].astype(bool).mean() * 100, "percent",
           "Share of intervals covering the simulated value")
    record("Elasticity Explorer", "Max Recovery Error",
           dose["error"].abs().max(), "log points",
           "Largest absolute deviation from the truth")
    at_20 = dose.loc[dose["discount_pct"] == 20]
    record("Elasticity Explorer", "Mean Lift at 20% Discount",
           at_20["estimated_lift_pct"].mean(), "percent",
           "Average estimated uplift at a 20% discount")

    # ---- Stockout risk & replenishment ------------------------------------
    record("Stockout Risk", "Cannibalisation 1 Neighbour %",
           float(np.expm1(spillover["estimate"].iloc[0]) * 100), "percent",
           "Effect on an untreated SKU when one category neighbour is promoted")
    record("Stockout Risk", "Cannibalisation 4+ Neighbours %",
           float(np.expm1(spillover["estimate"].iloc[-1]) * 100), "percent",
           "Effect at four or more concurrent promotions")
    record("Stockout Risk", "Median Service Level %",
           service["median_service_level"].median() * 100, "percent",
           "Median cost-derived service level across categories")
    record("Stockout Risk", "Lowest Service Level Category",
           service["median_service_level"].min() * 100, "percent",
           "The category the newsvendor ratio pushes lowest")

    # ---- What-if promotion simulator --------------------------------------
    record("What-If Simulator", "Promotions Selected", len(plan), "count",
           "Rows in the optimiser's recommended plan")
    record("What-If Simulator", "Plan Spend", plan["promotion_cost"].sum(), "GBP",
           "Retailer-funded give-away in the plan")
    record("What-If Simulator", "Plan Incremental Profit",
           plan["incremental_profit"].sum(), "GBP",
           "Deterministic profit the optimiser reports")
    record("What-If Simulator", "Plan Profit P10", uncertainty["p10"].iloc[0], "GBP",
           "10th percentile of the Monte Carlo distribution")
    record("What-If Simulator", "Plan Profit P50", uncertainty["p50"].iloc[0], "GBP",
           "Median simulated outcome")
    record("What-If Simulator", "Plan Profit P90", uncertainty["p90"].iloc[0], "GBP",
           "90th percentile")
    record("What-If Simulator", "Probability of Loss %",
           uncertainty["probability_of_loss"].iloc[0] * 100, "percent",
           "Share of draws below zero")
    record("What-If Simulator", "Candidates Profitable %",
           economics["profitable"].sum() / economics["candidates"].sum() * 100, "percent",
           "Share of candidate promotions clearing zero after cannibalisation")

    # The scenario measure is the one piece of non-trivial DAX arithmetic here,
    # so its identity is pinned: at a multiplier of 1 it must reproduce the
    # optimiser's own figure. If it does not, the expression double-counts
    # something - which is exactly the bug Phase 6 shipped and fixed.
    record("What-If Simulator", "Scenario Profit at Multiplier 1",
           _scenario_profit(plan, 1.0), "GBP",
           "Must equal Plan Incremental Profit; guards against double-counting the give-away")
    record("What-If Simulator", "Scenario Profit at Multiplier 0.88",
           _scenario_profit(plan, 0.88), "GBP",
           "Phase 4 concluded the uplift estimate is an upper bound, so 0.88 is the central case")

    return pd.DataFrame(rows)


def _scenario_profit(plan: pd.DataFrame, multiplier: float) -> float:
    """
    Python mirror of the `Scenario Incremental Profit` DAX measure.

    Promoted volume at the discounted margin, less baseline volume at the full
    margin, less cannibalisation. The give-away is deliberately not subtracted
    again - it is already inside promo_margin.
    """
    if plan.empty:
        return 0.0
    return float(
        (
            (plan["baseline_units"] + plan["incremental_units"] * multiplier)
            * plan["promo_margin"]
            - plan["baseline_units"] * plan["full_margin"]
            - plan["cannibalisation_loss"]
        ).sum()
    )


def build() -> Path:
    expected = compute_expected()
    path = output_dir() / "dax_parity.csv"
    expected.assign(expected_value=expected["expected_value"].round(4)).to_csv(path, index=False)
    LOGGER.info("Wrote %d measure expectations to %s", len(expected), path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    table = compute_expected()
    build()
    print()
    for page, group in table.groupby("page", sort=False):
        print(f"{page}")
        for row in group.itertuples(index=False):
            print(f"   {row.measure:<34} {row.expected_value:>16,.4f}  {row.unit}")
        print()
