"""
Build the Power BI data layer.

Design decision: aggregate before export
----------------------------------------
The raw fact table is 2.19M rows and 468MB of CSV. Pointing a semantic model at
that would make the repository unclonable and the report slow, for no analytical
gain - none of the five pages needs SKU-level daily detail.

So the fact grain here is **date x store x category** (146,200 rows, a few MB),
which supports every visual the report specification calls for, and the *results*
of Phases 3-6 ship as their own small tables. Those results tables are the point
of the decision layer: a hiring manager opening this should see the naive and
causal promotion estimates side by side, not be handed a pile of transactions.

Every file written here is small enough to commit, so the model loads from a
fresh clone without regenerating 626MB of source data first.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal import did, estimands  # noqa: E402
from features import star_schema  # noqa: E402
from optimization import inventory, monte_carlo, promo_lp  # noqa: E402
from stats import models  # noqa: E402
from utils import config  # noqa: E402

LOGGER = logging.getLogger("promopulse.powerbi.export")

OUTPUT_DIRNAME = "powerbi_data"


def output_dir() -> Path:
    path = Path(config.PROJECT_ROOT) / "powerbi" / OUTPUT_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(frame: pd.DataFrame, name: str, decimals: int = 4) -> Path:
    """
    Write a table, rounding floats first.

    Full float64 precision roughly doubles the file size and means nothing here -
    no visual distinguishes £1,234.5678 from £1,234.5678000001, and the repository
    has to stay clonable.
    """
    path = output_dir() / f"{name}.csv"
    rounded = frame.copy()
    float_columns = rounded.select_dtypes(include=["float64", "float32"]).columns
    rounded[float_columns] = rounded[float_columns].round(decimals)
    rounded.to_csv(path, index=False)
    LOGGER.info("%-26s %7d rows  %7.1f KB", name, len(rounded), path.stat().st_size / 1024)
    return path


# ---------------------------------------------------------------------------
# Dimensions and the aggregated fact
# ---------------------------------------------------------------------------

def export_dimensions(con) -> Dict[str, pd.DataFrame]:
    store = con.execute(
        """
        SELECT store_id, store_name, city, region, country, store_format,
               floor_area_sqm, average_daily_footfall, competition_intensity_score,
               store_income_index, local_deprivation_decile, rollout_cohort
        FROM dim_store ORDER BY store_id
        """
    ).df()
    product = con.execute(
        """
        SELECT sku_id, sku_name, category, subcategory, brand_type,
               regular_unit_price_gbp, unit_cost_gbp, baseline_gross_margin_pct,
               price_elasticity_segment, promotion_sensitivity_segment,
               demand_volatility_segment, seasonal_profile, is_perishable,
               shelf_life_days, reorder_lead_time_days
        FROM dim_product ORDER BY sku_id
        """
    ).df()
    calendar = con.execute(
        """
        SELECT date, year, quarter, month, month_name, week_of_year, day_of_week,
               day_name, is_weekend, is_month_end, is_payday_window, is_school_holiday,
               is_bank_holiday, bank_holiday_name, is_easter_period, is_christmas_period,
               is_black_friday_period, is_heatwave, weather_condition, temperature_celsius
        FROM dim_calendar ORDER BY date
        """
    ).df()

    # Category is the fact's grain but is not unique in dim_product, so it needs
    # its own dimension for the relationship to be one-to-many.
    category = con.execute(
        """
        SELECT category,
               COUNT(*) AS skus,
               AVG(baseline_gross_margin_pct) AS mean_margin_pct,
               SUM(CASE WHEN is_perishable THEN 1 ELSE 0 END) AS perishable_skus
        FROM dim_product GROUP BY 1 ORDER BY 1
        """
    ).df()

    _write(store, "dim_store")
    _write(product, "dim_product")
    _write(calendar, "dim_calendar")
    _write(category, "dim_category")
    return {"store": store, "product": product, "calendar": calendar, "category": category}


def export_fact(con) -> pd.DataFrame:
    """
    Daily fact aggregated to date x store x category.

    `potential_demand_units` and `lost_sales_estimate_units` are deliberately
    excluded: §7 bars them from any model, and a BI surface is exactly where a
    latent-demand column would get mistaken for something a planner can act on.
    Lost sales appear only as a derived cost measure, computed here and labelled
    as a simulation quantity.
    """
    fact = con.execute(
        """
        SELECT
            date,
            store_id,
            category,
            SUM(units_sold)                                  AS units_sold,
            SUM(sales_revenue_gbp)                           AS revenue_gbp,
            SUM(cost_of_goods_sold_gbp)                      AS cogs_gbp,
            SUM(retailer_promo_cost_gbp)                     AS promo_cost_gbp,
            SUM(gross_profit_gbp)                            AS gross_profit_gbp,
            SUM(CASE WHEN promo_flag THEN 1 ELSE 0 END)      AS promoted_sku_days,
            COUNT(*)                                         AS sku_days,
            SUM(CASE WHEN stockout_flag THEN 1 ELSE 0 END)   AS stockout_sku_days,
            SUM(CASE WHEN promo_flag THEN units_sold ELSE 0 END)   AS promoted_units,
            SUM(CASE WHEN promo_flag THEN sales_revenue_gbp ELSE 0 END) AS promoted_revenue_gbp,
            SUM(waste_units)                                 AS waste_units,
            AVG(discount_pct)                                AS mean_discount_pct
        FROM analytics_daily
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    ).df()
    # Currency to the penny, rates to two decimals - anything finer is noise.
    _write(fact, "fact_daily_category", decimals=2)
    return fact


# ---------------------------------------------------------------------------
# Results tables — the actual decision layer
# ---------------------------------------------------------------------------

def export_causal_estimates(panel: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Naive vs DiD vs IPW against the simulated truth (Phase 4)."""
    counterfactual_target = _phase4_target(panel, ground_truth)
    naive = did.naive_estimates(panel)
    variants = did.did_variants(panel)

    rows: List[Dict[str, object]] = []
    for record in naive.itertuples(index=False):
        rows.append({
            "method": record.estimator.replace("Naive: ", ""),
            "family": "Naive",
            "log_effect": record.estimate,
            "ci_low": np.nan, "ci_high": np.nan,
        })
    labels = {
        "twfe_all": "DiD: all untreated controls",
        "twfe_cannibal_ctrl": "DiD: + concurrent-promotion control",
        "twfe_never_treated": "DiD: never-promoted controls",
        "twfe_out_of_category": "DiD: uncannibalised controls",
        "twfe_clean_seasonal_fe": "DiD: uncannibalised + seasonal effects",
    }
    for record in variants.itertuples(index=False):
        rows.append({
            "method": labels.get(record.estimator, record.estimator),
            "family": "Difference-in-differences",
            "log_effect": record.estimate,
            "ci_low": record.ci_low, "ci_high": record.ci_high,
        })

    frame = pd.DataFrame(rows)
    frame["effect_pct"] = np.expm1(frame["log_effect"]) * 100
    frame["true_log_effect"] = counterfactual_target
    frame["true_effect_pct"] = float(np.expm1(counterfactual_target) * 100)
    frame["error_log"] = frame["log_effect"] - counterfactual_target
    frame["error_pp"] = frame["effect_pct"] - frame["true_effect_pct"]
    return _write(frame, "causal_estimates") and frame


def _phase4_target(panel: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """The row-level counterfactual benchmark Phase 4 established."""
    con = star_schema.connect()
    try:
        extra = con.execute(
            """
            SELECT date, store_id, sku_id, potential_demand_units,
                   opening_stock_units, delivery_units
            FROM analytics_daily ORDER BY date, store_id, sku_id
            """
        ).df()
    finally:
        con.close()
    merged = panel.merge(extra, on=["date", "store_id", "sku_id"])
    multiplier = estimands.treatment_multiplier(merged, ground_truth)
    treated = merged["promo_flag"] == 1
    counterfactual = np.minimum(
        merged.loc[treated, "potential_demand_units"] / multiplier[treated],
        merged.loc[treated, "opening_stock_units"] + merged.loc[treated, "delivery_units"],
    )
    return float(
        (np.log1p(merged.loc[treated, "units_sold"]) - np.log1p(counterfactual)).mean()
    )


def export_dose_response(panel: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Validated promotional dose-response by segment (Phase 3)."""
    by_segment = models.dose_response_by_group(
        panel, "price_elasticity_segment", ground_truth
    )
    frame = by_segment.rename(columns={"price_elasticity_segment": "segment"})
    frame["estimated_lift_pct"] = np.expm1(frame["estimate"]) * 100
    frame["true_lift_pct"] = np.expm1(frame["true_effect"]) * 100
    # Integer rather than boolean: Power BI infers a TRUE/FALSE text column from
    # CSV, which makes every DAX filter on it a string comparison.
    frame["ci_covers_truth"] = frame["ci_covers_truth"].astype(int)
    columns = [
        "segment", "discount_pct", "estimate", "ci_low", "ci_high", "true_effect",
        "error", "ci_covers_truth", "estimated_lift_pct", "true_lift_pct",
    ]
    _write(frame[columns], "dose_response")
    return frame


def export_spillover(panel: pd.DataFrame) -> pd.DataFrame:
    """Cannibalisation onto untreated neighbours (Phase 3)."""
    frame = models.spillover_diagnostic(panel)
    # The last bin is labelled "4+", so the column is mixed int/str. Cast to text
    # so Power BI types it consistently instead of coercing the bin labels.
    frame["others_on_promo"] = frame["others_on_promo"].astype(str)
    frame["effect_pct"] = np.expm1(frame["estimate"]) * 100
    _write(frame, "spillover")
    return frame


def export_service_levels(pairs: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    """Cost-derived service levels and reorder points (Phase 6)."""
    policy = inventory.build_policy(pairs, sigma, ["demand_volatility_segment", "promo_flag"])
    summary = (
        policy.groupby("category_label", observed=True)
        .agg(
            pairs=("sku_id", "size"),
            median_service_level=("service_level", "median"),
            mean_safety_stock=("safety_stock", "mean"),
            mean_reorder_point=("reorder_point", "mean"),
        )
        .reset_index()
        .rename(columns={"category_label": "category"})
        .sort_values("median_service_level")
    )
    _write(summary, "service_levels")
    _write(
        policy[["store_id", "sku_id", "category_label", "service_level",
                "safety_stock", "reorder_point", "is_perishable"]]
        .rename(columns={"category_label": "category"}),
        "reorder_policy",
    )
    return summary


def export_promo_plan(
    pairs: pd.DataFrame, curve, category_baseline: pd.DataFrame, budget: float
) -> pd.DataFrame:
    """The recommended promotion plan and its Monte Carlo range (Phase 6)."""
    candidates = promo_lp.build_candidates(pairs, curve, category_baseline)
    solution = promo_lp.solve(candidates, budget=budget, max_per_store_category=3)
    plan = solution["plan"]

    if not plan.empty:
        # promo_margin and full_margin ship too: the what-if simulator needs to
        # recompute profit at a different uplift, and doing that in DAX requires
        # both margins, not just the resulting profit.
        _write(
            plan[["store_id", "sku_id", "category", "discount_pct", "baseline_units",
                  "incremental_units", "promo_margin", "full_margin",
                  "promotion_cost", "cannibalisation_loss", "incremental_profit"]],
            "promo_plan",
        )
        draws = monte_carlo.simulate_plan(plan, n_draws=4000, seed=config.seed())
        summary = monte_carlo.summarise(draws, solution["total_profit"])
        _write(pd.DataFrame([summary]), "promo_plan_uncertainty")
        _write(
            pd.DataFrame({"draw": np.arange(len(draws)), "profit_gbp": draws["profit"]}),
            "promo_plan_draws",
        )

    # Candidate economics, so the report can show how few promotions pay.
    economics = candidates.groupby("discount_pct").agg(
        candidates=("incremental_profit", "size"),
        profitable=("incremental_profit", lambda s: int((s > 0).sum())),
        mean_incremental_profit=("incremental_profit", "mean"),
        mean_margin_sacrificed=("margin_sacrificed", "mean"),
        mean_cannibalisation=("cannibalisation_loss", "mean"),
    ).reset_index()
    _write(economics, "promo_economics")
    return plan


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build() -> Dict[str, object]:
    con = star_schema.connect()
    try:
        star_schema.load_raw_tables(con)
        star_schema.build_promotion_bridge(con)
        star_schema.build_analytics_view(con)
        LOGGER.info("Exporting dimensions and aggregated fact")
        export_dimensions(con)
        fact = export_fact(con)
    finally:
        con.close()

    ground_truth = estimands.load_ground_truth()
    panel = models.prepare(models.load_analysis_frame())

    LOGGER.info("Exporting causal estimates")
    causal = export_causal_estimates(panel, ground_truth)
    LOGGER.info("Exporting dose-response and spillover")
    dose = export_dose_response(panel, ground_truth)
    export_spillover(panel)

    LOGGER.info("Building optimisation inputs")
    recent = panel[panel["date"] >= panel["date"].max() - pd.Timedelta(days=90)]
    pairs = (
        recent[recent["promo_flag"] == 0]
        .groupby(["store_id", "sku_id"], observed=True)
        .agg(mean_daily_units=("units_sold", "mean")).reset_index()
        .merge(
            panel.groupby(["store_id", "sku_id"], observed=True).agg(
                category=("category", "first"),
                price_elasticity_segment=("price_elasticity_segment", "first"),
                demand_volatility_segment=("demand_volatility_segment", "first"),
                regular_unit_price_gbp=("regular_unit_price_gbp", "first"),
            ).reset_index(),
            on=["store_id", "sku_id"],
        )
    )
    products = pd.read_csv(config.path("raw") / "dim_product.csv", keep_default_na=False)
    pairs = pairs.merge(
        products[["sku_id", "unit_cost_gbp", "shelf_life_days", "is_perishable",
                  "reorder_lead_time_days"]],
        on="sku_id",
    )
    pairs["is_perishable"] = pairs["is_perishable"].astype(str).str.lower().eq("true")
    pairs["unit_margin"] = pairs["regular_unit_price_gbp"] - pairs["unit_cost_gbp"]
    pairs["promo_flag"] = 0

    # Forecast-error spread, reusing the Phase 5 holdout rather than refitting.
    sigma = _forecast_sigma()
    LOGGER.info("Exporting service levels")
    export_service_levels(pairs, sigma)

    category_baseline = (
        pairs.groupby(["store_id", "category"], observed=True)["mean_daily_units"]
        .sum().reset_index().rename(columns={"mean_daily_units": "baseline_units_per_day"})
    )
    promotions = pd.read_csv(config.path("raw") / "fact_promotions.csv")
    budget = float(
        (promotions["promotion_cost_gbp"] * (1 - promotions["vendor_funded_pct"] / 100)).sum() / 8
    )
    curve = promo_lp.curve_from_dose_response(
        dose.rename(columns={"segment": "price_elasticity_segment"})
    )
    LOGGER.info("Exporting promotion plan")
    export_promo_plan(pairs, curve, category_baseline, budget)

    return {"fact_rows": len(fact), "causal_rows": len(causal), "budget": budget}


def _forecast_sigma() -> pd.DataFrame:
    """
    Forecast-error standard deviation by segment.

    Refits the Phase 5 holdout model. That is the expensive step in this export
    (~2 minutes) but the alternative is hard-coding numbers out of a markdown
    report, which would silently go stale.
    """
    from ml import features as ml_features
    from ml import forecast as ml_forecast

    source = ml_features.load_source()
    frame, feature_names = ml_features.build_features(source)
    del source
    _, holdout_index = ml_features.time_split(frame)
    _, _, _, predictions = ml_forecast.run_holdout(
        frame, feature_names, list(ml_features.CATEGORICAL_FEATURES), holdout_index
    )
    return inventory.forecast_error_sigma(
        frame.iloc[holdout_index], predictions, ["demand_volatility_segment", "promo_flag"]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = build()
    print(f"\nPower BI data layer written to {output_dir()}")
    print(f"  aggregated fact rows: {result['fact_rows']:,}")
