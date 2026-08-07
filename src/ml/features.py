"""
Leakage-safe feature construction for demand forecasting.

The horizon is the whole problem
--------------------------------
Northstar's reorder lead times run 1-8 days, so a replenishment decision needs a
forecast roughly a week out. That makes the forecast target `units_sold` on day
`T + HORIZON`, with every feature computed from information available on day `T`.

This is where time-series feature engineering usually goes wrong. The panel ships
`lag_1_units_sold` and `rolling_7_day_avg_units_sold`, both computed from sales
strictly before the target day - which is correct for a *one-day* forecast and
completely wrong for a seven-day one. On day `T` you do not know day `T + 6`'s
sales. Using those columns as-is would leak six days of future information and
produce a model that looks excellent and cannot be deployed.

Every demand-history feature here is therefore shifted by `HORIZON` within its
store x SKU pair, so it only ever reads sales up to day `T`.

What *is* known in advance
--------------------------
Promotions are planned, not discovered: `fact_promotions` carries
`promotion_planned_flag`, and a retailer knows next week's promotional calendar
when it places replenishment orders. So the target day's promotion features -
flag, depth, mechanic, support channels, and the resulting shelf price - are
legitimately available at forecast time and are included.

Calendar and weather for the target day are treated the same way. Calendar is
known with certainty; a seven-day weather forecast is standard planning input,
and that assumption is stated in the report rather than buried here.

What is excluded
----------------
Everything downstream of the outcome: `potential_demand_units`,
`lost_sales_estimate_units`, the anomaly labels, stock positions on the target
day, and every ground-truth column. The assembled matrix is passed through the
§7 leakage checker before it is returned.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_quality import leakage  # noqa: E402
from features import star_schema  # noqa: E402

LOGGER = logging.getLogger("promopulse.ml.features")

HORIZON = 7

TARGET = "units_sold"

# Known with certainty when the order is placed.
CALENDAR_FEATURES = [
    "month",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "is_bank_holiday",
    "is_school_holiday",
    "is_payday_window",
    "is_month_end",
    "is_easter_period",
    "is_christmas_period",
    "is_black_friday_period",
]

# Planned in advance, so available at forecast time.
PROMOTION_FEATURES = [
    "promo_flag",
    "discount_pct",
    "display_support_flag",
    "email_or_app_support_flag",
    "leaflet_support_flag",
    "actual_unit_price_gbp",
    "regular_unit_price_gbp",
]

# Weather: a seven-day forecast is a normal planning input. Stated as an
# assumption in the report.
WEATHER_FEATURES = ["temperature_celsius", "rainfall_mm", "is_heatwave"]

STATIC_FEATURES = [
    "unit_cost_gbp",
    "baseline_gross_margin_pct",
    "shelf_life_days",
    "is_perishable",
    "minimum_display_stock",
    "reorder_lead_time_days",
    "average_daily_footfall",
    "competition_intensity_score",
    "store_income_index",
    "local_deprivation_decile",
    "floor_area_sqm",
]

CATEGORICAL_FEATURES = [
    "category",
    "brand_type",
    "seasonal_profile",
    "price_elasticity_segment",
    "promotion_sensitivity_segment",
    "demand_volatility_segment",
    "store_format",
    "region",
    "promo_type",
]

SOURCE_QUERY = """
    SELECT date, store_id, sku_id, units_sold, stockout_flag,
           promo_flag, promo_type, discount_pct,
           display_support_flag, email_or_app_support_flag, leaflet_support_flag,
           actual_unit_price_gbp, regular_unit_price_gbp,
           month, day_of_week, week_of_year, is_weekend, is_bank_holiday,
           is_school_holiday, is_payday_window, is_month_end,
           is_easter_period, is_christmas_period, is_black_friday_period,
           temperature_celsius, rainfall_mm, is_heatwave,
           category, brand_type, seasonal_profile, price_elasticity_segment,
           promotion_sensitivity_segment, demand_volatility_segment,
           unit_cost_gbp, baseline_gross_margin_pct, shelf_life_days, is_perishable,
           minimum_display_stock, reorder_lead_time_days,
           store_format, region, average_daily_footfall, competition_intensity_score,
           store_income_index, local_deprivation_decile, floor_area_sqm
    FROM analytics_daily
    ORDER BY store_id, sku_id, date
"""


def load_source() -> pd.DataFrame:
    """Pull the raw columns the feature builder needs, in pair-then-date order."""
    con = star_schema.connect()
    try:
        star_schema.load_raw_tables(con)
        star_schema.build_promotion_bridge(con)
        star_schema.build_analytics_view(con)
        return con.execute(SOURCE_QUERY).df()
    finally:
        con.close()


def build_features(frame: pd.DataFrame, horizon: int = HORIZON) -> Tuple[pd.DataFrame, List[str]]:
    """
    Assemble the model matrix for a `horizon`-day-ahead forecast.

    Returns the frame and the list of feature column names. Rows without enough
    history to populate the lags are dropped rather than imputed - filling them
    would invent demand history that did not exist.
    """
    work = frame.sort_values(["store_id", "sku_id", "date"]).copy()
    work["pair_id"] = work["store_id"] + "|" + work["sku_id"]
    grouped = work.groupby("pair_id", sort=False)[TARGET]

    features: List[str] = []

    # --- demand history, every term shifted by the horizon ------------------
    # `shift(horizon)` on the target day T+h reads day T. Anything less would be
    # reading sales that have not happened when the order is placed.
    for extra in (0, 7, 14):
        name = f"sales_lag_{horizon + extra}"
        work[name] = grouped.shift(horizon + extra)
        features.append(name)

    shifted = grouped.shift(horizon)
    by_pair = shifted.groupby(work["pair_id"], sort=False)
    for window in (7, 28):
        mean_name = f"sales_roll{window}_mean"
        work[mean_name] = by_pair.transform(lambda s, w=window: s.rolling(w, min_periods=w).mean())
        features.append(mean_name)
    work["sales_roll7_std"] = by_pair.transform(
        lambda s: s.rolling(7, min_periods=7).std()
    )
    features.append("sales_roll7_std")

    # Trend and volatility, expressed as ratios so they carry across SKU scales.
    work["sales_trend_ratio"] = work["sales_roll7_mean"] / work["sales_roll28_mean"].clip(lower=0.1)
    work["sales_cv"] = work["sales_roll7_std"] / work["sales_roll7_mean"].clip(lower=0.1)
    features += ["sales_trend_ratio", "sales_cv"]

    # Same weekday a week before the last observed day, for weekly shape.
    work["sales_same_dow"] = grouped.shift(horizon + 7 - (horizon % 7))
    features.append("sales_same_dow")

    # Promotion history: was this pair promoted in the observed window? Captures
    # post-promotion dips without reading the target day's outcome.
    promo_shifted = work.groupby("pair_id", sort=False)["promo_flag"].shift(horizon)
    work["promo_days_last_28"] = (
        promo_shifted.groupby(work["pair_id"], sort=False)
        .transform(lambda s: s.rolling(28, min_periods=28).sum())
    )
    features.append("promo_days_last_28")

    # --- known-at-forecast-time features ------------------------------------
    work["discount_depth"] = work["discount_pct"] / 100.0
    work["price_ratio"] = work["actual_unit_price_gbp"] / work["regular_unit_price_gbp"]
    features += ["discount_depth", "price_ratio"]

    # These arrive numeric (or boolean) from DuckDB; booleans are cast below.
    for column in CALENDAR_FEATURES + PROMOTION_FEATURES + WEATHER_FEATURES + STATIC_FEATURES:
        if column == "discount_pct":  # superseded by discount_depth
            continue
        features.append(column)

    for column in CATEGORICAL_FEATURES:
        work[column] = work[column].astype("category")
        features.append(column)

    # Cyclical calendar encodings help the linear models; trees ignore them.
    work["dow_sin"] = np.sin(2 * np.pi * work["day_of_week"] / 7)
    work["dow_cos"] = np.cos(2 * np.pi * work["day_of_week"] / 7)
    work["month_sin"] = np.sin(2 * np.pi * work["month"] / 12)
    work["month_cos"] = np.cos(2 * np.pi * work["month"] / 12)
    features += ["dow_sin", "dow_cos", "month_sin", "month_cos"]

    features = list(dict.fromkeys(features))
    for column in features:
        if str(work[column].dtype) == "bool":
            work[column] = work[column].astype(int)

    before = len(work)
    work = work.dropna(subset=features + [TARGET]).reset_index(drop=True)
    LOGGER.info(
        "Dropped %d of %d rows lacking %d days of history", before - len(work), before, horizon + 28
    )

    leakage.assert_no_leakage(features, context=f"{horizon}-day forecast feature set")
    return work, features


def seasonal_naive(frame: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """
    Baseline: predict the same weekday from the last fully observed week.

    With a seven-day horizon, `shift(7)` is both the most recent observation
    available and the same day of week, which is why this is the right naive
    comparator rather than a flat mean.
    """
    return frame.groupby("pair_id", sort=False)[TARGET].shift(horizon)


def time_split(
    frame: pd.DataFrame, n_folds: int = 4, holdout_days: int = 91
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Expanding-window splits plus a final untouched holdout.

    Every split is strictly chronological. A shuffled split on this panel would
    put a store x SKU's future beside its own past and inflate every metric.
    """
    dates = pd.to_datetime(frame["date"])
    last_date = dates.max()
    holdout_start = last_date - pd.Timedelta(days=holdout_days - 1)

    development = dates < holdout_start
    holdout_index = np.flatnonzero(~development)

    dev_dates = dates[development]
    span_start, span_end = dev_dates.min(), dev_dates.max()
    total_days = (span_end - span_start).days
    step = total_days // (n_folds + 1)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(1, n_folds + 1):
        train_end = span_start + pd.Timedelta(days=step * fold)
        valid_end = span_start + pd.Timedelta(days=step * (fold + 1))
        # A gap of `horizon` days: the model must not train on days whose outcome
        # would not yet be known when the validation forecast is made.
        gap_end = train_end + pd.Timedelta(days=HORIZON)
        train_index = np.flatnonzero(development & (dates <= train_end))
        valid_index = np.flatnonzero(development & (dates > gap_end) & (dates <= valid_end))
        if len(valid_index):
            folds.append((train_index, valid_index))
    return folds, holdout_index


def describe_split(frame: pd.DataFrame, folds, holdout: np.ndarray) -> pd.DataFrame:
    """Human-readable summary of the chronological splits, for the report."""
    dates = pd.to_datetime(frame["date"])
    rows: List[Dict[str, object]] = []
    for i, (train, valid) in enumerate(folds, start=1):
        rows.append({
            "fold": f"CV fold {i}",
            "train_start": dates.iloc[train].min().date(),
            "train_end": dates.iloc[train].max().date(),
            "valid_start": dates.iloc[valid].min().date(),
            "valid_end": dates.iloc[valid].max().date(),
            "train_rows": len(train),
            "valid_rows": len(valid),
        })
    rows.append({
        "fold": "Holdout",
        "train_start": dates.iloc[:0].min() if False else dates.min().date(),
        "train_end": (dates.iloc[holdout].min() - pd.Timedelta(days=1)).date(),
        "valid_start": dates.iloc[holdout].min().date(),
        "valid_end": dates.iloc[holdout].max().date(),
        "train_rows": len(frame) - len(holdout),
        "valid_rows": len(holdout),
    })
    return pd.DataFrame(rows)
