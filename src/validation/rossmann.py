"""
Rossmann Store Sales adapter — the "swapped data loader" of §6 Phase 7.

The point of this module is that it is *only* a loader. It produces a frame with
the same column contract the Northstar pipeline expects, so
`ml/forecast.run_cross_validation` and `run_holdout` execute unchanged against
real data. If the method only worked on data built to be solvable, that would
show up here.

What Rossmann is
----------------
1,017,209 store-days: 1,115 German drugstores, 2013-01-01 to 2015-07-31, with a
daily binary promotion flag, holiday indicators, and store attributes.

What Rossmann is not
--------------------
**There is no price, discount, margin or cost column anywhere in the dataset.**
Not a missing column - the concept does not exist in it. Sales are recorded in
euros at the store-day level with no product dimension at all.

That single fact decides most of Phase 7. The elasticity and dose-response work
of Phase 3 is built entirely on discount depth varying across promotions; with a
binary flag and no price, none of it can be estimated. This is not a limitation
of the method - it is what most real retail data looks like, and saying so is
the honest result.

Two structural differences that matter for the transfer
-------------------------------------------------------
* **Grain.** Northstar is date x store x SKU; Rossmann is date x store. The
  store is the panel unit, so `pair_id` maps to the store.
* **Closed days.** 17% of rows have `Open = 0` and zero sales. Those are
  calendar-predictable and trivially forecastable, so including them would
  flatter every metric. They are kept in the series for lag construction - a
  forecaster really does observe a zero on a Sunday - but excluded from
  evaluation, which is the convention the original competition used.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_quality import leakage  # noqa: E402
from utils import config  # noqa: E402

LOGGER = logging.getLogger("promopulse.validation.rossmann")

HORIZON = 7
TARGET = "sales"

# Known when the order is placed: the promotional and holiday calendar.
KNOWN_IN_ADVANCE = [
    "promo",
    "school_holiday",
    "is_state_holiday",
    "open",
    "day_of_week",
    "month",
    "week_of_year",
    "is_weekend",
    "day_of_month",
]

STATIC_FEATURES = [
    "competition_distance",
    "competition_open_months",
    "promo2_active",
    "promo2_months",
]

CATEGORICAL_FEATURES = ["store_type", "assortment", "state_holiday"]


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two competition files from data/external."""
    external = config.path("external")
    train_path = external / "train.csv"
    store_path = external / "store.csv"
    for path in (train_path, store_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Phase 7 needs the Rossmann Store Sales files "
                "(train.csv, store.csv) in data/external/."
            )
    sales = pd.read_csv(train_path, parse_dates=["Date"], dtype={"StateHoliday": str})
    stores = pd.read_csv(store_path)
    return sales, stores


def build_panel() -> pd.DataFrame:
    """Merge, rename to the pipeline's vocabulary, and derive store attributes."""
    sales, stores = load_raw()
    frame = sales.merge(stores, on="Store", how="left")

    panel = pd.DataFrame({
        "date": frame["Date"],
        "store_id": frame["Store"].astype(str),
        "sales": frame["Sales"].astype(float),
        "customers": frame["Customers"].astype(float),
        "open": frame["Open"].astype(int),
        "promo": frame["Promo"].astype(int),
        "school_holiday": frame["SchoolHoliday"].astype(int),
        "state_holiday": frame["StateHoliday"].fillna("0").astype(str),
        "day_of_week": frame["DayOfWeek"].astype(int),
        "store_type": frame["StoreType"].astype(str),
        "assortment": frame["Assortment"].astype(str),
        "competition_distance": frame["CompetitionDistance"].astype(float),
    })
    panel["is_state_holiday"] = (panel["state_holiday"] != "0").astype(int)
    panel["month"] = panel["date"].dt.month
    panel["week_of_year"] = panel["date"].dt.isocalendar().week.astype(int)
    panel["day_of_month"] = panel["date"].dt.day
    panel["is_weekend"] = panel["day_of_week"].isin([6, 7]).astype(int)
    panel["pair_id"] = panel["store_id"]

    # Median-fill the 3 stores with no recorded competitor distance rather than
    # dropping them; the flag below preserves that they were missing.
    panel["competition_distance_missing"] = panel["competition_distance"].isna().astype(int)
    panel["competition_distance"] = panel["competition_distance"].fillna(
        panel["competition_distance"].median()
    )

    competition_open = pd.to_datetime(
        dict(
            year=frame["CompetitionOpenSinceYear"].fillna(1900).astype(int),
            month=frame["CompetitionOpenSinceMonth"].fillna(1).astype(int),
            day=1,
        ),
        errors="coerce",
    )
    panel["competition_open_months"] = (
        (panel["date"] - competition_open).dt.days / 30.44
    ).clip(lower=0).fillna(0)

    promo2_start = promo2_adoption_date(frame)
    panel["promo2_start"] = promo2_start
    panel["promo2_active"] = (
        promo2_start.notna() & (panel["date"] >= promo2_start)
    ).astype(int)
    panel["promo2_months"] = (
        (panel["date"] - promo2_start).dt.days / 30.44
    ).clip(lower=0).fillna(0)

    return panel.sort_values(["store_id", "date"]).reset_index(drop=True)


def promo2_adoption_date(frame: pd.DataFrame) -> pd.Series:
    """
    Convert Promo2SinceYear/Week into a calendar date.

    Promo2 is a *continuing* promotional programme: once a store joins it stays
    in. That makes it an absorbing treatment with staggered adoption - the design
    PROJECT_ARCHITECTURE.md §3.2 assumed Rossmann did not have.
    """
    year = frame["Promo2SinceYear"]
    week = frame["Promo2SinceWeek"]
    valid = frame["Promo2"].eq(1) & year.notna() & week.notna()

    result = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if valid.any():
        iso = (
            year[valid].astype(int).astype(str)
            + "-"
            + week[valid].astype(int).astype(str).str.zfill(2)
            + "-1"
        )
        result.loc[valid] = pd.to_datetime(iso, format="%G-%V-%u", errors="coerce")
    return result


def build_features(
    panel: pd.DataFrame, horizon: int = HORIZON
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the same feature shapes Phase 5 uses, on Rossmann's grain.

    Every demand-history term is shifted by `horizon` within the store, exactly
    as in `ml/features.py` - at a seven-day horizon you do not know yesterday's
    sales. Lags are computed over the full daily series including closed days,
    because that is genuinely what a forecaster observes.
    """
    # Reindex to a complete store x date grid before shifting anything.
    #
    # Rossmann's panel is NOT balanced: around 180 stores were closed for
    # refurbishment for roughly six months in 2014 and have no rows at all for
    # that period. `groupby.shift(7)` shifts by seven *rows*, so on a gapped
    # series it silently reaches much further back than seven days - store 670,
    # for instance, has 758 rows across 942 calendar days. That does not leak
    # (a row shift on a sorted series can only reach further into the past) but
    # the feature would not be the quantity it claims to be.
    #
    # Northstar needs no equivalent step because its panel is complete by
    # construction: 20 x 150 x 731 = 2,193,000 rows exactly.
    work = panel.sort_values(["store_id", "date"]).copy()
    original_keys = set(zip(work["store_id"], work["date"]))

    full_dates = pd.date_range(work["date"].min(), work["date"].max(), freq="D")
    full_grid = pd.MultiIndex.from_product(
        [sorted(work["store_id"].unique()), full_dates], names=["store_id", "date"]
    )
    work = (
        work.set_index(["store_id", "date"])
        .reindex(full_grid)
        .reset_index()
        .sort_values(["store_id", "date"])
    )
    LOGGER.info(
        "Reindexed to a complete grid: %d rows (%d filled gaps)",
        len(work), len(work) - len(original_keys),
    )

    grouped = work.groupby("store_id", sort=False)[TARGET]
    features: List[str] = []

    for extra in (0, 7, 14):
        name = f"sales_lag_{horizon + extra}"
        work[name] = grouped.shift(horizon + extra)
        features.append(name)

    shifted = grouped.shift(horizon)
    by_store = shifted.groupby(work["store_id"], sort=False)
    for window in (7, 28):
        name = f"sales_roll{window}_mean"
        work[name] = by_store.transform(lambda s, w=window: s.rolling(w, min_periods=w).mean())
        features.append(name)
    work["sales_roll7_std"] = by_store.transform(lambda s: s.rolling(7, min_periods=7).std())
    features.append("sales_roll7_std")

    work["sales_trend_ratio"] = (
        work["sales_roll7_mean"] / work["sales_roll28_mean"].clip(lower=1.0)
    )
    work["sales_cv"] = work["sales_roll7_std"] / work["sales_roll7_mean"].clip(lower=1.0)
    features += ["sales_trend_ratio", "sales_cv"]

    # Promotional history, shifted like everything else.
    promo_shifted = work.groupby("store_id", sort=False)["promo"].shift(horizon)
    work["promo_days_last_28"] = promo_shifted.groupby(
        work["store_id"], sort=False
    ).transform(lambda s: s.rolling(28, min_periods=28).sum())
    features.append("promo_days_last_28")

    features += KNOWN_IN_ADVANCE + STATIC_FEATURES + ["competition_distance_missing"]
    for column in CATEGORICAL_FEATURES:
        work[column] = work[column].astype("category")
        features.append(column)

    work["dow_sin"] = np.sin(2 * np.pi * work["day_of_week"] / 7)
    work["dow_cos"] = np.cos(2 * np.pi * work["day_of_week"] / 7)
    work["month_sin"] = np.sin(2 * np.pi * work["month"] / 12)
    work["month_cos"] = np.cos(2 * np.pi * work["month"] / 12)
    features += ["dow_sin", "dow_cos", "month_sin", "month_cos"]

    features = list(dict.fromkeys(features))

    # Drop the filler rows the reindex introduced - they are not observations -
    # then drop any remaining row without a full history window. A store coming
    # back from a six-month closure genuinely has no 28-day history, so it stays
    # out until it does.
    work["_observed"] = [
        (store, date) in original_keys
        for store, date in zip(work["store_id"], work["date"])
    ]
    before = len(work)
    work = work[work["_observed"]].drop(columns="_observed")
    work = work.dropna(subset=features + [TARGET]).reset_index(drop=True)
    LOGGER.info(
        "Kept %d of %d grid rows (filler removed, insufficient history dropped)",
        len(work), before,
    )

    leakage.assert_no_leakage(features, context="Rossmann forecast feature set")
    return work, features


def evaluation_mask(frame: pd.DataFrame) -> np.ndarray:
    """
    Rows the forecast is scored on: open days only.

    Closed days are zero by definition and known from the trading calendar, so
    scoring them would inflate every metric without reflecting any real skill.
    This is the convention the original competition used.
    """
    return (frame["open"] == 1).to_numpy()


def time_split(
    frame: pd.DataFrame, n_folds: int = 4, holdout_days: int = 91
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Chronological expanding-window splits with a horizon-sized gap.

    Deliberately the same construction as `ml/features.time_split`, so the
    validation discipline is identical on both datasets.
    """
    dates = pd.to_datetime(frame["date"])
    holdout_start = dates.max() - pd.Timedelta(days=holdout_days - 1)
    development = dates < holdout_start
    holdout_index = np.flatnonzero(~development)

    span_start = dates[development].min()
    span_end = dates[development].max()
    step = (span_end - span_start).days // (n_folds + 1)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(1, n_folds + 1):
        train_end = span_start + pd.Timedelta(days=step * fold)
        valid_end = span_start + pd.Timedelta(days=step * (fold + 1))
        gap_end = train_end + pd.Timedelta(days=HORIZON)
        train_index = np.flatnonzero(development & (dates <= train_end))
        valid_index = np.flatnonzero(development & (dates > gap_end) & (dates <= valid_end))
        if len(valid_index):
            folds.append((train_index, valid_index))
    return folds, holdout_index
