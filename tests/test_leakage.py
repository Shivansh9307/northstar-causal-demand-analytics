"""
Tests for the §7 leakage checker.

The checker's job is to fail loudly. These tests assert it catches the columns it
must, tolerates the ones it must not block, and survives the obvious evasion: a
leaking column renamed on its way into a feature frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_quality import leakage  # noqa: E402

SAFE_COLUMNS = [
    "date",
    "store_id",
    "sku_id",
    "units_sold",
    "discount_pct",
    "promo_flag",
    "actual_unit_price_gbp",
    "average_daily_footfall",
    "baseline_gross_margin_pct",
    "rolling_7_day_avg_units_sold",
    "category",
    "rollout_cohort",
]


@pytest.mark.parametrize(
    "column",
    [
        "potential_demand_units",
        "lost_sales_estimate_units",
        "anomaly_flag",
        "anomaly_type",
    ],
)
def test_blocks_configured_forbidden_columns(column):
    with pytest.raises(leakage.LeakageError, match=column):
        leakage.assert_no_leakage(SAFE_COLUMNS + [column])


@pytest.mark.parametrize(
    "column",
    [
        "true_price_elasticity",
        "true_promo_uplift_pct",
        "true_realised_att_pct",
        "true_demand_trend_pct_per_year",
    ],
)
def test_blocks_ground_truth_parameters(column):
    with pytest.raises(leakage.LeakageError):
        leakage.assert_no_leakage(SAFE_COLUMNS + [column])


@pytest.mark.parametrize(
    "column",
    [
        "potential_demand",          # suffix dropped
        "PotentialDemandUnits",      # case changed
        "lost_sales",                # shortened
        "anomaly",                   # stem only
        "true_uplift",               # invented ground-truth name
        "realised_att_pct",          # ATT under a new name
    ],
)
def test_blocks_renamed_leaks(column):
    """A renamed leaking column leaks exactly as much as the original."""
    with pytest.raises(leakage.LeakageError):
        leakage.assert_no_leakage(SAFE_COLUMNS + [column])


def test_allows_legitimate_features():
    leakage.assert_no_leakage(SAFE_COLUMNS)


def test_ground_truth_join_keys_are_not_blocked():
    """sku_id and category appear in the ground-truth file but are real features."""
    assert leakage.find_leaks(["sku_id", "category"]) == []


def test_safe_feature_columns_filters_and_preserves_order():
    columns = ["date", "potential_demand_units", "units_sold", "true_price_elasticity", "promo_flag"]
    assert leakage.safe_feature_columns(columns) == ["date", "units_sold", "promo_flag"]


def test_assert_frame_is_safe_reports_context():
    frame = pd.DataFrame({"units_sold": [1], "potential_demand_units": [2]})
    with pytest.raises(leakage.LeakageError, match="my model"):
        leakage.assert_frame_is_safe(frame, context="my model")
