"""
Mutation tests for the generator's automated data-quality checks.

A QA check that cannot fail is worse than no check, because it reads as
assurance. Two checks in the original generator were exactly that:

* the gross-profit check asserted ``gross_profit == gross_profit`` after an
  expression that cancelled algebraically;
* the inventory reconciliation check subtracted a row *count* instead of units
  sold, then discarded the result and only tested ``closing_stock >= 0``.

Both passed on data that was in fact correct, so the failure was invisible.
These tests corrupt one value at a time and require the corresponding check to
raise. They are the reason PROJECT_ARCHITECTURE.md §6 Phase 1's "nothing
proceeds until QA is 100% clean" gate means anything.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "generation"))

import generate_retail_dataset as gen  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
SAMPLE_ROWS = 9_000

pytestmark = pytest.mark.skipif(
    not (RAW / "fact_daily_store_sku.csv").exists(),
    reason="Generated dataset not present; run src/generation/generate_retail_dataset.py first.",
)


@pytest.fixture(scope="module")
def fixtures():
    """A small slice of the generated dataset plus its dimensions."""
    daily = pd.read_csv(RAW / "fact_daily_store_sku.csv", nrows=SAMPLE_ROWS)
    inventory = pd.read_csv(RAW / "fact_inventory_delivery.csv", nrows=SAMPLE_ROWS)
    stores = pd.read_csv(RAW / "dim_store.csv")
    products = pd.read_csv(RAW / "dim_product.csv")
    calendar = pd.read_csv(RAW / "dim_calendar.csv")
    promotions = pd.read_csv(RAW / "fact_promotions.csv")
    # promotion_eligible is an internal generation flag and is not published to
    # dim_product; reconstruct it from who actually received a promotion.
    products["promotion_eligible"] = products["sku_id"].isin(promotions["sku_id"])
    metrics = {
        "rows": len(daily),
        "promo_rows": int(daily["promo_flag"].sum()),
        "stockouts": int(daily["stockout_flag"].sum()),
        "true_att_overall_pct": 0.0,
    }
    return daily, inventory, stores, products, calendar, promotions, metrics


def _run_checks(tmp_path, fixtures, mutate_daily=None, mutate_inventory=None) -> None:
    """Write a (possibly corrupted) copy of the slice and run validate_data on it."""
    daily, inventory, stores, products, calendar, promotions, metrics = fixtures
    daily, inventory = daily.copy(), inventory.copy()
    if mutate_daily is not None:
        mutate_daily(daily)
    if mutate_inventory is not None:
        mutate_inventory(inventory)

    daily.to_csv(tmp_path / "fact_daily_store_sku.csv", index=False)
    inventory.to_csv(tmp_path / "fact_inventory_delivery.csv", index=False)

    original_dir = gen.OUTPUT_DIR
    gen.OUTPUT_DIR = tmp_path
    logging.disable(logging.ERROR)
    try:
        gen.validate_data(stores, products, calendar, promotions, metrics)
    finally:
        logging.disable(logging.NOTSET)
        gen.OUTPUT_DIR = original_dir


def test_clean_data_passes(tmp_path, fixtures):
    """The uncorrupted slice must pass, or the mutation tests prove nothing."""
    _run_checks(tmp_path, fixtures)


def test_detects_broken_gross_profit_identity(tmp_path, fixtures):
    """Regression test for the tautological gross-profit assert."""
    with pytest.raises(AssertionError, match="Gross profit"):
        _run_checks(
            tmp_path,
            fixtures,
            mutate_daily=lambda d: d.__setitem__(
                "gross_profit_gbp",
                d["gross_profit_gbp"].mask(d.index == 7, d["gross_profit_gbp"] + 5),
            ),
        )


def test_detects_broken_inventory_reconciliation(tmp_path, fixtures):
    """Regression test for the reconciliation check that was never evaluated."""
    with pytest.raises(AssertionError, match="Inventory reconciliation"):
        _run_checks(
            tmp_path,
            fixtures,
            mutate_inventory=lambda i: i.__setitem__(
                "closing_stock_units",
                i["closing_stock_units"].mask(i.index == 11, i["closing_stock_units"] + 3),
            ),
        )


def test_detects_negative_inventory(tmp_path, fixtures):
    with pytest.raises(AssertionError, match="negative inventory"):
        _run_checks(
            tmp_path,
            fixtures,
            mutate_inventory=lambda i: i.__setitem__(
                "damaged_units", i["damaged_units"].mask(i.index == 3, -1)
            ),
        )


def test_detects_out_of_range_discount(tmp_path, fixtures):
    with pytest.raises(AssertionError, match="discount"):
        _run_checks(
            tmp_path,
            fixtures,
            mutate_daily=lambda d: d.__setitem__(
                "discount_pct", d["discount_pct"].mask(d.index == 5, 45.0)
            ),
        )


def test_detects_sales_exceeding_available_stock(tmp_path, fixtures):
    with pytest.raises(AssertionError, match="exceed available stock"):
        _run_checks(
            tmp_path,
            fixtures,
            mutate_daily=lambda d: d.__setitem__(
                "units_sold", d["units_sold"].mask(d.index == 9, 999_999)
            ),
        )


def test_ground_truth_is_not_in_raw_directory():
    """§5/§7: the validation artifact must not sit where a loader would glob it."""
    assert not (RAW / "ground_truth_simulation_parameters.csv").exists()
    assert (
        PROJECT_ROOT / "data" / "ground_truth" / "ground_truth_simulation_parameters.csv"
    ).exists()
