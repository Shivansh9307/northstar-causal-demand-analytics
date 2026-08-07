"""
Tests for the Power BI layer.

Power BI Desktop is not available here, so nothing below proves the report opens
or that the DAX evaluates. What these tests do cover is everything that can be
checked without it:

* the exported tables exist, are small enough to commit, and reconcile with the
  figures the phase reports quote;
* the measure library parses and every measure has a home table;
* the scenario measure's arithmetic identity holds, which is the one piece of
  non-trivial DAX logic and the place Phase 6's double-counting bug would recur.

The Phase 8 report states the same limitation rather than implying the model is
verified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from powerbi import parity, tmdl  # noqa: E402
from powerbi.export import output_dir  # noqa: E402

DATA = output_dir()
POWERBI = PROJECT_ROOT / "powerbi"

pytestmark = pytest.mark.skipif(
    not (DATA / "fact_daily_category.csv").exists(),
    reason="Power BI export not present; run src/powerbi/export.py first.",
)

EXPECTED_TABLES = [
    "dim_store", "dim_product", "dim_calendar", "dim_category",
    "fact_daily_category", "reorder_policy",
    "causal_estimates", "dose_response", "spillover", "service_levels",
    "promo_plan", "promo_plan_uncertainty", "promo_plan_draws", "promo_economics",
]


# ---------------------------------------------------------------------------
# The exported data layer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_table_exists_and_is_non_empty(table):
    path = DATA / f"{table}.csv"
    assert path.exists(), f"{table}.csv missing"
    assert len(pd.read_csv(path, nrows=5)) > 0


def test_export_stays_committable():
    """
    The whole point of aggregating was to keep the repository clonable. If this
    starts failing, the fact grain has crept back towards SKU level.
    """
    total_mb = sum(p.stat().st_size for p in DATA.glob("*.csv")) / (1024 * 1024)
    assert total_mb < 30, f"Power BI export is {total_mb:.1f} MB"


def test_fact_grain_is_unique():
    fact = pd.read_csv(DATA / "fact_daily_category.csv")
    assert not fact.duplicated(["date", "store_id", "category"]).any()


def test_latent_columns_are_not_exported():
    """
    §7 bars simulation outputs from any model. A BI surface is exactly where a
    latent-demand column would get mistaken for something actionable.
    """
    fact = pd.read_csv(DATA / "fact_daily_category.csv", nrows=5)
    for banned in ("potential_demand", "lost_sales", "anomaly"):
        assert not any(banned in c for c in fact.columns)


def test_relationship_keys_resolve():
    """Every relationship the model declares must join on values that exist."""
    for from_table, from_column, to_table, to_column in tmdl.RELATIONSHIPS:
        source = pd.read_csv(DATA / f"{from_table}.csv")
        target = pd.read_csv(DATA / f"{to_table}.csv")
        orphans = set(source[from_column].astype(str)) - set(target[to_column].astype(str))
        assert not orphans, f"{from_table}.{from_column} has orphans: {sorted(orphans)[:5]}"


def test_dimension_keys_are_unique():
    """A many-to-one relationship needs a unique key on the one side."""
    for table, key in (
        ("dim_store", "store_id"), ("dim_product", "sku_id"),
        ("dim_calendar", "date"), ("dim_category", "category"),
    ):
        frame = pd.read_csv(DATA / f"{table}.csv")
        assert frame[key].is_unique, f"{table}.{key} is not unique"


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

def test_parity_table_covers_every_page():
    expected = parity.compute_expected()
    pages = set(expected["page"])
    assert pages == {
        "Executive Summary", "Promotion ROI", "Elasticity Explorer",
        "Stockout Risk", "What-If Simulator",
    }


def test_parity_values_are_finite():
    expected = parity.compute_expected()
    assert np.isfinite(expected["expected_value"]).all()


def test_parity_reconciles_with_the_phase_reports():
    """
    Spot-check the figures the written reports quote. If the export drifts, the
    reports and the dashboard would silently disagree.
    """
    expected = parity.compute_expected().set_index("measure")["expected_value"]

    assert expected["Naive Promo Lift %"] == pytest.approx(126.7, abs=0.5)
    assert expected["True Promo Lift %"] == pytest.approx(81.0, abs=0.5)
    assert expected["Promotion Rate %"] == pytest.approx(8.49, abs=0.05)
    assert expected["Stockout Rate %"] == pytest.approx(0.282, abs=0.01)
    assert expected["Cannibalisation 1 Neighbour %"] == pytest.approx(-6.1, abs=0.3)


def test_scenario_measure_reproduces_the_plan_at_multiplier_one():
    """
    The identity the What-If page depends on. At an uplift multiplier of 1 the
    scenario expression must return the optimiser's own figure; if it does not,
    the DAX is deducting the promotional give-away twice, which is precisely the
    bug Phase 6 shipped and fixed.
    """
    plan = pd.read_csv(DATA / "promo_plan.csv")
    scenario = parity._scenario_profit(plan, 1.0)
    assert scenario == pytest.approx(plan["incremental_profit"].sum(), abs=0.01)


def test_scenario_profit_rises_with_the_multiplier():
    plan = pd.read_csv(DATA / "promo_plan.csv")
    values = [parity._scenario_profit(plan, m) for m in (0.6, 0.8, 1.0, 1.2)]
    assert values == sorted(values)


def test_plan_columns_support_the_scenario_measure():
    """The what-if recomputes profit, so it needs both margins, not just the result."""
    plan = pd.read_csv(DATA / "promo_plan.csv", nrows=1)
    for column in ("baseline_units", "incremental_units", "promo_margin",
                   "full_margin", "cannibalisation_loss"):
        assert column in plan.columns


# ---------------------------------------------------------------------------
# The measure library and generated model
# ---------------------------------------------------------------------------

def test_every_measure_has_a_home_table():
    measures = tmdl.parse_measures(POWERBI / "measures.dax")
    homed = {name for names in tmdl.MEASURE_HOME.values() for name in names}
    assert set(measures) == homed


def test_measures_parse_to_non_empty_expressions():
    measures = tmdl.parse_measures(POWERBI / "measures.dax")
    assert len(measures) > 30
    for name, expression in measures.items():
        assert expression.strip(), f"{name} parsed to an empty expression"


def test_measure_home_tables_exist():
    for table in tmdl.MEASURE_HOME:
        assert (DATA / f"{table}.csv").exists(), f"{table} has measures but no data"


def test_generated_model_files_exist():
    definition = POWERBI / f"{tmdl.PROJECT_NAME}.SemanticModel" / "definition"
    assert (definition / "model.tmdl").exists()
    assert (definition / "relationships.tmdl").exists()
    assert (definition / "expressions.tmdl").exists()
    assert (POWERBI / f"{tmdl.PROJECT_NAME}.pbip").exists()
    for table in EXPECTED_TABLES:
        assert (definition / "tables" / f"{table}.tmdl").exists()


def test_generated_tmdl_declares_every_column():
    """
    The model is generated from the data precisely so the schema cannot drift.
    This asserts that it did not.
    """
    definition = POWERBI / f"{tmdl.PROJECT_NAME}.SemanticModel" / "definition" / "tables"
    for table in EXPECTED_TABLES:
        columns = list(pd.read_csv(DATA / f"{table}.csv", nrows=1).columns)
        text = (definition / f"{table}.tmdl").read_text(encoding="utf-8")
        for column in columns:
            assert f"column {column}\n" in text, f"{table}.tmdl missing column {column}"


def test_measure_format_string_is_inside_the_measure_block():
    """
    A blank line between a measure's expression and its formatString closes the
    block, and TMDL then reads the property as the table's. This caught a real
    defect in the generator.
    """
    definition = POWERBI / f"{tmdl.PROJECT_NAME}.SemanticModel" / "definition" / "tables"
    text = (definition / "causal_estimates.tmdl").read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("formatString:"):
            assert lines[index - 1].strip(), "blank line precedes a formatString"
