"""
Tests for the DuckDB star schema.

The join in `analytics_daily` fans across four dimensions and a promotion bridge.
A silent duplicate on any of those would inflate every downstream aggregate while
still looking plausible, so grain conservation is asserted rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import star_schema  # noqa: E402
from utils import config  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (config.path("raw") / "fact_daily_store_sku.csv").exists(),
    reason="Generated dataset not present; run src/generation/generate_retail_dataset.py first.",
)


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    """A schema built into a throwaway database, so tests never touch the real one."""
    db = tmp_path_factory.mktemp("duckdb") / "test.duckdb"
    connection = star_schema.connect(db)
    star_schema.load_raw_tables(connection)
    star_schema.build_promotion_bridge(connection)
    star_schema.build_analytics_view(connection)
    yield connection
    connection.close()


def test_all_schema_checks_pass(con):
    failures = [r for r in star_schema.validate_schema(con) if not r["passed"]]
    assert failures == [], f"Schema validation failures: {failures}"


def test_analytics_daily_preserves_fact_grain(con):
    """The dimension joins must not fan out the fact table."""
    fact_rows = con.execute("SELECT COUNT(*) FROM fact_daily_store_sku").fetchone()[0]
    joined_rows = con.execute("SELECT COUNT(*) FROM analytics_daily").fetchone()[0]
    assert joined_rows == fact_rows


def test_analytics_daily_key_is_unique(con):
    duplicates = con.execute(
        "SELECT COUNT(*) FROM (SELECT date, store_id, sku_id FROM analytics_daily "
        "GROUP BY 1, 2, 3 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert duplicates == 0


def test_promotion_bridge_is_unique_per_store_sku_date(con):
    """Overlapping promotions must collapse to one row, resolved by deepest discount."""
    duplicates = con.execute(
        "SELECT COUNT(*) FROM (SELECT date, store_id, sku_id FROM bridge_promotion_day "
        "GROUP BY 1, 2, 3 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert duplicates == 0


def test_every_realised_promotion_was_scheduled(con):
    """promo_flag can only be true where the bridge says a promotion was running."""
    orphans = con.execute(
        "SELECT COUNT(*) FROM analytics_daily WHERE promo_flag AND NOT promo_scheduled_flag"
    ).fetchone()[0]
    assert orphans == 0


def test_scheduled_is_a_superset_of_realised(con):
    """
    Stock suppression means scheduled >= realised. If realised ever exceeded
    scheduled the bridge would be losing events.
    """
    scheduled, realised = con.execute(
        "SELECT SUM(CASE WHEN promo_scheduled_flag THEN 1 ELSE 0 END), "
        "       SUM(CASE WHEN promo_flag THEN 1 ELSE 0 END) FROM analytics_daily"
    ).fetchone()
    assert scheduled >= realised


def test_campaign_waves_are_staggered(con):
    """The DiD-identifying structure: cohorts enter a campaign at different offsets."""
    waves = con.execute(
        "SELECT DISTINCT campaign_wave_days FROM bridge_promotion_day "
        "WHERE campaign_id IS NOT NULL AND campaign_id <> '' ORDER BY 1"
    ).fetchall()
    assert [w[0] for w in waves] == [0, 21, 42]


def test_never_treated_control_pool_exists(con):
    """Phase 4 needs untreated units; assert the pool did not vanish."""
    never_treated = con.execute(
        "SELECT COUNT(*) FROM (SELECT store_id, sku_id FROM analytics_daily "
        "GROUP BY 1, 2 HAVING MAX(CASE WHEN promo_flag THEN 1 ELSE 0 END) = 0)"
    ).fetchone()[0]
    assert never_treated > 0


def test_feature_safe_columns_exclude_simulation_outputs(con):
    safe = set(star_schema.feature_safe_columns(con))
    for column in ("potential_demand_units", "lost_sales_estimate_units", "anomaly_flag", "anomaly_type"):
        assert column not in safe
    for column in ("units_sold", "promo_flag", "discount_pct"):
        assert column in safe
