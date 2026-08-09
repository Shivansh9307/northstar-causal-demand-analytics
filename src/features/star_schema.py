"""
Build a DuckDB star schema over data/raw and export the joined analytical table.

Design notes
------------
The two fact tables share the same grain (date x store_id x sku_id), so
`analytics_daily` joins them rather than treating inventory as a separate fact.
Only inventory-unique columns are pulled across; the columns duplicated in both
sources (opening/closing stock, deliveries, stockout flag) are taken from the
daily fact.

`bridge_promotion_day` expands `fact_promotions` from event grain to daily grain
so Phase 4 can attribute a promoted day to its `campaign_id` and rollout wave -
the daily fact carries `promo_flag` but not which campaign caused it.

That bridge is deliberately **intention to treat**: it records that a promotion
was scheduled. The fact table's `promo_flag` is **realised treatment** - the
generator suppresses the flag when a store had no stock to display, so a row can
be scheduled but not treated. Phase 4 should be explicit about which of the two
it is estimating; they are not the same estimand.

The exported parquet retains latent columns (`potential_demand_units`,
`lost_sales_estimate_units`, anomaly labels) because EDA and Phase 6 lost-sales
costing legitimately need them. They are gated at feature-selection time by
`src/data_quality/leakage.py`, not by omission here - see §7.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_quality import leakage  # noqa: E402
from utils import config  # noqa: E402

LOGGER = logging.getLogger("northstar.star_schema")

RAW_TABLES: Dict[str, str] = {
    "dim_store": "dim_store.csv",
    "dim_product": "dim_product.csv",
    "dim_calendar": "dim_calendar.csv",
    "fact_promotions": "fact_promotions.csv",
    "fact_daily_store_sku": "fact_daily_store_sku.csv",
    "fact_inventory_delivery": "fact_inventory_delivery.csv",
}

# Columns present in fact_inventory_delivery but not in the daily fact.
INVENTORY_ONLY_COLUMNS: List[str] = [
    "delivery_scheduled_flag",
    "delivery_delay_flag",
    "damaged_units",
    "expired_or_wasted_units",
    "stock_cover_days",
    "reorder_point_units",
    "expected_lead_time_days",
]

PRODUCT_ATTRIBUTES: List[str] = [
    "category",
    "subcategory",
    "brand_type",
    "price_elasticity_segment",
    "promotion_sensitivity_segment",
    "demand_volatility_segment",
    "seasonal_profile",
    "is_perishable",
    "shelf_life_days",
    "unit_cost_gbp",
    "baseline_gross_margin_pct",
    "minimum_display_stock",
    "reorder_lead_time_days",
]

STORE_ATTRIBUTES: List[str] = [
    "store_format",
    "region",
    "country",
    "city",
    "average_daily_footfall",
    "competition_intensity_score",
    "store_income_index",
    "local_deprivation_decile",
    "floor_area_sqm",
    "rollout_cohort",
]

# Calendar columns not already denormalised onto the daily fact.
CALENDAR_ATTRIBUTES: List[str] = [
    "year",
    "quarter",
    "month_name",
    "day_name",
    "is_month_end",
    "is_easter_period",
    "is_christmas_period",
    "is_black_friday_period",
    "is_heatwave",
    "bank_holiday_name",
]


def connect(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the project DuckDB database."""
    target = db_path or (config.path("processed") / "northstar.duckdb")
    target.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(target))


def load_raw_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Materialise the raw CSVs as DuckDB tables."""
    raw = config.path("raw")
    for table, filename in RAW_TABLES.items():
        source = raw / filename
        if not source.exists():
            raise FileNotFoundError(
                f"{source} not found. Run src/generation/generate_retail_dataset.py first."
            )
        LOGGER.info("Loading %s", table)
        con.execute(
            f"CREATE OR REPLACE TABLE {table} AS "
            f"SELECT * FROM read_csv_auto(?, header=true)",
            [str(source)],
        )


def build_promotion_bridge(con: duckdb.DuckDBPyConnection) -> None:
    """
    Expand promotion events to daily grain, one row per store x SKU x date.

    Overlapping events on the same store/SKU/date are resolved by deepest
    discount, matching how the generator's promo lookup resolved them.
    """
    con.execute(
        """
        CREATE OR REPLACE TABLE bridge_promotion_day AS
        SELECT * FROM (
            SELECT
                p.promotion_id,
                p.campaign_id,
                p.campaign_wave_days,
                p.store_id,
                p.sku_id,
                CAST(d.generate_series AS DATE) AS date,
                p.promo_type,
                p.discount_pct AS scheduled_discount_pct,
                p.campaign_theme,
                p.promotion_planned_flag,
                p.vendor_funded_pct,
                ROW_NUMBER() OVER (
                    PARTITION BY p.store_id, p.sku_id, CAST(d.generate_series AS DATE)
                    ORDER BY p.discount_pct DESC, p.promotion_id
                ) AS priority
            FROM fact_promotions p
            CROSS JOIN generate_series(
                CAST(p.promo_start_date AS DATE),
                CAST(p.promo_end_date AS DATE),
                INTERVAL 1 DAY
            ) AS d
        )
        WHERE priority = 1
        """
    )
    con.execute("ALTER TABLE bridge_promotion_day DROP COLUMN priority")


def build_analytics_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create the joined analytical view at date x store x SKU grain."""
    inventory = ", ".join(f"i.{c}" for c in INVENTORY_ONLY_COLUMNS)
    product = ", ".join(f"p.{c}" for c in PRODUCT_ATTRIBUTES)
    store = ", ".join(f"s.{c}" for c in STORE_ATTRIBUTES)
    calendar = ", ".join(f"c.{c}" for c in CALENDAR_ATTRIBUTES)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW analytics_daily AS
        SELECT
            f.*,
            {inventory},
            {product},
            {store},
            {calendar},
            b.campaign_id,
            b.campaign_wave_days,
            b.campaign_theme,
            b.scheduled_discount_pct,
            b.promotion_planned_flag,
            b.vendor_funded_pct,
            -- Scheduled but not realised: no stock to display, so no observed promo.
            (b.promotion_id IS NOT NULL) AS promo_scheduled_flag
        FROM fact_daily_store_sku f
        JOIN fact_inventory_delivery i USING (date, store_id, sku_id)
        JOIN dim_product p USING (sku_id)
        JOIN dim_store s USING (store_id)
        JOIN dim_calendar c USING (date)
        LEFT JOIN bridge_promotion_day b USING (date, store_id, sku_id)
        """
    )


def validate_schema(con: duckdb.DuckDBPyConnection) -> List[Dict[str, object]]:
    """Check grain, referential integrity and row conservation across the join."""
    results: List[Dict[str, object]] = []

    def check(name: str, sql: str, expected: int = 0) -> None:
        actual = con.execute(sql).fetchone()[0]
        results.append(
            {
            "check": name,
            "passed": actual == expected,
            "detail": f"{actual:,} (expected {expected:,})",
        }
        )

    fact_rows = con.execute("SELECT COUNT(*) FROM fact_daily_store_sku").fetchone()[0]

    check(
        "analytics_daily preserves fact grain",
        f"SELECT ABS(COUNT(*) - {fact_rows}) FROM analytics_daily",
    )
    check(
        "analytics_daily key is unique",
        "SELECT COUNT(*) FROM (SELECT date, store_id, sku_id FROM analytics_daily "
        "GROUP BY 1,2,3 HAVING COUNT(*) > 1)",
    )
    check(
        "bridge_promotion_day key is unique",
        "SELECT COUNT(*) FROM (SELECT date, store_id, sku_id FROM bridge_promotion_day "
        "GROUP BY 1,2,3 HAVING COUNT(*) > 1)",
    )
    check(
        "no orphan store keys",
        "SELECT COUNT(*) FROM fact_daily_store_sku f "
        "LEFT JOIN dim_store s USING (store_id) WHERE s.store_id IS NULL",
    )
    check(
        "no orphan SKU keys",
        "SELECT COUNT(*) FROM fact_daily_store_sku f "
        "LEFT JOIN dim_product p USING (sku_id) WHERE p.sku_id IS NULL",
    )
    check(
        "no orphan calendar keys",
        "SELECT COUNT(*) FROM fact_daily_store_sku f "
        "LEFT JOIN dim_calendar c USING (date) WHERE c.date IS NULL",
    )
    check(
        "every realised promo row has a scheduling record",
        "SELECT COUNT(*) FROM analytics_daily WHERE promo_flag AND NOT promo_scheduled_flag",
    )
    return results


def export_analytical_table(con: duckdb.DuckDBPyConnection) -> Path:
    """Write the joined table to data/processed as parquet."""
    target = config.path("processed") / "analytics_daily.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        "COPY (SELECT * FROM analytics_daily) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [str(target)],
    )
    return target


def feature_safe_columns(con: duckdb.DuckDBPyConnection) -> List[str]:
    """Columns of analytics_daily that may legitimately be used for training (§7)."""
    columns = [row[0] for row in con.execute("DESCRIBE analytics_daily").fetchall()]
    return leakage.safe_feature_columns(columns)


def build(db_path: Path | None = None) -> Dict[str, object]:
    """Build the whole schema and return a summary for reporting."""
    con = connect(db_path)
    try:
        load_raw_tables(con)
        build_promotion_bridge(con)
        build_analytics_view(con)
        results = validate_schema(con)

        failed = [r for r in results if not r["passed"]]
        if failed:
            raise AssertionError(
                "Star schema validation failed: "
                + "; ".join(f"{r['check']} -> {r['detail']}" for r in failed)
            )

        parquet_path = export_analytical_table(con)
        all_columns = [row[0] for row in con.execute("DESCRIBE analytics_daily").fetchall()]
        safe = leakage.safe_feature_columns(all_columns)
        summary = {
            "rows": con.execute("SELECT COUNT(*) FROM analytics_daily").fetchone()[0],
            "columns": len(all_columns),
            "feature_safe_columns": len(safe),
            "blocked_columns": sorted(set(all_columns) - set(safe)),
            "parquet": parquet_path,
            "parquet_mb": parquet_path.stat().st_size / (1024 * 1024),
            "checks": results,
        }
        return summary
    finally:
        con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = build()
    print(f"\nanalytics_daily: {result['rows']:,} rows x {result['columns']} columns")
    print(f"Feature-safe columns: {result['feature_safe_columns']}")
    print(f"Blocked by leakage checker: {result['blocked_columns']}")
    print(f"Parquet: {result['parquet']} ({result['parquet_mb']:.1f} MB)")
    print("\nSchema checks:")
    for check in result["checks"]:
        print(f"  {'PASS' if check['passed'] else 'FAIL'}  {check['check']}  [{check['detail']}]")
