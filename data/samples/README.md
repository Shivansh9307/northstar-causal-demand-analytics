# Sample data

A browsable slice of the generated dataset, so this repository can be explored
without running the generator and materialising ~620MB into `data/raw/`.

Regenerate with `uv run python src/generation/make_samples.py`.

## What is here

Dimensions are complete. The fact tables are sliced to 2 stores
x 5 SKUs across the **full** date range, so every foreign key resolves
and the seasonality is still visible. A random row sample would have broken
every join.

Stores: STR001, STR002
SKUs: SKU0001, SKU0002, SKU0003, SKU0004, SKU0005

| File | Contents |
|---|---|
| `dim_store.csv` | whole file |
| `dim_product.csv` | whole file |
| `dim_calendar.csv` | whole file |
| `data_dictionary.csv` | whole file |
| `README_DATA_GENERATION.md` | whole file |
| `fact_daily_store_sku.csv` | 7,310 rows |
| `fact_inventory_delivery.csv` | 7,310 rows |
| `fact_promotions.csv` | 99 rows |

## Reading it

`seasonal_profile` in `dim_product.csv` contains the literal string `"None"` as
a category. `pandas.read_csv` turns that into `NaN` by default — pass
`keep_default_na=False` or read it through the DuckDB star schema.

`units_sold` is stockout-censored and is the only observable sales quantity.
`potential_demand_units` and `lost_sales_estimate_units` are latent simulation
outputs, present here exactly as they are in `data/raw/`. They are marked in
`data_dictionary.csv` under `whether_safe_for_model_training`, and
`src/data_quality/leakage.py` fails the pipeline if one reaches a feature set.
