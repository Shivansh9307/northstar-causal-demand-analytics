"""
Write a small, browsable sample of the generated dataset to data/samples/.

Why this exists
---------------
`data/raw/` is ~620MB and gitignored, so a reviewer cloning this repository
cannot look at a single row without first running the generator. `.gitignore`
has always un-ignored `data/samples/` in anticipation of this, and
PROJECT_ARCHITECTURE.md §5 promises a sample is kept; it simply was never
written.

Why a slice and not a random sample
-----------------------------------
A random sample of 5,000 fact rows would reference stores and SKUs whose
dimension rows are elsewhere, so every join in the star schema would come back
half empty. That is worse than no sample: it looks like the data is broken.

Instead this takes **whole dimensions** and slices the facts to a fixed handful
of store x SKU pairs across the full date range. Every foreign key resolves,
the seasonality is visible end to end, and the result is a few MB.

What is *not* filtered
----------------------
The latent columns (`potential_demand_units`, `lost_sales_estimate_units`, the
anomaly labels) stay exactly as they sit in `data/raw/`. They are part of the
raw dataset, §7's leakage checker guards feature sets rather than files, and a
sample whose schema quietly differed from the real thing would mislead anyone
using it to understand the data. `data/ground_truth/` is committed separately
and is untouched here.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import config  # noqa: E402

LOGGER = logging.getLogger("northstar.samples")

# Copied whole: small, and a complete dimension keeps the sample self-describing.
WHOLE_FILES = [
    "dim_store.csv",
    "dim_product.csv",
    "dim_calendar.csv",
    "data_dictionary.csv",
    "README_DATA_GENERATION.md",
]

# Sliced to the pairs below. Read in chunks — fact_daily_store_sku is ~490MB.
SLICED_FILES = [
    "fact_daily_store_sku.csv",
    "fact_inventory_delivery.csv",
    "fact_promotions.csv",
]

N_STORES = 2
N_SKUS = 5
CHUNK_ROWS = 500_000


def _pick_keys(raw: Path) -> tuple[List[str], List[str]]:
    """
    The first N stores and SKUs by id.

    Deliberately not random: the sample should be byte-identical across runs for
    the same generated dataset, like everything else here.
    """
    stores = sorted(pd.read_csv(raw / "dim_store.csv")["store_id"].unique())[:N_STORES]
    skus = sorted(pd.read_csv(raw / "dim_product.csv")["sku_id"].unique())[:N_SKUS]
    return list(stores), list(skus)


def _slice_csv(source: Path, target: Path, stores: List[str], skus: List[str]) -> int:
    """Stream `source`, keeping rows for the chosen pairs. Returns rows written."""
    written = 0
    first = True
    for chunk in pd.read_csv(source, chunksize=CHUNK_ROWS, keep_default_na=False, na_values=[""]):
        keep = chunk
        if "store_id" in chunk.columns:
            keep = keep[keep["store_id"].isin(stores)]
        if "sku_id" in chunk.columns:
            keep = keep[keep["sku_id"].isin(skus)]
        if keep.empty:
            continue
        keep.to_csv(target, mode="w" if first else "a", header=first, index=False)
        written += len(keep)
        first = False
    if first:  # nothing matched — still write a header so the schema is visible
        pd.read_csv(source, nrows=0).to_csv(target, index=False)
    return written


def build() -> dict:
    raw = config.path("raw")
    samples = config.path("samples")
    samples.mkdir(parents=True, exist_ok=True)

    if not (raw / "dim_store.csv").exists():
        raise FileNotFoundError(
            f"No generated data at {raw}. Run src/generation/generate_retail_dataset.py first."
        )

    stores, skus = _pick_keys(raw)
    LOGGER.info("Sampling stores=%s skus=%s", stores, skus)

    written = {}
    for name in WHOLE_FILES:
        source = raw / name
        if not source.exists():
            LOGGER.warning("Skipping %s — not found", name)
            continue
        (samples / name).write_bytes(source.read_bytes())
        written[name] = "whole"

    for name in SLICED_FILES:
        source = raw / name
        if not source.exists():
            LOGGER.warning("Skipping %s — not found", name)
            continue
        rows = _slice_csv(source, samples / name, stores, skus)
        written[name] = rows
        LOGGER.info("%s -> %d rows", name, rows)

    (samples / "README.md").write_text(_readme(stores, skus, written), encoding="utf-8")

    total_mb = sum(p.stat().st_size for p in samples.glob("*")) / (1024 * 1024)
    return {"files": written, "stores": stores, "skus": skus, "megabytes": total_mb}


def _readme(stores: List[str], skus: List[str], written: dict) -> str:
    rows = "\n".join(
        f"| `{name}` | {'whole file' if count == 'whole' else f'{count:,} rows'} |"
        for name, count in written.items()
    )
    return f"""# Sample data

A browsable slice of the generated dataset, so this repository can be explored
without running the generator and materialising ~620MB into `data/raw/`.

Regenerate with `uv run python src/generation/make_samples.py`.

## What is here

Dimensions are complete. The fact tables are sliced to {len(stores)} stores
x {len(skus)} SKUs across the **full** date range, so every foreign key resolves
and the seasonality is still visible. A random row sample would have broken
every join.

Stores: {", ".join(stores)}
SKUs: {", ".join(skus)}

| File | Contents |
|---|---|
{rows}

## Reading it

`seasonal_profile` in `dim_product.csv` contains the literal string `"None"` as
a category. `pandas.read_csv` turns that into `NaN` by default — pass
`keep_default_na=False` or read it through the DuckDB star schema.

`units_sold` is stockout-censored and is the only observable sales quantity.
`potential_demand_units` and `lost_sales_estimate_units` are latent simulation
outputs, present here exactly as they are in `data/raw/`. They are marked in
`data_dictionary.csv` under `whether_safe_for_model_training`, and
`src/data_quality/leakage.py` fails the pipeline if one reaches a feature set.
"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = build()
    print(f"\nSamples written to {config.path('samples')}")
    print(f"  stores: {', '.join(result['stores'])}")
    print(f"  skus:   {', '.join(result['skus'])}")
    print(f"  size:   {result['megabytes']:.1f} MB")
