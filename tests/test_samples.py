"""
Tests for the committed sample dataset.

`data/samples/` is the only part of the generated data that ships in the
repository, so it is the first thing a reviewer sees. Two ways it could quietly
become useless: its schema drifts from `data/raw/`, or the fact slice references
keys whose dimension rows were not carried across, so every join returns nothing.
Both look like a broken dataset rather than a broken sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import config  # noqa: E402

SAMPLES = config.path("samples")
RAW = config.path("raw")

pytestmark = pytest.mark.skipif(
    not (SAMPLES / "dim_store.csv").exists(),
    reason="No samples; run src/generation/make_samples.py first.",
)

EXPECTED = [
    "dim_store.csv",
    "dim_product.csv",
    "dim_calendar.csv",
    "fact_daily_store_sku.csv",
    "fact_inventory_delivery.csv",
    "fact_promotions.csv",
    "data_dictionary.csv",
    "README_DATA_GENERATION.md",
    "README.md",
]


@pytest.mark.parametrize("name", EXPECTED)
def test_sample_file_exists(name):
    assert (SAMPLES / name).exists(), f"{name} missing from data/samples/"


def test_samples_stay_committable():
    """The point is a repository that clones quickly. 620MB of facts is why."""
    total_mb = sum(p.stat().st_size for p in SAMPLES.glob("*")) / (1024 * 1024)
    assert total_mb < 10, f"samples are {total_mb:.1f} MB"


@pytest.mark.skipif(not (RAW / "dim_store.csv").exists(), reason="raw data not generated")
@pytest.mark.parametrize(
    "name",
    ["dim_store.csv", "dim_product.csv", "dim_calendar.csv",
     "fact_daily_store_sku.csv", "fact_inventory_delivery.csv", "fact_promotions.csv"],
)
def test_sample_schema_matches_raw(name):
    """
    A sample whose columns differ from the real thing teaches the wrong shape.
    Latent columns are deliberately kept — they are part of `data/raw/`, and the
    leakage checker guards feature sets rather than files.
    """
    sample = list(pd.read_csv(SAMPLES / name, nrows=0).columns)
    raw = list(pd.read_csv(RAW / name, nrows=0).columns)
    assert sample == raw, f"{name} schema has drifted from data/raw/"


def test_fact_keys_resolve_against_the_sampled_dimensions():
    """
    The reason this is a slice and not a random sample. If a fact row references
    a store or SKU whose dimension row was not carried across, every join in the
    star schema comes back half empty.
    """
    stores = set(pd.read_csv(SAMPLES / "dim_store.csv")["store_id"])
    skus = set(pd.read_csv(SAMPLES / "dim_product.csv")["sku_id"])
    dates = set(pd.read_csv(SAMPLES / "dim_calendar.csv")["date"].astype(str))

    fact = pd.read_csv(SAMPLES / "fact_daily_store_sku.csv")
    assert set(fact["store_id"]) <= stores
    assert set(fact["sku_id"]) <= skus
    assert set(fact["date"].astype(str)) <= dates


def test_fact_slice_is_a_complete_panel():
    """
    Every sampled pair should span the full calendar. A gapped panel would make
    the sample misleading about the dataset's structure — and lag features
    computed on it would silently reach further back than intended.
    """
    fact = pd.read_csv(SAMPLES / "fact_daily_store_sku.csv")
    pairs = fact.groupby(["store_id", "sku_id"]).size()
    days = fact["date"].nunique()
    assert pairs.nunique() == 1, "sampled pairs have different row counts"
    assert pairs.iloc[0] == days, "a sampled pair does not span every date"


def test_ground_truth_is_not_in_the_sample():
    """
    §7: ground truth lives in data/ground_truth/ and nowhere a training loader
    might glob. That rule applies to the sample directory too.
    """
    for path in SAMPLES.glob("*.csv"):
        columns = pd.read_csv(path, nrows=0).columns
        leaked = [c for c in columns if c.startswith("true_")]
        assert not leaked, f"{path.name} carries ground-truth columns: {leaked}"
