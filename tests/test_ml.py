"""
Tests for the Phase 5 forecasting pipeline.

The failure this phase is most exposed to is temporal leakage: at a seven-day
horizon, a feature that reads day T+1 makes the model look excellent and renders
it undeployable. The lag construction is therefore checked against the raw panel
row by row, not just asserted in a docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_quality import leakage  # noqa: E402
from ml import features, forecast, stockout  # noqa: E402
from utils import config  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (config.path("raw") / "fact_daily_store_sku.csv").exists(),
    reason="Generated dataset not present; run src/generation/generate_retail_dataset.py first.",
)


@pytest.fixture(scope="module")
def built():
    """Features built on a four-store slice, keeping the full date range."""
    source = features.load_source()
    keep = sorted(source["store_id"].unique())[:4]
    source = source[source["store_id"].isin(keep)].copy()
    frame, names = features.build_features(source)
    return source, frame, names


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_lag_features_are_shifted_by_the_horizon(built):
    """
    `sales_lag_7` on day D must equal actual sales on day D-7 for the same pair.

    If this ever reads D-1, the model is using sales that have not happened when
    the replenishment order is placed.
    """
    source, frame, _ = built
    lookup = source.set_index(["store_id", "sku_id", "date"])["units_sold"]

    sample = frame.sample(n=400, random_state=42)
    expected = []
    for row in sample.itertuples(index=False):
        key = (row.store_id, row.sku_id, row.date - pd.Timedelta(days=features.HORIZON))
        expected.append(lookup.get(key, np.nan))

    assert np.allclose(sample["sales_lag_7"].to_numpy(), np.array(expected), equal_nan=True)


def test_rolling_features_never_read_past_the_forecast_origin(built):
    """
    The 7-day rolling mean must average days T-6..T, never anything after T.
    """
    source, frame, _ = built
    lookup = source.set_index(["store_id", "sku_id", "date"])["units_sold"]

    sample = frame.sample(n=150, random_state=7)
    for row in sample.itertuples(index=False):
        window = [
            lookup.get(
                (row.store_id, row.sku_id,
                 row.date - pd.Timedelta(days=features.HORIZON + offset)),
                np.nan,
            )
            for offset in range(7)
        ]
        if not np.isnan(window).any():
            assert row.sales_roll7_mean == pytest.approx(float(np.mean(window)), rel=1e-9)


def test_feature_set_passes_the_leakage_checker(built):
    _, _, names = built
    leakage.assert_no_leakage(names, context="phase 5 features")


def test_no_feature_is_a_proxy_for_the_target(built):
    """A near-perfect correlation would mean the outcome leaked in under a new name."""
    _, frame, names = built
    numeric = frame[names].select_dtypes(include=[np.number])
    correlations = numeric.corrwith(frame["units_sold"]).abs()
    assert correlations.max() < 0.95


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_splits_are_chronological_with_a_gap(built):
    """Train must end before validation starts, with at least the horizon between."""
    _, frame, _ = built
    folds, holdout = features.time_split(frame)
    dates = pd.to_datetime(frame["date"])

    for train_index, valid_index in folds:
        train_end = dates.iloc[train_index].max()
        valid_start = dates.iloc[valid_index].min()
        assert valid_start > train_end
        assert (valid_start - train_end).days >= features.HORIZON

    holdout_start = dates.iloc[holdout].min()
    non_holdout = np.setdiff1d(np.arange(len(frame)), holdout)
    assert dates.iloc[non_holdout].max() < holdout_start


def test_holdout_is_untouched_by_cross_validation(built):
    _, frame, _ = built
    folds, holdout = features.time_split(frame)
    holdout_set = set(holdout.tolist())
    for train_index, valid_index in folds:
        assert not holdout_set & set(train_index.tolist())
        assert not holdout_set & set(valid_index.tolist())


def test_seasonal_naive_is_the_matching_weekday(built):
    """At a seven-day horizon the naive forecast is both the last observation
    available and the same day of week."""
    _, frame, _ = built
    naive = features.seasonal_naive(frame)
    matched = naive.notna()
    assert np.allclose(
        naive[matched].to_numpy(), frame.loc[matched, "sales_lag_7"].to_numpy()
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_evaluate_on_a_known_case():
    actual = np.array([10.0, 20.0, 0.0, 5.0])
    predicted = np.array([12.0, 18.0, 1.0, 5.0])
    metrics = forecast.evaluate(actual, predicted)

    assert metrics["mae"] == pytest.approx(5.0 / 4)
    assert metrics["wape"] == pytest.approx(5.0 / 35.0)
    assert metrics["bias"] == pytest.approx(1.0 / 4)
    # MAPE skips the zero actual rather than dividing by it.
    assert metrics["mape_nonzero"] == pytest.approx((0.2 + 0.1 + 0.0) / 3)
    assert metrics["zero_share"] == pytest.approx(0.25)


def test_classification_metrics_on_a_known_confusion_matrix():
    actual = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    metrics = stockout.classification_metrics(actual, scores, threshold=0.5)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_accuracy_is_useless_at_this_base_rate():
    """The always-negative baseline must score high on accuracy and zero on recall."""
    actual = np.zeros(10_000, dtype=int)
    actual[:28] = 1
    scores = np.zeros(10_000)
    metrics = stockout.classification_metrics(actual, scores, threshold=0.5)
    assert metrics["accuracy"] > 0.99
    assert metrics["recall"] == 0.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_gradient_boosting_beats_the_seasonal_naive_baseline(built):
    """The §8 bar: the model must beat the baseline on a time-based holdout."""
    _, frame, names = built
    categorical = list(features.CATEGORICAL_FEATURES)
    _, holdout = features.time_split(frame)

    metrics, booster, _, predictions = forecast.run_holdout(
        frame, names, categorical, holdout
    )
    naive_wape = float(metrics.loc[metrics["model"] == "Seasonal naive", "wape"].iloc[0])
    model_wape = float(metrics.iloc[-1]["wape"])
    assert model_wape < naive_wape


def test_predictions_are_non_negative(built):
    _, frame, names = built
    categorical = list(features.CATEGORICAL_FEATURES)
    _, holdout = features.time_split(frame)
    _, _, _, predictions = forecast.run_holdout(frame, names, categorical, holdout)
    assert (predictions >= 0).all()


def test_risk_ranking_concentrates_stockouts(built):
    """The top decile must carry far more than its share of stockouts."""
    _, frame, _ = built
    rng = np.random.default_rng(0)
    actual = frame["stockout_flag"].astype(int).to_numpy()
    # A ranking correlated with the truth, standing in for a fitted model.
    scores = actual * 0.8 + rng.random(len(actual)) * 0.2
    deciles = stockout.lift_by_decile(actual, scores)
    assert deciles.iloc[0]["lift_vs_base"] > 5


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def test_translate_params_maps_onto_each_backend():
    """
    The two backends name the same three ideas differently, and `params` used to
    be forwarded raw to LightGBM and dropped entirely on the sklearn path. A
    grid written in LightGBM's vocabulary was therefore a silent no-op on any
    machine without libomp, which returns identical scores for every candidate
    and reads as "tuning buys nothing".
    """
    neutral = {"learning_rate": 0.03, "leaves": 48, "min_leaf": 500}

    assert forecast.translate_params(neutral, "lightgbm") == {
        "learning_rate": 0.03, "num_leaves": 48, "min_data_in_leaf": 500,
    }
    assert forecast.translate_params(neutral, "sklearn_hist") == {
        "learning_rate": 0.03, "max_leaf_nodes": 48, "min_samples_leaf": 500,
    }


def test_translate_params_rejects_unknown_names():
    """A typo must fail loudly; silently ignored, it looks like a null result."""
    with pytest.raises(ValueError, match="unknown tuning parameters"):
        forecast.translate_params({"num_leaves": 48}, "lightgbm")


def test_empty_params_changes_nothing():
    """The shipped model passes no overrides, so this is the production path."""
    for backend in ("lightgbm", "sklearn_hist"):
        assert forecast.translate_params({}, backend) == {}


def test_tuning_grid_contains_the_defaults():
    """
    Without a defaults row there is nothing to measure a gain against, and
    `vs_defaults` would be a comparison between two tuned candidates.
    """
    assert {} in forecast.TUNING_GRID


def test_tuning_reports_every_candidate_against_the_defaults(built):
    _, frame, names = built
    categorical = list(features.CATEGORICAL_FEATURES)
    folds, _ = features.time_split(frame)
    grid = [{}, {"leaves": 16}]

    table = forecast.tune_gradient_booster(
        frame, names, categorical, folds[:1], max_train_rows=40_000, grid=grid,
    )

    assert len(table) == len(grid)
    assert "defaults" in set(table["params"])
    baseline = table.loc[table["params"] == "defaults"]
    assert float(baseline["vs_defaults"].iloc[0]) == pytest.approx(0.0)
    assert table["mean_wape"].between(0, 5).all()
    # Sorted best-first, so the reported winner is row zero.
    assert table["mean_wape"].is_monotonic_increasing
