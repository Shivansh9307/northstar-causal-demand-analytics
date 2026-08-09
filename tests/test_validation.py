"""
Tests for the Rossmann external-validity adapter.

The claim Phase 7 makes is that the Northstar pipeline runs on real data via a
swapped loader alone. That is only meaningful if the loader really does honour
the same contract and the same leakage discipline, so both are asserted rather
than demonstrated once in a report.
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
from ml import forecast as ml_forecast  # noqa: E402
from utils import config  # noqa: E402
from validation import bacon  # noqa: E402
from validation import rossmann  # noqa: E402

HAS_ROSSMANN = (config.path("external") / "train.csv").exists()

needs_rossmann = pytest.mark.skipif(
    not HAS_ROSSMANN, reason="Rossmann files not present in data/external/."
)


@pytest.fixture(scope="module")
def panel():
    # Skipping from the fixture rather than the module keeps the Goodman-Bacon
    # tests running. They are arithmetic on constructed panels and need no data,
    # and CI has no Rossmann files — a module-level skip would have hidden the
    # one test that checks the decomposition is correct.
    if not HAS_ROSSMANN:
        pytest.skip("Rossmann files not present in data/external/.")
    return rossmann.build_panel()


@pytest.fixture(scope="module")
def built(panel):
    return rossmann.build_features(panel)


# ---------------------------------------------------------------------------
# The dataset's shape, which the report's central claim depends on
# ---------------------------------------------------------------------------

@needs_rossmann
def test_rossmann_has_no_price_or_margin_column():
    """
    Phase 7's headline finding. If a future Rossmann variant ever shipped price
    data, the report's central claim would need rewriting - so this asserts the
    absence rather than trusting a sentence in the markdown.
    """
    sales, stores = rossmann.load_raw()
    columns = [c.lower() for c in list(sales.columns) + list(stores.columns)]
    for banned in ("price", "discount", "margin", "cost", "revenue"):
        assert not any(banned in c for c in columns), f"unexpected {banned} column appeared"


def test_promo_is_binary(panel):
    """No dose, which is why the Phase 3 dose-response cannot transfer."""
    assert set(panel["promo"].unique()) <= {0, 1}


def test_promo2_is_absorbing(panel):
    """
    Once a store joins Promo2 it never leaves. That is what makes it a staggered
    *adoption* design rather than an on/off treatment like Northstar's.
    """
    by_store = panel.sort_values("date").groupby("store_id")["promo2_active"]
    # A store's series may only ever go 0 -> 1, never back.
    reversals = by_store.apply(lambda s: bool((s.diff() < 0).any()))
    assert not reversals.any()


def test_promo2_has_both_control_groups(panel):
    """A staggered design needs never-treated and not-yet-treated units."""
    by_store = panel.groupby("store_id")["promo2_active"]
    never = int((by_store.max() == 0).sum())
    adopts_in_window = int(by_store.agg(lambda s: s.min() == 0 and s.max() == 1).sum())
    assert never > 100
    assert adopts_in_window > 100


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_lags_are_shifted_by_the_horizon(built, panel):
    """
    Same discipline as Phase 5: `sales_lag_7` on day D must be sales on D-7 for
    that store, not D-1.
    """
    frame, _ = built
    lookup = panel.set_index(["store_id", "date"])["sales"]

    sample = frame.sample(n=300, random_state=42)
    expected = [
        lookup.get((row.store_id, row.date - pd.Timedelta(days=rossmann.HORIZON)), np.nan)
        for row in sample.itertuples(index=False)
    ]
    assert np.allclose(sample["sales_lag_7"].to_numpy(), np.array(expected), equal_nan=True)


def test_customers_is_not_a_feature(built):
    """
    `Customers` is measured on the day being forecast, so using it would leak.
    It is the most tempting column in the dataset and the easiest mistake here.
    """
    _, features = built
    assert "customers" not in features


def test_feature_set_passes_the_leakage_checker(built):
    _, features = built
    leakage.assert_no_leakage(features, context="rossmann features")


def test_no_feature_is_a_proxy_for_the_target(built):
    frame, features = built
    scored = frame[rossmann.evaluation_mask(frame)]
    numeric = scored[features].select_dtypes(include=[np.number])
    correlations = numeric.corrwith(scored[rossmann.TARGET]).abs()
    assert correlations.max() < 0.95


# ---------------------------------------------------------------------------
# Splits and the shared code path
# ---------------------------------------------------------------------------

def test_splits_are_chronological_with_a_gap(built):
    frame, _ = built
    scored = frame[rossmann.evaluation_mask(frame)].reset_index(drop=True)
    folds, holdout = rossmann.time_split(scored)
    dates = pd.to_datetime(scored["date"])

    for train_index, valid_index in folds:
        assert dates.iloc[valid_index].min() > dates.iloc[train_index].max()
        gap = (dates.iloc[valid_index].min() - dates.iloc[train_index].max()).days
        assert gap >= rossmann.HORIZON

    non_holdout = np.setdiff1d(np.arange(len(scored)), holdout)
    assert dates.iloc[non_holdout].max() < dates.iloc[holdout].min()


def test_evaluation_excludes_closed_days(built):
    """Closed days are zero by definition; scoring them would inflate every metric."""
    frame, _ = built
    scored = frame[rossmann.evaluation_mask(frame)]
    assert (scored["open"] == 1).all()
    assert scored[rossmann.TARGET].min() >= 0


def test_northstar_pipeline_runs_unchanged_on_rossmann(built):
    """
    The actual Phase 7 claim: `ml/forecast` executes against Rossmann with only a
    different loader, and the model still beats the seasonal-naive baseline.
    """
    frame, features = built
    scored = frame[rossmann.evaluation_mask(frame)].reset_index(drop=True)
    # A store subset keeps the test quick without changing the code path.
    keep = sorted(scored["store_id"].unique())[:60]
    scored = scored[scored["store_id"].isin(keep)].reset_index(drop=True)

    _, holdout = rossmann.time_split(scored)
    metrics, _, _, predictions = ml_forecast.run_holdout(
        scored, features, list(rossmann.CATEGORICAL_FEATURES), holdout,
        target=rossmann.TARGET, naive_column="sales_lag_7",
    )
    naive_wape = float(metrics.loc[metrics["model"] == "Seasonal naive", "wape"].iloc[0])
    model_wape = float(metrics.iloc[-1]["wape"])

    assert model_wape < naive_wape
    assert (predictions >= 0).all()


# ---------------------------------------------------------------------------
# Goodman-Bacon decomposition
#
# Arithmetic on constructed panels, so these run without the Rossmann files.
# ---------------------------------------------------------------------------


def _staggered_panel(effect=0.5, units=60, periods=12, seed=3, heterogeneous=False):
    """A staggered-adoption panel with a known treatment effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(units):
        cohort = None if unit % 3 == 0 else 4 + (unit % 4) * 2
        for period in range(periods):
            treated = int(cohort is not None and period >= cohort)
            # Heterogeneous: later cohorts respond more strongly, which is the
            # condition under which the bad comparisons actually bite.
            size = effect * (1 + 0.5 * (cohort or 0) / periods) if heterogeneous else effect
            rows.append({
                "store_id": unit,
                "period": period,
                "log_units": unit / units + 0.05 * period + size * treated
                + rng.normal(scale=0.01),
                "treated": treated,
            })
    return pd.DataFrame(rows)


def _twfe(frame):
    """Two-way FE coefficient by explicit demeaning, independent of the decomposition."""
    work = frame.copy()
    for column in ("log_units", "treated"):
        work[column] = (
            work[column]
            - work.groupby("store_id")[column].transform("mean")
            - work.groupby("period")[column].transform("mean")
            + work[column].mean()
        )
    return float(
        np.linalg.lstsq(
            work[["treated"]].to_numpy(), work["log_units"].to_numpy(), rcond=None
        )[0][0]
    )


def test_decomposition_reproduces_the_twfe_coefficient():
    """
    The theorem, and the only check that the weights are right.

    Goodman-Bacon (2021) proves the weighted 2x2s sum exactly to the regression
    coefficient. Any error in a weight formula breaks that identity, so this is
    a far stronger test than asserting the output looks plausible - a
    decomposition that merely returned sensible-looking shares would pass an
    eyeball check and fail here.
    """
    frame = _staggered_panel()
    table = bacon.decompose(frame)

    assert table["weight"].sum() == pytest.approx(1.0)
    assert bacon.twfe_from_decomposition(table) == pytest.approx(_twfe(frame), abs=1e-10)


def test_decomposition_reproduces_twfe_under_heterogeneous_effects():
    """The identity is arithmetic, so it must hold when TWFE is *biased* too."""
    frame = _staggered_panel(heterogeneous=True)
    table = bacon.decompose(frame)
    assert bacon.twfe_from_decomposition(table) == pytest.approx(_twfe(frame), abs=1e-10)


def test_all_three_comparison_types_appear():
    table = bacon.decompose(_staggered_panel())
    assert set(table["comparison"]) == {
        bacon.TREATED_VS_NEVER, bacon.EARLY_VS_LATE, bacon.LATE_VS_EARLY,
    }


def test_units_treated_before_the_panel_opens_are_dropped():
    """
    They have no pre-period, so no 2x2 can include them - but a TWFE regression
    keeps them as ordinary controls without saying so.
    """
    frame = _staggered_panel()
    frame.loc[frame["store_id"] == 1, "treated"] = 1
    table = bacon.decompose(frame)
    assert table.attrs["n_dropped_always_treated"] >= 1


def test_decomposition_needs_a_treated_group():
    frame = _staggered_panel()
    frame["treated"] = 0
    with pytest.raises(ValueError, match="no unit is treated"):
        bacon.decompose(frame)
