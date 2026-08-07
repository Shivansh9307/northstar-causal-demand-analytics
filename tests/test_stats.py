"""
Tests for the Phase 3 regression machinery.

The load-bearing claim in Phase 3 is that two-way demeaning is equivalent to
including unit and time dummies. If that were wrong, every coefficient in the
report would be wrong in a way no amount of eyeballing would reveal, so it is
tested against a dummy-variable fit on a balanced panel where the two must agree
exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_quality import leakage  # noqa: E402
from stats import models  # noqa: E402
from utils import config  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (config.path("raw") / "fact_daily_store_sku.csv").exists(),
    reason="Generated dataset not present; run src/generation/generate_retail_dataset.py first.",
)


@pytest.fixture(scope="module")
def balanced_panel():
    """A small balanced panel with a known slope."""
    rng = np.random.default_rng(0)
    units, periods = 12, 20
    rows = []
    for u in range(units):
        unit_effect = rng.normal(0, 2)
        for t in range(periods):
            x = rng.normal(0, 1)
            z = rng.normal(0, 1)
            rows.append({
                "pair_id": f"U{u}",
                "date": f"2023-01-{t + 1:02d}",
                "x": x,
                "z": z,
                "y": 1.7 * x - 0.4 * z + unit_effect + 0.3 * t + rng.normal(0, 0.5),
            })
    return pd.DataFrame(rows)


def test_two_way_within_matches_dummy_variable_fit(balanced_panel):
    """
    The within estimator must reproduce the dummy-variable estimator exactly on a
    balanced panel. This is the assumption the whole phase rests on.
    """
    demeaned = models.two_way_within(balanced_panel, ["y", "x", "z"])
    within = sm.OLS(demeaned["y"], demeaned[["x", "z"]]).fit()

    dummies = pd.concat(
        [
            balanced_panel[["x", "z"]],
            pd.get_dummies(balanced_panel["pair_id"], drop_first=True).astype(float),
            pd.get_dummies(balanced_panel["date"], drop_first=True).astype(float),
        ],
        axis=1,
    )
    lsdv = sm.OLS(balanced_panel["y"], sm.add_constant(dummies)).fit()

    for term in ("x", "z"):
        assert within.params[term] == pytest.approx(lsdv.params[term], rel=1e-9, abs=1e-9)


def test_two_way_within_removes_unit_and_time_means(balanced_panel):
    demeaned = models.two_way_within(balanced_panel, ["y"])
    by_unit = demeaned["y"].groupby(balanced_panel["pair_id"]).mean()
    by_time = demeaned["y"].groupby(balanced_panel["date"]).mean()
    assert np.allclose(by_unit, 0, atol=1e-10)
    assert np.allclose(by_time, 0, atol=1e-10)


def test_rank_deficient_design_raises(balanced_panel):
    """
    A singular design must fail loudly. statsmodels returns coefficients with NaN
    standard errors instead of raising, which silently produced garbage before
    the guard was added.
    """
    frame = balanced_panel.copy()
    frame["x_copy"] = frame["x"]  # exact duplicate -> singular
    with pytest.raises(ValueError, match="rank deficient"):
        models.fit_within_ols(frame, "y", ["x", "x_copy", "z"], "singular")


def test_dose_response_spec_excludes_promo_mechanics():
    """
    The four discounting mechanics sum exactly to the six discount indicators,
    because Display-only is the only 0% mechanic. Including both is singular.
    """
    spec = models.dose_response_spec([f"disc_{d}" for d in models.DISCOUNT_LEVELS])
    assert not any(term.startswith("mech_") for term in spec)


# ---------------------------------------------------------------------------
# Integration checks against the real panel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def panel():
    """A store-stratified slice of the real panel, small enough to fit quickly."""
    frame = models.prepare(models.load_analysis_frame())
    keep = sorted(frame["store_id"].unique())[:6]
    return frame[frame["store_id"].isin(keep)].copy()


@pytest.fixture(scope="module")
def ground_truth():
    return pd.read_csv(
        config.path("ground_truth") / "ground_truth_simulation_parameters.csv"
    )


def test_regression_frame_is_leakage_free(panel):
    """The §7 gate must hold on whatever actually reaches a model."""
    leakage.assert_frame_is_safe(
        panel[[c for c in panel.columns if c != "true_effect"]], context="phase 3 panel"
    )


def test_dose_response_is_monotonic_and_positive(panel, ground_truth):
    """Deeper discounts must lift demand, and by more."""
    curve, _ = models.dose_response(panel, ground_truth)
    assert (curve["estimate"] > 0).all()
    assert curve["estimate"].is_monotonic_increasing


def test_dose_response_recovers_simulated_truth(panel, ground_truth):
    """
    The validated claim of Phase 3. On a six-store slice the intervals are wider
    than in the full report, so the tolerance is on the point estimate.
    """
    curve, _ = models.dose_response(panel, ground_truth)
    assert curve["error"].abs().max() < 0.05


def test_support_channels_recover_simulated_truth(panel, ground_truth):
    _, fit = models.dose_response(panel, ground_truth)
    support = models.support_channel_effects(fit, ground_truth)
    assert support["error"].abs().max() < 0.02


def test_spillover_is_negative_and_saturates(panel):
    """
    Cannibalisation must depress untreated neighbours, deepening with the number
    of promoted neighbours until it saturates.

    The generator floors the multiplier at `max(0.82, 1 - factor * count)`. With
    a mean cannibalisation factor around 0.058 that floor binds at roughly three
    concurrent promotions, so the effect flattens rather than continuing to fall.
    Asserting strict monotonicity here would be asserting something the data
    generating process does not do.
    """
    spill = models.spillover_diagnostic(panel)
    estimates = spill["estimate"].to_numpy()

    assert (estimates < 0).all()
    assert (spill["p_value"] < 0.01).all()
    # Deepens over the unsaturated range.
    assert estimates[1] < estimates[0]
    assert estimates[2] < estimates[1]
    # Never reverses beyond noise once saturated.
    assert np.all(np.diff(estimates) < 0.01)
    # The floor is real: the deepest effect stays well short of total suppression.
    assert estimates.min() > np.log(0.82) - 0.05


def test_price_and_dose_are_collinear(panel):
    """
    The identification failure is a property of the data, not of one fit. If this
    ever stops holding, the elasticity claims in the report must be revisited.
    """
    promoted = panel[panel["promo_flag"] == 1]
    corr = np.corrcoef(promoted["log_price_ratio"], promoted["discount_pct"] / 10)[0, 1]
    assert corr < -0.99
