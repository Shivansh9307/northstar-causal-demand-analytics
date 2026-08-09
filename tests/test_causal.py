"""
Tests for the Phase 4 causal machinery.

The most dangerous failure in this phase is silent: benchmarking against the
wrong quantity. `true_promo_uplift_pct` is a structural coefficient, the
arithmetic ATT and the log-scale ATT differ by a Jensen gap, and observed sales
are censored. Any of those confusions produces a confident, meaningless recovery
number, so the target construction is tested before the estimators are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from causal import did, estimands, psm  # noqa: E402
from utils import config  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (config.path("raw") / "fact_daily_store_sku.csv").exists(),
    reason="Generated dataset not present; run src/generation/generate_retail_dataset.py first.",
)


@pytest.fixture(scope="module")
def panel():
    """A six-store slice, prepared exactly as the report prepares the full panel."""
    from causal import phase4_report

    frame = phase4_report.load_frame()
    keep = sorted(frame["store_id"].unique())[:6]
    return frame[frame["store_id"].isin(keep)].copy()


@pytest.fixture(scope="module")
def truth():
    return estimands.load_ground_truth()


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_panel():
    """
    The whole panel, lean columns only.

    `true_realised_att_pct` averages over every treated row of a SKU across all
    stores, so the reconciliation below only holds on the complete panel - a
    slice has a different mix of discounts and support flags.
    """
    from features import star_schema

    con = star_schema.connect()
    try:
        star_schema.load_raw_tables(con)
        star_schema.build_promotion_bridge(con)
        star_schema.build_analytics_view(con)
        return con.execute(
            """
            SELECT sku_id, promo_flag, promo_type, discount_pct,
                   actual_unit_price_gbp, regular_unit_price_gbp,
                   display_support_flag, email_or_app_support_flag, leaflet_support_flag
            FROM analytics_daily ORDER BY date, store_id, sku_id
            """
        ).df()
    finally:
        con.close()


def test_reconstructed_multiplier_matches_generator(full_panel, truth):
    """
    The reconstruction must reproduce the effect the generator recorded during
    simulation. If it does not, every recovery number in Phase 4 is measured
    against the wrong target.
    """
    result = estimands.verify_against_generator(full_panel, truth)
    assert result["max_abs_difference_pp"] < 0.01
    assert result["skus_compared"] > 100


def test_verification_rejects_a_partial_panel(panel, truth):
    """
    The guard must fire on a subset rather than quietly reconciling against a
    differently-weighted average - that would let a wrong benchmark through.
    """
    with pytest.raises(AssertionError, match="cannot be trusted"):
        estimands.verify_against_generator(panel, truth)


def test_multiplier_is_one_on_untreated_rows(panel, truth):
    multiplier = estimands.treatment_multiplier(panel, truth)
    untreated = panel["promo_flag"] == 0
    assert np.allclose(multiplier[untreated], 1.0)


def test_multiplier_exceeds_one_on_treated_rows(panel, truth):
    multiplier = estimands.treatment_multiplier(panel, truth)
    treated = panel["promo_flag"] == 1
    assert (multiplier[treated] > 1.0).all()


def test_jensen_gap_is_positive(panel, truth):
    """
    The arithmetic ATT must exceed the log-scale ATT whenever the multiplier
    varies. Benchmarking a log coefficient against the arithmetic figure would
    manufacture an error of exactly this size.
    """
    values = estimands.estimands(panel, truth)
    assert values["jensen_gap_pct"] > 0
    assert values["arithmetic_att_pct"] > values["log_att_as_pct"]


def test_censoring_reduces_the_observable_effect(panel, truth):
    """Stockouts bite harder on promoted rows, so the observable effect is smaller."""
    values = estimands.estimands(panel, truth)
    assert values["realised_share_treated"] < values["realised_share_control"]
    assert values["log_att_observed"] < values["log_att"]


def test_promo_uplift_column_is_not_the_att(truth):
    """
    Guard against the Phase 1R mistake returning: the structural coefficient and
    the realised ATT are different by a wide margin.
    """
    structural = truth["true_promo_uplift_pct"].mean()
    realised = truth["true_realised_att_pct"].mean()
    assert realised > 3 * structural


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def test_naive_overstates_the_effect(panel, truth):
    values = estimands.estimands(panel, truth)
    naive = did.naive_estimates(panel)
    cross_sectional = naive.loc[0, "estimate"]
    assert cross_sectional > values["log_att_observed"]


def test_did_reduces_naive_bias(panel):
    """
    The whole point of the phase: correcting for time and control selection must
    move the estimate materially towards the truth.
    """
    naive = did.naive_estimates(panel).loc[0, "estimate"]
    variants = did.did_variants(panel).set_index("estimator")
    best = variants.loc["twfe_clean_seasonal_fe", "estimate"]
    assert best < naive


def test_uncannibalised_controls_lower_the_estimate(panel):
    """
    Phase 3 found controls in a promoted category are depressed. Removing them
    must reduce the estimated effect, not raise it.
    """
    variants = did.did_variants(panel).set_index("estimator")
    assert variants.loc["twfe_out_of_category", "estimate"] < variants.loc["twfe_all", "estimate"]


def test_event_time_reference_is_zero(panel):
    events = did.event_study(panel, window=7)
    reference = events.loc[events["event_time"] == -1, "estimate"].iloc[0]
    assert reference == 0.0


def test_event_study_jumps_at_treatment(panel):
    """The effect on the first promoted day must exceed every pre-period lead."""
    events = did.event_study(panel, window=7)
    at_start = events.loc[events["event_time"] == 0, "estimate"].iloc[0]
    leads = events[events["event_time"] < 0]["estimate"]
    assert at_start > leads.max() * 5


def test_build_event_time_assigns_the_nearest_start(panel):
    timed = did.build_event_time(panel, window=14)
    assert timed["event_time"].abs().max() <= 14
    # Day zero must coincide with an actual promotion.
    day_zero = timed[timed["event_time"] == 0]
    assert (day_zero["promo_flag"] == 1).all()


# ---------------------------------------------------------------------------
# Propensity weighting
# ---------------------------------------------------------------------------

def test_propensity_design_is_leakage_free(panel):
    from data_quality import leakage

    design = psm.build_design(panel)
    leakage.assert_no_leakage(list(design.columns), context="propensity design")


def test_weighting_improves_balance_on_first_promo_day(panel):
    """
    Restricting to the first day of each promotion makes the lagged demand
    history genuinely pre-treatment, and balance should then be achievable.
    """
    ordered = panel.sort_values(["pair_id", "date"]).reset_index(drop=True)
    previous = ordered.groupby("pair_id")["promo_flag"].shift(1).fillna(0)
    first_day = ordered[~((ordered["promo_flag"] == 1) & (previous == 1))]

    fitted = psm.fit_propensity(first_day, sample_size=200_000, seed=42)
    balance = psm.standardised_differences(
        first_day, fitted["design"], fitted["propensity"],
        psm.CONTINUOUS_COVARIATES + psm.BINARY_COVARIATES,
    )
    assert fitted["converged"]
    assert balance["smd_after"].abs().max() < balance["smd_before"].abs().max()
    assert int(balance["balanced_after"].sum()) >= len(balance) - 1


def test_ipw_and_did_bracket_the_truth(panel, truth):
    """
    The headline reconciliation: the two corrected estimators sit on opposite
    sides of the simulated effect, because their assumptions fail in opposite
    directions.
    """
    values = estimands.estimands(panel, truth)
    target = values["log_att_observed"]

    variants = did.did_variants(panel).set_index("estimator")
    did_estimate = variants.loc["twfe_clean_seasonal_fe", "estimate"]

    fitted = psm.fit_propensity(panel, sample_size=200_000, seed=42)
    ipw_estimate = psm.ipw_att(panel, fitted["propensity"])["att_regression"]

    assert ipw_estimate < target < did_estimate


# ---------------------------------------------------------------------------
# Propensity score matching
# ---------------------------------------------------------------------------


def test_matching_pairs_treated_rows_with_nearer_controls(panel):
    """
    The point of matching is that the paired control is closer on the score than
    an arbitrary control would be. If that does not hold the matcher is not
    matching, and every downstream number is a slower way to compute the naive
    difference.
    """
    fitted = psm.fit_propensity(panel, sample_size=200_000, seed=42)
    result = psm.match_att(panel, fitted["propensity"])
    balance = psm.matched_balance(
        panel, fitted["design"], fitted["propensity"],
        psm.CONTINUOUS_COVARIATES + psm.BINARY_COVARIATES,
    )
    assert balance["smd_after"].abs().max() < balance["smd_before"].abs().max()
    assert result["share_treated_matched"] > 0.5
    assert result["ci_low"] < result["att_regression"] < result["ci_high"]


def test_matching_caliper_binds(panel):
    """
    A caliper that never excludes anything is not a caliper. Shrinking it must
    drop treated rows; if it does not, the distance comparison is not being
    applied and a treated row with no comparable control would still get one.
    """
    fitted = psm.fit_propensity(panel, sample_size=200_000, seed=42)
    wide = psm.match_att(panel, fitted["propensity"], caliper_sd=0.2)
    narrow = psm.match_att(panel, fitted["propensity"], caliper_sd=0.0005)

    assert narrow["caliper"] < wide["caliper"]
    assert narrow["n_treated_matched"] < wide["n_treated_matched"]
    assert narrow["n_treated_dropped"] > 0


def test_matching_recovers_a_known_effect():
    """
    On data where the answer is constructed, matching must return it.

    Treatment is assigned by a covariate that also drives the outcome, so the
    naive difference is biased by design and only an estimator that conditions
    on the covariate can recover the +0.40 that was actually applied.
    """
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 4_000
    x = rng.normal(size=n)
    treated = rng.random(n) < 1 / (1 + np.exp(-1.5 * x))
    effect = 0.40
    log_units = 0.9 * x + effect * treated + rng.normal(scale=0.25, size=n)

    frame = pd.DataFrame({
        "promo_flag": treated.astype(int),
        "log_units": log_units,
        "pair_id": np.arange(n) % 200,
    })
    propensity = pd.Series(1 / (1 + np.exp(-1.5 * x)), index=frame.index)

    naive = log_units[treated].mean() - log_units[~treated].mean()
    matched = psm.match_att(frame, propensity, caliper_sd=0.05)["att"]

    assert abs(naive - effect) > 0.3, "the test data is not confounded enough to be a test"
    assert abs(matched - effect) < 0.05, f"matching recovered {matched:.3f}, not {effect}"


def test_matching_needs_both_arms():
    import pandas as pd

    frame = pd.DataFrame({
        "promo_flag": [0, 0, 0],
        "log_units": [1.0, 2.0, 3.0],
        "pair_id": [1, 1, 2],
    })
    with pytest.raises(ValueError, match="treated and untreated"):
        psm.match_att(frame, pd.Series([0.1, 0.2, 0.3], index=frame.index))
