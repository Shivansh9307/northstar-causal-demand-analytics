"""
Propensity score estimation, IPW and balance diagnostics.

Why this is a genuine cross-check and not a repeat of the DiD
------------------------------------------------------------
DiD and IPW lean on different assumptions. DiD needs parallel trends and buys
identification from within-unit variation over time. IPW needs selection on
observables and buys it from comparing treated rows to untreated rows that look
alike on measured covariates. If both land in the same place, that is meaningful
corroboration; if they diverge, at least one assumption is failing and the report
should say which.

What is allowed in the propensity model
---------------------------------------
Only variables a planner could have seen before deciding to promote:

* SKU attributes - margin, brand type, seasonal profile, elasticity segment;
* store attributes - footfall, format, competition;
* calendar - month, day of week, holiday and payday flags;
* **lagged** demand history - the rolling 7- and 28-day averages, which the
  generator computes from sales strictly before the current day. That history is
  what carries the "weakening momentum" selection driver, and it is the reason a
  propensity model can adjust for it at all (Phase 1R made that trend real
  instead of an unobservable random draw).

Contemporaneous outcomes, the discount itself and anything downstream of
treatment are excluded, and the whole design matrix is passed through the §7
leakage checker.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_quality import leakage  # noqa: E402

LOGGER = logging.getLogger("northstar.psm")

CONTINUOUS_COVARIATES = [
    "baseline_gross_margin_pct",
    "average_daily_footfall",
    "competition_intensity_score",
    "log_lag_rolling_7",
    "log_lag_rolling_28",
    "momentum_ratio",
]

CATEGORICAL_COVARIATES = [
    "category",
    "brand_type",
    "seasonal_profile",
    "price_elasticity_segment",
    "store_format",
    "month",
    "day_of_week",
]

BINARY_COVARIATES = [
    "is_weekend",
    "is_bank_holiday",
    "is_payday_window",
    "is_school_holiday",
]


def build_design(frame: pd.DataFrame) -> pd.DataFrame:
    """Assemble the pre-treatment covariate matrix."""
    work = frame.copy()
    work["log_lag_rolling_7"] = np.log1p(work["rolling_7_day_avg_units_sold"].clip(lower=0))
    work["log_lag_rolling_28"] = np.log1p(work["rolling_28_day_avg_units_sold"].clip(lower=0))
    # Short-run momentum: recent demand against the slower baseline. Negative
    # values are the "weakening line" a merchandiser reaches for a promotion on.
    work["momentum_ratio"] = (
        work["rolling_7_day_avg_units_sold"].clip(lower=0.1)
        / work["rolling_28_day_avg_units_sold"].clip(lower=0.1)
    )

    for flag in BINARY_COVARIATES:
        work[flag] = work[flag].astype(int)

    design = pd.concat(
        [
            work[CONTINUOUS_COVARIATES + BINARY_COVARIATES].astype(float),
            *[
                pd.get_dummies(work[column], prefix=column, drop_first=True).astype(float)
                for column in CATEGORICAL_COVARIATES
            ],
        ],
        axis=1,
    )
    leakage.assert_no_leakage(list(design.columns), context="propensity model design")
    return design


def fit_propensity(
    frame: pd.DataFrame, sample_size: int, seed: int
) -> Dict[str, object]:
    """
    Fit a logistic propensity model.

    Fitted on a seeded sample for tractability, then scored on every row - the
    coefficients are what is expensive, prediction is not.
    """
    design = build_design(frame)
    treatment = frame["promo_flag"].astype(int).to_numpy()

    rng = np.random.default_rng(seed)
    if len(design) > sample_size:
        index = rng.choice(len(design), size=sample_size, replace=False)
    else:
        index = np.arange(len(design))

    exog = sm.add_constant(design, has_constant="add")
    LOGGER.info("Fitting propensity model on %d rows, %d covariates", len(index), exog.shape[1])
    model = sm.Logit(treatment[index], exog.iloc[index].to_numpy())
    result = model.fit(disp=False, maxiter=200)

    propensity = result.predict(exog.to_numpy())
    return {
        "design": design,
        "propensity": pd.Series(propensity, index=frame.index),
        "pseudo_r2": float(result.prsquared),
        "n_fit_rows": int(len(index)),
        "n_covariates": int(exog.shape[1]),
        "converged": bool(result.mle_retvals["converged"]),
    }


def ipw_att(
    frame: pd.DataFrame, propensity: pd.Series, trim: float = 0.01
) -> Dict[str, float]:
    """
    Inverse-probability-weighted ATT on the log outcome.

    ATT weights treated rows at 1 and untreated rows at e/(1-e), reweighting the
    controls to look like the treated population. Rows outside the common support
    are trimmed rather than allowed to carry extreme weights.
    """
    treated = frame["promo_flag"].astype(bool).to_numpy()
    outcome = frame["log_units"].to_numpy()
    e = propensity.to_numpy().clip(trim, 1 - trim)

    in_support = (propensity.to_numpy() > trim) & (propensity.to_numpy() < 1 - trim)
    treated_in = treated & in_support
    control_in = (~treated) & in_support

    weights = e[control_in] / (1 - e[control_in])
    treated_mean = outcome[treated_in].mean()
    control_mean = np.average(outcome[control_in], weights=weights)
    att = float(treated_mean - control_mean)

    # Cluster-robust standard error via a weighted regression on the pair.
    subset = frame.loc[in_support].copy()
    subset["ipw"] = np.where(
        subset["promo_flag"].astype(bool), 1.0, e[in_support] / (1 - e[in_support])
    )
    exog = sm.add_constant(subset["promo_flag"].astype(float))
    wls = sm.WLS(subset["log_units"], exog, weights=subset["ipw"]).fit(
        cov_type="cluster", cov_kwds={"groups": subset["pair_id"]}
    )
    conf = wls.conf_int()

    return {
        "att": att,
        "att_regression": float(wls.params["promo_flag"]),
        "ci_low": float(conf.loc["promo_flag", 0]),
        "ci_high": float(conf.loc["promo_flag", 1]),
        "n_treated": int(treated_in.sum()),
        "n_control": int(control_in.sum()),
        "n_trimmed": int((~in_support).sum()),
        "max_weight": float(weights.max()),
    }


def _nearest_neighbour_pairs(
    frame: pd.DataFrame, propensity: pd.Series, caliper_sd: float
) -> Dict[str, object]:
    """
    Pair each treated row with its nearest untreated row on the propensity logit.

    Matching on the logit rather than the raw score is deliberate: the score
    compresses near 0 and 1, so a fixed caliper there spans far more covariate
    distance than the same caliper mid-distribution.

    Nearest neighbour is found by binary search on the sorted control scores.
    Once sorted, the insertion point and the element before it are the only two
    candidates, which turns an O(n_treated x n_control) scan into a sort plus a
    searchsorted — the difference between seconds and hours at this row count.
    """
    treated_mask = frame["promo_flag"].astype(bool).to_numpy()
    e = np.clip(propensity.to_numpy(), 1e-6, 1 - 1e-6)
    score = np.log(e / (1 - e))
    caliper = caliper_sd * float(score.std())

    treated_idx = np.flatnonzero(treated_mask)
    control_idx = np.flatnonzero(~treated_mask)
    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise ValueError("matching needs both treated and untreated rows")

    order = np.argsort(score[control_idx], kind="stable")
    sorted_score = score[control_idx][order]
    sorted_idx = control_idx[order]

    positions = np.searchsorted(sorted_score, score[treated_idx])
    left = np.clip(positions - 1, 0, len(sorted_score) - 1)
    right = np.clip(positions, 0, len(sorted_score) - 1)

    dist_left = np.abs(score[treated_idx] - sorted_score[left])
    dist_right = np.abs(score[treated_idx] - sorted_score[right])
    take_left = dist_left <= dist_right
    nearest = np.where(take_left, left, right)
    within = np.where(take_left, dist_left, dist_right) <= caliper

    if not within.any():
        raise ValueError("no treated row found a control inside the caliper")

    return {
        "treated": treated_idx[within],
        "control": sorted_idx[nearest[within]],
        "caliper": float(caliper),
        "within": within,
        "treated_mask": treated_mask,
    }


def match_att(
    frame: pd.DataFrame,
    propensity: pd.Series,
    caliper_sd: float = 0.2,
) -> Dict[str, float]:
    """
    Nearest-neighbour propensity score matching, with replacement.

    §6 Phase 4.3 asks for "propensity score matching / IPW". IPW was built and
    matching was not, and the two are not interchangeable even though both lean
    on selection on observables:

    * IPW keeps every in-support row and reweights it. One control with an
      extreme score can carry a large share of the estimate.
    * Matching discards controls it cannot pair, which trades sample size for a
      comparison the reader can actually picture, and makes the failure mode
      visible - if the caliper drops most of the treated rows, the overlap was
      not there and no weighting scheme would have fixed it.

    Matching on the score itself rather than the covariates is the standard
    reduction: conditioning on `e(X)` is sufficient when treatment is ignorable
    given `X`.

    With replacement: treated rows outnumber usable controls in the dense part of
    the score distribution, and matching without replacement would make the
    estimate depend on the order rows happen to arrive in. The cost is that one
    control can carry several treated rows, which `max_control_reuse` reports so
    the reader can judge how thin the comparison got.

    Returns the ATT with a cluster-robust CI on the matched sample. No sampling
    is involved, so the result is deterministic.
    """
    pairs = _nearest_neighbour_pairs(frame, propensity, caliper_sd)
    matched_treated, matched_control = pairs["treated"], pairs["control"]
    outcome = frame["log_units"].to_numpy()

    att = float(outcome[matched_treated].mean() - outcome[matched_control].mean())

    # A matched control reused k times counts k times, so the CI is computed on
    # the stacked matched sample rather than on unique rows. Clustering stays on
    # the pair, as everywhere else in this phase.
    stacked = pd.concat(
        [frame.iloc[matched_treated], frame.iloc[matched_control]],
        axis=0,
    )
    exog = sm.add_constant(stacked["promo_flag"].astype(float))
    fitted = sm.OLS(stacked["log_units"], exog).fit(
        cov_type="cluster", cov_kwds={"groups": stacked["pair_id"]}
    )
    conf = fitted.conf_int()

    _, reuse_counts = np.unique(matched_control, return_counts=True)
    within = pairs["within"]

    return {
        "att": att,
        "att_regression": float(fitted.params["promo_flag"]),
        "ci_low": float(conf.loc["promo_flag", 0]),
        "ci_high": float(conf.loc["promo_flag", 1]),
        "caliper": pairs["caliper"],
        "n_treated_matched": int(len(matched_treated)),
        "n_treated_dropped": int((~within).sum()),
        "n_control_used": int(len(reuse_counts)),
        "share_treated_matched": float(within.mean()),
        "max_control_reuse": int(reuse_counts.max()),
    }


def matched_balance(
    frame: pd.DataFrame,
    design: pd.DataFrame,
    propensity: pd.Series,
    columns: Sequence[str],
    caliper_sd: float = 0.2,
) -> pd.DataFrame:
    """
    Standardised mean differences before and after matching.

    The weighted equivalent of `standardised_differences`, computed on the
    matched sample instead. Balance is the only thing that makes a matched
    estimate credible, so reporting the estimate without it would repeat exactly
    the omission this function exists to close.
    """
    pairs = _nearest_neighbour_pairs(frame, propensity, caliper_sd)
    matched_treated, matched_control = pairs["treated"], pairs["control"]
    treated_mask = pairs["treated_mask"]

    rows: List[Dict[str, object]] = []
    for column in columns:
        values = design[column].to_numpy(dtype=float)
        pooled_sd = np.sqrt(
            (values[treated_mask].var() + values[~treated_mask].var()) / 2
        )
        pooled_sd = pooled_sd if pooled_sd > 1e-12 else 1.0
        before = (values[treated_mask].mean() - values[~treated_mask].mean()) / pooled_sd
        after = (values[matched_treated].mean() - values[matched_control].mean()) / pooled_sd
        rows.append({
            "covariate": column,
            "smd_before": float(before),
            "smd_after": float(after),
            "balanced_after": bool(abs(after) < 0.1),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("smd_before", key=abs, ascending=False)
        .reset_index(drop=True)
    )


def standardised_differences(
    frame: pd.DataFrame,
    design: pd.DataFrame,
    propensity: pd.Series,
    columns: Sequence[str],
    trim: float = 0.01,
) -> pd.DataFrame:
    """
    Standardised mean differences before and after weighting.

    Convention: |SMD| below 0.1 is treated as adequate balance.
    """
    treated = frame["promo_flag"].astype(bool).to_numpy()
    e = propensity.to_numpy()
    in_support = (e > trim) & (e < 1 - trim)
    weights = np.where(treated, 1.0, e.clip(trim, 1 - trim) / (1 - e.clip(trim, 1 - trim)))

    rows: List[Dict[str, object]] = []
    for column in columns:
        values = design[column].to_numpy(dtype=float)
        t_mask = treated & in_support
        c_mask = (~treated) & in_support

        t_mean, c_mean = values[t_mask].mean(), values[c_mask].mean()
        pooled_sd = np.sqrt((values[t_mask].var() + values[c_mask].var()) / 2)
        pooled_sd = pooled_sd if pooled_sd > 1e-12 else 1.0
        before = (t_mean - c_mean) / pooled_sd

        w = weights[c_mask]
        c_mean_w = np.average(values[c_mask], weights=w)
        after = (t_mean - c_mean_w) / pooled_sd

        rows.append({
            "covariate": column,
            "smd_before": float(before),
            "smd_after": float(after),
            "balanced_after": bool(abs(after) < 0.1),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("smd_before", key=abs, ascending=False)
        .reset_index(drop=True)
    )


def overlap_summary(frame: pd.DataFrame, propensity: pd.Series) -> Dict[str, float]:
    """Common-support diagnostics for the propensity distribution."""
    treated = frame["promo_flag"].astype(bool)
    return {
        "treated_min": float(propensity[treated].min()),
        "treated_max": float(propensity[treated].max()),
        "control_min": float(propensity[~treated].min()),
        "control_max": float(propensity[~treated].max()),
        "treated_median": float(propensity[treated].median()),
        "control_median": float(propensity[~treated].median()),
        "share_control_above_treated_min": float(
            (propensity[~treated] > propensity[treated].min()).mean()
        ),
    }
