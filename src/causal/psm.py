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

LOGGER = logging.getLogger("promopulse.psm")

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
    return pd.DataFrame(rows).sort_values("smd_before", key=abs, ascending=False).reset_index(drop=True)


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
        "share_control_above_treated_min": float((propensity[~treated] > propensity[treated].min()).mean()),
    }
