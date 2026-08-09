"""
Phase 3 regression models (PROJECT_ARCHITECTURE.md §6 Phase 3).

Identification note — read before interpreting any coefficient here
-------------------------------------------------------------------
In this panel, **price moves only through promotions**. Zero rows are discounted
outside a promotion. So a regression of log demand on log price cannot, on its
own, separate the price response from the promotional uplift that arrives at the
same moment: both are functions of the same discount.

Three features of the data provide the separating variation, and every
specification below is built around them:

1. **Display-only promotions** (11.3% of promoted rows) carry a 0% discount but
   full display and support activity. They identify the non-price component of a
   promotion, so `log_price_ratio` is left measuring the *incremental* effect of
   discounting on top of simply being promoted.
2. **Discount depth is assigned independently of promotion type** (roughly
   uniform 5-30% within each discounting mechanic), so the mechanic dummies are
   not confounded with depth.
3. **Price-elasticity and promotion-sensitivity segments are correlated but not
   collinear** (corr of the underlying parameters is -0.33), leaving
   cross-sectional variation to estimate segment-specific elasticities.

Even so, the decomposition leans on functional form. The report quantifies what
remains with VIF and validates the result against the known simulated elasticity
rather than asserting the estimate is right.

Estimator
---------
Two-way within (fixed effects absorbed by demeaning) on store x SKU pair and on
date, with standard errors clustered on the pair. Pair effects absorb base
demand, store size and SKU price level; date effects absorb seasonality,
holidays, weather and any other common shock. Demeaning rather than dummies
keeps a 2.19M-row design matrix tractable - 731 date dummies plus 3,000 pair
dummies would not be.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_quality import leakage  # noqa: E402
from features import star_schema  # noqa: E402

LOGGER = logging.getLogger("promopulse.stats")

PAIR_KEYS = ["store_id", "sku_id"]

PROMO_MECHANICS = ["Bundle", "Clubcard-style Price", "Multi-buy", "Percent Off"]

DISCOUNT_LEVELS = [5, 10, 15, 20, 25, 30]

SUPPORT_FLAGS = [
    "display_support_flag",
    "email_or_app_support_flag",
    "leaflet_support_flag",
]


@dataclass
class FitResult:
    """A fitted specification with the pieces the report needs."""

    name: str
    params: pd.Series
    bse: pd.Series
    conf_int: pd.DataFrame
    pvalues: pd.Series
    nobs: int
    rsquared: float
    n_clusters: int

    def row(self, term: str) -> Dict[str, float]:
        return {
            "estimate": float(self.params[term]),
            "std_err": float(self.bse[term]),
            "ci_low": float(self.conf_int.loc[term, 0]),
            "ci_high": float(self.conf_int.loc[term, 1]),
            "p_value": float(self.pvalues[term]),
        }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_analysis_frame() -> pd.DataFrame:
    """
    Pull the modelling columns from the star schema.

    Only feature-safe columns are requested; the selection is passed through the
    §7 leakage checker before it is returned, so a leaking column cannot enter a
    model by accident.
    """
    con = star_schema.connect()
    try:
        star_schema.load_raw_tables(con)
        star_schema.build_promotion_bridge(con)
        star_schema.build_analytics_view(con)
        frame = con.execute(
            """
            SELECT
                date, store_id, sku_id,
                units_sold,
                actual_unit_price_gbp, regular_unit_price_gbp,
                discount_pct, promo_flag, promo_type,
                display_support_flag, email_or_app_support_flag, leaflet_support_flag,
                store_footfall,
                category, price_elasticity_segment, promotion_sensitivity_segment,
                demand_volatility_segment,
                -- Read via DuckDB, not pandas: the literal category "None" in
                -- seasonal_profile is in pandas' default na_values list, so a
                -- direct read_csv turns it into NaN.
                seasonal_profile, brand_type, month, day_of_week
            FROM analytics_daily
            -- Explicit ordering: DuckDB's parallel scan returns rows in a
            -- nondeterministic order, which made the sampled count-model fit
            -- (and its dispersion estimate) vary between runs.
            ORDER BY date, store_id, sku_id
            """
        ).df()
    finally:
        con.close()

    leakage.assert_frame_is_safe(frame, context="Phase 3 regression frame")
    return frame


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive model variables."""
    out = frame.copy()
    out["promo_flag"] = out["promo_flag"].astype(int)
    out["price_ratio"] = out["actual_unit_price_gbp"] / out["regular_unit_price_gbp"]
    out["log_price_ratio"] = np.log(out["price_ratio"])
    out["log_units"] = np.log1p(out["units_sold"])
    out["log_footfall"] = np.log(out["store_footfall"].clip(lower=1))

    for flag in ("display_support_flag", "email_or_app_support_flag", "leaflet_support_flag"):
        out[flag] = out[flag].astype(int)

    # Display-only is the within-promotion reference category: promoted, no discount.
    for mechanic in PROMO_MECHANICS:
        out[f"mech_{_slug(mechanic)}"] = (out["promo_type"] == mechanic).astype(int)

    out["pair_id"] = out["store_id"] + "|" + out["sku_id"]
    return out


def _slug(text: str) -> str:
    return text.lower().replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Two-way within transformation
# ---------------------------------------------------------------------------

def two_way_within(
    frame: pd.DataFrame,
    columns: Sequence[str],
    unit_key: str = "pair_id",
    time_key: str = "date",
) -> pd.DataFrame:
    """
    Absorb unit and time fixed effects by two-way demeaning.

    x_tilde = x - mean_unit(x) - mean_time(x) + mean(x)

    For a balanced panel this is exactly equivalent to including a full set of
    unit and time dummies, which is why it is safe to do it this way rather than
    materialising 3,731 indicator columns. `tests/test_stats.py` asserts that
    equivalence against a dummy-variable fit on a small subsample.
    """
    demeaned = pd.DataFrame(index=frame.index)
    for column in columns:
        series = frame[column].astype(float)
        unit_mean = series.groupby(frame[unit_key]).transform("mean")
        time_mean = series.groupby(frame[time_key]).transform("mean")
        demeaned[column] = series - unit_mean - time_mean + series.mean()
    return demeaned


def fit_within_ols(
    frame: pd.DataFrame,
    outcome: str,
    regressors: Sequence[str],
    name: str,
    n_absorbed: int | None = None,
    unit_key: str = "pair_id",
    time_key: str = "date",
) -> FitResult:
    """
    OLS on two-way demeaned data with standard errors clustered on the pair.

    Clustering matters more than usual here: 2.19M rows are only ~3,000
    independent units, and unclustered errors would be roughly 25x too small.
    """
    columns = [outcome, *regressors]
    demeaned = two_way_within(frame, columns, unit_key=unit_key, time_key=time_key)
    if demeaned.isna().any().any():
        raise ValueError(
            f"Demeaning produced NaN for '{name}'. Check that '{unit_key}' and "
            f"'{time_key}' have no missing values — pandas reads the literal "
            "string 'None' as NaN, which silently breaks grouping keys."
        )
    endog = demeaned[outcome]
    exog = demeaned[list(regressors)]

    # Fail loudly on a singular design. statsmodels will happily return
    # nonsense coefficients with NaN standard errors instead of raising, which
    # is exactly the silent-failure mode this project fixed in Phase 1R.
    rank = np.linalg.matrix_rank(exog.to_numpy())
    if rank < exog.shape[1]:
        raise ValueError(
            f"Design matrix for '{name}' is rank deficient: rank {rank} < "
            f"{exog.shape[1]} regressors. Some term is an exact linear "
            f"combination of the others. Regressors: {list(regressors)}"
        )

    model = sm.OLS(endog, exog)
    groups = frame["pair_id"]
    result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

    # Demeaning consumes degrees of freedom the OLS object does not know about.
    if n_absorbed is None:
        n_absorbed = frame["pair_id"].nunique() + frame["date"].nunique() - 1

    conf = result.conf_int()
    conf.columns = [0, 1]
    return FitResult(
        name=name,
        params=result.params,
        bse=result.bse,
        conf_int=conf,
        pvalues=result.pvalues,
        nobs=int(result.nobs),
        rsquared=float(result.rsquared),
        n_clusters=int(groups.nunique()),
    )


# ---------------------------------------------------------------------------
# Elasticity by group
# ---------------------------------------------------------------------------

def elasticity_by_group(
    frame: pd.DataFrame,
    group_column: str,
    base_regressors: Sequence[str],
    name: str,
) -> pd.DataFrame:
    """
    Estimate a separate price elasticity per group by interacting
    log_price_ratio with the group indicator.

    Interactions are formed *before* demeaning - demeaning a product is not the
    product of demeaned terms.
    """
    work = frame.copy()
    groups = sorted(work[group_column].dropna().unique())
    interaction_terms: List[str] = []
    for group in groups:
        term = f"lpr_x_{_slug(str(group))}"
        work[term] = work["log_price_ratio"] * (work[group_column] == group)
        interaction_terms.append(term)

    regressors = [*interaction_terms, *[r for r in base_regressors if r != "log_price_ratio"]]
    fit = fit_within_ols(work, "log_units", regressors, name)

    rows = []
    for group, term in zip(groups, interaction_terms):
        stats = fit.row(term)
        rows.append(
            {
                group_column: group,
                "n_rows": int((work[group_column] == group).sum()),
                **stats,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Non-parametric dose response
# ---------------------------------------------------------------------------

def add_discount_dummies(frame: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """One indicator per discount level. Display-only (0%) is the reference."""
    out = frame.copy()
    terms = []
    for level in DISCOUNT_LEVELS:
        term = f"disc_{level}"
        out[term] = (out["discount_pct"] == level).astype(float)
        terms.append(term)
    return out, terms


def dose_response_spec(discount_terms: Iterable[str]) -> List[str]:
    """
    Regressors for the dose-response model.

    The promotion-mechanic dummies are deliberately excluded: Display-only is the
    only mechanic carrying a 0% discount, so the four discounting mechanics sum
    exactly to the six discount indicators and the design would be singular.
    Mechanic is assigned independently of depth, so omitting it marginalises over
    mechanics rather than biasing the dose coefficients.
    """
    return [*discount_terms, "promo_flag", *SUPPORT_FLAGS, "log_footfall"]


def true_price_dose_effect(frame: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.Series:
    """
    The log effect the generator actually applied through the two price-linked
    channels: the elasticity response to the discount, plus the dose-dependent
    promotional uplift.

        elasticity * log(1 - d/100)  +  log(1 + uplift * d/10)

    This is the quantity the dose-response coefficients should recover, measured
    relative to the Display-only reference.
    """
    truth = ground_truth.set_index("sku_id")
    elasticity = frame["sku_id"].map(truth["true_price_elasticity"]).astype(float)
    uplift = frame["sku_id"].map(truth["true_promo_uplift_pct"]).astype(float) / 100
    discount = frame["discount_pct"].astype(float)
    return elasticity * np.log(1 - discount / 100) + np.log1p(uplift * discount / 10)


def dose_response(frame: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the promotional dose-response curve and check it against the truth.

    No functional form is imposed on the discount: each level gets its own
    indicator, so the curve is whatever the data says it is.
    """
    work, terms = add_discount_dummies(frame)
    work["true_effect"] = true_price_dose_effect(work, ground_truth)
    promoted = work["promo_flag"] == 1
    reference = work.loc[promoted & (work["discount_pct"] == 0), "true_effect"].mean()

    fit = fit_within_ols(work, "log_units", dose_response_spec(terms), "dose_response")

    rows = []
    for level, term in zip(DISCOUNT_LEVELS, terms):
        stats = fit.row(term)
        true_value = (
            work.loc[promoted & (work["discount_pct"] == level), "true_effect"].mean()
            - reference
        )
        rows.append({
            "discount_pct": level,
            "n_rows": int((work["discount_pct"] == level).sum()),
            **stats,
            "true_effect": float(true_value),
            "error": stats["estimate"] - float(true_value),
            "ci_covers_truth": bool(stats["ci_low"] <= true_value <= stats["ci_high"]),
            "estimated_lift_pct": float(np.expm1(stats["estimate"]) * 100),
            "true_lift_pct": float(np.expm1(true_value) * 100),
        })
    return pd.DataFrame(rows), fit


def support_channel_effects(fit: FitResult, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Compare the estimated non-price promotional channels against the truth."""
    truths = {
        "display_support_flag": np.log1p(ground_truth["true_display_uplift_pct"].mean() / 100),
        "email_or_app_support_flag": np.log1p(
            ground_truth["true_email_app_uplift_pct"].mean() / 100
        ),
        # The generator applies a flat 7% leaflet uplift, not a per-SKU parameter.
        "leaflet_support_flag": float(np.log1p(0.07)),
    }
    labels = {
        "display_support_flag": "In-store display",
        "email_or_app_support_flag": "Email / app",
        "leaflet_support_flag": "Leaflet",
    }
    rows = []
    for term, true_value in truths.items():
        stats = fit.row(term)
        rows.append({
            "channel": labels[term],
            **stats,
            "true_effect": true_value,
            "error": stats["estimate"] - true_value,
            "ci_covers_truth": bool(stats["ci_low"] <= true_value <= stats["ci_high"]),
        })
    return pd.DataFrame(rows)


def dose_response_by_group(
    frame: pd.DataFrame, group_column: str, ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """Dose-response curve estimated separately within each group."""
    work, terms = add_discount_dummies(frame)
    work["true_effect"] = true_price_dose_effect(work, ground_truth)
    groups = sorted(work[group_column].dropna().unique())

    interaction_terms: List[str] = []
    for group in groups:
        for level, term in zip(DISCOUNT_LEVELS, terms):
            name = f"{term}_x_{_slug(str(group))}"
            work[name] = work[term] * (work[group_column] == group)
            interaction_terms.append(name)

    regressors = [*interaction_terms, "promo_flag", *SUPPORT_FLAGS, "log_footfall"]
    fit = fit_within_ols(work, "log_units", regressors, f"dose_by_{group_column}")

    promoted = work["promo_flag"] == 1
    rows = []
    for group in groups:
        in_group = work[group_column] == group
        reference = work.loc[
            promoted & in_group & (work["discount_pct"] == 0), "true_effect"
        ].mean()
        for level, term in zip(DISCOUNT_LEVELS, terms):
            stats = fit.row(f"{term}_x_{_slug(str(group))}")
            true_value = (
                work.loc[
                    promoted & in_group & (work["discount_pct"] == level), "true_effect"
                ].mean()
                - reference
            )
            rows.append({
                group_column: group,
                "discount_pct": level,
                **stats,
                "true_effect": float(true_value),
                "error": stats["estimate"] - float(true_value),
                "ci_covers_truth": bool(stats["ci_low"] <= true_value <= stats["ci_high"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spillover / SUTVA diagnostic
# ---------------------------------------------------------------------------

def spillover_diagnostic(frame: pd.DataFrame, max_others: int = 4) -> pd.DataFrame:
    """
    Measure whether untreated rows are affected by *other* SKUs being promoted in
    the same store and category on the same day.

    Standard difference estimators assume no interference between units (SUTVA).
    The generator deliberately violates it: promoting one SKU cannibalises demand
    for its non-promoted category neighbours. If that is happening, the control
    group is depressed exactly when treatment is heaviest, and the estimated
    promotional effect is inflated.

    Estimated on untreated rows only, with pair and date fixed effects absorbed,
    so the result is not confounded by promotions clustering on high-demand days.
    """
    work = frame[[
        "date", "store_id", "sku_id", "category", "promo_flag", "log_units", "pair_id",
    ]].copy()
    category_promos = work.groupby(["date", "store_id", "category"])["promo_flag"].transform("sum")
    work["others_on_promo"] = (category_promos - work["promo_flag"]).clip(upper=max_others)

    control = work[work["promo_flag"] == 0].copy()
    terms = []
    for count in range(1, max_others + 1):
        term = f"others_{count}"
        control[term] = (control["others_on_promo"] == count).astype(float)
        terms.append(term)

    fit = fit_within_ols(control, "log_units", terms, "spillover")
    rows = []
    for count, term in zip(range(1, max_others + 1), terms):
        stats = fit.row(term)
        rows.append({
            "others_on_promo": count if count < max_others else f"{max_others}+",
            "n_rows": int((control["others_on_promo"] == count).sum()),
            **stats,
            "pct_effect": float(np.expm1(stats["estimate"]) * 100),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Count models and the Poisson-vs-NB decision
# ---------------------------------------------------------------------------

def overdispersion_test(counts: np.ndarray, fitted: np.ndarray) -> Dict[str, float]:
    """
    Cameron & Trivedi (1990) regression-based overdispersion test.

    Under Poisson, Var(y) = mu. Under NB2, Var(y) = mu + alpha * mu^2. Regress
    ((y - mu)^2 - y) / mu on mu with no constant; the slope estimates alpha and
    its t-statistic tests H0: alpha = 0 (equidispersion) against alpha > 0.
    """
    mu = np.asarray(fitted, dtype=float)
    y = np.asarray(counts, dtype=float)
    aux_y = ((y - mu) ** 2 - y) / mu
    aux_x = mu
    model = sm.OLS(aux_y, aux_x).fit(cov_type="HC1")
    return {
        "alpha": float(model.params[0]),
        "std_err": float(model.bse[0]),
        "t_stat": float(model.tvalues[0]),
        "p_value": float(model.pvalues[0]),
    }


def fit_count_models(frame: pd.DataFrame, sample_size: int, seed: int) -> Dict[str, object]:
    """
    Fit Poisson and Negative Binomial on a common specification and compare.

    A GLM cannot absorb fixed effects by demeaning, so the pair's baseline volume
    enters as an offset instead and calendar effects enter as dummies. The point
    of this fit is the Poisson-vs-NB comparison, not a second elasticity estimate.
    """
    rng = np.random.default_rng(seed)
    if len(frame) > sample_size:
        index = rng.choice(len(frame), size=sample_size, replace=False)
        sample = frame.iloc[index].copy()
    else:
        sample = frame.copy()

    # Offset: the pair's average volume, computed on the full panel.
    pair_mean = frame.groupby("pair_id")["units_sold"].mean().clip(lower=0.1)
    sample["offset"] = np.log(sample["pair_id"].map(pair_mean).astype(float))

    design_columns = [
        "log_price_ratio",
        "promo_flag",
        *[f"mech_{_slug(m)}" for m in PROMO_MECHANICS],
        "display_support_flag",
        "email_or_app_support_flag",
        "leaflet_support_flag",
        "log_footfall",
    ]
    exog = sample[design_columns].astype(float)
    exog = pd.concat(
        [
            exog,
            pd.get_dummies(sample["month"], prefix="month", drop_first=True).astype(float),
            pd.get_dummies(sample["day_of_week"], prefix="dow", drop_first=True).astype(float),
        ],
        axis=1,
    )
    exog = sm.add_constant(exog)
    endog = sample["units_sold"].astype(float)
    offset = sample["offset"].to_numpy()

    LOGGER.info("Fitting Poisson GLM on %d rows", len(sample))
    poisson = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=offset).fit()
    dispersion = overdispersion_test(endog.to_numpy(), poisson.fittedvalues.to_numpy())

    alpha = max(dispersion["alpha"], 1e-6)
    LOGGER.info("Fitting Negative Binomial GLM with alpha=%.4f", alpha)
    negbin = sm.GLM(
        endog, exog, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset
    ).fit()

    pearson_chi2_ratio = float(poisson.pearson_chi2 / poisson.df_resid)
    return {
        "sample_rows": len(sample),
        "dispersion_test": dispersion,
        "poisson_aic": float(poisson.aic),
        "negbin_aic": float(negbin.aic),
        "poisson_pearson_ratio": pearson_chi2_ratio,
        "negbin_pearson_ratio": float(negbin.pearson_chi2 / negbin.df_resid),
        "poisson_llf": float(poisson.llf),
        "negbin_llf": float(negbin.llf),
        "alpha": alpha,
        "poisson_elasticity": float(poisson.params["log_price_ratio"]),
        "negbin_elasticity": float(negbin.params["log_price_ratio"]),
        "negbin_elasticity_ci": [
            float(negbin.conf_int().loc["log_price_ratio", 0]),
            float(negbin.conf_int().loc["log_price_ratio", 1]),
        ],
        "poisson_resid": poisson.resid_pearson,
        "negbin_resid": negbin.resid_pearson,
        "negbin_fitted": negbin.fittedvalues,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def vif_table(frame: pd.DataFrame, regressors: Sequence[str]) -> pd.DataFrame:
    """
    Variance inflation factors on the demeaned design.

    Computed in the estimation space (after absorbing fixed effects), because
    that is where the collinearity actually bites.
    """
    demeaned = two_way_within(frame, regressors)
    rows = []
    for column in regressors:
        others = [c for c in regressors if c != column]
        y = demeaned[column]
        X = sm.add_constant(demeaned[others])
        r2 = sm.OLS(y, X).fit().rsquared
        rows.append({"term": column, "vif": 1.0 / max(1e-12, 1.0 - r2)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def residual_diagnostics(
    frame: pd.DataFrame, outcome: str, regressors: Sequence[str]
) -> Dict[str, object]:
    """Fitted values and residuals from the within model, for plotting and tests."""
    columns = [outcome, *regressors]
    demeaned = two_way_within(frame, columns)
    result = sm.OLS(demeaned[outcome], demeaned[list(regressors)]).fit()
    resid = result.resid
    fitted = result.fittedvalues

    # Breusch-Pagan on the within residuals.
    bp_model = sm.OLS(resid**2, sm.add_constant(demeaned[list(regressors)])).fit()
    bp_stat = float(bp_model.rsquared * len(resid))

    return {
        "resid": resid,
        "fitted": fitted,
        "breusch_pagan_stat": bp_stat,
        "breusch_pagan_df": len(regressors),
        "resid_skew": float(pd.Series(resid).skew()),
        "resid_kurtosis": float(pd.Series(resid).kurtosis()),
    }
