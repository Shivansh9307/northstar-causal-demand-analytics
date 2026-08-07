"""
Defining what Phase 4 is actually trying to recover.

Getting this wrong is the easiest way to produce a confident, meaningless
validation, so the target is defined explicitly and checked against the
generator before any estimator is run.

Three quantities are easy to confuse
------------------------------------
1. **`true_promo_uplift_pct`** in the ground-truth file. This is a *structural
   coefficient* applied per 10 percentage points of discount, and it compounds
   with a separate price response. It is **not** an average treatment effect and
   must never be the benchmark for a DiD estimate. (Established in Phase 1R.)

2. **The ATT on latent demand.** The generator multiplies latent demand by
   `price_effect * promo_effect * support_effect` on promoted rows. Averaged over
   treated rows, `E[multiplier - 1]` is the true arithmetic ATT - this is the
   `true_realised_att_pct` column.

3. **The ATT a log-outcome regression targets.** A regression of `log(units)` on
   a treatment indicator estimates `E[log Y(1) - log Y(0)]`, which is
   `E[log multiplier]`, not `log E[multiplier]`. Jensen's inequality makes the
   second strictly larger whenever the multiplier varies. Comparing a log-scale
   coefficient against the arithmetic ATT would manufacture a recovery "error"
   that is pure functional-form confusion.

This module reconstructs the row-level multiplier from observed columns plus the
ground-truth parameters, verifies it reproduces the generator's stored
`true_realised_att_pct`, and then reports both scales.

Censoring
---------
All of the above are effects on *latent* demand. Observed `units_sold` is
stockout-censored, and censoring bites harder on promoted rows precisely because
demand is higher. Any estimator run on observed sales therefore targets something
smaller than the latent ATT. `censoring_gap` quantifies that.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import config  # noqa: E402

# Multiplicative effect the generator attaches to each promotion mechanic,
# applied on top of the discount-dose uplift.
PROMO_TYPE_MULTIPLIER: Dict[str, float] = {
    "Multi-buy": 1.08,
    "Clubcard-style Price": 1.05,
    "Bundle": 1.10,
    "Display-only": 1.06,
    "Percent Off": 1.00,
    "None": 1.00,
}

LEAFLET_UPLIFT = 0.07


def load_ground_truth() -> pd.DataFrame:
    return pd.read_csv(
        config.path("ground_truth") / "ground_truth_simulation_parameters.csv"
    )


def treatment_multiplier(frame: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.Series:
    """
    Reconstruct the exact multiplicative treatment effect on latent demand.

    Mirrors the generator: latent demand is scaled by

        price_effect   = (actual / regular) ** elasticity
        promo_effect   = (1 + uplift * discount/10) * mechanic_multiplier
        support_effect = 1 + display*d_uplift + email*e_uplift + leaflet*0.07

    Equals 1.0 on untreated rows, where the discount is zero, the mechanic is
    "None" and every support flag is false.
    """
    truth = ground_truth.set_index("sku_id")
    elasticity = frame["sku_id"].map(truth["true_price_elasticity"]).astype(float)
    uplift = frame["sku_id"].map(truth["true_promo_uplift_pct"]).astype(float) / 100
    display_uplift = frame["sku_id"].map(truth["true_display_uplift_pct"]).astype(float) / 100
    email_uplift = frame["sku_id"].map(truth["true_email_app_uplift_pct"]).astype(float) / 100

    promoted = frame["promo_flag"].astype(bool)
    discount = frame["discount_pct"].astype(float)

    price_ratio = frame["actual_unit_price_gbp"] / frame["regular_unit_price_gbp"]
    price_effect = np.power(price_ratio, elasticity)

    mechanic = frame["promo_type"].map(PROMO_TYPE_MULTIPLIER).fillna(1.0).astype(float)
    promo_effect = (1 + uplift * (discount / 10) * promoted) * mechanic

    support_effect = (
        1
        + frame["display_support_flag"].astype(float) * display_uplift
        + frame["email_or_app_support_flag"].astype(float) * email_uplift
        + frame["leaflet_support_flag"].astype(float) * LEAFLET_UPLIFT
    )
    return price_effect * promo_effect * support_effect


def estimands(frame: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict[str, float]:
    """
    The target quantities, on both scales, plus the censoring gap.

    `arithmetic_att_pct` should reproduce the generator's stored
    `true_realised_att_pct`; `verify_against_generator` asserts that.
    """
    multiplier = treatment_multiplier(frame, ground_truth)
    treated = frame["promo_flag"].astype(bool)
    treated_multiplier = multiplier[treated]

    arithmetic = float((treated_multiplier - 1).mean() * 100)
    log_scale = float(np.log(treated_multiplier).mean())

    # Censoring: how much of latent demand is actually realised as sales.
    realised_treated = float(
        frame.loc[treated, "units_sold"].sum()
        / max(frame.loc[treated, "potential_demand_units"].sum(), 1)
    )
    realised_control = float(
        frame.loc[~treated, "units_sold"].sum()
        / max(frame.loc[~treated, "potential_demand_units"].sum(), 1)
    )

    # An estimator run on observed sales cannot see the demand that stockouts
    # destroyed. Promoted rows lose more of it, so the observable effect sits
    # below the latent one by roughly the difference in realised share, on the
    # log scale. This is the number a DiD on log(units) should be judged against.
    censoring_log_shift = float(np.log(realised_treated / realised_control))
    log_att_observed = log_scale + censoring_log_shift

    return {
        "arithmetic_att_pct": arithmetic,
        "log_att": log_scale,
        "log_att_as_pct": float(np.expm1(log_scale) * 100),
        "jensen_gap_pct": arithmetic - float(np.expm1(log_scale) * 100),
        "treated_rows": int(treated.sum()),
        "realised_share_treated": realised_treated,
        "realised_share_control": realised_control,
        "censoring_log_shift": censoring_log_shift,
        "log_att_observed": log_att_observed,
        "log_att_observed_as_pct": float(np.expm1(log_att_observed) * 100),
    }


def verify_against_generator(
    frame: pd.DataFrame, ground_truth: pd.DataFrame, tolerance_pct: float = 1.0
) -> Dict[str, float]:
    """
    Check the reconstructed multiplier against the value the generator recorded
    during simulation.

    If these disagree, the reconstruction is wrong and every recovery number in
    Phase 4 would be measured against the wrong target - so this raises rather
    than warns.

    **Requires the full panel.** `true_realised_att_pct` is each SKU's average
    over all of its treated rows across every store, so a subset of stores or
    dates has a different mix of discounts and support flags and will not
    reconcile. That is a property of the benchmark, not a defect.
    """
    multiplier = treatment_multiplier(frame, ground_truth)
    treated = frame["promo_flag"].astype(bool)

    per_sku = (
        pd.DataFrame({"sku_id": frame.loc[treated, "sku_id"], "effect": multiplier[treated] - 1})
        .groupby("sku_id")["effect"].mean() * 100
    )
    stored = ground_truth.set_index("sku_id")["true_realised_att_pct"].dropna()
    common = per_sku.index.intersection(stored.index)
    difference = (per_sku.loc[common] - stored.loc[common]).abs()

    result = {
        "skus_compared": int(len(common)),
        "max_abs_difference_pp": float(difference.max()),
        "mean_abs_difference_pp": float(difference.mean()),
    }
    if result["max_abs_difference_pp"] > tolerance_pct:
        raise AssertionError(
            "Reconstructed treatment multiplier disagrees with the generator's recorded "
            f"true_realised_att_pct by up to {result['max_abs_difference_pp']:.3f} pp "
            f"(tolerance {tolerance_pct} pp). The Phase 4 benchmark cannot be trusted "
            "until this reconciles."
        )
    return result
