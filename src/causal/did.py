"""
Difference-in-differences and the event study behind it.

Design
------
Treatment here is **not absorbing**. A store x SKU pair goes on promotion for a
median of eight days and then comes off, roughly eight times across the panel.
That rules out the canonical staggered-adoption estimators, which assume a unit
stays treated once treated, and makes the natural estimator a two-way fixed
effects regression of log demand on a time-varying treatment indicator:

    log(units + 1) = beta * promo + pair FE + date FE + e

Pair effects absorb every fixed difference between products and stores; date
effects absorb the seasonality, holidays and paydays that Phase 2 showed drive
most of the naive gap. `beta` is the DiD estimate.

The staggered campaign rollout still matters: cohorts enter a campaign 0, 21 and
42 days apart, so on any given date some campaign members are treated and others
are not yet. That is what makes the date effects identifiable without leaning on
the never-treated pool alone.

The Phase 3 spillover problem
-----------------------------
Phase 3 established that promoting one SKU depresses its non-promoted category
neighbours by 6-16%. Those neighbours are exactly the control rows a naive DiD
uses, so the estimate is inflated. Four control strategies are estimated side by
side rather than picking one and hoping:

* `twfe_all`          - all untreated rows as controls (contaminated)
* `twfe_cannibal_ctrl`- adds the count of concurrent category promotions
* `twfe_never_treated`- controls restricted to never-promoted pairs
* `twfe_out_of_category` - controls restricted to rows whose store x category had
                        no promotion that day
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stats import models  # noqa: E402

LOGGER = logging.getLogger("promopulse.did")


# ---------------------------------------------------------------------------
# Naive comparisons
# ---------------------------------------------------------------------------

def naive_estimates(frame: pd.DataFrame) -> pd.DataFrame:
    """
    The estimates a careless analysis would report, on the same log scale as the
    DiD so they are directly comparable.
    """
    treated = frame["promo_flag"] == 1
    rows = []

    # Cross-sectional: promoted rows vs everything else.
    rows.append({
        "estimator": "Naive: promoted rows vs all others",
        "estimate": float(frame.loc[treated, "log_units"].mean() - frame.loc[~treated, "log_units"].mean()),
    })

    # Within-pair before/after: holds product and store identity fixed, but not time.
    pair_means = frame.groupby(["pair_id", "promo_flag"])["log_units"].mean().unstack()
    pair_means = pair_means.dropna()
    rows.append({
        "estimator": "Naive: within-pair promoted vs not",
        "estimate": float((pair_means[1] - pair_means[0]).mean()),
    })

    # Ever-promoted pairs vs never-promoted pairs, all rows.
    ever = frame.groupby("pair_id")["promo_flag"].transform("max") == 1
    rows.append({
        "estimator": "Naive: ever-promoted vs never-promoted pairs",
        "estimate": float(frame.loc[ever, "log_units"].mean() - frame.loc[~ever, "log_units"].mean()),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Control-set construction
# ---------------------------------------------------------------------------

def add_category_promo_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Concurrent promotions in the same store x category x day, excluding self."""
    out = frame.copy()
    total = out.groupby(["date", "store_id", "category"])["promo_flag"].transform("sum")
    out["others_on_promo"] = (total - out["promo_flag"]).astype(float)
    out["category_has_promo"] = (total > 0).astype(int)
    return out


def did_variants(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the DiD under four control strategies.

    The spread between them is the honest measure of how much the spillover
    matters, and it is reported rather than resolved by fiat.
    """
    work = add_category_promo_counts(frame)
    work["seasonal_date"] = work["seasonal_profile"].astype(str) + "|" + work["date"].astype(str)
    ever_treated = work.groupby("pair_id")["promo_flag"].transform("max") == 1
    results: List[Dict[str, object]] = []

    def record(
        name: str,
        subset: pd.DataFrame,
        regressors: Sequence[str],
        note: str,
        time_key: str = "date",
    ) -> None:
        fit = models.fit_within_ols(
            subset, "log_units", list(regressors), name, time_key=time_key
        )
        stats = fit.row("promo_flag")
        results.append({
            "estimator": name,
            "note": note,
            "n_rows": int(len(subset)),
            "n_clusters": fit.n_clusters,
            **stats,
        })

    record(
        "twfe_all", work, ["promo_flag"],
        "All untreated rows as controls",
    )
    record(
        "twfe_cannibal_ctrl", work, ["promo_flag", "others_on_promo"],
        "Controls for concurrent category promotions",
    )

    # Treated rows plus never-promoted pairs only: no already-treated unit is
    # ever used as a control.
    never_treated_subset = work[(work["promo_flag"] == 1) | (~ever_treated)]
    record(
        "twfe_never_treated", never_treated_subset, ["promo_flag"],
        "Controls restricted to never-promoted pairs",
    )

    # Treated rows plus untreated rows from store x categories with no promotion
    # running that day: controls that cannibalisation cannot have touched.
    clean_controls = work[(work["promo_flag"] == 1) | (work["category_has_promo"] == 0)]
    record(
        "twfe_out_of_category", clean_controls, ["promo_flag"],
        "Controls restricted to uncannibalised store x category x days",
    )

    # Same clean controls, with seasonal-profile-specific day effects. Global date
    # effects cannot absorb the fact that Christmas-profile SKUs rise before
    # Christmas, which is also when they get promoted.
    record(
        "twfe_clean_seasonal_fe", clean_controls, ["promo_flag"],
        "Uncannibalised controls + seasonal-profile x date effects",
        time_key="seasonal_date",
    )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Event study
# ---------------------------------------------------------------------------

def build_event_time(frame: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Assign each row an event time relative to the nearest promotion start for its
    pair.

    Because treatment cycles on and off, a row can sit between two episodes; it
    is assigned to whichever start is closer. Rows further than `window` days
    from any start are left as NaN and act as the untreated baseline.
    """
    work = frame.sort_values(["pair_id", "date"]).copy()
    previous = work.groupby("pair_id")["promo_flag"].shift(1).fillna(0)
    work["is_start"] = ((work["promo_flag"] == 1) & (previous == 0)).astype(int)

    starts = work.loc[work["is_start"] == 1, ["pair_id", "date"]].rename(
        columns={"date": "start_date"}
    ).sort_values("start_date")

    ordered = work.sort_values("date")
    backward = pd.merge_asof(
        ordered[["pair_id", "date"]], starts,
        left_on="date", right_on="start_date", by="pair_id", direction="backward",
    )["start_date"]
    forward = pd.merge_asof(
        ordered[["pair_id", "date"]], starts,
        left_on="date", right_on="start_date", by="pair_id", direction="forward",
    )["start_date"]

    ordered = ordered.assign(
        days_since=(ordered["date"].to_numpy() - backward.to_numpy()) / np.timedelta64(1, "D"),
        days_until=(forward.to_numpy() - ordered["date"].to_numpy()) / np.timedelta64(1, "D"),
    )
    since = ordered["days_since"]
    until = ordered["days_until"]
    use_since = since.notna() & (until.isna() | (since <= until))
    event_time = np.where(use_since, since, -until)
    ordered["event_time"] = np.where(np.abs(event_time) <= window, event_time, np.nan)
    return ordered


def event_study(frame: pd.DataFrame, window: int = 14, reference: int = -1) -> pd.DataFrame:
    """
    Leads and lags around promotion start, with pair and date effects absorbed.

    The leads are the parallel-trends test: if promotions were placed on pairs
    already diverging from their controls, the pre-period coefficients will not
    be flat at zero.
    """
    work = build_event_time(frame, window=window)
    offsets = [k for k in range(-window, window + 1) if k != reference]

    terms = []
    for k in offsets:
        term = f"ev_{'m' if k < 0 else 'p'}{abs(k)}"
        work[term] = (work["event_time"] == k).astype(float)
        terms.append(term)

    fit = models.fit_within_ols(work, "log_units", terms, "event_study")
    rows = []
    for k, term in zip(offsets, terms):
        stats = fit.row(term)
        rows.append({"event_time": k, "n_rows": int((work["event_time"] == k).sum()), **stats})
    rows.append({
        "event_time": reference, "n_rows": int((work["event_time"] == reference).sum()),
        "estimate": 0.0, "std_err": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": np.nan,
    })
    return pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)


def parallel_trends_test(events: pd.DataFrame, reference: int = -1) -> Dict[str, float]:
    """
    Summarise the pre-period. Under parallel trends every lead should be
    indistinguishable from zero.
    """
    leads = events[(events["event_time"] < 0) & (events["event_time"] != reference)]
    significant = int((leads["p_value"] < 0.05).sum())
    return {
        "n_leads": int(len(leads)),
        "n_significant": significant,
        "max_abs_lead": float(leads["estimate"].abs().max()),
        "mean_abs_lead": float(leads["estimate"].abs().mean()),
        "share_significant": significant / max(len(leads), 1),
    }
