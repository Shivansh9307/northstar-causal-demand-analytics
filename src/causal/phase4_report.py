"""
Phase 4 report: causal inference.

Produces reports/phase4_causal.md and its figures.

The through-line is the comparison PROJECT_ARCHITECTURE.md §6 Phase 4 asks for -
naive, then DiD, then propensity weighting, each measured against the known
simulated effect. What the data adds is that the two corrected estimators fail in
*opposite* directions for identifiable reasons, which is more informative than
either number alone.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal import did, estimands, psm  # noqa: E402
from features import star_schema  # noqa: E402
from stats import models  # noqa: E402
from utils import config  # noqa: E402

LOGGER = logging.getLogger("promopulse.phase4")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 160,
})

ANALYSIS_QUERY = """
    SELECT date, store_id, sku_id, units_sold, potential_demand_units,
           opening_stock_units, delivery_units,
           promo_flag, promo_type, discount_pct,
           actual_unit_price_gbp, regular_unit_price_gbp,
           display_support_flag, email_or_app_support_flag, leaflet_support_flag,
           store_footfall, category, price_elasticity_segment,
           promotion_sensitivity_segment, seasonal_profile, brand_type,
           month, day_of_week, baseline_gross_margin_pct, average_daily_footfall,
           competition_intensity_score, store_format,
           rolling_7_day_avg_units_sold, rolling_28_day_avg_units_sold,
           is_weekend, is_bank_holiday, is_payday_window, is_school_holiday
    FROM analytics_daily
    ORDER BY date, store_id, sku_id
"""


def _despine(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)


def _table(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = []
    for record in frame.itertuples(index=False):
        cells = []
        for value in record:
            if isinstance(value, (bool, np.bool_)):
                cells.append("yes" if value else "**no**")
            elif isinstance(value, (float, np.floating)):
                cells.append(floatfmt.format(value))
            elif isinstance(value, (int, np.integer)):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def load_frame() -> pd.DataFrame:
    con = star_schema.connect()
    try:
        star_schema.load_raw_tables(con)
        star_schema.build_promotion_bridge(con)
        star_schema.build_analytics_view(con)
        frame = con.execute(ANALYSIS_QUERY).df()
    finally:
        con.close()
    return models.prepare(frame)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_estimator_comparison(table: pd.DataFrame, target: float, figures: Path) -> Path:
    """Forest plot of every estimator against the simulated truth."""
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    y = np.arange(len(table))
    colours = [ORANGE if kind == "naive" else BLUE for kind in table["kind"]]

    for i, row in enumerate(table.itertuples(index=False)):
        ax.plot([row.ci_low, row.ci_high], [i, i], color=colours[i], linewidth=2.5,
                solid_capstyle="round")
        ax.plot(row.estimate, i, "o", markersize=8, color=colours[i],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{row.estimate:+.3f}", xy=(row.ci_high, i), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK_2,
                    fontsize=8.5, fontweight="bold")

    ax.axvline(target, color=AQUA, linewidth=2.5, linestyle="--", zorder=0)
    # Anchored to the bottom: the top row's value label already sits up there.
    ax.annotate(f"True effect {target:+.3f}", xy=(target, 0.0),
                xycoords=("data", "axes fraction"), xytext=(-8, 8),
                textcoords="offset points", color="#0f7a56", fontsize=9,
                va="bottom", ha="right", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(table["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Estimated effect on log demand")
    ax.set_title("Naive estimates overshoot; corrected estimators bracket the truth")
    ax.margins(x=0.16)
    _despine(ax)
    fig.tight_layout()
    path = figures / "10_estimator_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_event_study(events: pd.DataFrame, target: float, figures: Path) -> Path:
    """Leads and lags: the parallel-trends test and the treatment dynamics."""
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    pre = events[events["event_time"] < 0]
    post = events[events["event_time"] >= 0]

    ax.fill_between(events["event_time"], events["ci_low"], events["ci_high"],
                    color=BLUE, alpha=0.16, linewidth=0)
    ax.plot(pre["event_time"], pre["estimate"], color=MUTED, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1,
        label="Pre-promotion (parallel-trends test)")
    ax.plot(post["event_time"], post["estimate"], color=BLUE, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1, label="Promotion window")
    ax.axhline(0, color=BASELINE, linewidth=1.2)
    ax.axvline(-0.5, color=ORANGE, linewidth=1.5, linestyle="--")
    ax.axhline(target, color=AQUA, linewidth=1.8, linestyle=":")
    ax.annotate("True static effect", xy=(1.0, target), xycoords=("axes fraction", "data"),
                xytext=(-6, 6), textcoords="offset points", color="#0f7a56",
                fontsize=8.5, ha="right", va="bottom")
    ax.annotate("promotion starts", xy=(-0.5, 1.0), xycoords=("data", "axes fraction"),
                xytext=(6, -10), textcoords="offset points", color=ORANGE,
                fontsize=8.5, va="top")

    ax.set_xlabel("Days relative to promotion start")
    ax.set_ylabel("Effect on log demand")
    ax.set_title("Effect builds over the first three days, then decays as promotions end")
    ax.legend(loc="upper left", labelcolor=INK_2)
    _despine(ax)
    fig.tight_layout()
    path = figures / "11_event_study.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_balance(all_rows: pd.DataFrame, first_day: pd.DataFrame, figures: Path) -> Path:
    """Love plot: covariate balance before and after weighting, both samples."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
    # The two panels need independent x-scales - the right one's worst imbalance
    # is an order of magnitude smaller - so each title states its own maximum to
    # stop the panels reading as comparable spreads.
    panels = [
        (axes[0], all_rows,
         "All treated rows\n(lagged history contains treated days)\n"
         f"worst |SMD| after weighting: {all_rows['smd_after'].abs().max():.2f}"),
        (axes[1], first_day,
         "First promotion day only\n(history genuinely pre-treatment)\n"
         f"worst |SMD| after weighting: {first_day['smd_after'].abs().max():.3f}"),
    ]
    order = all_rows.sort_values("smd_before", key=abs)["covariate"].tolist()

    for ax, table, title in panels:
        indexed = table.set_index("covariate").loc[order]
        y = np.arange(len(indexed))
        ax.scatter(indexed["smd_before"].abs(), y, s=52, color=ORANGE,
                   edgecolor=SURFACE, linewidth=1.2, label="Before weighting", zorder=3)
        ax.scatter(indexed["smd_after"].abs(), y, s=52, color=BLUE,
                   edgecolor=SURFACE, linewidth=1.2, label="After weighting", zorder=3)
        for i in range(len(indexed)):
            ax.plot([abs(indexed["smd_before"].iloc[i]), abs(indexed["smd_after"].iloc[i])],
                    [i, i], color=GRID, linewidth=1.5, zorder=1)
        ax.axvline(0.1, color=MUTED, linewidth=1.5, linestyle="--")
        ax.annotate("balance\nthreshold 0.1", xy=(0.1, 0), xycoords=("data", "axes fraction"),
                    xytext=(6, 6), textcoords="offset points", color=MUTED,
                    fontsize=8, va="bottom", ha="left")
        ax.set_yticks(y)
        ax.set_yticklabels(indexed.index)
        ax.set_xlabel("|Standardised mean difference|")
        ax.set_title(title, fontsize=9.5)
        ax.margins(x=0.08)
        _despine(ax)
    axes[0].legend(loc="center right", labelcolor=INK_2)
    fig.tight_layout()
    path = figures / "12_covariate_balance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report() -> Path:
    figures = config.path("figures")
    figures.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading panel")
    frame = load_frame()
    ground_truth = estimands.load_ground_truth()

    LOGGER.info("Verifying reconstructed treatment multiplier")
    verification = estimands.verify_against_generator(frame, ground_truth)
    targets = estimands.estimands(frame, ground_truth)

    # Row-level counterfactual: without the promotion, latent demand would have
    # been potential/multiplier, low enough that stock would not have bound.
    multiplier = estimands.treatment_multiplier(frame, ground_truth)
    treated_mask = frame["promo_flag"] == 1
    counterfactual = np.minimum(
        frame.loc[treated_mask, "potential_demand_units"] / multiplier[treated_mask],
        frame.loc[treated_mask, "opening_stock_units"] + frame.loc[treated_mask, "delivery_units"],
    )
    target = float(
        (np.log1p(frame.loc[treated_mask, "units_sold"]) - np.log1p(counterfactual)).mean()
    )

    LOGGER.info("Naive estimates")
    naive = did.naive_estimates(frame)

    LOGGER.info("DiD variants")
    variants = did.did_variants(frame)

    LOGGER.info("Event study")
    events = did.event_study(frame, window=14)
    trends = did.parallel_trends_test(events)

    LOGGER.info("Propensity weighting")
    ordered = frame.sort_values(["pair_id", "date"]).reset_index(drop=True)
    previous = ordered.groupby("pair_id")["promo_flag"].shift(1).fillna(0)
    ordered["mid_promo"] = ((ordered["promo_flag"] == 1) & (previous == 1)).astype(int)
    first_day_frame = ordered[ordered["mid_promo"] == 0]

    psm_all = psm.fit_propensity(ordered, sample_size=400_000, seed=config.seed())
    att_all = psm.ipw_att(ordered, psm_all["propensity"])
    balance_all = psm.standardised_differences(
        ordered, psm_all["design"], psm_all["propensity"],
        psm.CONTINUOUS_COVARIATES + psm.BINARY_COVARIATES,
    )
    overlap = psm.overlap_summary(ordered, psm_all["propensity"])

    psm_first = psm.fit_propensity(first_day_frame, sample_size=400_000, seed=config.seed())
    att_first = psm.ipw_att(first_day_frame, psm_first["propensity"])
    balance_first = psm.standardised_differences(
        first_day_frame, psm_first["design"], psm_first["propensity"],
        psm.CONTINUOUS_COVARIATES + psm.BINARY_COVARIATES,
    )

    # Assemble the comparison table.
    rows = []
    for record in naive.itertuples(index=False):
        rows.append({
            "label": record.estimator.replace("Naive: ", ""), "kind": "naive",
            "estimate": record.estimate, "ci_low": record.estimate, "ci_high": record.estimate,
        })
    variant_labels = {
        "twfe_all": "DiD: all untreated rows as controls",
        "twfe_cannibal_ctrl": "DiD: + concurrent-promotion control",
        "twfe_never_treated": "DiD: never-promoted pairs as controls",
        "twfe_out_of_category": "DiD: uncannibalised controls",
        "twfe_clean_seasonal_fe": "DiD: uncannibalised + seasonal day effects",
    }
    for record in variants.itertuples(index=False):
        rows.append({
            "label": variant_labels.get(record.estimator, record.estimator), "kind": "did",
            "estimate": record.estimate, "ci_low": record.ci_low, "ci_high": record.ci_high,
        })
    rows.append({"label": "IPW: all treated rows", "kind": "psm",
                 "estimate": att_all["att_regression"],
                 "ci_low": att_all["ci_low"], "ci_high": att_all["ci_high"]})
    rows.append({"label": "IPW: first promotion day only", "kind": "psm",
                 "estimate": att_first["att_regression"],
                 "ci_low": att_first["ci_low"], "ci_high": att_first["ci_high"]})
    comparison = pd.DataFrame(rows)
    comparison["error"] = comparison["estimate"] - target
    comparison["pct_effect"] = np.expm1(comparison["estimate"]) * 100
    comparison["error_pp"] = comparison["pct_effect"] - np.expm1(target) * 100

    LOGGER.info("Rendering figures")
    fig_compare = figure_estimator_comparison(comparison, target, figures)
    fig_events = figure_event_study(events, target, figures)
    fig_balance = figure_balance(balance_all, balance_first, figures)

    # The headline naive comparison is the cross-sectional one. Using the largest
    # naive error would pick the ever-vs-never row, which errs the other way and
    # would make the correction look better than it is.
    naive_headline = comparison.iloc[0]
    naive_bias = abs(naive_headline["error"])
    best_did = comparison[comparison["kind"] == "did"].loc[
        comparison[comparison["kind"] == "did"]["error"].abs().idxmin()
    ]

    rel = lambda p: f"figures/{p.name}"  # noqa: E731
    lines: List[str] = [
        "# PromoPulse — Phase 4: Causal Inference",
        "",
        "Regenerate with `uv run python src/causal/phase4_report.py`.",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- The naive promotional lift is **{np.expm1(naive_headline['estimate']) * 100:.0f}%**. "
        f"The true effect is **{np.expm1(target) * 100:.0f}%**. Naive overstates it by "
        f"{naive_bias:.3f} log points ({naive_headline['error_pp']:+.0f} pp).",
        f"- The best difference-in-differences specification recovers "
        f"**{best_did['estimate']:+.3f}** against a truth of **{target:+.3f}** — an error of "
        f"**{best_did['error']:+.3f} log points** ({best_did['error_pp']:+.1f} pp on a "
        f"{np.expm1(target) * 100:.0f}% effect), removing "
        f"{(1 - abs(best_did['error']) / naive_bias) * 100:.0f}% of the naive bias.",
        f"- Propensity weighting lands on the **other side** of the truth "
        f"({att_all['att_regression']:+.3f}, error {att_all['att_regression'] - target:+.3f}). "
        "DiD and IPW fail in opposite directions, for reasons identified in sections 5 and 6.",
        f"- **Parallel trends does not hold cleanly**: {trends['n_significant']} of "
        f"{trends['n_leads']} pre-period leads are significant, though small "
        f"(mean |lead| {trends['mean_abs_lead']:.3f} against an effect of {target:.3f}).",
        "",
        "---",
        "",
        "## 1. What is being recovered",
        "",
        "Three quantities are easy to conflate, and picking the wrong one manufactures a "
        "recovery error out of nothing:",
        "",
        _table(pd.DataFrame([
            {"quantity": "true_promo_uplift_pct (ground-truth column)",
             "value": f"{ground_truth['true_promo_uplift_pct'].mean():.1f}%",
             "why not the benchmark": "A structural coefficient per 10pp of discount, not an ATT"},
            {"quantity": "Arithmetic ATT on latent demand",
             "value": f"{targets['arithmetic_att_pct']:.1f}%",
             "why not the benchmark": "Right concept, wrong scale for a log-outcome regression"},
            {"quantity": "Log-scale ATT on latent demand",
             "value": f"{targets['log_att']:.4f} log pts",
             "why not the benchmark": "Ignores stockout censoring of observed sales"},
            {"quantity": "**Log-scale ATT on observed sales**",
             "value": f"**{target:.4f} log pts ({np.expm1(target) * 100:.1f}%)**",
             "why not the benchmark": "**This is the benchmark used below**"},
        ])),
        "",
        f"The gap between the arithmetic and log scales is {targets['jensen_gap_pct']:.1f} "
        "percentage points — pure Jensen's inequality. Comparing a log-scale coefficient "
        "against the arithmetic ATT would have reported a large fake error.",
        "",
        "The benchmark is built row by row: without its promotion, a treated row's latent "
        "demand would have been `potential_demand / multiplier`, which is low enough that "
        "stock would not have bound. The multiplier is reconstructed from observed flags plus "
        "the ground-truth parameters and **reconciles with the value the generator recorded "
        f"during simulation to within {verification['max_abs_difference_pp']:.4f} pp** across "
        f"{verification['skus_compared']} SKUs, so the target is not itself a modelling choice.",
        "",
        "## 2. All estimates",
        "",
        f"![Estimator comparison]({rel(fig_compare)})",
        "",
        _table(comparison[[
            "label", "estimate", "ci_low", "ci_high", "error", "pct_effect", "error_pp",
        ]]
               .rename(columns={"label": "estimator", "estimate": "log effect",
                                "ci_low": "CI low", "ci_high": "CI high", "error": "error",
                                "pct_effect": "as %", "error_pp": "error (pp)"})),
        "",
        "## 3. Naive estimates and why they fail",
        "",
        "Phase 2 already showed the naive gap is mostly *timing* rather than composition: "
        "promotions land on Christmas, Easter and payday windows, which are high-demand days "
        "anyway. The within-pair naive estimate holds product and store identity fixed and "
        "barely moves, which is the signature of time confounding rather than selection on "
        "product characteristics.",
        "",
        "The third naive row is the informative one: comparing *ever-promoted* pairs to "
        "*never-promoted* pairs across all days gives only "
        f"{np.expm1(comparison.iloc[2]['estimate']) * 100:.1f}%. Almost none of the naive lift "
        "comes from promoted products being intrinsically better sellers — it is when they are "
        "promoted that does the work.",
        "",
        "## 4. Difference-in-differences",
        "",
        "Treatment is **not absorbing**: a pair goes on promotion for a median of eight days "
        "and comes off, roughly eight times across the panel. That rules out the canonical "
        "staggered-adoption estimators, which assume treatment sticks. The estimator is a "
        "two-way fixed effects regression of log demand on a time-varying treatment indicator, "
        "with pair effects absorbing every fixed product and store difference and date effects "
        "absorbing the seasonality that drives the naive bias.",
        "",
        "The staggered campaign rollout still earns its keep: cohorts enter a campaign 0, 21 "
        "and 42 days apart, so on most dates some campaign members are treated and others are "
        "not yet, which is what identifies the date effects without leaning on the small "
        "never-treated pool.",
        "",
        "### Control selection dominates everything else",
        "",
        _table(variants[["estimator", "note", "n_rows", "estimate", "ci_low", "ci_high"]]
               .assign(error=variants["estimate"] - target)
               .rename(columns={"estimator": "specification", "note": "control strategy",
                                "n_rows": "rows", "estimate": "estimate",
                                "ci_low": "CI low", "ci_high": "CI high", "error": "error"})),
        "",
        "Phase 3 established that promoting one SKU depresses its non-promoted category "
        "neighbours by 6-16%. Those neighbours are exactly the rows a naive DiD uses as "
        "controls, so the counterfactual is understated and the effect inflated. The table is "
        "that finding priced out:",
        "",
        "- Using **all untreated rows** as controls leaves the full contamination in place.",
        "- Adding a **count of concurrent category promotions** as a covariate barely helps — "
        "the spillover is not linear in that count, and it saturates.",
        "- Restricting to **never-promoted pairs** is *worse*, not better. Those 480 pairs are "
        "the never-eligible SKUs, and they still sit in categories where other products are "
        "being promoted, so they are cannibalised too — while also being a small, unusual "
        "slice of the assortment.",
        "- Restricting to **store x category x days with no promotion running** removes the "
        "contamination at source and cuts the error by more than half.",
        "- Adding **seasonal-profile-specific day effects** helps a little more: a global date "
        "effect cannot absorb the fact that Christmas-profile SKUs are already climbing in "
        "December, which is exactly when they get promoted.",
        "",
        "## 5. Event study: parallel trends and treatment dynamics",
        "",
        f"![Event study]({rel(fig_events)})",
        "",
        "### The parallel-trends test does not fully pass",
        "",
        f"{trends['n_significant']} of {trends['n_leads']} pre-period leads are statistically "
        f"distinguishable from zero, drifting up to {trends['max_abs_lead']:.3f} log points "
        "immediately before treatment. Demand is already rising before the promotion starts.",
        "",
        "This is small relative to the effect being estimated "
        f"(mean lead {trends['mean_abs_lead']:.3f} against {target:.3f}, about "
        f"{trends['mean_abs_lead'] / target * 100:.0f}%), but it is real and it biases upward. "
        "The cause is visible in the design: promotions are timed to seasonal peaks, and a "
        "global date effect cannot absorb demand that is rising for one SKU's seasonal profile "
        "and not another's. Seasonal-profile day effects reduce the estimate but do not "
        "eliminate the pre-trend, so **the DiD estimate should be read as an upper bound**.",
        "",
        "### The effect is dynamic, and that is not a bias",
        "",
        f"The effect is {events.loc[events['event_time'] == 0, 'estimate'].iloc[0]:.3f} on the "
        f"first day, peaks at {events['estimate'].max():.3f} on day "
        f"{int(events.loc[events['estimate'].idxmax(), 'event_time'])}, then decays as "
        "promotions of varying length end.",
        "",
        "The build-up is a genuine feature of the data generating process, not an artefact. The "
        "generator carries demand memory forward (`memory = 0.65*memory + 0.35*demand`) and "
        "feeds it into an autocorrelation term, so a promotion raises demand, which raises "
        "memory, which amplifies demand further. **The total causal effect of a promotion "
        "therefore exceeds its static multiplier**, and part of the DiD's apparent overshoot "
        "against the static benchmark is really this dynamic channel being captured correctly.",
        "",
        "## 6. Propensity weighting",
        "",
        f"![Covariate balance]({rel(fig_balance)})",
        "",
        "The propensity model uses only what a planner could have seen before choosing to "
        "promote: SKU attributes, store attributes, calendar, and **lagged** demand history. "
        "That history matters — it is what carries the weakening-momentum selection driver "
        "that Phase 1R made real rather than leaving as an unobservable random draw. "
        f"Pseudo-R² is {psm_all['pseudo_r2']:.3f} across {psm_all['n_covariates']} covariates.",
        "",
        "### Conditioning on lagged demand is a bad control here",
        "",
        _table(pd.DataFrame([
            {"sample": "All treated rows", "ATT": att_all["att_regression"],
             "error": att_all["att_regression"] - target,
             "covariates balanced":
                 f"{int(balance_all['balanced_after'].sum())}/{len(balance_all)}",
             "momentum SMD after":
                 balance_all.set_index('covariate').loc['momentum_ratio', 'smd_after']},
            {"sample": "First promotion day only", "ATT": att_first["att_regression"],
             "error": att_first["att_regression"] - target,
             "covariates balanced":
                 f"{int(balance_first['balanced_after'].sum())}/{len(balance_first)}",
             "momentum SMD after":
                 balance_first.set_index('covariate').loc['momentum_ratio', 'smd_after']},
        ])),
        "",
        "On all treated rows the weighting **overshoots** on the demand-history covariates — "
        f"momentum balance goes from "
        f"{balance_all.set_index('covariate').loc['momentum_ratio', 'smd_before']:.2f} before to "
        f"{balance_all.set_index('covariate').loc['momentum_ratio', 'smd_after']:.2f} after, "
        "crossing zero rather than approaching it, and only "
        f"{int(balance_all['balanced_after'].sum())} of {len(balance_all)} covariates end inside "
        "the 0.1 threshold.",
        "",
        "The reason is that on day three of an eight-day promotion, the \"lagged\" 7-day average "
        "already contains days one and two — which were treated. Conditioning on it blocks the "
        "autocorrelation channel that is part of the treatment effect, so the estimate is "
        "biased *down*.",
        "",
        "Restricting to the first day of each promotion, where the history is genuinely "
        f"pre-treatment, balances **{int(balance_first['balanced_after'].sum())} of "
        f"{len(balance_first)}** covariates and drives momentum imbalance to "
        f"{balance_first.set_index('covariate').loc['momentum_ratio', 'smd_after']:.3f}. The "
        f"estimate moves to {att_first['att_regression']:+.3f} — now above the truth, because "
        "this specification fixes the bad control but leaves the cannibalisation contamination "
        "of the control group untouched.",
        "",
        "### Overlap",
        "",
        f"- Treated propensities span "
        f"[{overlap['treated_min']:.4f}, {overlap['treated_max']:.4f}], "
        f"controls [{overlap['control_min']:.4f}, {overlap['control_max']:.4f}].",
        f"- {overlap['share_control_above_treated_min'] * 100:.1f}% of control rows sit above the "
        "minimum treated propensity, so common support is wide.",
        f"- {att_all['n_trimmed']:,} rows outside [0.01, 0.99] are trimmed; the largest surviving "
        f"control weight is {att_all['max_weight']:.0f}.",
        "",
        "## 7. Reconciling the two corrected estimates",
        "",
        "DiD and IPW rest on different assumptions and fail in opposite directions, which is "
        "more informative than either alone:",
        "",
        _table(pd.DataFrame([
            {"estimator": "DiD (uncannibalised + seasonal day effects)",
             "estimate": best_did["estimate"], "error": best_did["error"],
             "direction": "over", "why": "residual pre-trend; captures dynamic amplification"},
            {"estimator": "IPW (all treated rows)",
             "estimate": att_all["att_regression"], "error": att_all["att_regression"] - target,
             "direction": "under", "why": "conditions on post-treatment lagged demand"},
            {"estimator": "Simple average of the two",
             "estimate": (best_did["estimate"] + att_all["att_regression"]) / 2,
             "error": (best_did["estimate"] + att_all["att_regression"]) / 2 - target,
             "direction": "—", "why": "not a principled estimator, but the bracket is real"},
        ])),
        "",
        "The bracket is the honest headline. Neither estimator nails the number; together they "
        "bound it, and each one's failure mode is identified rather than waved at.",
        "",
        "## 8. What I would not claim",
        "",
        "- **That parallel trends holds.** It does not, quite. The pre-period leads drift "
        f"upward and {trends['n_significant']} of {trends['n_leads']} are significant. Promotions "
        "are timed to seasonal peaks and a global date effect cannot fully absorb SKU-specific "
        "seasonality. The DiD estimate is an upper bound.",
        "- **That the control group is clean.** Cannibalisation means untreated rows in a "
        "promoted category are themselves affected by treatment — a SUTVA violation. The "
        "uncannibalised-control specification addresses it at source, but the never-treated "
        "pool is small and not representative of the assortment.",
        "- **That this is the per-SKU causal effect.** With cannibalisation present, the "
        "commercially relevant quantity is the net effect on the category, which is a different "
        "estimand. Phase 6 should optimise against the net figure, not this one.",
        "- **That the recovery error is small.** It is "
        f"{abs(best_did['error_pp']):.0f} percentage points on a "
        f"{np.expm1(target) * 100:.0f}% effect. The method demonstrably removes most of the "
        "naive bias, and the residual is explained rather than hidden — but a "
        f"{abs(best_did['error_pp']):.0f}pp error would matter for a real promotional budget.",
        "- **That any of this transfers without the ground truth.** Every conclusion above was "
        "checkable because the simulated answer exists. On real data the same diagnostics — "
        "pre-trend tests, balance tables, control-set sensitivity — are available, but the "
        "final scoring is not.",
        "",
        "---",
        "",
        "## What Phase 5 should carry forward",
        "",
        "1. **Lagged demand features are contaminated inside promotion windows.** The rolling "
        "averages contain earlier treated days. For forecasting that is fine and even desirable; "
        "for anything causal it is a bad control.",
        "2. **Use uncannibalised comparisons** wherever a counterfactual is needed.",
        "3. **The dynamic build-up is real.** A model that treats a promotion as a constant "
        "shift will misfit the first three days and the tail.",
        "4. **Time-based splits only**, and cluster on the pair — both already load-bearing here.",
        "",
    ]

    path = config.path("reports") / "phase4_causal.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print(f"\nPhase 4 report written to {build_report()}")
