"""
Phase 6 report: prescriptive optimisation.

Produces reports/phase6_optimization.md and its figures.

Three deliverables, in the order §6 Phase 6 asks for: a reorder-point policy
built on forecast uncertainty, a promotion budget allocated by integer program,
and a Monte Carlo profit range rather than a point estimate.

The fourth thing here is not in the architecture but is the reason the earlier
phases were worth doing: the same optimiser is run three times, on a naive
promotional estimate, on the causal one, and on the simulated truth, and all
three plans are scored under the truth. The gap between them is what the causal
work is worth in pounds.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal import estimands  # noqa: E402
from ml import features as ml_features  # noqa: E402
from ml import forecast as ml_forecast  # noqa: E402
from optimization import inventory, monte_carlo, promo_lp  # noqa: E402
from stats import models  # noqa: E402
from utils import config  # noqa: E402

LOGGER = logging.getLogger("northstar.phase6")

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

SEGMENT_COLUMNS = ["demand_volatility_segment", "promo_flag"]


def _despine(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)


def _table(frame: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = []
    for record in frame.itertuples(index=False):
        cells = []
        for value in record:
            if isinstance(value, (bool, np.bool_)):
                cells.append("yes" if value else "no")
            elif isinstance(value, (float, np.floating)):
                cells.append(floatfmt.format(value))
            elif isinstance(value, (int, np.integer)):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


# ---------------------------------------------------------------------------
# Effect curves
# ---------------------------------------------------------------------------

def naive_dose_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """
    The dose-response an analyst gets *without* fixed effects.

    Pooled OLS, no pair or date absorption - so it inherits all the timing
    confounding Phase 2 identified and reads promotions as far more effective
    than they are.
    """
    work, terms = models.add_discount_dummies(frame)
    regressors = [*terms, "promo_flag", *models.SUPPORT_FLAGS, "log_footfall"]
    exog = sm.add_constant(work[regressors].astype(float))
    fit = sm.OLS(work["log_units"], exog).fit(
        cov_type="cluster", cov_kwds={"groups": work["pair_id"]}
    )
    rows = []
    for segment in sorted(work["price_elasticity_segment"].dropna().unique()):
        for depth, term in zip(models.DISCOUNT_LEVELS, terms):
            rows.append({
                "price_elasticity_segment": segment,
                "discount_pct": depth,
                "estimate": float(fit.params[term]),
            })
    return pd.DataFrame(rows)


def true_dose_curve(frame: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """The effect the generator actually applied, by segment and depth."""
    work = frame.copy()
    work["true_effect"] = models.true_price_dose_effect(work, ground_truth)
    promoted = work["promo_flag"] == 1

    rows = []
    for segment, group in work[promoted].groupby("price_elasticity_segment", observed=True):
        reference = group.loc[group["discount_pct"] == 0, "true_effect"].mean()
        for depth in models.DISCOUNT_LEVELS:
            at_depth = group.loc[group["discount_pct"] == depth, "true_effect"].mean()
            rows.append({
                "price_elasticity_segment": segment,
                "discount_pct": depth,
                "estimate": float(at_depth - reference),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_service_levels(policy: pd.DataFrame, figures: Path) -> Path:
    """
    Cost-derived service levels by category.

    A histogram of these is unreadable - the distribution is sharply bimodal,
    with ambient lines stacked at the upper clip and perishables spread from the
    lower one - so this shows the median and interquartile range per category
    instead, which is also the form a category manager would act on.
    """
    summary = (
        policy.groupby("category_label", observed=True)["service_level"]
        .agg(median="median", low=lambda s: s.quantile(0.25), high=lambda s: s.quantile(0.75))
        .reset_index().sort_values("median")
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    y = np.arange(len(summary))
    for i, row in enumerate(summary.itertuples(index=False)):
        ax.plot([row.low * 100, row.high * 100], [i, i], color=BLUE, linewidth=6,
                solid_capstyle="round", alpha=0.35)
        ax.plot(row.median * 100, i, "o", markersize=9, color=BLUE,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{row.median * 100:.0f}%", xy=(row.median * 100, i), xytext=(0, 11),
                    textcoords="offset points", ha="center", color=INK_2, fontsize=8.5)
    ax.axvline(95, color=ORANGE, linewidth=2, linestyle="--")
    ax.annotate("flat 95% policy", xy=(95, 0.0), xycoords=("data", "axes fraction"),
                xytext=(-8, 8), textcoords="offset points", color=ORANGE,
                fontsize=9, va="bottom", ha="right", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(summary["category_label"])
    ax.set_xlabel("Cost-derived service level (%), median and interquartile range")
    ax.set_title("Fresh categories should run lower availability, not higher")
    ax.margins(y=0.06)
    _despine(ax)
    fig.tight_layout()
    path = figures / "17_service_levels.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_estimate_quality(experiment: pd.DataFrame, figures: Path) -> Path:
    """Realised profit of plans built on three different beliefs."""
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    y = np.arange(len(experiment))
    colours = [ORANGE, BLUE, AQUA][: len(experiment)]
    # Adaptive units, as in the Monte Carlo figure: these values are in the
    # hundreds, so a £000 axis would round every label to "£0k".
    magnitude = experiment["realised_profit"].abs().max()
    scale, unit = (1000.0, "£000") if magnitude >= 10_000 else (1.0, "£")

    ax.barh(y, experiment["realised_profit"] / scale, height=0.55, color=colours,
            edgecolor=SURFACE, linewidth=1.5)
    for i, row in enumerate(experiment.itertuples(index=False)):
        value = row.realised_profit / scale
        label = f"£{value:,.0f}k" if scale > 1 else f"£{row.realised_profit:,.0f}"
        ax.annotate(label, xy=(value, i),
                    xytext=(8 if value >= 0 else -8, 0), textcoords="offset points",
                    va="center", ha="left" if value >= 0 else "right",
                    color=INK_2, fontsize=9, fontweight="bold")
    ax.axvline(0, color=BASELINE, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(experiment["plan_built_on"])
    ax.invert_yaxis()
    ax.set_xlabel(f"Realised incremental profit under the true promotional response ({unit})")
    ax.set_title("What the causal correction is worth")
    ax.margins(x=0.16)
    _despine(ax)
    fig.tight_layout()
    path = figures / "18_estimate_quality.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_monte_carlo(draws: pd.DataFrame, summary: Dict[str, float],
                       sensitivity: pd.DataFrame, figures: Path) -> Path:
    """Profit distribution and its sensitivity to the effect estimate."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # Adaptive units: these plans come out in the hundreds of pounds, and dividing
    # by a thousand would round every annotation to "£0k".
    magnitude = max(abs(draws["profit"]).max(), abs(summary["deterministic"]))
    scale, unit = (1000.0, "£000") if magnitude >= 10_000 else (1.0, "£")
    fmt = (lambda v: f"£{v / scale:,.0f}k") if scale > 1 else (lambda v: f"£{v:,.0f}")

    profit = draws["profit"].to_numpy() / scale
    axes[0].hist(profit, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    for key, colour, label in (
        ("p10", ORANGE, "P10"), ("p50", INK_2, "P50"), ("p90", AQUA, "P90"),
    ):
        axes[0].axvline(summary[key] / scale, color=colour, linewidth=2, linestyle="--")
        axes[0].annotate(f"{label} {fmt(summary[key])}",
                         xy=(summary[key] / scale, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(4, -10), textcoords="offset points",
                         color=colour, fontsize=8.5, va="top", rotation=90)
    axes[0].axvline(summary["deterministic"] / scale, color="#8a2be2", linewidth=2)
    axes[0].annotate(f"plan estimate\n{fmt(summary['deterministic'])}",
                     xy=(summary["deterministic"] / scale, 0.55),
                     xycoords=("data", "axes fraction"), xytext=(8, 0),
                     textcoords="offset points", color="#8a2be2", fontsize=8.5, va="center")
    axes[0].set_xlabel(f"Incremental profit ({unit})")
    axes[0].set_ylabel("Simulation draws")
    axes[0].set_title("The plan's estimate sits near the optimistic tail")
    _despine(axes[0])

    axes[1].fill_between(sensitivity["effect_multiplier"], sensitivity["p10"] / scale,
                         sensitivity["p90"] / scale, color=BLUE, alpha=0.18, linewidth=0)
    axes[1].plot(sensitivity["effect_multiplier"], sensitivity["mean_profit"] / scale,
                 color=BLUE, marker="o", markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1, label="Mean (P10–P90 band)")
    axes[1].axhline(0, color=BASELINE, linewidth=1.2)
    axes[1].axvline(1.0, color=MUTED, linewidth=1.5, linestyle="--")
    axes[1].annotate("estimate taken at face value", xy=(1.0, 1.0),
                     xycoords=("data", "axes fraction"), xytext=(-6, -10),
                     textcoords="offset points", color=MUTED, fontsize=8.5,
                     va="top", ha="right")
    axes[1].set_xlabel("Multiplier on the estimated promotional uplift")
    axes[1].set_ylabel(f"Incremental profit ({unit})")
    axes[1].set_title("How wrong can the effect estimate be before the plan loses money?")
    axes[1].legend(loc="upper left", labelcolor=INK_2, fontsize=8.5)
    _despine(axes[1])

    fig.tight_layout()
    path = figures / "19_monte_carlo.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report() -> Path:
    figures = config.path("figures")
    figures.mkdir(parents=True, exist_ok=True)
    ground_truth = estimands.load_ground_truth()

    LOGGER.info("Loading panel")
    panel = models.prepare(models.load_analysis_frame())

    LOGGER.info("Estimating effect curves")
    causal_dose = models.dose_response_by_group(panel, "price_elasticity_segment", ground_truth)
    causal_curve = promo_lp.curve_from_dose_response(causal_dose)
    naive_curve = promo_lp.curve_from_dose_response(naive_dose_curve(panel))
    true_curve = promo_lp.curve_from_dose_response(true_dose_curve(panel, ground_truth))

    curve_comparison = pd.DataFrame([
        {
            "discount %": depth,
            "naive": naive_curve["Medium"][depth],
            "causal (Phase 3)": causal_curve["Medium"][depth],
            "simulated truth": true_curve["Medium"][depth],
        }
        for depth in promo_lp.DISCOUNT_DEPTHS
    ])

    LOGGER.info("Building pair-level inputs")
    recent = panel[panel["date"] >= panel["date"].max() - pd.Timedelta(days=90)]
    pairs = (
        recent[recent["promo_flag"] == 0]
        .groupby(["store_id", "sku_id"], observed=True)
        .agg(mean_daily_units=("units_sold", "mean"))
        .reset_index()
        .merge(
            panel.groupby(["store_id", "sku_id"], observed=True).agg(
                category=("category", "first"),
                price_elasticity_segment=("price_elasticity_segment", "first"),
                demand_volatility_segment=("demand_volatility_segment", "first"),
                regular_unit_price_gbp=("regular_unit_price_gbp", "first"),
            ).reset_index(),
            on=["store_id", "sku_id"],
        )
    )
    products = pd.read_csv(config.path("raw") / "dim_product.csv", keep_default_na=False)
    pairs = pairs.merge(
        products[["sku_id", "unit_cost_gbp", "shelf_life_days", "is_perishable",
                  "reorder_lead_time_days"]],
        on="sku_id",
    )
    pairs["is_perishable"] = pairs["is_perishable"].astype(str).str.lower().eq("true")
    pairs["unit_margin"] = pairs["regular_unit_price_gbp"] - pairs["unit_cost_gbp"]

    category_baseline = (
        pairs.groupby(["store_id", "category"], observed=True)["mean_daily_units"]
        .sum().reset_index().rename(columns={"mean_daily_units": "baseline_units_per_day"})
    )

    # ---- inventory -------------------------------------------------------
    LOGGER.info("Fitting forecast for error distribution")
    source = ml_features.load_source()
    ml_frame, feature_names = ml_features.build_features(source)
    del source
    _, holdout_index = ml_features.time_split(ml_frame)
    _, _, X_holdout, predictions = ml_forecast.run_holdout(
        ml_frame, feature_names, list(ml_features.CATEGORICAL_FEATURES), holdout_index
    )
    holdout_frame = ml_frame.iloc[holdout_index]

    sigma_lookup = inventory.forecast_error_sigma(
        holdout_frame, predictions, SEGMENT_COLUMNS
    )

    demand_basis = (
        holdout_frame.groupby(["store_id", "sku_id"], observed=True)
        .agg(units_sold=("units_sold", "mean"))
        .reset_index()
        .merge(pairs[["store_id", "sku_id", "reorder_lead_time_days"]], on=["store_id", "sku_id"])
    )
    policy_pairs = pairs.assign(promo_flag=0)
    policy_comparison = inventory.compare_policies(
        policy_pairs, sigma_lookup, SEGMENT_COLUMNS, demand_basis
    )
    optimal_policy = inventory.build_policy(policy_pairs, sigma_lookup, SEGMENT_COLUMNS)

    # ---- promotion budget ------------------------------------------------
    LOGGER.info("Solving promotion ILP")
    # Budget anchored to what Northstar actually spent: total retailer-funded
    # promotional cost across the panel, divided by its eight quarters.
    promotions = pd.read_csv(config.path("raw") / "fact_promotions.csv")
    retailer_funded = (
        promotions["promotion_cost_gbp"] * (1 - promotions["vendor_funded_pct"] / 100)
    ).sum()
    budget = float(retailer_funded / 8)  # eight quarters in the panel

    candidates = promo_lp.build_candidates(pairs, causal_curve, category_baseline)
    solution = promo_lp.solve(candidates, budget=budget, max_per_store_category=3)
    plan = solution["plan"]
    depth_mix = promo_lp.plan_summary(plan)

    # Cannibalisation is the assumption the whole recommendation turns on, so it
    # is swept rather than fixed. Zero is what an optimiser that ignores
    # cross-SKU effects implicitly assumes.
    LOGGER.info("Cannibalisation sensitivity")
    cannibalisation_rows: List[Dict[str, object]] = []
    for rate, label in (
        (0.0, "Ignored (0%)"),
        (0.02, "Conservative (2%)"),
        (
            promo_lp.CANNIBALISATION_FIRST_PROMO,
            f"Measured in Phase 3 ({promo_lp.CANNIBALISATION_FIRST_PROMO:.1%})",
        ),
    ):
        swept = promo_lp.build_candidates(
            pairs, causal_curve, category_baseline, cannibalisation_rate=rate
        )
        swept_solution = promo_lp.solve(swept, budget=budget, max_per_store_category=3)
        # Whatever the planner assumed, score the resulting plan under the
        # measured rate - the world does not change because the model ignored it.
        realised = promo_lp.evaluate_plan_under(
            swept_solution["plan"], pairs, causal_curve, category_baseline
        )
        cannibalisation_rows.append({
            "assumption": label,
            "viable_candidates": swept_solution["n_viable"],
            "promotions_selected": swept_solution["n_selected"],
            "profit_the_plan_predicted": swept_solution["total_profit"],
            "profit_under_measured_rate": realised["realised_profit"],
        })
    cannibalisation_sweep = pd.DataFrame(cannibalisation_rows)

    # How many promotions did Northstar actually run per quarter, for scale?
    observed_promotions_per_quarter = len(pd.read_csv(
        config.path("raw") / "fact_promotions.csv"
    )) / 8

    gross_gain = (
        candidates["promoted_units"] * candidates["promo_margin"]
        - candidates["baseline_units"] * candidates["full_margin"]
    )
    viable_before_cannibalisation = float((gross_gain > 0).mean())
    by_depth_gross = (
        candidates.assign(gross=gross_gain)
        .groupby("discount_pct")["gross"]
        .apply(lambda s: float((s > 0).mean()) * 100)
        .reset_index()
        .rename(columns={
            "discount_pct": "discount %",
            "gross": "% profitable before cannibalisation",
        })
    )

    # ---- estimate-quality experiment -------------------------------------
    LOGGER.info("Estimate-quality experiment")
    experiment_rows: List[Dict[str, object]] = []
    for label, curve in (
        ("Naive estimate (no fixed effects)", naive_curve),
        ("Causal estimate (Phase 3 dose-response)", causal_curve),
        ("Simulated truth (unavailable in practice)", true_curve),
    ):
        built = promo_lp.build_candidates(pairs, curve, category_baseline)
        solved = promo_lp.solve(built, budget=budget, max_per_store_category=3)
        realised = promo_lp.evaluate_plan_under(
            solved["plan"], pairs, true_curve, category_baseline
        )
        experiment_rows.append({
            "plan_built_on": label,
            "promotions": solved["n_selected"],
            "planned_profit": solved["total_profit"],
            "spend": realised["spend"],
            "realised_profit": realised["realised_profit"],
            "loss_making_picks": realised["n_loss_making"],
        })
    experiment = pd.DataFrame(experiment_rows)
    experiment["gap_vs_best"] = experiment["realised_profit"].max() - experiment["realised_profit"]

    # ---- Monte Carlo -----------------------------------------------------
    LOGGER.info("Monte Carlo")
    draws = monte_carlo.simulate_plan(plan, n_draws=4000, seed=config.seed())
    mc_summary = monte_carlo.summarise(draws, solution["total_profit"])
    mc_sensitivity = monte_carlo.sensitivity(
        plan, solution["total_profit"], [0.6, 0.7, 0.8, 0.88, 1.0, 1.1, 1.2]
    )

    LOGGER.info("Rendering figures")
    fig_service = figure_service_levels(optimal_policy, figures)
    perishable = optimal_policy["is_perishable"]
    perishable_service_level = optimal_policy.loc[perishable, "service_level"].median() * 100
    ambient_service_level = optimal_policy.loc[~perishable, "service_level"].median() * 100
    fig_experiment = figure_estimate_quality(experiment, figures)
    fig_mc = figure_monte_carlo(draws, mc_summary, mc_sensitivity, figures)

    best_policy = policy_comparison.iloc[0]
    flat95 = policy_comparison[policy_comparison["policy"] == "Flat 95% service level"].iloc[0]
    saving = flat95["total_cost_gbp"] - best_policy["total_cost_gbp"]
    naive_row = experiment.iloc[0]
    causal_row = experiment.iloc[1]
    truth_row = experiment.iloc[2]

    rel = lambda p: f"figures/{p.name}"  # noqa: E731
    lines: List[str] = [
        "# Northstar — Phase 6: Prescriptive Optimisation",
        "",
        "Regenerate with `uv run python src/optimization/phase6_report.py`.",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Reorder points.** Deriving the service level per SKU from its own underage and "
        f"overage costs beats a flat 95% policy by **£{saving:,.0f}** over the holdout quarter "
        f"({best_policy['policy']}). Perishables optimally run *lower* service levels, not "
        "higher.",
        f"- **Promotion budget.** The binding constraint is not the £{budget:,.0f} budget — it "
        f"is profitability. Of {solution['n_candidates']:,} candidate promotions, "
        f"{viable_before_cannibalisation * 100:.0f}% are profitable on their own margin, but "
        f"only **{solution['n_viable']:,}** survive once cannibalisation of the rest of the "
        f"category is charged. The optimiser recommends **{solution['n_selected']} promotions** "
        f"against the roughly {observed_promotions_per_quarter:,.0f} Northstar currently runs "
        "each quarter.",
        f"- **The causal work changes the decision, not just the number.** A plan built on the "
        f"naive promotional estimate picks {int(naive_row['promotions'])} promotions of which "
        f"**all {int(naive_row['loss_making_picks'])} lose money** under the true response, "
        f"delivering £{naive_row['realised_profit']:,.0f}. The plan built on the Phase 3 causal "
        f"estimate delivers £{causal_row['realised_profit']:,.0f} — "
        f"{causal_row['realised_profit'] / max(truth_row['realised_profit'], 1e-9) * 100:.0f}% "
        "of what perfect knowledge achieves.",
        f"- **The range, not the number.** Monte Carlo puts the plan between "
        f"**£{mc_summary['p10']:,.0f} (P10)** and **£{mc_summary['p90']:,.0f} (P90)**, median "
        f"£{mc_summary['p50']:,.0f}, against the optimiser's deterministic "
        f"£{mc_summary['deterministic']:,.0f} — a figure exceeded in only "
        f"{100 - mc_summary['probability_below_deterministic'] * 100:.0f}% of draws.",
        "- **The promotional pound figures are small on purpose, and section 2 explains why**: "
        "this simulation gives promotions no traffic-building effect, so every unit they "
        "generate is either own-SKU uplift or volume taken from a neighbour. The transferable "
        "result is the method and the relative comparison, not the absolute level.",
        "",
        "---",
        "",
        "## 1. Reorder points from forecast uncertainty",
        "",
        "The policy is the standard one:",
        "",
        "```",
        "reorder point = expected demand over lead time + safety stock",
        "safety stock  = z(service level) x sigma of forecast error over lead time",
        "```",
        "",
        "Two choices carry the weight, and earlier phases settled both.",
        "",
        "**Sigma is forecast error, not demand variance.** Sizing on how much demand varies "
        "protects against the wrong thing — what causes a stockout is the part of demand the "
        "forecast missed. Phase 5's held-out quarter supplies that distribution, and Phase 5 "
        "also showed it is not homogeneous, so it is estimated per segment:",
        "",
        _table(sigma_lookup.rename(columns={
            "demand_volatility_segment": "volatility segment", "promo_flag": "on promotion",
            "sigma_daily": "daily error SD", "bias_daily": "daily bias", "rows": "rows",
        }), "{:.3f}"),
        "",
        "**The service level is derived, not picked.** The newsvendor critical ratio "
        "`Cu / (Cu + Co)` sets it per SKU, where `Cu` is margin forgone on an unserved sale and "
        "`Co` is holding cost plus expected spoilage.",
        "",
        f"![Service levels]({rel(fig_service)})",
        "",
        "This produces a result that a flat policy cannot: **perishables optimally run lower "
        f"service levels** (median {perishable_service_level:.1f}% "
        f"against {ambient_service_level:.1f}% "
        "for ambient lines). Holding an extra unit of salad that will be thrown away costs the "
        "full unit cost; holding an extra tin costs a few pence of capital. Chasing 98% "
        "availability on fresh produce destroys margin.",
        "",
        "Two caveats on those numbers. The ratio is clipped to [50%, 99.5%] — below 50% a "
        "replenishment policy stops being credible to a planner, and above 99.5% the "
        "z-multiplier explodes on thin data. Several ambient categories sit at the upper clip, "
        "so their *ordering* is meaningful but the exact figure is the bound, not an estimate. "
        "And the model prices spoilage as a probability of losing the full unit cost, ignoring "
        "markdown recovery, which pushes fresh service levels lower than a retailer running "
        "reduced-to-clear would choose.",
        "",
        _table(policy_comparison[[
            "policy", "stockout_rate", "units_short", "lost_margin_gbp",
            "holding_cost_gbp", "total_cost_gbp", "mean_safety_stock",
        ]].rename(columns={
            "policy": "policy", "stockout_rate": "stockout rate", "units_short": "units short",
            "lost_margin_gbp": "lost margin £", "holding_cost_gbp": "holding £",
            "total_cost_gbp": "total cost £", "mean_safety_stock": "mean safety stock",
        })),
        "",
        "## 2. Promotion budget allocation",
        "",
        f"An integer program over {solution['n_candidates']:,} (store, SKU, depth) options, "
        f"maximising incremental profit subject to a £{budget:,.0f} quarterly budget, one depth "
        "per store × SKU, and at most three promotions per store × category.",
        "",
        "### The accounting that decides everything",
        "",
        "Valuing a promotion as *incremental units × margin* is wrong, and generously so. "
        "Discounting also cuts the margin on volume that would have sold anyway:",
        "",
        "```",
        "incremental profit = (baseline + incremental) x discounted margin",
        "                   - baseline x full margin",
        "                   - promotion cost",
        "                   - cannibalisation",
        "```",
        "",
        f"On the promoted SKU's own P&L, **{viable_before_cannibalisation * 100:.0f}%** of "
        "candidates clear zero — and the shallower the discount, the more of them do:",
        "",
        _table(by_depth_gross, "{:.1f}"),
        "",
        "Deep discounts are where the margin sacrificed on baseline volume overwhelms the "
        "uplift. That much is standard. What changes the answer entirely is the next term.",
        "",
        "### Cannibalisation is the whole ballgame",
        "",
        f"Phase 3 measured a **{promo_lp.CANNIBALISATION_FIRST_PROMO:.1%}** depression on "
        "non-promoted SKUs in the same store and category. A category holds a dozen or more "
        "SKUs, so 6% of the category's margin is a much larger number than one SKU's promotional "
        "gain. Charging it takes the viable share from "
        f"**{viable_before_cannibalisation * 100:.0f}%** to "
        f"**{solution['n_viable'] / max(solution['n_candidates'], 1) * 100:.2f}%**.",
        "",
        _table(cannibalisation_sweep.rename(columns={
            "assumption": "cannibalisation assumed",
            "viable_candidates": "viable candidates",
            "promotions_selected": "promotions selected",
            "profit_the_plan_predicted": "profit the plan predicted £",
            "profit_under_measured_rate": "profit under the measured rate £",
        })),
        "",
        "The middle column is what an optimiser believes; the last is what it gets. Ignoring "
        "cannibalisation does not make it go away — it produces a plan that promises "
        f"£{cannibalisation_sweep.iloc[0]['profit_the_plan_predicted']:,.0f} and delivers "
        f"£{cannibalisation_sweep.iloc[0]['profit_under_measured_rate']:,.0f}.",
        "",
        "**This is the single largest modelling assumption in the phase, and it is an "
        "empirical measurement rather than a judgement call** — which is precisely why Phase 3 "
        "was worth doing. But it rests on a linearisation (below), and a reader should treat "
        f"the {solution['n_selected']}-promotion recommendation as directional: the robust "
        "conclusion is *far fewer, shallower, and spread across categories*, not that exactly "
        f"{solution['n_selected']} promotions is optimal.",
        "",
        "### Why the pound figures here are small, and what that means",
        "",
        "A recommendation of a handful of promotions against the ~"
        f"{observed_promotions_per_quarter:,.0f} Northstar runs, delivering hundreds rather than "
        "hundreds of thousands of pounds, deserves an explanation rather than a shrug.",
        "",
        "Two mechanics in the data generating process drive it:",
        "",
        "1. **Uplift is credited at the discounted margin; cannibalisation is charged at the "
        "full one.** Category *volume* does rise when promotions run — total units in a store × "
        "category climb steadily with the number of concurrent promotions. But the extra volume "
        "arrives on a discounted line while the volume it displaces was earning full margin, so "
        "category *profit* can fall even as category *units* rise. That is a real retail "
        "phenomenon and the model is capturing it correctly.",
        "2. **There is no traffic-building effect to offset it.** In this simulation store "
        "footfall is a function of the calendar and noise; promotions do not draw shoppers in, "
        "grow baskets, or win share from competitors. Every unit a promotion generates is either "
        "the SKU's own uplift or volume taken from a category neighbour.",
        "",
        "Real retailers promote partly for footfall, basket and competitive-share reasons that "
        "this data does not represent. **So the finding is a statement about this data "
        "generating process, not advice to a real grocer**: given these mechanics, the optimiser "
        "correctly concludes that promotion at the observed scale destroys margin. The "
        "transferable results are the *method* — the accounting, the cannibalisation charge, the "
        "sensitivity structure — and the relative comparison in section 3, which holds whatever "
        "the absolute level.",
        "",
        "### The recommended plan",
        "",
        _table(depth_mix.rename(columns={
            "discount_pct": "discount %", "promotions": "promotions", "spend": "spend £",
            "profit": "incremental profit £", "incremental_units": "incremental units",
            "profit_per_pound": "profit per £ spent",
        })),
        "",
        f"Budget utilisation: **{solution['budget_used'] * 100:.1f}%** "
        f"(£{solution['total_cost']:,.0f} of £{budget:,.0f}). Solver status: "
        f"{solution['status']}.",
        "",
        "### Cannibalisation is linearised, and that is an approximation",
        "",
        "Phase 3 measured cannibalisation as genuinely non-linear — 6% for the first concurrent "
        "promotion in a store × category, deepening to 16% by the fourth, then saturating. An "
        "integer program cannot express that directly. Two devices stand in: every promotion is "
        "charged the first-promotion marginal loss against its category's untreated baseline, "
        "and a cap of three promotions per store × category keeps the plan inside the range "
        "where that linear charge is roughly right rather than out where the effect has "
        "saturated and the charge would overstate it.",
        "",
        "## 3. What the causal correction is worth",
        "",
        f"![Estimate quality]({rel(fig_experiment)})",
        "",
        "This is the part that justifies Phases 3 and 4 commercially. The same optimiser, the "
        "same budget, the same constraints — run three times on three different beliefs about "
        "how promotions work, then every plan scored under the **true** promotional response:",
        "",
        _table(experiment.rename(columns={
            "plan_built_on": "plan built on", "promotions": "promotions",
            "planned_profit": "profit the plan predicted £", "spend": "spend £",
            "realised_profit": "profit actually delivered £",
            "loss_making_picks": "loss-making picks", "gap_vs_best": "gap vs best £",
        })),
        "",
        "The naive estimate does not merely predict too much profit — it **picks the wrong "
        "promotions**. Overstating uplift makes deep discounts look attractive on SKUs where "
        f"the margin sacrificed on baseline volume swamps the gain, so "
        f"{int(naive_row['loss_making_picks'])} of its {int(naive_row['promotions'])} selections "
        "are loss-making under the true response.",
        "",
        "The three effect curves, at Medium price elasticity:",
        "",
        _table(curve_comparison, "{:.3f}"),
        "",
        f"The causal plan captures "
        f"**{causal_row['realised_profit'] / max(truth_row['realised_profit'], 1e-9) * 100:.1f}%** "
        "of what perfect knowledge would have delivered. The remaining gap is the price of "
        "estimation error that Phase 4 was explicit about — the DiD estimate is an upper bound, "
        "and an upper bound over-allocates.",
        "",
        "## 4. Profit range, not a point estimate",
        "",
        f"![Monte Carlo]({rel(fig_mc)})",
        "",
        "The plan rests on three estimated quantities, so quoting a single profit figure would "
        "be a fiction. Four thousand draws over:",
        "",
        "- **promotional response** — centred at 0.88x the estimate, because Phase 4 found the "
        "DiD figure overshot the simulated truth and concluded it should be read as an upper "
        "bound;",
        "- **baseline demand** — spread from Phase 5's held-out forecast error;",
        "- **cannibalisation** — spread around Phase 3's measured 6%.",
        "",
        _table(pd.DataFrame([{
            "deterministic plan estimate": mc_summary["deterministic"],
            "mean": mc_summary["mean"], "P10": mc_summary["p10"],
            "P50": mc_summary["p50"], "P90": mc_summary["p90"],
            "probability of loss": mc_summary["probability_of_loss"],
        }]), "{:,.3f}"),
        "",
        f"**The honest headline is £{mc_summary['p10']:,.0f} to £{mc_summary['p90']:,.0f}**, "
        f"median £{mc_summary['p50']:,.0f}. The deterministic plan estimate of "
        f"£{mc_summary['deterministic']:,.0f} is exceeded in only "
        f"{100 - mc_summary['probability_below_deterministic'] * 100:.0f}% of draws — it is not "
        "a forecast, it is the optimiser's best case.",
        "",
        "### How wrong can the effect estimate be?",
        "",
        _table(mc_sensitivity.rename(columns={
            "effect_multiplier": "uplift multiplier", "mean_profit": "mean profit £",
            "p10": "P10 £", "p90": "P90 £", "probability_of_loss": "probability of loss",
        })),
        "",
        "## 5. Limitations",
        "",
        "- **The forecast target is censored.** Phase 5 forecasts `units_sold`, not demand. On "
        "days when stock bound, observed sales understate what customers wanted, so both the "
        "mean and the error spread feeding the safety-stock calculation are biased low — in the "
        "same direction, on the same days. Service levels here are therefore slightly "
        "optimistic.",
        "- **Lead-time error scaling assumes independence.** `sigma x sqrt(L)` treats "
        "consecutive days' forecast errors as independent. They are not: demand is "
        "autocorrelated and a forecast wrong on Monday is usually wrong on Tuesday. This "
        "understates the true lead-time spread.",
        "- **Cannibalisation is linear here and non-linear in reality.** See section 2.",
        "- **The plan assumes the promotional calendar is otherwise unchanged.** Competitor "
        "response, supplier funding negotiations and shelf-space constraints are all outside "
        "this model.",
        "- **`true_curve` is not available in practice.** It exists here only because the data "
        "is simulated. Its role is to score the other two plans, not to build one.",
        "",
        "---",
        "",
        "## Recommendation",
        "",
        f"**Inventory.** Adopt the cost-derived reorder points — worth £{saving:,.0f} a quarter "
        "against a flat 95% policy, earned chiefly by *reducing* safety stock on perishables "
        "rather than adding it everywhere.",
        "",
        "**Promotions.** Run far fewer, shallower promotions, and spread them across categories "
        "rather than concentrating them. Northstar currently runs roughly "
        f"{observed_promotions_per_quarter:,.0f} promotions a quarter; once cannibalisation is "
        "charged at the rate Phase 3 measured, only a small fraction of them create value. The "
        f"specific figure of {solution['n_selected']} is directional — it depends on a "
        "linearised cannibalisation charge — but the direction is robust across every "
        "assumption tested here.",
        "",
        f"**Budget the range, not the number**: £{mc_summary['p10']:,.0f}–"
        f"£{mc_summary['p90']:,.0f} of incremental profit, median £{mc_summary['p50']:,.0f}. The "
        f"optimiser's own £{mc_summary['deterministic']:,.0f} is its best case, not its "
        "expectation.",
        "",
        "**Re-estimate the promotional response causally each quarter.** On this budget that "
        f"choice alone is worth £{naive_row['gap_vs_best'] - causal_row['gap_vs_best']:,.0f} "
        "against using a naive estimate.",
        "",
    ]

    path = config.path("reports") / "phase6_optimization.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print(f"\nPhase 6 report written to {build_report()}")
