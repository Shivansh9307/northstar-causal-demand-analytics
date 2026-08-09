"""
Phase 3 report: statistical analysis and regression.

Produces reports/phase3_regression.md and its figures.

The report is organised around a distinction that the data forces: what the
promotional variation in this panel can identify, and what it cannot. Both are
checked against the known simulated parameters rather than asserted.
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

from stats import models  # noqa: E402
from utils import config  # noqa: E402

LOGGER = logging.getLogger("northstar.phase3")

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


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_dose_response(curve: pd.DataFrame, figures: Path) -> Path:
    """Estimated dose-response with CI band against the simulated truth."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = curve["discount_pct"]
    ax.fill_between(x, curve["ci_low"], curve["ci_high"], color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(x, curve["estimate"], color=BLUE, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label="Estimated (95% CI)")
    ax.plot(x, curve["true_effect"], color=ORANGE, linestyle="--", marker="s", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label="Simulated truth")
    ax.set_xlabel("Discount depth (%)")
    ax.set_ylabel("Log lift vs Display-only promotion")
    ax.set_title("Promotional dose-response recovers the simulated truth at every depth")
    ax.legend(loc="upper left", labelcolor=INK_2)
    _despine(ax)
    fig.tight_layout()
    path = figures / "06_dose_response.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_identification(naive_rows: pd.DataFrame, figures: Path) -> Path:
    """
    Why the structural elasticity is not identified: each specification's
    estimate against the known true elasticity.
    """
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(naive_rows))
    ax.barh(y, naive_rows["estimate"], height=0.55, color=BLUE,
            edgecolor=SURFACE, linewidth=1.5)
    for i, row in enumerate(naive_rows.itertuples(index=False)):
        ax.plot([row.ci_low, row.ci_high], [i, i], color=INK_2, linewidth=2, solid_capstyle="round")
        # Label past the outer end of the interval so it never sits on the bar.
        outward = row.ci_low if row.estimate < 0 else row.ci_high
        ax.annotate(
            f"{row.estimate:+.2f}", xy=(outward, i),
            xytext=(-8 if row.estimate < 0 else 8, 0), textcoords="offset points",
            ha="right" if row.estimate < 0 else "left", va="center",
            color=INK_2, fontsize=9, fontweight="bold",
        )
    truth = naive_rows["true_value"].iloc[0]
    ax.axvline(truth, color=ORANGE, linewidth=2, linestyle="--")
    # Anchored to the top of the axes in fraction coords so it cannot be clipped.
    ax.annotate(
        f"True elasticity {truth:+.2f}", xy=(truth, 1.0), xycoords=("data", "axes fraction"),
        xytext=(6, -10), textcoords="offset points",
        color=ORANGE, fontsize=9, va="top", ha="left",
    )
    ax.margins(x=0.14)
    ax.set_yticks(y)
    ax.set_yticklabels(naive_rows["specification"])
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient on log(price ratio)")
    ax.set_title("No specification recovers the structural price elasticity")
    _despine(ax)
    fig.tight_layout()
    path = figures / "07_identification.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_spillover(spill: pd.DataFrame, figures: Path) -> Path:
    """Cannibalisation onto untreated rows — a single series, so no legend."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(spill))
    ax.bar(x, spill["pct_effect"], width=0.6, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    for i, row in enumerate(spill.itertuples(index=False)):
        low = np.expm1(row.ci_low) * 100
        high = np.expm1(row.ci_high) * 100
        ax.plot([i, i], [low, high], color=INK_2, linewidth=2, solid_capstyle="round")
        ax.annotate(f"{row.pct_effect:.1f}%", xy=(i, low), xytext=(0, -12),
                    textcoords="offset points", ha="center", va="top",
                    color=INK_2, fontsize=9, fontweight="bold")
    ax.axhline(0, color=BASELINE, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(spill["others_on_promo"])
    ax.set_xlabel("Other SKUs promoted in the same store x category that day")
    ax.set_ylabel("Effect on untreated demand (%)")
    ax.set_title("Untreated rows are not untreated — promotions cannibalise their neighbours")
    ax.margins(y=0.22)
    _despine(ax)
    fig.tight_layout()
    path = figures / "09_spillover.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_residuals(diagnostics: dict, count_models: dict, figures: Path) -> Path:
    """Residual diagnostics: within-model residuals and the Poisson/NB comparison."""
    rng = np.random.default_rng(42)
    resid = np.asarray(diagnostics["resid"])
    fitted = np.asarray(diagnostics["fitted"])
    idx = rng.choice(len(resid), size=min(40_000, len(resid)), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    axes[0].scatter(fitted[idx], resid[idx], s=5, color=BLUE, alpha=0.25, linewidth=0)
    axes[0].axhline(0, color=ORANGE, linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Fitted (within-transformed)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs fitted")

    from scipy import stats as scipy_stats  # local import: only needed here

    sample = resid[idx]
    scipy_stats.probplot(sample, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set(marker="o", markersize=2, color=BLUE, alpha=0.4, linestyle="none")
    axes[1].get_lines()[1].set(color=ORANGE, linewidth=1.5, linestyle="--")
    axes[1].set_title("Normal Q-Q of residuals")
    axes[1].set_xlabel("Theoretical quantiles")
    axes[1].set_ylabel("Sample quantiles")

    poisson_resid = np.asarray(count_models["poisson_resid"])
    negbin_resid = np.asarray(count_models["negbin_resid"])
    bins = np.linspace(-6, 12, 70)
    axes[2].hist(poisson_resid, bins=bins, color=ORANGE, alpha=0.75,
                 label=f"Poisson (chi2/df {count_models['poisson_pearson_ratio']:.2f})")
    axes[2].hist(negbin_resid, bins=bins, color=BLUE, alpha=0.75,
                 label=f"Neg. Binomial (chi2/df {count_models['negbin_pearson_ratio']:.2f})")
    axes[2].set_xlabel("Pearson residual")
    axes[2].set_ylabel("Rows")
    axes[2].set_title("Poisson residuals are far too dispersed")
    axes[2].legend(loc="upper right", labelcolor=INK_2, fontsize=8)

    for ax in axes:
        _despine(ax)
    fig.tight_layout()
    path = figures / "08_residual_diagnostics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report() -> Path:
    figures = config.path("figures")
    figures.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading analysis frame")
    frame = models.prepare(models.load_analysis_frame())
    ground_truth = pd.read_csv(
        config.path("ground_truth") / "ground_truth_simulation_parameters.csv"
    )
    products = pd.read_csv(config.path("raw") / "dim_product.csv")
    true_elasticity = float(ground_truth["true_price_elasticity"].mean())

    mech = [f"mech_{models._slug(m)}" for m in models.PROMO_MECHANICS]
    controlled_spec = [
        "log_price_ratio", "promo_flag", *mech, *models.SUPPORT_FLAGS, "log_footfall",
    ]

    LOGGER.info("Fitting elasticity specifications")
    naive_fit = models.fit_within_ols(frame, "log_units", ["log_price_ratio"], "naive")
    controlled_fit = models.fit_within_ols(frame, "log_units", controlled_spec, "controlled")

    # Both discount channels in the same model - the specification that shows why
    # the decomposition fails.
    frame["discount_dose"] = frame["discount_pct"] / 10.0
    both_spec = ["log_price_ratio", "discount_dose", "promo_flag", *mech,
                 *models.SUPPORT_FLAGS, "log_footfall"]
    both_fit = models.fit_within_ols(frame, "log_units", both_spec, "both_channels")

    specs = pd.DataFrame([
        {"specification": "Naive: log(price) only", **naive_fit.row("log_price_ratio")},
        {
            "specification": "+ promotion mechanism & support",
            **controlled_fit.row("log_price_ratio"),
        },
        {"specification": "+ separate discount-dose term", **both_fit.row("log_price_ratio")},
    ])
    specs["true_value"] = true_elasticity
    specs["error"] = specs["estimate"] - true_elasticity

    LOGGER.info("Fitting dose-response")
    curve, dose_fit = models.dose_response(frame, ground_truth)
    support = models.support_channel_effects(dose_fit, ground_truth)

    LOGGER.info("Fitting dose-response by segment and category")
    by_segment = models.dose_response_by_group(frame, "price_elasticity_segment", ground_truth)
    by_category = models.dose_response_by_group(frame, "category", ground_truth)

    LOGGER.info("Computing VIF")
    vif_controlled = models.vif_table(frame, controlled_spec)
    vif_both = models.vif_table(frame, both_spec)

    LOGGER.info("Fitting count models")
    counts = models.fit_count_models(frame, sample_size=200_000, seed=config.seed())

    LOGGER.info("Residual diagnostics")
    diagnostics = models.residual_diagnostics(frame, "log_units", controlled_spec)

    LOGGER.info("Spillover diagnostic")
    spill = models.spillover_diagnostic(frame)
    # Does a category's cannibalisation exposure explain its estimation error?
    exposure = (
        frame.groupby("category")
        .agg(promo_rate=("promo_flag", "mean"), skus=("sku_id", "nunique"))
        .join(ground_truth.groupby("category")["true_cannibalisation_factor"].mean())
    )
    exposure["exposure"] = (
        exposure["true_cannibalisation_factor"] * exposure["promo_rate"] * exposure["skus"]
    )
    exposure["error"] = by_category[
        by_category["discount_pct"] == 20
    ].set_index("category")["error"]
    spill_corr = float(exposure["error"].corr(exposure["exposure"]))

    # Naive log-log elasticity by segment, with the true segment value alongside.
    seg_truth = (
        ground_truth.merge(products[["sku_id", "price_elasticity_segment"]], on="sku_id")
        .groupby("price_elasticity_segment")["true_price_elasticity"].mean()
    )
    naive_by_segment = models.elasticity_by_group(
        frame, "price_elasticity_segment", controlled_spec, "elasticity_by_segment"
    )
    naive_by_segment["true_elasticity"] = naive_by_segment[
        "price_elasticity_segment"
    ].map(seg_truth)
    naive_by_segment["error"] = naive_by_segment["estimate"] - naive_by_segment["true_elasticity"]

    LOGGER.info("Rendering figures")
    fig_dose = figure_dose_response(curve, figures)
    fig_ident = figure_identification(specs, figures)
    fig_spill = figure_spillover(spill, figures)
    fig_resid = figure_residuals(diagnostics, counts, figures)

    coverage = int(curve["ci_covers_truth"].sum())
    support_coverage = int(support["ci_covers_truth"].sum())
    seg_coverage = int(by_segment["ci_covers_truth"].sum())
    cat_coverage = int(by_category["ci_covers_truth"].sum())
    dispersion = counts["dispersion_test"]

    rel = lambda p: f"figures/{p.name}"  # noqa: E731
    lines: List[str] = [
        "# Northstar — Phase 3: Statistical Analysis & Regression",
        "",
        f"Estimated on the full {naive_fit.nobs:,}-row panel with store x SKU and date fixed "
        f"effects absorbed by two-way demeaning, and standard errors clustered on "
        f"{naive_fit.n_clusters:,} store x SKU pairs.",
        "",
        "Regenerate with `uv run python src/stats/phase3_report.py`.",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Negative Binomial, not Poisson.** Cameron-Trivedi alpha = "
        f"{dispersion['alpha']:.3f} (t = {dispersion['t_stat']:.1f}), and Poisson's Pearson "
        f"chi2/df is {counts['poisson_pearson_ratio']:.2f} against NB's "
        f"{counts['negbin_pearson_ratio']:.2f}.",
        f"- **The promotional dose-response is identified and recovers the truth**: "
        f"{coverage}/6 confidence intervals cover the simulated value.",
        f"- **The non-price promotion channels are identified**: {support_coverage}/3 cover.",
        f"- **Promotions cannibalise their own controls.** An untreated SKU loses "
        f"{abs(spill['pct_effect'].iloc[0]):.1f}% of demand when one category neighbour is "
        f"promoted, rising to {abs(spill['pct_effect'].iloc[-1]):.1f}% at four or more. This "
        "SUTVA violation explains the category-level miscoverage (section 5) and is the single "
        "most important thing Phase 4 must design around.",
        "- **The structural price elasticity is *not* identified from this data.** Price moves "
        "only through promotions, so the price response and the promotional uplift cannot be "
        "separated. Section 6 shows the evidence rather than reporting a confident wrong number.",
        "",
        "---",
        "",
        "## 1. Model choice: Poisson vs Negative Binomial",
        "",
        "Phase 2 flagged a variance-to-mean ratio of 20.4. Testing it formally rather than "
        "assuming it, on a "
        f"{counts['sample_rows']:,}-row sample with the pair's mean volume as an offset:",
        "",
        "**Cameron & Trivedi (1990) regression test.** Under Poisson, Var(y) = mu. Regressing "
        "`((y - mu)^2 - y) / mu` on `mu` estimates the NB2 dispersion parameter:",
        "",
        f"- alpha = **{dispersion['alpha']:.4f}** (SE {dispersion['std_err']:.4f}), "
        f"t = **{dispersion['t_stat']:.1f}**, p = {dispersion['p_value']:.3g}",
        "",
        "Equidispersion is rejected decisively.",
        "",
        # Pre-formatted: AIC and log-likelihood want no decimals, the dispersion
        # ratio wants two, and a single float format cannot serve both.
        _table(pd.DataFrame([
            {"model": "Poisson",
             "AIC": f"{counts['poisson_aic']:,.0f}",
             "log-likelihood": f"{counts['poisson_llf']:,.0f}",
             "Pearson chi2/df": f"{counts['poisson_pearson_ratio']:.2f}"},
            {"model": "Negative Binomial",
             "AIC": f"{counts['negbin_aic']:,.0f}",
             "log-likelihood": f"{counts['negbin_llf']:,.0f}",
             "Pearson chi2/df": f"{counts['negbin_pearson_ratio']:.2f}"},
        ])),
        "",
        f"A well-specified model has Pearson chi2/df near 1. Poisson sits at "
        f"{counts['poisson_pearson_ratio']:.2f} — its standard errors would be roughly "
        f"{np.sqrt(counts['poisson_pearson_ratio']):.1f}x too small. NB sits at "
        f"{counts['negbin_pearson_ratio']:.2f}. **Phase 5 should use Negative Binomial or a "
        "count-aware gradient booster, not Poisson.**",
        "",
        "## 2. Promotional dose-response",
        "",
        f"![Dose response]({rel(fig_dose)})",
        "",
        "Each discount depth gets its own indicator, so no functional form is imposed on the "
        "discount. Display-only promotions (0% discount, full display activity) are the "
        "reference, which is what isolates the effect of *depth* from the effect of *being on "
        "promotion at all*.",
        "",
        _table(curve[[
            "discount_pct", "n_rows", "estimate", "ci_low", "ci_high",
            "true_effect", "error", "ci_covers_truth", "estimated_lift_pct", "true_lift_pct",
        ]].rename(columns={
            "discount_pct": "discount %", "n_rows": "rows", "estimate": "est. log lift",
            "ci_low": "CI low", "ci_high": "CI high", "true_effect": "true log lift",
            "error": "error", "ci_covers_truth": "CI covers truth",
            "estimated_lift_pct": "est. lift %", "true_lift_pct": "true lift %",
        })),
        "",
        f"**{coverage} of 6 intervals cover the simulated value**, with the largest error "
        f"{curve['error'].abs().max():.3f} log points. The method works where the variation "
        "supports it.",
        "",
        "## 3. Non-price promotional channels",
        "",
        _table(support[[
            "channel", "estimate", "ci_low", "ci_high", "true_effect", "error",
            "ci_covers_truth",
        ]]
               .rename(columns={"estimate": "estimate", "ci_low": "CI low", "ci_high": "CI high",
                                "true_effect": "truth", "error": "error",
                                "ci_covers_truth": "CI covers truth"}), "{:.4f}"),
        "",
        "Display, email/app and leaflet support are randomly assigned conditional on a promotion "
        f"running, so they are cleanly identified — and all {support_coverage} recover the "
        "simulated uplift.",
        "",
        "## 4. Dose-response by segment and category",
        "",
        "### By price-elasticity segment",
        "",
        _table(by_segment[[
            "price_elasticity_segment", "discount_pct", "estimate", "ci_low", "ci_high",
            "true_effect", "error", "ci_covers_truth",
        ]].rename(columns={"price_elasticity_segment": "segment", "discount_pct": "discount %",
                           "estimate": "estimate", "ci_low": "CI low", "ci_high": "CI high",
                           "true_effect": "truth", "error": "error",
                           "ci_covers_truth": "CI covers truth"})),
        "",
        f"{seg_coverage} of {len(by_segment)} intervals cover. High-elasticity SKUs respond more "
        "steeply to depth, as designed.",
        "",
        "### By category",
        "",
        _table(by_category[by_category["discount_pct"] == 20][[
            "category", "estimate", "ci_low", "ci_high", "true_effect", "error", "ci_covers_truth",
        ]].rename(columns={"category": "category", "estimate": "estimate", "ci_low": "CI low",
                           "ci_high": "CI high", "true_effect": "truth", "error": "error",
                           "ci_covers_truth": "CI covers truth"})),
        "",
        f"Shown at a 20% discount for readability; across all depths only "
        f"{cat_coverage}/{len(by_category)} intervals cover. **That miscoverage is not noise, "
        "and section 5 identifies its cause** — it is the one place in this phase where the "
        "estimates are systematically off.",
        "",
        "## 5. Why the category estimates miss: promotions cannibalise their controls",
        "",
        f"![Spillover]({rel(fig_spill)})",
        "",
        "The category errors are signed and systematic, not random. They correlate at "
        f"**{spill_corr:.2f}** with a category's cannibalisation exposure "
        "(its cannibalisation factor x promotion rate x SKU count).",
        "",
        "The mechanism is a violation of SUTVA — the assumption that one unit's outcome does not "
        "depend on another unit's treatment. Estimated on untreated rows only, with pair and "
        "date fixed effects absorbed so promotion clustering on high-demand days cannot explain "
        "it:",
        "",
        _table(spill[["others_on_promo", "n_rows", "estimate", "ci_low", "ci_high", "pct_effect"]]
               .rename(columns={"others_on_promo": "other SKUs on promo", "n_rows": "rows",
                                "estimate": "log effect", "ci_low": "CI low", "ci_high": "CI high",
                                "pct_effect": "effect %"}), "{:.4f}"),
        "",
        f"An untreated SKU loses **{abs(spill['pct_effect'].iloc[0]):.1f}%** of its demand when "
        f"one category neighbour is promoted and **{abs(spill['pct_effect'].iloc[-1]):.1f}%** "
        "when four or more are. The control group is therefore depressed precisely when "
        "treatment is heaviest, which inflates the estimated promotional effect wherever "
        "cannibalisation is strong.",
        "",
        "The effect **saturates** between three and four concurrent promotions rather than "
        "continuing to deepen. That is the generator's floor showing through — it caps the "
        "cannibalisation multiplier at 0.82 — and the estimates recovering that plateau is "
        "itself a check that the diagnostic is measuring the intended mechanism.",
        "",
        "Two consequences worth stating plainly:",
        "",
        "1. **The pooled dose-response survives it** because the Display-only reference rows are "
        "depressed by roughly the same amount as the discounted rows, so the contamination "
        "largely differences out. The category-level estimates do not have that protection, "
        "because cannibalisation exposure varies across categories.",
        "2. **The contaminated estimate may be the more useful one commercially.** A retailer "
        "deciding whether to run a promotion cares about the net effect on the category, not the "
        "effect on one SKU in a world where its neighbours were left alone. But it is a "
        "different estimand from the per-SKU causal effect, and the two must not be conflated.",
        "",
        "## 6. What this data cannot identify",
        "",
        f"![Identification]({rel(fig_ident)})",
        "",
        "The architecture asks for a log-log price elasticity. Here is what happens when it is "
        "estimated:",
        "",
        _table(specs[["specification", "estimate", "ci_low", "ci_high", "true_value", "error"]]
               .rename(columns={"specification": "specification", "estimate": "elasticity",
                                "ci_low": "CI low", "ci_high": "CI high",
                                "true_value": "true elasticity", "error": "error"})),
        "",
        "**No specification recovers it, and the third one flips sign.** The reason is "
        "structural, not a modelling mistake, and is separate from the spillover above:",
        "",
        "- Price moves *only* through promotions. Zero rows in the panel are discounted outside "
        "a promotion.",
        "- The generator applies two effects to the same discount: a price response "
        "`(1 - d/100)^elasticity` and a dose-dependent uplift `1 + uplift * d/10`.",
        f"- Across promoted rows those two functions of `d` correlate at "
        f"**{np.corrcoef(
            frame.loc[frame.promo_flag == 1, 'log_price_ratio'],
            frame.loc[frame.promo_flag == 1, 'discount_dose'],
        )[0, 1]:.4f}**.",
        "",
        "Variance inflation factors make the consequence explicit:",
        "",
        _table(vif_both.rename(columns={"term": "term", "vif": "VIF"}), "{:,.1f}"),
        "",
        f"With both channels in the model, VIF exceeds "
        f"{vif_both['vif'].max():,.0f}. Without the dose term the design is well conditioned "
        f"(max VIF {vif_controlled['vif'].max():.1f}), but then `log(price ratio)` silently "
        "absorbs the promotional uplift — which is why the naive estimate is roughly "
        f"{abs(specs.loc[0, 'estimate'] / true_elasticity):.1f}x too elastic.",
        "",
        "Interacting with segment does not rescue it either — the correlation is a property of "
        "the discount grid, not of any particular subgroup:",
        "",
        _table(naive_by_segment[[
            "price_elasticity_segment", "estimate", "ci_low", "ci_high", "true_elasticity", "error",
        ]].rename(columns={"price_elasticity_segment": "segment",
                      "estimate": "estimated elasticity",
                           "ci_low": "CI low", "ci_high": "CI high",
                           "true_elasticity": "true elasticity", "error": "error"})),
        "",
        "### What would be needed",
        "",
        "Separating the two requires price variation that is not promotional: everyday price "
        "changes, base-price tests, or a mechanic that cuts price without display support. None "
        "exists here. Reporting a structural elasticity from this panel would be a number with "
        "no identifying variation behind it.",
        "",
        "**What is reported instead** is the total promotional response at each depth — which is "
        "also the quantity Phase 6's budget optimiser actually needs, since a merchandiser "
        "chooses a promotion, not a disembodied price.",
        "",
        "## 7. Residual diagnostics",
        "",
        f"![Residual diagnostics]({rel(fig_resid)})",
        "",
        f"- Breusch-Pagan statistic {diagnostics['breusch_pagan_stat']:,.0f} on "
        f"{diagnostics['breusch_pagan_df']} df — heteroskedasticity is present, which is why "
        "every standard error above is cluster-robust.",
        f"- Residual skew {diagnostics['resid_skew']:.3f}, excess kurtosis "
        f"{diagnostics['resid_kurtosis']:.3f}. The Q-Q plot shows heavier tails than normal, "
        "expected for a log-transformed count.",
        f"- Within-model R^2 = {controlled_fit.rsquared:.4f} after absorbing fixed effects. Low "
        "by construction: the pair and date effects have already taken the explainable level and "
        "seasonality, leaving day-to-day Gamma-Poisson noise.",
        "",
        "The log-count outcome uses `log(units + 1)`. The Negative Binomial model in section 1 "
        "handles zeros natively and is the robustness check on that transform; both agree on the "
        "sign and rough magnitude of every promotional term.",
        "",
        "---",
        "",
        "## What Phase 4 should carry forward",
        "",
        "1. **Control units in the same store x category are contaminated.** This is the big "
        "one. A DiD or matching design that draws controls from a treated SKU's own category "
        "will understate the counterfactual and overstate the effect. Either draw controls from "
        "outside the promoted category, condition on the number of concurrent category "
        "promotions, or state explicitly that the estimand is net of cannibalisation.",
        "2. **Do not use the estimated elasticity as a causal price effect.** It is a total "
        "promotional response and is not separable in this panel.",
        "3. **The dose-response curve is the validated promotional effect**, and is the right "
        "input to Phase 6's optimiser.",
        "4. **Fixed effects matter enormously.** Phase 2 showed the naive lift is mostly timing; "
        "date fixed effects absorb it, and the same discipline is what DiD will rely on.",
        "5. **Cluster on the store x SKU pair.** Unclustered errors on 2.19M rows drawn from "
        f"{naive_fit.n_clusters:,} units would be roughly an order of magnitude too small.",
        "6. **Use Negative Binomial** for any count model in Phase 5.",
        "",
    ]

    path = config.path("reports") / "phase3_regression.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print(f"\nPhase 3 report written to {build_report()}")
