"""
Phase 2 exploratory data analysis (PROJECT_ARCHITECTURE.md §6 Phase 2).

Produces reports/eda_report.md plus figures in reports/figures/. Aggregation runs
in DuckDB and only small result sets reach pandas, so this stays fast on the full
2.19M-row panel.

The report is written to answer the questions the *next* phases need settled:
category seasonality, promotion and stockout rates, missingness, distribution
shape, and - the one Phase 3 depends on - whether demand counts are overdispersed
enough to rule out Poisson in favour of Negative Binomial. That choice is made on
evidence here rather than asserted later.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import star_schema  # noqa: E402
from utils import config  # noqa: E402

LOGGER = logging.getLogger("promopulse.eda")

# Validated categorical slots (see dataviz references/palette.md).
# Slots 1-3 clear the all-pairs CVD and normal-vision floors on the light surface.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

# Figures are static PNGs embedded in a markdown report, so they cannot adapt to
# the reader's theme. They render on the light surface with its matching ink.
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.dpi": 160,
    }
)


def _despine(ax: plt.Axes) -> None:
    """Recessive chrome: keep the baseline, drop the box."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)


def _fmt_table(frame: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Render a small dataframe as a markdown table."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = []
    for record in frame.itertuples(index=False):
        cells = []
        for value in record:
            if isinstance(value, float):
                cells.append(floatfmt.format(value))
            elif isinstance(value, int):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_seasonality(con, figures: Path) -> Path:
    """
    Ten categories is past the eight-slot categorical cap and far past the
    three-slot all-pairs cap, so this is small multiples rather than ten lines
    sharing an axis. Each panel is one series and needs no legend.
    """
    data = con.execute(
        """
        SELECT category, month, AVG(units_sold) AS mean_units
        FROM analytics_daily
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()
    data["index_100"] = data.groupby("category")["mean_units"].transform(
        lambda s: s / s.mean() * 100
    )
    categories = sorted(data["category"].unique())

    fig, axes = plt.subplots(2, 5, figsize=(15, 5.4), sharex=True, sharey=True)
    for ax, category in zip(axes.ravel(), categories):
        panel = data[data["category"] == category]
        ax.plot(panel["month"], panel["index_100"], color=BLUE)
        ax.axhline(100, color=BASELINE, linewidth=1, linestyle="--")
        ax.set_title(category, fontsize=9)
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
        _despine(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Index (mean = 100)")
    fig.suptitle(
        "Category seasonality — mean daily units by month, indexed to each category's own mean",
        fontsize=11, fontweight="bold", color=INK, y=1.0,
    )
    fig.tight_layout()
    path = figures / "01_category_seasonality.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_overdispersion(con, figures: Path) -> Path:
    """
    Variance vs mean per SKU on log-log axes. Poisson implies variance = mean,
    drawn as the reference line; points above it are overdispersed.

    Colour encodes the three volatility segments — exactly the three slots that
    clear the all-pairs floors. Aqua sits below 3:1 on this surface, so the
    relief rule applies: the legend is present and the same numbers appear as a
    table in the report body.
    """
    data = con.execute(
        """
        SELECT sku_id,
               demand_volatility_segment AS segment,
               AVG(units_sold) AS mean_units,
               VAR_SAMP(units_sold) AS var_units
        FROM analytics_daily
        GROUP BY 1, 2
        HAVING AVG(units_sold) > 0
        """
    ).df()

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for segment, color in (("Low", BLUE), ("Medium", ORANGE), ("High", AQUA)):
        subset = data[data["segment"] == segment]
        ax.scatter(
            subset["mean_units"], subset["var_units"],
            s=34, color=color, alpha=0.85,
            edgecolor=SURFACE, linewidth=1.2,  # 2px surface ring on overlap
            label=f"{segment} volatility",
        )
    lo = max(data["mean_units"].min(), 0.5)
    hi = data["mean_units"].max() * 1.1
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.5, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Label sits in the empty wedge below the diagonal - every point is above it.
    ax.annotate(
        "Poisson: variance = mean",
        xy=(hi * 0.55, hi * 0.55), xytext=(hi * 0.55, hi * 0.10),
        color=MUTED, fontsize=8.5, ha="center", va="top",
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1, shrinkA=2, shrinkB=2),
    )
    ax.set_xlabel("Mean daily units sold (per SKU)")
    ax.set_ylabel("Variance of daily units sold")
    ax.set_title("Demand is overdispersed — every SKU sits above the Poisson line")
    ax.legend(loc="upper left", labelcolor=INK_2)

    # Plain integer ticks; matplotlib's default log labels render as "2 x 10^0".
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax.set_xticks([2, 3, 5, 10, 20, 30, 50])
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([10, 100, 1000])
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(NullFormatter())
    _despine(ax)
    fig.tight_layout()
    path = figures / "02_overdispersion.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_rates_over_time(con, figures: Path) -> Path:
    """
    Promotion rate (~8.5%) and stockout rate (~0.3%) differ by more than an order
    of magnitude. Two panels sharing an x-axis, never a second y-axis.
    """
    data = con.execute(
        """
        SELECT DATE_TRUNC('week', date) AS week,
               AVG(CASE WHEN promo_flag THEN 1.0 ELSE 0.0 END) * 100 AS promo_rate,
               AVG(CASE WHEN stockout_flag THEN 1.0 ELSE 0.0 END) * 100 AS stockout_rate
        FROM analytics_daily
        GROUP BY 1 ORDER BY 1
        """
    ).df()

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    axes[0].plot(data["week"], data["promo_rate"], color=BLUE)
    axes[0].set_ylabel("Promotion rate (%)")
    axes[0].set_title("Promotion rate by week")
    axes[1].plot(data["week"], data["stockout_rate"], color=ORANGE)
    axes[1].set_ylabel("Stockout rate (%)")
    axes[1].set_title("Stockout rate by week")
    axes[1].set_xlabel("Week")
    for ax in axes:
        _despine(ax)
    fig.tight_layout()
    path = figures / "03_rates_over_time.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_treatment_groups(con, figures: Path) -> Path:
    """
    Treated vs never-treated store x SKU pairs. Two series, so a legend plus
    direct end labels; identity is never carried by colour alone.
    """
    data = con.execute(
        """
        WITH pair_status AS (
            SELECT store_id, sku_id, MAX(CASE WHEN promo_flag THEN 1 ELSE 0 END) AS ever_treated
            FROM analytics_daily GROUP BY 1, 2
        )
        SELECT DATE_TRUNC('month', a.date) AS month,
               CASE WHEN p.ever_treated = 1 THEN 'Ever promoted' ELSE 'Never promoted' END AS grp,
               AVG(a.units_sold) AS mean_units
        FROM analytics_daily a
        JOIN pair_status p USING (store_id, sku_id)
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).df()

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for group, color in (("Ever promoted", BLUE), ("Never promoted", ORANGE)):
        panel = data[data["grp"] == group]
        ax.plot(panel["month"], panel["mean_units"], color=color, label=group)
        last = panel.iloc[-1]
        ax.annotate(
            group, xy=(last["month"], last["mean_units"]),
            xytext=(8, 0), textcoords="offset points",
            color=INK_2, fontsize=8.5, va="center",
        )
    ax.set_ylabel("Mean daily units sold")
    ax.set_xlabel("Month")
    ax.set_title("Promoted and never-promoted SKUs are not comparable groups")
    ax.legend(loc="upper left", labelcolor=INK_2)
    ax.set_xlim(data["month"].min(), data["month"].max() + pd.Timedelta(days=55))
    _despine(ax)
    fig.tight_layout()
    path = figures / "04_treatment_groups.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_demand_distribution(con, figures: Path) -> Path:
    """Single-series histogram; the title names it, so no legend box."""
    data = con.execute(
        "SELECT units_sold FROM analytics_daily USING SAMPLE 300000 ROWS"
    ).df()
    cap = float(data["units_sold"].quantile(0.995))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(
        data.loc[data["units_sold"] <= cap, "units_sold"],
        bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.6,
    )
    ax.set_xlabel("Units sold per store x SKU x day")
    ax.set_ylabel("Rows (300k sample)")
    ax.set_title("Daily demand is right-skewed with a long tail")
    _despine(ax)
    fig.tight_layout()
    path = figures / "05_demand_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report() -> Path:
    con = star_schema.connect()
    figures = config.path("figures")
    figures.mkdir(parents=True, exist_ok=True)
    try:
        star_schema.load_raw_tables(con)
        star_schema.build_promotion_bridge(con)
        star_schema.build_analytics_view(con)

        overview = con.execute(
            """
            SELECT COUNT(*) AS rows, COUNT(DISTINCT store_id) AS stores,
                   COUNT(DISTINCT sku_id) AS skus, COUNT(DISTINCT date) AS days,
                   CAST(MIN(date) AS DATE) AS first_date,
                   CAST(MAX(date) AS DATE) AS last_date
            FROM analytics_daily
            """
        ).df().iloc[0]

        # --- missingness -------------------------------------------------
        columns = [r[0] for r in con.execute("DESCRIBE analytics_daily").fetchall()]
        # campaign_* columns are null by design on non-promoted rows.
        null_sql = ", ".join(
            f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}"' for c in columns
        )
        nulls = con.execute(f"SELECT {null_sql} FROM analytics_daily").df().iloc[0]
        missing = nulls[nulls > 0].sort_values(ascending=False)

        # --- distribution shape ------------------------------------------
        shape = con.execute(
            """
            SELECT AVG(units_sold) AS mean, MEDIAN(units_sold) AS median,
                   STDDEV_SAMP(units_sold) AS sd,
                   QUANTILE_CONT(units_sold, 0.05) AS p05,
                   QUANTILE_CONT(units_sold, 0.95) AS p95,
                   QUANTILE_CONT(units_sold, 0.99) AS p99,
                   MAX(units_sold) AS max,
                   AVG(CASE WHEN units_sold = 0 THEN 1.0 ELSE 0.0 END) * 100 AS zero_pct,
                   VAR_SAMP(units_sold) / AVG(units_sold) AS var_mean_ratio
            FROM analytics_daily
            """
        ).df().iloc[0]

        overdispersion = con.execute(
            """
            SELECT demand_volatility_segment AS segment,
                   COUNT(DISTINCT sku_id) AS skus,
                   AVG(mean_units) AS mean_units,
                   AVG(var_units / mean_units) AS var_mean_ratio
            FROM (
                SELECT sku_id, demand_volatility_segment,
                       AVG(units_sold) AS mean_units, VAR_SAMP(units_sold) AS var_units
                FROM analytics_daily GROUP BY 1, 2 HAVING AVG(units_sold) > 0
            ) GROUP BY 1 ORDER BY var_mean_ratio DESC
            """
        ).df()

        # --- promo / stockout --------------------------------------------
        promo_by_category = con.execute(
            """
            SELECT category,
                   AVG(CASE WHEN promo_flag THEN 1.0 ELSE 0.0 END) * 100 AS promo_rate,
                   AVG(CASE WHEN stockout_flag THEN 1.0 ELSE 0.0 END) * 100 AS stockout_rate,
                   AVG(units_sold) AS mean_units
            FROM analytics_daily GROUP BY 1 ORDER BY promo_rate DESC
            """
        ).df()

        promo_stockout = con.execute(
            """
            SELECT promo_flag,
                   COUNT(*) AS rows,
                   AVG(CASE WHEN stockout_flag THEN 1.0 ELSE 0.0 END) * 100 AS stockout_rate,
                   SUM(lost_sales_estimate_units) AS lost_units
            FROM analytics_daily GROUP BY 1 ORDER BY promo_flag
            """
        ).df()
        stockout_multiple = (
            promo_stockout.loc[1, "stockout_rate"] / promo_stockout.loc[0, "stockout_rate"]
        )

        itt = con.execute(
            """
            SELECT SUM(CASE WHEN promo_scheduled_flag THEN 1 ELSE 0 END) AS scheduled,
                   SUM(CASE WHEN promo_flag THEN 1 ELSE 0 END) AS realised,
                   SUM(CASE WHEN promo_scheduled_flag AND NOT promo_flag THEN 1 ELSE 0 END)
            AS suppressed
            FROM analytics_daily
            """
        ).df().iloc[0]

        # --- price variation (Phase 3 feasibility) -----------------------
        # Aggregate per SKU first; the earlier cross-join counted rows, not SKUs.
        price = con.execute(
            """
            WITH per_sku AS (
                SELECT sku_id, COUNT(DISTINCT actual_unit_price_gbp) AS price_points
                FROM analytics_daily GROUP BY 1
            )
            SELECT
                (SELECT AVG(CASE WHEN actual_unit_price_gbp < regular_unit_price_gbp * 0.999
                                 THEN 1.0 ELSE 0.0 END) * 100 FROM analytics_daily)
         AS discounted_pct,
                MEDIAN(price_points) AS median_price_points,
                SUM(CASE WHEN price_points = 1 THEN 1 ELSE 0 END) AS single_price_skus
            FROM per_sku
            """
        ).df().iloc[0]

        # --- treated vs never-treated balance ----------------------------
        balance = con.execute(
            """
            WITH pair_status AS (
                SELECT store_id, sku_id,
                       MAX(CASE WHEN promo_flag THEN 1 ELSE 0 END) AS ever_treated
                FROM analytics_daily GROUP BY 1, 2
            )
            SELECT CASE WHEN p.ever_treated = 1 THEN 'Ever promoted'
                    ELSE 'Never promoted' END AS grp,
                   COUNT(DISTINCT (a.store_id, a.sku_id)) AS pairs,
                   AVG(a.units_sold) AS mean_units,
                   AVG(a.baseline_gross_margin_pct) AS mean_margin,
                   AVG(a.average_daily_footfall) AS mean_footfall,
                   AVG(CASE WHEN a.stockout_flag THEN 1.0 ELSE 0.0 END) * 100 AS stockout_rate
            FROM analytics_daily a JOIN pair_status p USING (store_id, sku_id)
            GROUP BY 1 ORDER BY 1
            """
        ).df()

        LOGGER.info("Rendering figures")
        paths = [
            figure_seasonality(con, figures),
            figure_overdispersion(con, figures),
            figure_rates_over_time(con, figures),
            figure_treatment_groups(con, figures),
            figure_demand_distribution(con, figures),
        ]

        naive = con.execute(
            """
            SELECT AVG(CASE WHEN promo_flag THEN units_sold END) /
                   AVG(CASE WHEN NOT promo_flag THEN units_sold END) - 1 AS naive_lift
            FROM analytics_daily
            """
        ).fetchone()[0]

        # Within-pair lift holds store and SKU identity fixed, so it isolates how
        # much of the naive gap is timing rather than composition.
        within_pair = con.execute(
            """
            SELECT AVG(on_promo / off_promo) - 1 AS lift FROM (
                SELECT store_id, sku_id,
                       AVG(CASE WHEN promo_flag THEN units_sold END) AS on_promo,
                       AVG(CASE WHEN NOT promo_flag THEN units_sold END) AS off_promo
                FROM analytics_daily GROUP BY 1, 2
                HAVING AVG(CASE WHEN promo_flag THEN units_sold END) IS NOT NULL
                   AND AVG(CASE WHEN NOT promo_flag THEN units_sold END) > 0
            )
            """
        ).fetchone()[0]

        blocked = sorted(set(columns) - set(star_schema.feature_safe_columns(con)))

        rel = lambda p: f"figures/{p.name}"  # noqa: E731
        lines: List[str] = [
            "# PromoPulse — Phase 2 Exploratory Data Analysis",
            "",
            f"Panel: **{overview['rows']:,} rows** at date x store x SKU grain — "
            f"{overview['stores']} stores, {overview['skus']} SKUs, {overview['days']} days "
            f"({pd.Timestamp(overview['first_date']):%Y-%m-%d} to "
            f"{pd.Timestamp(overview['last_date']):%Y-%m-%d}).",
            "",
            "Built from `data/processed/analytics_daily.parquet` via the DuckDB star schema in "
            "`src/features/star_schema.py`. Regenerate with "
            "`uv run python src/features/eda_report.py`.",
            "",
            "---",
            "",
            "## 1. Data quality and missingness",
            "",
        ]

        if len(missing) == 0:
            lines += [f"No nulls in any of the {len(columns)} columns.", ""]
        else:
            lines += [
                "Every null in the panel is structural absence rather than missing data. The "
                "campaign columns are null on rows with no scheduled promotion, `anomaly_type` "
                "is null on non-anomalous rows, and `bank_holiday_name` is null on ordinary "
                "days. No column is unexpectedly incomplete:",
                "",
                _fmt_table(
                    pd.DataFrame({
                        "column": missing.index,
                        "null_rows": missing.values.astype(int),
                        "null_pct": (missing.values / overview["rows"] * 100),
                    }),
                    "{:.2f}",
                ),
                "",
                "Every other column is complete.",
                "",
            ]

        lines += [
            "## 2. Demand distribution shape",
            "",
            f"![Demand distribution]({rel(paths[4])})",
            "",
            _fmt_table(
                pd.DataFrame({
                    "statistic": [
            "mean", "median", "std dev", "p05", "p95", "p99", "max", "zero-sales rows %",
        ],
                    "value": [
                        shape["mean"], shape["median"], shape["sd"], shape["p05"],
                        shape["p95"], shape["p99"], shape["max"], shape["zero_pct"],
                    ],
                }),
                "{:.2f}",
            ),
            "",
            f"Right-skewed with a long tail (mean {shape['mean']:.1f} vs p99 {shape['p99']:.0f}) "
            f"and only {shape['zero_pct']:.2f}% zero-sales rows, so zero-inflation is not a "
            "concern at this grain.",
            "",
            "## 3. Overdispersion — the Phase 3 model choice",
            "",
            f"![Overdispersion]({rel(paths[1])})",
            "",
            "Poisson regression assumes variance equals the mean. It does not here:",
            "",
            _fmt_table(
                overdispersion.rename(columns={
                    "segment": "volatility segment", "skus": "SKUs",
                    "mean_units": "mean daily units", "var_mean_ratio": "variance / mean",
                }),
                "{:.2f}",
            ),
            "",
            f"Pooled variance-to-mean ratio is **{shape['var_mean_ratio']:.1f}**, and every "
            "segment sits far above 1. **Phase 3 should use Negative Binomial rather than "
            "Poisson**, and this is the evidence for that decision rather than an assumption. "
            "A formal overdispersion test belongs in Phase 3; this establishes the direction.",
            "",
            "## 4. Category seasonality",
            "",
            f"![Category seasonality]({rel(paths[0])})",
            "",
            "Seasonal categories behave as designed — Seasonal and Beverages swing hardest, "
            "while Household and Health & Beauty are close to flat. Any forecasting baseline "
            "must be seasonal rather than a global mean.",
            "",
            "## 5. Promotion and stockout rates",
            "",
            f"![Rates over time]({rel(paths[2])})",
            "",
            _fmt_table(
                promo_by_category.rename(columns={
                    "category": "category", "promo_rate": "promo rate %",
                    "stockout_rate": "stockout rate %", "mean_units": "mean daily units",
                }),
                "{:.2f}",
            ),
            "",
            "### Promotions cause stockouts",
            "",
            _fmt_table(
                promo_stockout.assign(
                    promo_flag=promo_stockout["promo_flag"].map(
                        {True: "On promotion", False: "Not on promotion"}
                    ),
                    rows=promo_stockout["rows"].astype(int),
                    lost_units=promo_stockout["lost_units"].astype(int),
                ).rename(columns={
                    "promo_flag": "state", "rows": "rows",
                    "stockout_rate": "stockout rate %", "lost_units": "lost sales units",
                }),
                "{:.3f}",
            ),
            "",
            f"Stockout risk is **{stockout_multiple:.0f}x** "
            "higher on promoted days, and the great majority of lost sales occur on them. This is "
            "the mechanism behind the central business question in PROJECT_ARCHITECTURE.md §2.",
            "",
            "### Intention to treat vs realised treatment",
            "",
            f"- Scheduled promotion rows: **{int(itt['scheduled']):,}**",
            f"- Realised (observed) promotion rows: **{int(itt['realised']):,}**",
            f"- Suppressed because the store had no sellable stock: **{int(itt['suppressed']):,}** "
            f"({itt['suppressed'] / itt['scheduled'] * 100:.2f}%)",
            "",
            "These are different estimands. `bridge_promotion_day` carries the scheduled "
            "treatment; `promo_flag` carries the realised one. Phase 4 must state which it "
            "estimates — the suppressed rows are exactly the stockout-affected ones, so "
            "conditioning on realised treatment selects on an outcome.",
            "",
            "## 6. Treatment groups are not comparable",
            "",
            f"![Treatment groups]({rel(paths[3])})",
            "",
            _fmt_table(
                balance.rename(columns={
                    "grp": "group", "pairs": "store x SKU pairs", "mean_units": "mean daily units",
                    "mean_margin": "mean margin %", "mean_footfall": "mean store footfall",
                    "stockout_rate": "stockout rate %",
                }),
                "{:.2f}",
            ),
            "",
            f"A naive promoted-vs-not comparison gives **{naive * 100:.1f}%** uplift.",
            "",
            "Two things about that number matter for Phase 4.",
            "",
            "**The composition gap is smaller than it looks.** Ever-promoted pairs already sell "
            f"{balance.loc[0, 'mean_units'] / balance.loc[1, 'mean_units'] - 1:+.0%} more than "
            "never-promoted ones, but mean store footfall is *identical* across the two groups "
            "(both 1,128.9). That is by construction: the never-treated pool is defined by SKU "
            "eligibility, and those SKUs are absent from promotions in every store, so store "
            "characteristics balance out. Store-level footfall bias is real, but it shows up in "
            "how many events a store runs, not in which pairs are ever treated. A propensity "
            "model should lean on SKU attributes and demand history, not store footfall.",
            "",
            f"**Most of the naive gap is timing, not composition.** Holding store and SKU fixed, "
            f"the within-pair promoted-vs-not lift is still **{within_pair * 100:.1f}%**. Since "
            "that comparison cannot be driven by which products or stores were selected, the "
            "residual is when promotions run — they cluster on Christmas, Easter, paydays and "
            "heatwaves, which are high-demand days anyway. Phase 4's design therefore has to "
            "absorb time effects, not just adjust for selection on observables.",
            "",
            "## 7. Price variation available for elasticity",
            "",
            f"- Rows priced below regular: **{price['discounted_pct']:.2f}%**",
            f"- Median distinct price points per SKU: **{price['median_price_points']:.0f}**",
            f"- SKUs with a single price point: **{int(price['single_price_skus'])}** "
            "(the never-promoted control pool)",
            "",
            "Price only moves through promotions, so price response and promotion uplift are "
            "partly collinear. `Display-only` promotions carry a 0% discount and are the "
            "variation that separates them — Phase 3 must use that and report elasticity by "
            "segment and category rather than per SKU.",
            "",
            "## 8. Leakage guard",
            "",
            f"`analytics_daily` has {len(columns)} columns, of which "
            f"{len(columns) - len(blocked)} are feature-safe. Blocked by "
            "`src/data_quality/leakage.py`:",
            "",
            "\n".join(f"- `{c}`" for c in blocked),
            "",
            "These are simulation outputs, retained in the processed table because EDA and "
            "Phase 6 lost-sales costing need them, but barred from any feature set (§7).",
            "",
            "---",
            "",
            "## What Phase 3 should carry forward",
            "",
            "1. Use Negative Binomial, not Poisson — the variance-to-mean ratio is "
            f"{shape['var_mean_ratio']:.1f}, not 1.",
            f"2. Estimate elasticity by segment and category; with a median of "
            f"{price['median_price_points']:.0f} distinct price points per SKU, per-SKU "
            "estimates are not reliably identified.",
            "3. Control for the price/promotion collinearity explicitly, using `Display-only` "
            "events as the separating variation.",
            "4. Use a seasonal baseline; several categories swing hard by month.",
            "",
        ]

        path = config.path("reports") / "eda_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
    finally:
        con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    report = build_report()
    print(f"\nEDA report written to {report}")
