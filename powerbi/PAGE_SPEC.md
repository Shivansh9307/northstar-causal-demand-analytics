# Power BI page specification

> **These pages are built.** All five exist as PBIR definition files under
> `Northstar.Report/definition/pages/` — 66 visuals, opened and verified in
> Desktop, screenshots in `screenshots/`. This file is the specification they
> were built from, kept as the record of intent. Deviations taken during the
> build, and the defects it exposed, are in `reports/phase8_powerbi.md`.

The semantic model in `Northstar.SemanticModel/` is generated and carries all 41
measures, and `Northstar.Report/` is scaffolded with these five pages already
named and empty. This file specifies what goes on them, precisely enough to
assemble in Power BI Desktop.

**Why this is a spec rather than a built report.** A PBIR visual carries both
position and query bindings. Authoring thirty of them without being able to open
Desktop and look at them is where blind authoring fails hardest — the likely
outcome is a file that opens to broken or empty visuals, which is worse in a
portfolio than an honest spec. The model and measures are the hard part and they
are done; the pages are half an hour of drag-and-drop.

---

## Before you start

1. **Do this before opening anything.** In **File → Options and settings →
   Options → Preview features**, tick *Store semantic model using TMDL format*
   and *Store reports using enhanced metadata format (PBIR)*. This project is
   stored in both formats.

   With PBIR off, Desktop **silently ignores** `Northstar.Report/definition/`.
   There is no error — the model loads fine and you get one blank page instead
   of five, which looks like the pages were never generated. Worse, saving in
   that state rewrites the report in the legacy format and the five scaffolded
   pages are gone.

2. Open `Northstar.pbip`. It opens on **Executive Summary** with all five pages
   present and empty.

3. **Repoint the data folder.** `DataFolder` is generated with an absolute path,
   so on any machine but the one that generated it, set it under
   **Home → Transform data → Edit parameters** to your own
   `.../powerbi_data/` — keep the trailing separator — then **Apply changes**
   and **Refresh**.

   To skip that step, generate for the target machine directly:

   ```bash
   uv run python src/powerbi/tmdl.py \
     --data-folder 'C:/Users/you/Desktop/power-bi-northstar/powerbi_data/'
   ```

   Regenerate without the flag afterwards, so the repo isn't left holding
   someone else's path.

4. On first open the model is unrefreshed, so expect banners about calculated
   objects needing a refresh and tables having no data. They clear once the
   parameter resolves and you refresh — they are not errors.

5. Check any measure against `powerbi_data/dax_parity.csv`. Drop it on a card
   with no filters; it should match the `expected_value` column.

The **Uplift Scenario** what-if parameter Page 5 needs is generated as part of the
model — you no longer create it by hand. It is a calculated table over
`GENERATESERIES(0.5, 1.3, 0.05)`. Set the slicer's default to `0.88` when you
build the page: that value is not arbitrary, as Phase 4 found the DiD estimate
overshot the simulated truth, so the central case assumes the estimate is
optimistic.

---

## Page 1 — Executive Summary

*Question it answers: how did the business trade, and how much of that was bought
with promotional margin?*

| Visual | Type | Fields |
|---|---|---|
| KPI row | 4 × Card | `Total Revenue`, `Total Gross Profit`, `Gross Margin %`, `Units Sold` |
| Trading trend | Line | Axis `dim_calendar[date]` (month), Values `Total Revenue`, `Total Gross Profit` |
| Category mix | Bar | Axis `dim_category[category]`, Values `Total Revenue`, sorted descending |
| Promotional intensity | Card ×2 | `Promotion Rate %`, `Promoted Revenue Share %` |
| Availability | Card | `Stockout Rate %` |
| Slicers | — | `dim_calendar[year]`, `dim_store[region]`, `dim_category[category]` |

Callout text: *"16% of revenue is earned on promotion. Page 2 asks how much of
that was incremental."*

---

## Page 2 — Promotion ROI (naive vs causal, side by side)

*The centrepiece. Two numbers that answer the same question and disagree by 46
percentage points.*

| Visual | Type | Fields |
|---|---|---|
| The comparison | 3 × Card, side by side | `Naive Promo Lift %`, `Causal Promo Lift %`, `True Promo Lift %` |
| Bias | 2 × Card | `Naive Bias pp`, `Causal Bias pp` |
| Correction | Card | `Bias Removed %` |
| Health label | Card | `Estimate Health` |
| All specifications | Bar | Axis `causal_estimates[method]`, Value `causal_estimates[effect_pct]`, with a constant line at `True Promo Lift %` |
| Control-set sensitivity | Table | `causal_estimates[method]`, `[family]`, `[effect_pct]`, `[ci_low]`, `[ci_high]`, `[error_pp]` |

Layout: put the naive card on the left in a warning colour and the causal card on
the right. The gap between them is the page.

Callout: *"Naive says +127%. The causal estimate says +96%. The simulated truth
is +81%. The correction removes 64% of the bias — and the residual is why Phase 4
calls its estimate an upper bound."*

---

## Page 3 — Elasticity Explorer

*Where the method demonstrably works: the dose-response recovers the simulated
answer.*

| Visual | Type | Fields |
|---|---|---|
| Dose-response | Line + markers | Axis `dose_response[discount_pct]`, Values `Estimated Lift %` and `True Lift %`, Legend by `dose_response[segment]` |
| Recovery | Card ×2 | `CI Coverage %`, `Max Recovery Error` |
| At 20% | Card | `Mean Lift at 20% Discount` |
| Gap by segment | Bar | Axis `dose_response[segment]`, Value `Lift Recovery Gap pp` |
| Detail | Table | segment, discount_pct, estimate, ci_low, ci_high, true_effect, ci_covers_truth |
| Slicer | — | `dose_response[segment]` |

Callout: *"Estimated and true curves lie on top of each other. This is the half of
the method that validates cleanly — the structural price elasticity, by contrast,
is not identified in this data at all."*

---

## Page 4 — Stockout Risk & Replenishment

*What to do about availability, and the counter-intuitive answer on fresh.*

| Visual | Type | Fields |
|---|---|---|
| Service levels | Bar | Axis `service_levels[category]`, Value `service_levels[median_service_level]`, constant line at 0.95 |
| Insight label | Card | `Service Level Insight` |
| Extremes | Card ×2 | `Median Service Level %`, `Lowest Service Level %` |
| Cannibalisation | Bar | Axis `spillover[others_on_promo]`, Value `spillover[effect_pct]` |
| Cannibalisation callouts | Card ×2 | `Cannibalisation 1 Neighbour %`, `Cannibalisation 4+ Neighbours %` |
| Policy detail | Table | `reorder_policy[store_id]`, `[sku_id]`, `[category]`, `[service_level]`, `[safety_stock]`, `[reorder_point]` |
| Slicers | — | `dim_store[store_format]`, `dim_category[category]` |

Callout: *"Fresh Produce should run 52% availability, not 95%. Holding an extra
unit of salad forfeits its whole cost; an extra tin costs pennies of capital."*

---

## Page 5 — What-If Promotion Simulator

*Driven by the Phase 6 optimiser output, with the uncertainty attached.*

| Visual | Type | Fields |
|---|---|---|
| Slider | Slicer | `'Uplift Scenario'[Uplift Scenario]` |
| Scenario result | Card | `Scenario Incremental Profit` |
| Verdict | Card | `Scenario Verdict` |
| Against plan | Card | `Scenario vs Plan` |
| Plan summary | Card ×3 | `Promotions Selected`, `Plan Spend`, `Plan Incremental Profit` |
| Uncertainty | Card ×3 | `Plan Profit P10`, `Plan Profit P50`, `Plan Profit P90` |
| Risk | Card | `Probability of Loss %` |
| Profit distribution | Histogram (or Column on binned `promo_plan_draws[profit_gbp]`) | with reference lines at P10/P50/P90 |
| Why so few | Card + Table | `Candidates Profitable %`; table of `promo_economics` by discount depth |
| The plan | Table | `promo_plan[store_id]`, `[sku_id]`, `[category]`, `[discount_pct]`, `[incremental_profit]` |

Callout: *"The optimiser reports £337. The median simulated outcome is −£268, and
74% of draws lose money. Budget the range, not the number."*

Sanity check while building: set the slider to `1.00` and
`Scenario Incremental Profit` must equal `Plan Incremental Profit` (£337.03). If
it does not, the scenario expression is double-counting the promotional
give-away — the exact bug Phase 6 shipped and fixed.

---

## Design notes

- **Do not use red/green alone** to distinguish naive from causal on Page 2.
  Label both cards; colour is secondary.
- **Every figure on Pages 2 and 3 that mentions "true" comes from the simulation
  and is unavailable in real life.** Mark those visuals so nobody reads them as a
  production capability — Phase 7 showed that on real data there is no truth
  column to check against.
- Keep the £ format strings the model already carries; do not override with
  percentages on currency measures.
