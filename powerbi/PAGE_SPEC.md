# Power BI page specification

The semantic model in `Northstar.SemanticModel/` is generated and carries all 41
measures. This file specifies the five report pages precisely enough to assemble
them in Power BI Desktop.

**Why this is a spec rather than a built report.** The visual layer of a PBIP is
a large, position-sensitive JSON document. Authoring one without being able to
open Desktop and look at it is where blind authoring fails hardest — the likely
outcome is a file that opens to broken or empty visuals, which is worse in a
portfolio than an honest spec. The model and measures are the hard part and they
are done; the pages are half an hour of drag-and-drop.

---

## Before you start

1. Open `Northstar.pbip` in Power BI Desktop.
2. **Fix the data path.** `definition/expressions.tmdl` holds a `DataFolder`
   parameter written with an absolute path at generation time. If you cloned this
   repository somewhere else, edit it to your own
   `.../powerbi/powerbi_data/` (Transform data → Manage parameters), or re-run
   `uv run python src/powerbi/tmdl.py` to regenerate it.
3. Create the what-if parameter Page 5 needs:
   Modeling → New parameter → Numeric range, named **Uplift Scenario**,
   Minimum `0.5`, Maximum `1.3`, Increment `0.05`, Default `0.88`.
   The default is not arbitrary — Phase 4 found the DiD estimate overshot the
   simulated truth, so the central case assumes the estimate is optimistic.
4. Check any measure against `powerbi_data/dax_parity.csv`. Drop it on a card
   with no filters; it should match the `expected_value` column.

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
