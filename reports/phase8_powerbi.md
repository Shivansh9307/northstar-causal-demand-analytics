# Northstar — Phase 8: Power BI Decision Layer

Rebuild with:

```bash
uv run python src/powerbi/export.py    # curated data layer (~3 min)
uv run python src/powerbi/parity.py    # expected value of every measure
uv run python src/powerbi/tmdl.py      # generate the semantic model
```

---

## What is verified

All five pages are built — **66 visuals** authored as PBIR definition files —
opened, refreshed and checked page by page in Power BI Desktop 2.155.756.0.

| Artefact | Status |
|---|---|
| Exported data layer (14 tables) | **Verified** — tests cover grain, keys, relationship integrity, size and content |
| Measure expectations (`dax_parity.csv`) | **Verified** — computed from the exported data and reconciled against the phase reports |
| `measures.dax` (45 measures) | **Executed** — every figure below read off the rendered report |
| TMDL semantic model | **Loaded and refreshed** in Desktop |
| Report pages | **Built** — `powerbi/screenshots/` |

Every headline figure matches its expected value, including both what-if cases:

| Measure | Shown | Expected |
|---|---|---|
| Total Revenue / Gross Profit | £182.54M / £69.06M | 182,537,804.93 / 69,059,188.19 |
| Gross Margin % / Units Sold | 37.83 / 39.24M | 37.8328 / 39,235,280 |
| Promotion Rate % / Promoted Revenue Share % | 8.49 / 15.96 | 8.4889 / 15.9633 |
| Stockout Rate % | 0.28 | 0.2823 |
| Naive / Causal / True Promo Lift % | 126.74 / 96.44 / 81.03 | 126.7368 / 96.4411 / 81.0255 |
| Naive / Causal Bias pp | 45.71 / 15.42 | 45.7113 / 15.4156 |
| Bias Removed % | 63.72 | 63.7211 |
| CI Coverage % / Max Recovery Error | 83.33 / 0.027 | 83.3333 / 0.0266 |
| Mean Lift at 20% Discount | 97.42 | 97.4221 |
| Median / Lowest Service Level % | 99.45 / 52.31 | 99.45 / 52.31 |
| Cannibalisation 1 / 4+ neighbours | −6.12 / −16.37 | −6.1244 / −16.3727 |
| Plan Incremental Profit | £337.03 | 337.0253 |
| P10 / P50 / P90 | (£761.98) / (£267.74) / £264.76 | −761.9826 / −267.7425 / 264.7569 |
| Probability of Loss % / Candidates Profitable % | 74.20 / 0.15 | 74.2 / 0.15 |
| Scenario @ 1.00 / @ 0.88 | £337.03 / (£259.55) | 337.0255 / −259.5533 |

**The scenario identity holds in the engine, not just in Python.** At multiplier
1.00 `Scenario Incremental Profit` reads £337.03 with `Scenario vs Plan` at 0.00
— equal to `Plan Incremental Profit`. That is the same property the Phase 6
regression test pins: the promotional give-away is not deducted twice.

---

## What the build exposed

The model was generated without a DAX engine to check it against, and four
defects survived that gap. Each is now fixed in `src/powerbi/tmdl.py` and pinned
by a test, because regenerating would otherwise reintroduce all of them.

**1. Boolean columns loaded as empty tables.** pandas writes booleans as the text
`True`/`False`; the generated M cast them to `Int64.Type`. Power Query cannot
convert `"True"` to a whole number, so *every row errored* and the table loaded
empty — behind nothing louder than a "some tables have incomplete or no data"
banner. Eleven columns across `dim_calendar`, `dim_product` and `reorder_policy`.
`dim_calendar` failing was the expensive one: it is the date dimension, so the
relationship to `fact_daily_category` went dead and took `Revenue LY`,
`Revenue YoY %` and every date slicer with it.

**2. A DAX type error downstream of the same cause.** `Service Level Insight`
compares `reorder_policy[is_perishable] = TRUE()`. Against an `int64` column DAX
refuses — *"cannot compare values of type Integer with values of type
True/False"*. The measure was right; the column type was wrong.

**3. PBIR has generations, and the scaffold was written against the wrong one.**
Desktop 2.155 reads 2.x. Seeing definition version `1.0.0` it silently treated
the report as PBIR-Legacy, found no legacy layout, opened **one blank page**, and
refused to save with *"unable to upgrade report to PBIR format — the report has
no pages"*. No error named the version. Ground truth came from letting Desktop
upgrade a throwaway copy and reading what it wrote: `report/2.1.0`,
`page/2.0.0`, `visualContainer/2.1.0`, definition version `2.0.0` — and no
`$schema` at all in `definition.pbir` or the `.pbip`.

This one carries a lesson beyond Power BI. The scaffold *passed* JSON Schema
validation the whole time, because it was validated against the 1.x schemas its
author chose to vendor. **Validating against a schema proves conformance to that
schema, not fitness for the consumer.**

**4. The what-if slider could not reach its own parity value.** The column was
`double` over `GENERATESERIES(0.5, 1.3, 0.05)`. Stepping 0.05 through binary
floating point drifts, so the series ended at 1.25 rather than 1.30, and no
stored value exactly equalled what the slicer computed. Worse, `dax_parity.csv`
expects a **0.88** multiplier, which is not a member of a 0.05 series at all —
the spec and the parity file contradicted each other. Now `decimal`, built from
exact integers at 0.01 steps, so 0.88 is a real member.

Also corrected: `dim_calendar` now carries `dataCategory: Time` so `DATEADD` in
`Revenue LY` is dependable, and numeric format strings are no longer emitted on
text and boolean columns.

**Four measures and two calculated columns were added during the build.** The
model sets `discourageImplicitMeasures`, so a raw numeric column cannot go in a
visual's Values well; `Effect %`, `Service Level`, `Spillover Effect %` and
`Draw Count` stand in front of stored columns for four charts, and
`profit_bucket` and `month_start` supply axes the exported CSVs do not carry.
They are in `measures.dax` and the generator so regeneration cannot drop them.

---

## 1. The data layer

The raw fact table is 2.19M rows and 468MB of CSV. Pointing a semantic model at
that would make the repository unclonable and the report slow, for no analytical
gain — none of the five pages needs SKU-level daily detail.

So the fact grain is **date × store × category** (146,200 rows), and the
*results* of Phases 3–6 ship as their own small tables. Those results tables are
the actual decision layer: someone opening this should see the naive and causal
promotion estimates side by side, not be handed a pile of transactions.

| Table | Rows | Purpose |
|---|---|---|
| `dim_store`, `dim_product`, `dim_calendar`, `dim_category` | 20 / 150 / 731 / 10 | Dimensions |
| `fact_daily_category` | 146,200 | Aggregated trading fact |
| `causal_estimates` | 8 | Naive vs DiD vs truth (Phase 4) |
| `dose_response` | 18 | Validated promotional curve by segment (Phase 3) |
| `spillover` | 4 | Cannibalisation onto untreated neighbours (Phase 3) |
| `service_levels`, `reorder_policy` | 10 / 3,000 | Newsvendor policy (Phase 6) |
| `promo_plan`, `promo_plan_uncertainty`, `promo_plan_draws`, `promo_economics` | 10 / 1 / 4,000 / 6 | Optimiser output and its Monte Carlo range (Phase 6) |
| `dax_parity` | 25 | Expected value of every measure |

Total under 15MB, so the model loads from a fresh clone without regenerating
626MB of source data first. A test asserts that ceiling, so if the fact grain
ever creeps back towards SKU level it fails rather than silently bloating the
repository.

`potential_demand_units`, `lost_sales_estimate_units` and the anomaly labels are
deliberately excluded. §7 bars them from any model, and a BI surface is exactly
where a latent-demand column gets mistaken for something a planner can act on.

## 2. Parity, and what it proves

§6 Phase 8 asks that "DAX measures should reproduce the Python-calculated figures
as a parity check". A parity check nobody can run is a promise, so
`src/powerbi/parity.py` computes what every measure *should* return, straight
from the exported tables, and writes the answers to `dax_parity.csv`. Each
measure in `measures.dax` carries its expected value as a comment.

That gives a reviewer something concrete: drop the measure on a card with no
filters, and it should match.

**What it proves:** the numbers are pinned down, reproducible, and consistent
with what the written reports claim. A test spot-checks them against the phase
reports, so the dashboard and the prose cannot drift apart silently.

**What it does not prove:** that the DAX expressions are correct. Only Power BI
can establish that. Parity constrains the target, not the implementation.

The one exception is the scenario measure, whose arithmetic *is* verified. It
recomputes plan profit at a different uplift, and at a multiplier of 1 it must
reproduce the optimiser's own figure:

```
Scenario at multiplier 1.00 : £337.03
Plan Incremental Profit     : £337.03
```

That identity is asserted in tests because it is where Phase 6's double-counting
bug would recur — the promotional give-away is already inside `promo_margin`, and
subtracting it again would make the measure disagree.

## 3. The semantic model

`src/powerbi/tmdl.py` generates the model rather than hand-writing it. Every
column name, data type and M transform is derived from the exported CSVs at build
time.

That was a deliberate choice about risk. Hand-authoring twenty TMDL files means
twenty chances to mistype a column or guess a type wrong, and none of those
mistakes surface until Desktop refuses to load the model — which I cannot check.
Generating removes the entire class of error: if the export changes, the model
changes with it, and a test asserts every exported column appears in the
generated TMDL.

Five relationships, all tested for orphan keys and for uniqueness on the
one-side:

```
fact_daily_category[date]       -> dim_calendar[date]
fact_daily_category[store_id]   -> dim_store[store_id]
fact_daily_category[category]   -> dim_category[category]
reorder_policy[store_id]        -> dim_store[store_id]
reorder_policy[sku_id]          -> dim_product[sku_id]
```

`dim_category` exists because category is the fact's grain but is not unique in
`dim_product`, so relating them directly would have produced a many-to-many.

### Known rough edge

The `DataFolder` parameter is written with an absolute path at generation time.
Cloning the repository elsewhere means editing it or re-running `tmdl.py`.
`PAGE_SPEC.md` says so as step 2. A relative path would be tidier but Power Query
resolves relative paths against the Desktop process's working directory, not the
project file, which is less predictable than an obvious parameter.

## 4. The pages

`powerbi/PAGE_SPEC.md` specifies all five pages required by §6 Phase 8 — visual
type, fields, slicers, layout and the callout each page should carry.

The specification includes two things a bare field list would not:

- **A build-time sanity check.** Setting the What-If slider to 1.00 must make
  `Scenario Incremental Profit` equal `Plan Incremental Profit`. If it does not,
  the expression is double-counting the give-away.
- **A marking convention.** Every figure on Pages 2 and 3 labelled "true" comes
  from the simulation and does not exist in production. Phase 7 established that
  on real data there is no truth column to check against, so those visuals need
  marking or a reader will take them for a live capability.

## 5. What a human did

Each of these was carried out on a Windows machine running Desktop 2.155.756.0,
and the results are what the tables above record.

1. Opened `Northstar.pbip` and loaded the model. It did not work first time; the
   four defects in "What the build exposed" are what stood between the generated
   project and a model that loads, and each needed its own round trip because
   Desktop reports only the first failure.
2. The `Uplift Scenario` what-if parameter is **generated in TMDL**, so nothing
   was created by hand. An earlier draft of this report claimed it could only be
   made in Desktop; that was wrong. Desktop's Modeling → New parameter builds a
   calculated table over `GENERATESERIES` and nothing else, which TMDL expresses
   directly.
3. Built the five pages from `PAGE_SPEC.md` as PBIR definition files — 66 visuals
   under `Northstar.Report/definition/pages/`.
4. Ticked every measure against `dax_parity.csv`. The parity table above is that
   pass, including both what-if multipliers.
5. Screenshotted all five pages to `powerbi/screenshots/`, where the README
   embeds them.

---

## Honest summary

The analytical substance of this phase — a curated decision layer, 45 measures,
and a machine-checked set of expected values tying the dashboard to the written
analysis — is done and tested. So is the assembly: five pages, 66 visuals, opened
and refreshed, every headline figure read off the rendered report and reconciled.

`dax_parity.csv` remains the proof that the numbers are real, and it is worth
being precise about what it proves. It is computed in pandas, so it pins the
values; it does not execute DAX, so it cannot show that these expressions return
them. Only Desktop can, which is why the figures in the parity table above were
read off the rendered report rather than trusted from the CSV. The gap between
those two things is the whole reason this phase needed a human at all.

---

## Superseded: the state this was written against (pre-assembly, Phase 8 build)

Kept as the build trail. This was the summary while the semantic model was
generated but no page had been authored, and it is no longer true.

> The analytical substance of this phase — a curated decision layer, 41 measures,
> and a machine-checked set of expected values tying the dashboard to the written
> analysis — is done and tested. The assembly is not, and cannot be from here.
>
> A reviewer should read `measures.dax` and `PAGE_SPEC.md` as the deliverable and
> `dax_parity.csv` as the proof that the numbers behind them are real, while
> treating the `.pbip` as an unopened draft until someone opens it.
>
> **What still needs a human**
>
> 1. Open `Northstar.pbip` in Desktop and confirm the model loads.
> 2. Create the `Uplift Scenario` what-if parameter — it cannot be declared in
>    TMDL, only in Desktop.
> 3. Build the five pages from `PAGE_SPEC.md`.
> 4. Tick each measure off against `dax_parity.csv`.
> 5. Screenshot to `powerbi/screenshots/` for the README.

Two things in that block were wrong rather than merely out of date, and both are
worth keeping visible. The measure count was 45, not 41. And item 2 was a factual
error about TMDL's capabilities, not a scheduling note — the parameter is
generated, and believing otherwise would have meant a hand-built object that no
test could reach.
