# PromoPulse — Phase 8: Power BI Decision Layer

Rebuild with:

```bash
uv run python src/powerbi/export.py    # curated data layer (~3 min)
uv run python src/powerbi/parity.py    # expected value of every measure
uv run python src/powerbi/tmdl.py      # generate the semantic model
```

---

## What is verified, and what is not

This is the one phase where a substantial deliverable could not be checked by the
person who built it. Power BI Desktop is not available in this environment, so:

| Artefact | Status |
|---|---|
| Exported data layer (14 tables) | **Verified** — 31 tests cover grain, keys, relationship integrity, size and content |
| Measure expectations (`dax_parity.csv`) | **Verified** — computed from the exported data and reconciled against the phase reports |
| Scenario measure arithmetic | **Verified** — the identity at multiplier 1 is asserted in tests |
| `measures.dax` (41 measures) | **Authored, not executed** — parses and every measure has a home table, but no DAX engine has evaluated it |
| TMDL semantic model | **Generated, not opened** — schema is derived from the data so it cannot drift, but Desktop has never loaded it |
| Report pages | **Specified, not built** — see `powerbi/PAGE_SPEC.md` |

Nothing here should be read as "the dashboard works". The honest claim is that
the model and measures are authored, the numbers behind them are pinned down and
tested, and the pages are specified precisely enough to assemble.

**Why the pages are a spec rather than a file.** A PBIP's visual layer is a
large, position-sensitive JSON document. Hand-authoring one without being able to
open Desktop and look at it is where blind authoring fails hardest, and the
likely outcome is a file that opens to broken or empty visuals — worse in a
portfolio than an honest specification. The semantic model is the hard part and
it is done.

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

## 5. What still needs a human

1. Open `Northstar.pbip` in Desktop and confirm the model loads. If it does not,
   the fix is likely in `expressions.tmdl` (the data path) or a type inference in
   `tmdl.py::infer_types`.
2. Create the `Uplift Scenario` what-if parameter — it cannot be declared in
   TMDL, only in Desktop.
3. Build the five pages from `PAGE_SPEC.md`.
4. Tick each measure off against `dax_parity.csv`.
5. Screenshot to `powerbi/screenshots/` for the README.

---

## Honest summary

The analytical substance of this phase — a curated decision layer, 41 measures,
and a machine-checked set of expected values tying the dashboard to the written
analysis — is done and tested. The assembly is not, and cannot be from here.

A reviewer should read `measures.dax` and `PAGE_SPEC.md` as the deliverable and
`dax_parity.csv` as the proof that the numbers behind them are real, while
treating the `.pbip` as an unopened draft until someone opens it.
