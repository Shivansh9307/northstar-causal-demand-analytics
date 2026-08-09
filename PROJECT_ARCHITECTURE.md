# Northstar — Causal Demand, Promotion & Inventory Intelligence

**Repo name:** `northstar-causal-demand-analytics`
**Owner:** Shivansh Chauhan
**Scope:** Statistical analysis, regression, causal inference, machine learning and prescriptive optimization for a UK multi-store retailer.

This document is the **architecture plan**. Read it in full before writing any code, confirm the build order, and work phase by phase rather than attempting the whole repo in one pass.

---

## 1. Purpose

Northstar Retail Group is a fictional UK grocery chain that promotes constantly and cannot say what those promotions earn. The project answers that question end to end: data generation, regression, causal estimation, forecasting, prescriptive optimization, and a BI layer that carries the answer to whoever has to act on it.

Four design decisions shape everything downstream.

**The dataset is synthetic and records its own ground truth.** On real retail data a causal estimate can only be judged on whether it looks plausible, because the true effect is never observed. Here the generator writes down the effect it applied, so every estimate has a computable recovery error against a known answer. That is the reason for building the data rather than downloading it.

**The output is prescriptive, not predictive.** The final artifact is a promotion allocation and a set of reorder points with a profit range attached, not a forecast with an accuracy score. A forecast does not say what to do with itself.

**The method is re-run against real data.** A generator and an estimator written by the same person can quietly share an assumption. Phase 7 puts the pipeline in front of Rossmann Store Sales, which was not built to satisfy anything.

**The result lands in Power BI.** A recommendation that exists only in a notebook does not reach the person who makes the decision it describes.

---

## 2. The Central Business Question

> **Northstar Retail Group runs promotions across 60 stores and 500 SKUs, but nobody can confidently say which promotions actually drove incremental profit versus which ones would have sold anyway — and stockouts during promotions are quietly destroying the upside. Which products and stores should receive promotional investment next quarter, and how should replenishment be adjusted to avoid losing the sales the promotion was meant to capture?**

Every phase of this project answers a piece of that question. If a piece of analysis doesn't move you closer to answering it, it doesn't belong in this repo.

---

## 3. Data Strategy

### 3.1 Primary dataset — synthetic, with ground truth (build first)

Use the `generate_retail_dataset.py` specification already drafted (Northstar Retail Group, 60 stores / 500 SKUs / 3 years in `FULL_MODE=True`, reduced 20/150/2yr in dev mode). Key non-negotiables carried over from that spec:

- Promotion assignment is **deliberately biased** toward weakening sales, high margin, high footfall and seasonal clustering. It is never randomly assigned. This bias is the whole reason the causal inference phase has anything to correct.
- Staggered store rollouts are included specifically to enable difference-in-differences.
- Stockouts censor observed sales. `potential_demand_units` holds the uncensored latent demand and must **never** be used as a model feature.
- `ground_truth_simulation_parameters.csv` holds the true elasticity, true promo uplift, true cannibalisation factor, etc. This file is a **validation-only** artifact. Any pipeline step that touches it for training must fail a leakage check (see §7).
- Seed fixed at 42 throughout for reproducibility.
- Automated data-quality checks must all pass before EDA begins: no negative inventory, no duplicate keys, discount range 0–30%, unit cost below price, the inventory reconciliation identity holding, and so on. Print their output and save it to `reports/data_quality_report.md`.

### 3.2 Secondary dataset — real data for external validation (build later, Phase 7)

**Rossmann Store Sales** (Kaggle) — real daily store-level sales, promotions, holidays, competitor distance. Used only to re-run the **elasticity regression and demand forecasting pipeline** (not the full causal/optimization stack, since Rossmann lacks the staggered-rollout structure) against real-world data.

The point of the exercise is that synthetic data can flatter a method. When the same person writes the generator and the estimator, an assumption shared by both is invisible: the pipeline recovers the answer because both halves agree on how the world works, not because the method is sound. Running it against data nobody shaped for the purpose is what separates those two cases. It also establishes which findings depend on the simulation and which survive contact with a real dataset.

---

## 4. Method Stack

| Capability | Where it shows up |
|---|---|
| Statistical analysis | EDA, hypothesis testing (promo effect significance, seasonal ANOVA), distribution diagnostics, VIF/multicollinearity checks |
| Regression | OLS log-log elasticity regression, Poisson/Negative Binomial regression for count demand, logistic regression for stockout risk, DiD as a regression specification |
| Causal inference | Difference-in-differences (staggered rollout), propensity score matching / IPW, naive-vs-corrected estimate comparison against ground truth |
| Machine learning | Gradient boosted demand forecast (LightGBM/XGBoost) with time-based CV, stockout classifier, SHAP interpretability |
| Modelling / optimization | Reorder point & safety stock optimization, promotion budget allocation (linear/integer programming), Monte Carlo profit simulation under uncertainty |
| Communication / BI | Power BI semantic model (PBIP/TMDL), DAX measures mirroring Python results, exec-facing what-if simulator |

---

## 5. Repository Structure

```
northstar-causal-demand-analytics/
├── PROJECT_ARCHITECTURE.md        <- this file
├── README.md                      <- business case narrative (written last, Phase 9)
├── config/
│   └── config.yaml                <- FULL_MODE toggle, seed, paths, model params
├── data/
│   ├── raw/                       <- generated synthetic CSVs (large files gitignored, sample kept)
│   ├── external/                  <- Rossmann Store Sales (Phase 7)
│   ├── processed/                 <- cleaned/joined analytical tables (parquet)
│   └── ground_truth/              <- ground_truth_simulation_parameters.csv — NEVER a model feature
├── src/
│   ├── generation/                <- generate_retail_dataset.py + validators
│   ├── data_quality/              <- automated QA checks, leakage checker
│   ├── features/                  <- lags, rolling windows, calendar features
│   ├── stats/                     <- OLS / Poisson / NB regression, elasticity estimation
│   ├── causal/                    <- DiD, propensity score matching, naive-vs-corrected comparison
│   ├── ml/                        <- demand forecast models, stockout classifier, SHAP
│   ├── optimization/               <- reorder point optimizer, promo budget LP/ILP, Monte Carlo sim
│   ├── validation/                <- ground-truth recovery tests, Rossmann external validity
│   └── utils/
├── notebooks/                     <- exploratory only, numbered (01_eda.ipynb etc), no orphan analysis
├── tests/                         <- pytest unit tests for src/
├── reports/
│   ├── figures/
│   ├── data_quality_report.md
│   └── case_study.md              <- narrative write-up with £ business impact
├── powerbi/
│   ├── Northstar.pbip
│   ├── measures.dax
│   └── screenshots/
└── requirements.txt
```

---

## 6. Build Phases

**Phase 0 — Foundations**
Lock config.yaml (seed=42, `FULL_MODE=False` for all development), confirm folder structure, set up `requirements.txt` and a virtual environment. Do not generate the full 60-store/3-year dataset until every downstream phase has been proven end-to-end in dev mode.

**Phase 1 — Synthetic Data Generation**
Implement `generate_retail_dataset.py` per §3.1. Run and print all automated QA checks. Nothing proceeds until QA is 100% clean.

**Phase 2 — Data Modelling & EDA**
Build a star schema (DuckDB or plain SQL views). Produce an EDA report: category seasonality, promo rate, stockout rate, missingness, distribution shapes. Save to `reports/`.

**Phase 3 — Statistical Analysis & Regression**
Log-log OLS elasticity regression by price-elasticity segment. Poisson/Negative Binomial regression for demand counts (justify choice via overdispersion test, not assumption). Residual diagnostics and VIF. Output: an elasticity table with confidence intervals by category and segment.

**Phase 4 — Causal Inference (the centrepiece)**
1. Compute the **naive** promotion effect (simple before/after or treated/untreated comparison) and show it's biased.
2. Correct it with **difference-in-differences** using the staggered rollout stores.
3. Cross-check with **propensity score matching / IPW** to adjust for the deliberate assignment bias built into Phase 1's data.
4. **Validate**: compare your recovered treatment effect against `true_promo_uplift_pct` in the ground truth file and report the recovery error explicitly, e.g. *"DiD estimate: 14.2% uplift vs true simulated uplift: 15.0% (0.8pp error, 95% CI covers true value)."* This comparison is what confirms the method is recovering something real, rather than producing a plausible-looking number that happens to fit the noise.

**Phase 5 — Machine Learning Demand Forecasting**
Seasonal-naive baseline → regularized regression (Ridge/Lasso) on leakage-safe features → gradient boosted forecast (LightGBM/XGBoost) with **time-based** cross-validation (never randomly shuffled). Separate stockout-risk classifier with precision/recall reported given class imbalance (stockouts are rare events, so accuracy is a meaningless metric here). SHAP importance, cross-checked for consistency against the Phase 3 elasticity findings.

**Phase 6 — Prescriptive Optimization**
Reorder point / safety stock optimization using forecast uncertainty, not just the point forecast. Promotion budget allocation formulated as an LP/ILP (maximize incremental profit subject to budget and inventory constraints, using PuLP or similar). Monte Carlo simulation to express the recommendation as a profit **range**, not a single number.

**Phase 7 — External Validity Check**
Re-run the elasticity regression and demand forecasting pipeline (same code path, swapped data loader) against Rossmann Store Sales. Report where the method held up and where it didn't. A pipeline that performs identically on synthetic and real data would be unusual and worth checking rather than assuming it's simply correct.

**Phase 8 — Power BI Decision Layer**
PBIP/TMDL semantic model. Pages: Executive Summary, Promotion ROI (naive vs causal-corrected side by side), Elasticity Explorer, Stockout Risk & Replenishment, What-If Promotion Simulator (driven by Phase 6 optimization output). DAX measures should reproduce the Python-calculated figures as a parity check.

**Phase 9 — Documentation & Case Study**
`README.md` structured as: situation → business question → method → result → business impact in £ → explicit limitations. `reports/case_study.md` translates findings into a decision, e.g. *"Reallocating 20% of promotional spend from low-elasticity to under-promoted high-elasticity SKUs is projected to lift gross margin by £X–£Y while cutting promotion-driven stockout losses by Z%."*

**Phase 10 — Repo Polish**
Tests, basic CI (lint + test on push), pinned dependencies, `.gitignore` for large data files, LICENSE, GitHub topics (`causal-inference`, `regression`, `demand-forecasting`, `inventory-optimization`, `python`, `power-bi`).

---

## 7. Modelling Standards

- **Never** let `potential_demand_units` or any `ground_truth_simulation_parameters.csv` column enter a model's feature set. Training on the latent demand the simulation generated makes the validation circular: the model is scored against a quantity it was handed. The failure is also silent, because every metric improves and nothing looks wrong, while the recovery error quietly stops testing anything at all. Build an automated leakage checker in `src/data_quality/` that fails the pipeline rather than warning.
- **Always** report the naive or biased estimate next to the corrected causal one. A corrected number on its own gives a reader no way to tell whether the correction changed anything, and the size of the gap between them is itself a result.
- **Always** report confidence intervals or uncertainty ranges. An estimate of 15% with an interval of ±2pp supports a decision that the same 15% with ±30pp does not, and a bare point estimate hides which of the two you are holding.
- Use **time-based** train/test splits for anything sequential. Random shuffling scatters rows from the test period into training, so the holdout score measures partial memorisation of the window it was supposed to hold out. What you get is a model that validates well and degrades on arrival.
- Report **business-relevant** metrics (MAPE, £ profit impact, stockout-driven lost sales) alongside, not instead of, the standard ML metrics. An accuracy figure does not convert into a reorder quantity or a promotional budget, so on its own there is nothing to act on.
- The README must contain an explicit **"What I would not claim"** section, naming the assumptions whose violation would change the conclusions. DiD assumes parallel trends, so state that and state what could break it. Without that section a reader cannot separate the results that are conditional from the ones that hold regardless.

---

## 8. Definition of Done

Each item names the artifact that satisfies it. Nothing is ticked that cannot be pointed at.

- [x] All Phase 1 QA checks pass and are documented — `reports/data_quality_report.md`, regenerated
      by the generator itself. Each check is mutation-tested in `tests/test_data_quality.py`, so a
      check that cannot fail is itself a test failure.
- [x] Elasticity table produced with CIs, by segment — `reports/phase3_regression.md` and
      `powerbi_data/dose_response.csv` (18 segment × discount-depth cells, 6/6 CIs covering truth).
      Read the qualifier with it: what is identified is the **dose-response curve**, not a
      structural price elasticity. Price moves only through promotions here, so the two collinear
      channels cannot be separated, and §7's naive −3.88 is not reported as an elasticity.
- [x] Naive vs DiD vs PSM promotion effect estimates all computed and compared against ground
      truth, with recovery error reported — `reports/phase4_causal.md` sections 4–6. Naive 0.819,
      five DiD specifications (best 0.675), IPW 0.510, all scored against the 0.5935 log-point
      target from `src/causal/estimands.py`. The propensity arm carries balance and common-support
      diagnostics (`figures/12_covariate_balance.png`).
- [x] Demand forecast model beats seasonal-naive baseline on time-based holdout, MAPE reported —
      `reports/phase5_forecasting.md`, WAPE 0.583 → 0.367 on the Oct–Dec holdout, pinned by
      `tests/test_ml.py::test_gradient_boosting_beats_the_seasonal_naive_baseline`. MAPE is in every results
      table because this list asks for it; WAPE is the primary metric and §5 of that report
      explains why counts this small make MAPE misleading.
- [x] Stockout classifier evaluated on precision/recall, not accuracy — PR-AUC 0.331 against a
      0.0066 base rate, `reports/phase5_forecasting.md` section 6. Accuracy appears in the table
      only to show it is uninformative at this class balance.
- [x] Reorder point optimizer and promo budget LP both produce concrete, justified recommendations
      — `powerbi_data/service_levels.csv`, `reorder_policy.csv` (3,000 pairs) and `promo_plan.csv`
      (10 promotions), with the derivation in `reports/phase6_optimization.md`.
- [x] Monte Carlo simulation produces a profit range, not a point estimate —
      `powerbi_data/promo_plan_uncertainty.csv` and `promo_plan_draws.csv` (4,000 draws). The plan
      is worth £337; the median outcome is −£268 with a 74.2% chance of loss, and the README quotes
      both.
- [x] Pipeline re-run successfully against Rossmann external data, results documented honestly —
      `reports/phase7_external_validity.md`. The honesty is the finding: forecasting transferred
      (WAPE 0.313 → 0.089), the causal design did not, and §3.2's assumption that Rossmann lacks a
      staggered rollout was wrong and is corrected rather than quietly dropped.
- [x] Power BI report built on PBIP/TMDL with at least the 4 pages listed in Phase 8 — five pages,
      66 visuals, opened and refreshed in Desktop 2.155. Parity for every measure in
      `powerbi_data/dax_parity.csv`; the build and the four defects it exposed are in
      `reports/phase8_powerbi.md`.
- [x] README tells the full business story in under a 5-minute read, ends with an explicit
      limitations section — both pinned, by `tests/test_docs.py::test_readme_stays_a_five_minute_read`
      and `::test_readme_has_the_limitations_section`, which also requires each named limitation to
      survive a rewrite.
- [x] No ground-truth or leakage columns present in any trained model's feature set (automated
      check passes) — `src/data_quality/leakage.py` raises rather than warns, and is wired into
      every place a feature set is formed (`stats/models.py`, `ml/features.py`, `causal/psm.py`,
      `validation/rossmann.py`, `features/star_schema.py`). `tests/test_leakage.py` covers renamed
      leaks as well as named ones.

---

## 9. Build Discipline

- Work through the phases **in order**. Do not attempt to generate the full `FULL_MODE=True` dataset (~30M rows) until the entire pipeline has been proven correct in dev mode on the smaller synthetic dataset. Regenerating a dataset that size after finding a Phase 4 or Phase 5 bug costs hours that dev mode would have saved.
- Weigh non-trivial dependencies (EconML, DoWhy, PuLP, LightGBM/XGBoost, great_expectations, etc.) before adding them rather than pulling them in by reflex.
- Commit at the end of each phase with a message describing what changed and why, so the history reflects the actual build order rather than one squashed commit at the end.
- If a phase's output contradicts the ground truth in a way that can't be explained (e.g., DiD recovery error is large), stop and flag it rather than silently adjusting the model until numbers look right. Tuning until the answer matches defeats the entire point of the validation design.
- Ask before scope-creeping in extras (e.g., double/debiased ML via EconML). They're valuable stretch goals, but the Definition of Done in §8 is the bar.
