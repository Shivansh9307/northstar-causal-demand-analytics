# PromoPulse: Causal Demand, Promotion & Inventory Intelligence for Northstar Retail Group

**Repo name:** `northstar-causal-demand-analytics`
**Owner:** Shivansh Chauhan
**Purpose:** Flagship portfolio project demonstrating statistical analysis, regression, causal inference, machine learning, and prescriptive optimization for a UK multi-store retailer — built to be read closely by a Fortune 500 hiring manager screening for a Data Analyst / BI Analyst role.

This document is the **architecture plan**. Read it in full before writing any code, confirm the build order, and work phase by phase rather than attempting the whole repo in one pass.

---

## 1. Why This Project Exists (hiring-manager lens)

Most "data analyst portfolio" repos fail for the same three reasons:

1. **They stop at prediction.** A forecast with an R² score doesn't tell a hiring manager you can make a decision — it tells them you can call `.fit()`.
2. **They confuse correlation with causation.** "Sales went up during the promotion" is not the same claim as "the promotion caused sales to go up," and most portfolios don't know the difference.
3. **They're generic.** Titanic, Iris, a Kaggle churn set with an XGBoost model and a confusion matrix — a hiring manager has seen this exact repo two hundred times this year.

This project is designed to avoid all three. It:

- Uses a **synthetic dataset with known ground truth**, so causal and forecasting claims can be validated against the actual simulated answer — not just judged by "does the number look reasonable."
- Goes **predictive → prescriptive**: the final output isn't a forecast, it's a promotion budget allocation and reorder-point recommendation with a profit range attached.
- Is **cross-validated against real public retail data** (Rossmann Store Sales), so the methodology isn't only proven to work on data built to be solvable.
- Ends in a **Power BI decision layer**, not just a Jupyter notebook — because that combination (rigorous Python-side causal inference + a polished BI front end) is rare and plays directly to your existing strengths.

---

## 2. The Central Business Question

> **Northstar Retail Group runs promotions across 60 stores and 500 SKUs, but nobody can confidently say which promotions actually drove incremental profit versus which ones would have sold anyway — and stockouts during promotions are quietly destroying the upside. Which products and stores should receive promotional investment next quarter, and how should replenishment be adjusted to avoid losing the sales the promotion was meant to capture?**

Every phase of this project answers a piece of that question. If a piece of analysis doesn't move you closer to answering it, it doesn't belong in the flagship repo.

---

## 3. Data Strategy

### 3.1 Primary dataset — synthetic, with ground truth (build first)

Use the `generate_retail_dataset.py` specification already drafted (Northstar Retail Group, 60 stores / 500 SKUs / 3 years in `FULL_MODE=True`, reduced 20/150/2yr in dev mode). Key non-negotiables carried over from that spec:

- Promotion assignment is **deliberately biased** (weakening sales, high margin, high footfall, seasonal clustering) — never randomly assigned. This is what makes the causal inference phase meaningful.
- Staggered store rollouts are included specifically to enable difference-in-differences.
- Stockouts censor observed sales; `potential_demand_units` (uncensored latent demand) is generated but must **never** be used as a model feature.
- `ground_truth_simulation_parameters.csv` holds the true elasticity, true promo uplift, true cannibalisation factor, etc. This file is a **validation-only** artifact. Any pipeline step that touches it for training must fail a leakage check (see §7).
- Seed fixed at 42 throughout for reproducibility.
- Automated data-quality checks (no negative inventory, no duplicate keys, discount range 0–30%, unit cost < price, inventory reconciliation identity holds, etc.) must all pass before EDA begins, and their output should be printed and saved to `reports/data_quality_report.md`.

### 3.2 Secondary dataset — real data for external validation (build later, Phase 7)

**Rossmann Store Sales** (Kaggle) — real daily store-level sales, promotions, holidays, competitor distance. Used only to re-run the **elasticity regression and demand forecasting pipeline** (not the full causal/optimization stack, since Rossmann lacks the staggered-rollout structure) against real-world data, to demonstrate the method isn't just solving a puzzle you built yourself.

This phase is what lets your README say, credibly: *"the same feature engineering and model architecture were validated on an independent real-world dataset."* That sentence is worth more than another 10% of model accuracy.

---

## 4. Method Stack (mapped explicitly to target JD language)

| JD skill | Where it shows up |
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
4. **Validate**: compare your recovered treatment effect against `true_promo_uplift_pct` in the ground truth file and report the recovery error explicitly, e.g. *"DiD estimate: 14.2% uplift vs true simulated uplift: 15.0% (0.8pp error, 95% CI covers true value)."* This single comparison is the strongest artifact in the whole repo — it proves the method works, not just that it produces a plausible-looking number.

**Phase 5 — Machine Learning Demand Forecasting**
Seasonal-naive baseline → regularized regression (Ridge/Lasso) on leakage-safe features → gradient boosted forecast (LightGBM/XGBoost) with **time-based** cross-validation (never randomly shuffled). Separate stockout-risk classifier with precision/recall reported given class imbalance (stockouts are rare events — accuracy is a meaningless metric here). SHAP importance, cross-checked for consistency against the Phase 3 elasticity findings.

**Phase 6 — Prescriptive Optimization**
Reorder point / safety stock optimization using forecast uncertainty, not just the point forecast. Promotion budget allocation formulated as an LP/ILP (maximize incremental profit subject to budget and inventory constraints, using PuLP or similar). Monte Carlo simulation to express the recommendation as a profit **range**, not a single number.

**Phase 7 — External Validity Check**
Re-run the elasticity regression and demand forecasting pipeline (same code path, swapped data loader) against Rossmann Store Sales. Report honestly where the method held up and where it didn't — that honesty is itself a signal of seniority.

**Phase 8 — Power BI Decision Layer**
PBIP/TMDL semantic model. Pages: Executive Summary, Promotion ROI (naive vs causal-corrected side by side), Elasticity Explorer, Stockout Risk & Replenishment, What-If Promotion Simulator (driven by Phase 6 optimization output). DAX measures should reproduce the Python-calculated figures as a parity check.

**Phase 9 — Documentation & Case Study**
`README.md` structured as: situation → business question → method → result → business impact in £ → explicit limitations. `reports/case_study.md` translates findings into a decision, e.g. *"Reallocating 20% of promotional spend from low-elasticity to under-promoted high-elasticity SKUs is projected to lift gross margin by £X–£Y while cutting promotion-driven stockout losses by Z%."*

**Phase 10 — Repo Polish**
Tests, basic CI (lint + test on push), pinned dependencies, `.gitignore` for large data files, LICENSE, GitHub topics (`causal-inference`, `regression`, `demand-forecasting`, `inventory-optimization`, `python`, `power-bi`).

---

## 7. Non-Negotiable Rules (statistical maturity signals)

- **Never** let `potential_demand_units` or any `ground_truth_simulation_parameters.csv` column enter a model's feature set. Build an automated leakage checker in `src/data_quality/` that fails the pipeline if it detects this.
- **Always** report the naive/biased estimate next to the corrected causal estimate — showing you know the difference is worth more than the corrected number alone.
- **Always** report confidence intervals or uncertainty ranges. A point estimate with no uncertainty reads as junior.
- **Time-based** train/test splits only for anything sequential. Random shuffling on time series data is an instant credibility loss with a technical reviewer.
- Report **business-relevant** metrics (MAPE, £ profit impact, stockout-driven lost sales) alongside — not instead of — standard ML metrics. Accuracy alone means nothing to a commercial stakeholder.
- The README must contain an explicit **"What I would not claim"** section — e.g., stating clearly that the DiD design assumes parallel trends and naming what could violate it. This is the single highest-leverage sentence type for signalling seniority to an experienced reviewer.

---

## 8. Definition of Done

- [ ] All Phase 1 QA checks pass and are documented
- [ ] Elasticity table produced with CIs, by segment
- [ ] Naive vs DiD vs PSM promotion effect estimates all computed and compared against ground truth, with recovery error reported
- [ ] Demand forecast model beats seasonal-naive baseline on time-based holdout, MAPE reported
- [ ] Stockout classifier evaluated on precision/recall, not accuracy
- [ ] Reorder point optimizer and promo budget LP both produce concrete, justified recommendations
- [ ] Monte Carlo simulation produces a profit range, not a point estimate
- [ ] Pipeline re-run successfully against Rossmann external data, results documented honestly
- [ ] Power BI report built on PBIP/TMDL with at least the 4 pages listed in Phase 8
- [ ] README tells the full business story in under a 5-minute read, ends with an explicit limitations section
- [ ] No ground-truth or leakage columns present in any trained model's feature set (automated check passes)

---

## 9. Build Discipline

- Work through the phases **in order**. Do not attempt to generate the full `FULL_MODE=True` dataset (~30M rows) until the entire pipeline has been proven correct in dev mode on the smaller synthetic dataset — this avoids burning time regenerating a huge dataset after finding a bug in Phase 4 or 5.
- Weigh non-trivial dependencies (EconML, DoWhy, PuLP, LightGBM/XGBoost, great_expectations, etc.) before adding them rather than pulling them in by reflex.
- Commit at the end of each phase with a clear message, so the repo history itself tells the story of the build (useful for a hiring manager who checks commit history).
- If a phase's output contradicts the ground truth in a way that can't be explained (e.g., DiD recovery error is large), stop and flag it rather than silently adjusting the model until numbers look right — that would defeat the entire point of the validation design.
- Ask before scope-creeping in extras (e.g., double/debiased ML via EconML) — they're valuable stretch goals but the core Definition of Done in §8 is the actual bar for "flagship" status.