# Response to the Phase 8 / final-polish review

What changed, where, and which test now holds it in place. Points not acted on are
listed with the reason, and three of the review's premises turned out not to hold —
those are corrected here rather than quietly worked around.

---

## 1. The build-state contradiction — fixed

**Raised:** `PAGE_SPEC.md` and `reports/phase8_powerbi.md` each claimed the five Power BI
pages were both built and not built.

**Established from the repo, not from either document:** five page directories under
`Northstar.Report/definition/pages/`, 66 `visual.json` files, five screenshots, and an
`Uplift Scenario` parameter generated in TMDL. The build is done; the prose was stale.

**Changed:** `phase8_powerbi.md` §5 and its Honest summary now describe the built state.
`PAGE_SPEC.md` lost the "Why this is a spec rather than a built report" argument and the
"scaffolded … named and empty" clause. Both keep the superseded text verbatim under a
dated `## Superseded:` heading — the four defects the assembly exposed only make sense
next to the prediction that it could not be done blind.

**Correction to the brief:** the review placed `§5 "What still needs a human"` in
`PAGE_SPEC.md`. Both that section and `"Honest summary"` are in `phase8_powerbi.md`;
`PAGE_SPEC.md` has neither. Its contradiction was in the opening paragraphs instead, so
the fix landed in a different place than described.

**Also wrong rather than stale:** the measure count. `phase8_powerbi.md` said 45 in one
place and 41 in another, `PAGE_SPEC.md` said 41; `tmdl.parse_measures` returns **45**.
And `PAGE_SPEC.md` described the what-if series as `GENERATESERIES(0.5, 1.3, 0.05)` —
the drifting `double` form that cannot land on 0.88, fixed in the model months earlier.

**Pinned by** `tests/test_docs.py`:
`test_no_document_asks_for_work_the_repo_already_contains`,
`test_documents_claim_the_page_and_visual_counts_that_exist`,
`test_measure_count_in_prose_matches_the_library`. Text under a superseded heading is
exempt by design; that is what lets the build trail survive the check.

---

## 2. Definition of Done — ticked

All eleven items in `PROJECT_ARCHITECTURE.md` §8 were unticked while every one was met.
Each now carries a pointer to the artifact that satisfies it. Three carry a qualifier
rather than a bare tick, because the honest answer is narrower than the item asks:

- the elasticity item is met by the **dose-response curve**, not a structural price
  elasticity, which is not identified in this data;
- MAPE is reported because §8 asks for it, but WAPE is primary;
- §8 asks for at least four Power BI pages; five exist.

**Pinned by** `test_definition_of_done_points_at_things_that_exist`, which fails on any
unticked item, any `::test_name` that no test file defines, and any cited path that does
not exist. It was written after the first draft of the list cited a test name that does
not exist.

---

## 3. DataFolder friction — reduced

The committed parameter was `C:/Users/Shivansh Chauhan/Desktop/power-bi-northstar/powerbi_data/`:
one person's directory layout, published, resolving on exactly one machine.

**Changed:** the committed literal is a placeholder; `src/powerbi/tmdl.py` prints the
exact value to paste on generation, already normalised; `PAGE_SPEC.md` gains an
"If you cloned this" block at the top, where the question is actually asked.

**Pinned by** `test_committed_data_folder_is_not_a_developer_home_path`, confirmed
failing against the old literal before the replacement.

**Scope note:** the parameter itself stays. Power Query resolves relative paths against
the Desktop process's working directory rather than the project file, so the reasoning
in `phase8_powerbi.md` for an explicit parameter is sound. One line of
`expressions.tmdl` was hand-edited and `tmdl.py` was deliberately **not** re-run —
regenerating would have rewritten the semantic model on a macOS machine and swapped one
machine-specific path for another. Nothing under `Northstar.Report/` was touched.

---

## 4a. PSM/IPW — the premise was wrong; the real gap is closed

**Raised:** the propensity arm is thin next to DiD, and needs balance before and after
weighting, common-support diagnostics, a CI, and a recovery error reported beside naive
and DiD.

**All five already existed** before this review: `standardised_differences`
(`src/causal/psm.py`) and `figures/12_covariate_balance.png`; `overlap_summary` and the
"100.0% of control rows sit above the minimum treated propensity" finding; the CI from
`ipw_att`; the −0.083 recovery error; and the §5 comparison table that carries naive,
DiD and IPW together.

**The real gap was narrower.** §6 Phase 4.3 asks for "propensity score matching / IPW"
and only the weighting half was built. `psm.match_att` now does nearest-neighbour
matching on the propensity logit, with replacement, inside a 0.2-SD caliper, with
`psm.matched_balance` as its balance table.

| sample | IPW | matching | error vs 0.5935 |
|---|---|---|---|
| all treated rows | +0.510 | **+0.557** | −0.083 → **−0.037** |
| first promotion day only | +0.752 | +0.695 | +0.158 → +0.101 |

Matching is the closest single estimate in the phase, and the report argues against
reading a winner off that. It does not fix the bad-control problem dragging both below
the target; it weights the same contaminated comparison differently, and the ordering
reverses on the first-day sample.

**Pinned by** four tests in `tests/test_causal.py`, including
`test_matching_recovers_a_known_effect` (a constructed +0.40 effect on data confounded
badly enough that the naive difference misses by more than 0.3) and
`test_matching_caliper_binds`.

No existing estimate moved — the diff to `phase4_causal.md` is insertions only.

---

## 4b. Goodman-Bacon — measured

**Raised:** Phase 7 named the staggered-adoption problem and never addressed it.

**Chose the decomposition over Callaway–Sant'Anna.** It answers the question actually
raised, needs no new dependency, and is checkable against a theorem rather than against
a reference implementation that cannot be run here.

`src/validation/bacon.py`, on the Promo2 design:

| comparison | weight | avg effect |
|---|---|---|
| treated vs never-treated | 0.915 | +0.014 |
| later vs earlier (already-treated control) | **0.062** | −0.066 |
| earlier vs later (later as control) | 0.024 | +0.061 |

**6.2% of the weight sits on the bad comparisons**, so contamination is not what makes
this estimate unusable. The parallel-trends failure is, which section 5 established
before the decomposition ran. The decomposition was run expecting to explain the
estimate away and it does not — so a Callaway–Sant'Anna estimator would replace bad
comparisons with clean ones and still difference against a control group that was
already drifting. Reported as a sensitivity on the estimator, not as a rescued number,
exactly as the brief asked.

**Pinned by** `test_decomposition_reproduces_the_twfe_coefficient`: the weighted 2x2s
must sum to the regression coefficient exactly, and do so to 7e-16, including under
heterogeneous effects where TWFE is itself biased. A decomposition returning
plausible-looking shares would pass an eyeball check and fail this.

---

## 5. Hyperparameter search — run, reported, not adopted

Eight candidates over learning rate, leaf count and minimum leaf size, scored on the
existing expanding-window folds — same chronology, same 7-day gap, nothing shuffled.

| parameters | mean WAPE | vs defaults |
|---|---|---|
| learning_rate=0.03 | 0.3929 | −0.0006 |
| leaves=48 | 0.3930 | −0.0005 |
| min_leaf=500 | 0.3932 | −0.0003 |
| **defaults** | **0.3934** | — |
| learning_rate=0.1 | 0.3946 | +0.0011 |

**The best candidate is worth 0.0006 WAPE, about 0.1% of the error**, and that is the
result rather than a disappointment. It says the model is not parameter-starved: the
ladder's step from seasonal naive (0.583) to boosting (0.367) is worth 0.216 WAPE, some
360 times more than the best knob on the booster.

**The shipped model keeps the defaults**, for two reasons. Phase 6 sizes safety stock
from this model's forecast error, so adopting a tuned model would move every service
level, every reorder point, the £16,320/quarter figure and the Power BI export — a
revaluation the whole downstream chain would have to absorb for a gain of 0.0006. And
the winner was chosen on the same folds that scored it, so part of that gain is bought
from the validation set rather than from the model.

**One defect found while building this.** `GradientBooster.params` was forwarded raw to
LightGBM and **dropped entirely on the sklearn path**, which is the backend on any
machine without `libomp` — including this one. A grid written in LightGBM's vocabulary
would have been a silent no-op returning eight identical scores, which reads exactly
like "tuning buys nothing". Parameters are now backend-neutral and translated per
backend, pinned by `test_translate_params_maps_onto_each_backend` and
`test_translate_params_rejects_unknown_names`. Had this not been caught, the honest-
looking null result above would have been an artifact.

**Pinned by** `test_tuning_reports_every_candidate_against_the_defaults` and
`test_tuning_grid_contains_the_defaults` — without a defaults row, `vs_defaults` would
silently compare two tuned candidates.

The diff to `phase5_forecasting.md` is insertions plus the removal of the old
"no hyperparameter search" limitation. No production metric moved.

---

## Not acted on

- **`PROJECT_ARCHITECTURE.md` §6 Phase 4.4 still says to validate against
  `true_promo_uplift_pct`.** That is the wrong quantity — a structural coefficient
  applied per 10pp of discount, not an ATT — and Phase 4 correctly uses
  `true_realised_att_pct`. Correcting the spec retroactively was out of scope for this
  brief, as with MAPE in §7/§8. Worth a separate change.
- **Callaway–Sant'Anna itself.** See 4b: it would not repair identification on a design
  that fails parallel trends, and the decomposition answers the question that was asked.
- **Nothing in the Power BI report was rebuilt or regenerated.** One line of
  `expressions.tmdl` changed by hand. No page, visual, screenshot or exported table was
  touched, and no analytical output added here needs to appear on the dashboard.

## What was deliberately left alone

The README's "What I would not claim" section is unchanged, and every negative result
stands: the Monte Carlo's 74.2% probability of loss, the Rossmann parallel-trends
failure, the unidentified price elasticity. Two tests would fail if any of them were
softened — `test_readme_has_the_limitations_section` and
`test_monte_carlo_range_is_quoted_consistently`.
