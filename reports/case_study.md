# Case study — what Northstar should do differently

A decision memo, not a summary. The analysis is in the phase reports; this is what follows from it.

---

## The recommendation, in one paragraph

Northstar should **stop promoting at its current volume**, **cut safety stock on perishables
rather than adding it**, and **re-estimate promotional response causally every quarter**. The
first two are worth roughly **£16,300 a quarter** in inventory cost alone and turn a promotional
programme that mostly destroys margin into a much smaller one that does not. The third is what
makes the first two trustworthy: on the same budget, a plan built on the naive promotional
estimate loses £349 while the causally-estimated plan earns £220.

---

## 1. Cut safety stock on perishables — £16,320 a quarter

**Decision:** replace the flat 95% service-level policy with per-SKU levels derived from each
line's own underage and overage costs.

**Why it works:** a 95% target applied to everything is a decision not to think about cost
asymmetry. The newsvendor ratio says the optimal service level is a function of what a lost sale
costs versus what an unsold unit costs — and for fresh produce those are wildly different.

| Category | Cost-optimal service level |
|---|---|
| Fresh Produce | **52%** |
| Bakery | 59% |
| Dairy & Eggs | 84% |
| Ambient, Beverages, Frozen, Household, Snacks | 99%+ |

Holding an extra unit of salad forfeits its entire cost when it spoils. Holding an extra tin of
beans costs a few pence of working capital for a week. Running fresh at 95% availability is
buying insurance that costs more than the risk.

**Value:** £16,320 over the holdout quarter against the flat policy — earned *chiefly by
reducing* safety stock, not adding it.

**What would change this:** the figure depends on the perishability and margin assumptions in the
cost model. The direction — perishables lower, ambient higher — is robust across every
assumption tested; the exact £ is not.

---

## 2. Run far fewer, shallower promotions

**Decision:** cut the promotional programme from roughly 3,013 promotions a quarter to a small
double-digit number, and spread what remains across categories rather than concentrating it.

**Why it works:** of 18,000 candidate (store, SKU, depth) combinations, **51% are profitable on
their own P&L**. Once each is charged for the demand it steals from its own category at the rate
Phase 3 measured, **27 survive**. The optimiser picks 10 under budget and inventory constraints.

Two errors are doing the damage in the current programme:

- **Crediting incremental units at full margin.** A promotion sells more units *and* sells the
  baseline units at a discount. Netting the margin sacrificed on volume that would have sold
  anyway is not a refinement — it is the difference between a positive and a negative number.
- **Ignoring cannibalisation entirely.** At 6.1% per promoted neighbour, a category running four
  concurrent promotions is moving volume between its own shelves and calling all of it
  incremental.

**What would change this:** the specific figure of 10 depends on a linearised cannibalisation
charge, and the report sweeps that rate precisely because it drives the answer. Treat 10 as
directional. The direction holds everywhere in the sweep.

---

## 3. Re-estimate causally every quarter — £569 on this budget

**Decision:** make the causal re-estimation a standing quarterly process, not a one-off study.

**Why it works:** this is the finding that justifies the whole method stack.

| Plan built on | Promotions picked | Profit under the true response | Loss-making picks |
|---|---|---|---|
| Naive estimate | 19 | **−£348.83** | 19 of 19 |
| Causal estimate | 10 | **+£219.80** — 96.8% of perfect knowledge | 1 of 10 |
| Perfect knowledge (unavailable in practice) | 9 | +£227.12 | 0 of 9 |

The naive estimate does not merely overstate the return. It overstates it *unevenly*, and so
picks a different and worse set of promotions. An optimiser fed a biased input optimises
confidently in the wrong direction — which is why the £569 gap is a decision-quality figure, not
a forecasting-accuracy one.

---

## 4. Budget the range, not the number

**Decision:** carry the promotional plan in the budget at its uncertainty range, and treat the
point estimate as the optimistic end.

The optimiser reports **£337** for its recommended plan. The Monte Carlo over 4,000 draws of the
forecast error distribution says:

| | |
|---|---|
| P10 | −£762 |
| P50 (median) | **−£268** |
| P90 | +£265 |
| Probability of loss | **74%** |

Both numbers are correct. £337 is what the plan earns if the causal estimate is right; −£268 is
what it earns at the median once estimation error is propagated. Given Phase 4 established the
DiD estimate is an *upper bound*, the honest central case assumes the estimate is optimistic —
which is why the dashboard's what-if slider opens at 0.88 rather than 1.00, where the plan turns
a **−£259.55** loss.

A programme this marginal should not be scaled until the estimate is tightened.

---

## What this case study does not establish

The promotional £ figures here are small in absolute terms because the simulation gives
promotions **no traffic-building effect** — every incremental unit is own-SKU uplift or volume
taken from a neighbour. Real promotions bring people into stores who buy other things, and that
channel is worth more than anything measured here. The transferable findings are the *method* and
the *sign of the bias*, not the pound values.

Nor is any of this a price-elasticity result. Price moves only through promotions in this data,
so the two cannot be separated (Phase 3, VIF > 2000). What is validated is the dose-response
curve — how much lift a given discount depth buys — which is what the optimiser actually consumes.
