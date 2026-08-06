# PromoPulse — Data Quality Report

Generated from seed 42 in DEV mode (20 stores x 150 SKUs, 2023-01-01 to 2024-12-31).

**22 of 22 checks passed.**

| Check | Result | Detail |
|---|---|---|
| Unique store primary keys | PASS | — |
| Unique SKU primary keys | PASS | — |
| Unique calendar dates | PASS | — |
| Unique promotion IDs | PASS | — |
| Unit cost below retail price | PASS | — |
| Promotion discount within 0-30% | PASS | observed range 0-30 |
| Promotion store foreign keys valid | PASS | — |
| Promotion SKU foreign keys valid | PASS | — |
| Promoted SKUs are all promotion-eligible | PASS | never-treated control pool must stay untreated |
| No blank or invalid daily foreign keys | PASS | 0 violations |
| No negative daily quantities or prices | PASS | 0 violations |
| Daily discount within 0-30% | PASS | 0 violations |
| Promoted price never exceeds regular price | PASS | 0 violations |
| Units sold never exceed available stock | PASS | 0 violations |
| Gross profit reconciles to revenue - COGS - promo cost | PASS | max absolute difference £0.0000 |
| No duplicate daily fact keys | PASS | — |
| No negative inventory quantities | PASS | 0 violations |
| Inventory reconciliation identity holds | PASS | 0 violations, max difference 0 units |
| No duplicate inventory fact keys | PASS | — |
| Daily row count matches simulation | PASS | 2,193,000 rows |
| Inventory row count matches simulation | PASS | 2,193,000 rows |
| Ground truth is outside data/raw | PASS | leakage guard per §5/§7 |

## Key rates

- Daily fact rows: 2,193,000
- Promotion density: 8.49%
- Stockout rate: 0.282%
- True average treatment effect on latent demand: 102.63%

## Modelling warning

`potential_demand_units`, `lost_sales_estimate_units`, the anomaly columns and every column in `data/ground_truth/` are outcomes of the simulation, not inputs. They must never enter a model feature set. `true_promo_uplift_pct` is a structural coefficient applied per 10 percentage points of discount and compounds with the separate price response — compare causal estimates against `true_realised_att_pct` instead.
