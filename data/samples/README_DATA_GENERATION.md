# PromoPulse: Causal Promotion, Demand and Inventory Optimisation

## Fictional business scenario
Northstar Retail Group is a fictional UK mid-market grocery and convenience retailer operating City Convenience, High Street, Suburban Supermarket and Retail Park Superstore locations across England, Scotland and Wales.

## Data model
- `dim_store.csv`: store master data.
- `dim_product.csv`: SKU and commercial-product attributes.
- `dim_calendar.csv`: daily calendar, events and synthetic weather.
- `fact_promotions.csv`: promotion-event records.
- `fact_inventory_delivery.csv`: daily inventory movement and stockout outcomes.
- `fact_daily_store_sku.csv`: primary analytical fact table at date × store × SKU grain.
- `../ground_truth/ground_truth_simulation_parameters.csv`: known simulation parameters for
  retrospective validation only. Deliberately stored outside `data/raw/` so no training
  loader can pick it up by globbing the raw directory.

Relationships:
- Daily facts link to `dim_store` through `store_id`.
- Daily facts link to `dim_product` through `sku_id`.
- Daily facts link to `dim_calendar` through `date`.
- Promotion events can be joined to the daily fact table using store, SKU and the promotion-date range.

## Entity counts and coverage
- Stores: 20
- SKUs: 150
- Calendar days: 731
- Daily modelling rows: 2,193,000
- Date range: 2023-01-01 to 2024-12-31

## Synthetic demand logic
Latent demand is generated through a noisy multiplicative process combining product base demand, store demand factor, weekday effect, seasonality, holiday and event effects, weather, price response, promotion uplift, marketing support, autocorrelation and stochastic variation. Demand is generated using a Gamma-Poisson mixture to create over-dispersion similar to retail demand.

## Promotion-selection bias
Promotions are intentionally not random. Products with stronger margin, seasonal relevance, own-label status and weakening demand momentum are more likely to receive promotions. That momentum is a real declining trend applied to latent demand, so the decline is visible in observed sales history and can legitimately be adjusted for by a propensity model. Higher-footfall stores receive more promotion events. Promotions cluster around Christmas, Easter, heatwaves, bank holidays and payday windows.

## Staggered campaign rollout
Roughly 60% of promotion events belong to multi-store campaigns identified by `campaign_id`. Stores join a campaign by `rollout_cohort` (assigned on footfall rank), entering after the wave offset recorded in `campaign_wave_days`. This produces genuine staggered adoption for difference-in-differences. The remaining events are store-local tactical promotions. SKUs that are never promoted anywhere form a never-treated control pool.

## Treatment-effect ground truth
`true_promo_uplift_pct` is a **structural coefficient applied per 10 percentage points of discount**, and it compounds with the separate price-elasticity response. It is not an average treatment effect and must not be compared directly against a DiD or PSM estimate. Use `true_realised_att_pct`, which is the exact simulated effect on latent demand averaged over each SKU's treated rows. Note that observed sales are stockout-censored, so an estimate recovered from `units_sold` will sit below the latent ATT.

## Stockout censoring
`potential_demand_units` represents latent uncensored demand. `units_sold` is constrained by available stock after deliveries, damage and waste. Therefore stockouts censor observed sales and create `lost_sales_estimate_units`. Inventory reconciliation is:
`closing_stock = max(0, opening_stock + delivery_units - units_sold - damaged_units - expired_or_wasted_units)`.

## Data-quality checks completed
The generator checks unique dimension keys, fact key construction, foreign-key validity, non-negative quantities, price and cost validity, discount bounds, inventory reconciliation, stock constraints, gross-profit reconciliation, never-treated pool integrity, headers and blank keys. Full results are written to `reports/data_quality_report.md`. Each check is verified by mutation testing in `tests/` — a corrupted value must make the corresponding check fail.

## Important modelling warning
Do not use `potential_demand_units`, `lost_sales_estimate_units`, anomaly labels or `ground_truth_simulation_parameters.csv` as predictive or causal-model features. `ground_truth_simulation_parameters.csv` exists only to assess whether a model recovered the known simulated truth after modelling.

## Suggested analysis tasks
a. Regression analysis of price elasticity  
b. Difference-in-differences study of promotion effect  
c. Propensity-score-weighted promotion-effect study  
d. Time-series demand forecasting  
e. Stockout prediction  
f. Profit-maximising promotion simulation  
g. Replenishment recommendation  
