"""
Safety stock and reorder point optimisation.

The rule this implements
------------------------
    reorder point = expected demand over lead time + safety stock
    safety stock  = z(service level) x sigma of forecast error over lead time

Two choices in there carry all the weight, and both were settled by earlier
phases rather than assumed here.

**Sigma is forecast error, not demand variance.** A policy sized on how much
demand varies protects against the wrong thing: what actually causes a stockout
is the part of demand the forecast failed to anticipate. Phase 5 produced that
error distribution on a held-out quarter, and it is what feeds in here. Phase 5
also showed error is far from homogeneous - MAE on promoted rows is 2.5x the
off-promotion figure - so sigma is estimated per segment rather than pooled.

**The service level is derived, not picked.** A flat "95% for everything" ignores
that a promoted chocolate bar and a bag of salad have very different costs of
being wrong. The newsvendor critical ratio sets it per SKU:

    service level = Cu / (Cu + Co)

with Cu the underage cost (margin forgone on a sale that could not be served) and
Co the overage cost (holding, plus expected waste, which is large for a
three-day-shelf-life item and negligible for tinned goods).

Known limitation, inherited from Phase 5: the forecast targets `units_sold`,
which is stockout-censored. On days when stock bound, observed sales understate
demand, so both the mean and the error spread are biased low - in the same
direction, on the same days. The service levels below are therefore slightly
optimistic, and the report says so rather than burying it.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOGGER = logging.getLogger("northstar.optimization.inventory")

# Annual holding cost as a share of unit cost, a standard retail planning figure.
ANNUAL_HOLDING_RATE = 0.25
DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class CostInputs:
    """Per-unit economics for one SKU."""

    unit_margin: float
    unit_cost: float
    shelf_life_days: int
    is_perishable: bool

    @property
    def holding_cost_per_day(self) -> float:
        return self.unit_cost * ANNUAL_HOLDING_RATE / DAYS_PER_YEAR

    def overage_cost(self, cover_days: float) -> float:
        """
        Cost of one unit of surplus: carrying it, plus the chance it spoils.

        A perishable line with a short shelf life risks the whole unit cost, and
        that risk rises the longer the surplus sits relative to its shelf life.
        """
        holding = self.holding_cost_per_day * cover_days
        if not self.is_perishable:
            return holding
        spoilage_probability = float(np.clip(cover_days / max(self.shelf_life_days, 1), 0.0, 1.0))
        return holding + spoilage_probability * self.unit_cost

    def underage_cost(self) -> float:
        """Margin forgone when demand cannot be served."""
        return self.unit_margin


def critical_ratio(costs: CostInputs, cover_days: float) -> float:
    """Newsvendor optimal service level, clipped to a sane operating band."""
    underage = costs.underage_cost()
    overage = costs.overage_cost(cover_days)
    ratio = underage / max(underage + overage, 1e-9)
    # Below 50% a replenishment policy stops being credible to a planner; above
    # 99.5% the z-multiplier explodes on thin data.
    return float(np.clip(ratio, 0.50, 0.995))


def forecast_error_sigma(
    holdout: pd.DataFrame,
    predictions: np.ndarray,
    segment_columns: Sequence[str],
    target: str = "units_sold",
) -> pd.DataFrame:
    """
    Daily forecast-error standard deviation by segment.

    Uses the held-out quarter, so the spread reflects genuine out-of-sample
    error rather than residuals the model has already fitted.
    """
    work = holdout.copy()
    work["forecast_error"] = predictions - work[target].to_numpy()
    grouped = work.groupby(list(segment_columns), observed=True)["forecast_error"]
    summary = grouped.agg(["std", "mean", "size"]).reset_index()
    return summary.rename(columns={"std": "sigma_daily", "mean": "bias_daily", "size": "rows"})


def lead_time_sigma(sigma_daily: float, lead_time_days: float) -> float:
    """
    Scale a daily error standard deviation to the lead-time window.

    sqrt(L) assumes errors are independent across days. They are not - demand
    is autocorrelated and a forecast that is wrong on Monday is usually wrong on
    Tuesday - so this understates the true spread. It is the standard planning
    approximation and is flagged as a limitation rather than silently used.
    """
    return float(sigma_daily * np.sqrt(max(lead_time_days, 1.0)))


def build_policy(
    pairs: pd.DataFrame,
    sigma_lookup: pd.DataFrame,
    segment_columns: Sequence[str],
    service_level: float | None = None,
) -> pd.DataFrame:
    """
    Compute reorder points for every store x SKU pair.

    `service_level=None` derives it per SKU from the newsvendor ratio; passing a
    float applies that flat level to everything, which is what the comparison in
    the report uses as the incumbent policy.
    """
    work = pairs.merge(sigma_lookup, on=list(segment_columns), how="left")
    work["sigma_daily"] = work["sigma_daily"].fillna(work["sigma_daily"].median())

    rows: List[Dict[str, object]] = []
    for record in work.itertuples(index=False):
        lead_time = float(record.reorder_lead_time_days)
        costs = CostInputs(
            unit_margin=float(record.unit_margin),
            unit_cost=float(record.unit_cost_gbp),
            shelf_life_days=int(record.shelf_life_days),
            is_perishable=bool(record.is_perishable),
        )
        cover = lead_time + 1.0
        level = critical_ratio(costs, cover) if service_level is None else service_level
        z = float(stats.norm.ppf(level))

        sigma_lt = lead_time_sigma(float(record.sigma_daily), lead_time)
        safety_stock = max(0.0, z * sigma_lt)
        expected_demand = float(record.mean_daily_units) * lead_time

        rows.append({
            "store_id": record.store_id,
            "sku_id": record.sku_id,
            "category_label": getattr(record, "category", "Unknown"),
            "service_level": level,
            "z": z,
            "sigma_daily": float(record.sigma_daily),
            "sigma_lead_time": sigma_lt,
            "expected_lead_time_demand": expected_demand,
            "safety_stock": safety_stock,
            "reorder_point": expected_demand + safety_stock,
            "unit_margin": costs.unit_margin,
            "overage_cost": costs.overage_cost(cover),
            "is_perishable": costs.is_perishable,
        })
    return pd.DataFrame(rows)


def evaluate_policy(
    policy: pd.DataFrame,
    demand: pd.DataFrame,
    target: str = "units_sold",
) -> Dict[str, float]:
    """
    Score a policy on the holdout: expected shortfall against holding cost.

    A stockout day is one where lead-time demand exceeded the reorder point.
    Shortfall is valued at margin, surplus at the overage cost, so the two sides
    are on the same footing and the comparison is a £ number rather than a
    service-level percentage that hides its own trade-off.
    """
    merged = demand.merge(policy, on=["store_id", "sku_id"], how="inner")
    lead_demand = merged[target] * merged["reorder_lead_time_days"]

    shortfall = np.maximum(lead_demand - merged["reorder_point"], 0.0)
    surplus = np.maximum(merged["reorder_point"] - lead_demand, 0.0)

    lost_margin = float((shortfall * merged["unit_margin"]).sum())
    holding = float((surplus * merged["overage_cost"]).sum())

    return {
        "rows": int(len(merged)),
        "stockout_days": int((shortfall > 0).sum()),
        "stockout_rate": float((shortfall > 0).mean()),
        "units_short": float(shortfall.sum()),
        "lost_margin_gbp": lost_margin,
        "holding_cost_gbp": holding,
        "total_cost_gbp": lost_margin + holding,
        "mean_safety_stock": float(merged["safety_stock"].mean()),
        "mean_service_level": float(merged["service_level"].mean()),
    }


def compare_policies(
    pairs: pd.DataFrame,
    sigma_lookup: pd.DataFrame,
    segment_columns: Sequence[str],
    demand: pd.DataFrame,
    flat_levels: Sequence[float] = (0.90, 0.95, 0.98),
) -> pd.DataFrame:
    """Flat service levels against the cost-derived policy, scored on the same holdout."""
    rows: List[Dict[str, object]] = []
    for level in flat_levels:
        policy = build_policy(pairs, sigma_lookup, segment_columns, service_level=level)
        rows.append({
            "policy": f"Flat {level:.0%} service level",
            **evaluate_policy(policy, demand),
        })

    optimal = build_policy(pairs, sigma_lookup, segment_columns, service_level=None)
    rows.append({
        "policy": "Newsvendor, cost-derived per SKU",
        **evaluate_policy(optimal, demand),
    })
    return pd.DataFrame(rows).sort_values("total_cost_gbp").reset_index(drop=True)
