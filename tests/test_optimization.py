"""
Tests for the Phase 6 optimisers.

The bug this phase actually shipped and then fixed was an accounting one: the
discount was deducted twice, once through the reduced promotional margin and
again as a separate promotional cost, which made every candidate look
loss-making. Arithmetic that wrong is invisible in a report full of plausible
tables, so the profit identity is pinned down here on hand-computed cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optimization import inventory, monte_carlo, promo_lp  # noqa: E402


# ---------------------------------------------------------------------------
# Newsvendor
# ---------------------------------------------------------------------------

def test_critical_ratio_is_lower_for_perishables():
    """
    A short-shelf-life line risks losing the whole unit cost to spoilage, so its
    optimal service level must sit below an equivalent ambient line's.
    """
    shared = dict(unit_margin=2.0, unit_cost=3.0)
    perishable = inventory.CostInputs(**shared, shelf_life_days=3, is_perishable=True)
    ambient = inventory.CostInputs(**shared, shelf_life_days=365, is_perishable=False)

    assert inventory.critical_ratio(perishable, cover_days=3) < inventory.critical_ratio(
        ambient, cover_days=3
    )


def test_critical_ratio_rises_with_margin():
    """Higher forgone margin means underage hurts more, so stock deeper."""
    low = inventory.CostInputs(unit_margin=0.5, unit_cost=3.0, shelf_life_days=365,
                               is_perishable=False)
    high = inventory.CostInputs(unit_margin=5.0, unit_cost=3.0, shelf_life_days=365,
                                is_perishable=False)
    assert inventory.critical_ratio(high, 3) > inventory.critical_ratio(low, 3)


def test_critical_ratio_stays_inside_the_operating_band():
    extreme = inventory.CostInputs(unit_margin=1000.0, unit_cost=0.01, shelf_life_days=365,
                                   is_perishable=False)
    trivial = inventory.CostInputs(unit_margin=0.001, unit_cost=50.0, shelf_life_days=1,
                                   is_perishable=True)
    assert inventory.critical_ratio(extreme, 3) <= 0.995
    assert inventory.critical_ratio(trivial, 3) >= 0.50


def test_lead_time_sigma_scales_with_square_root():
    assert inventory.lead_time_sigma(10.0, 4.0) == pytest.approx(20.0)
    assert inventory.lead_time_sigma(10.0, 1.0) == pytest.approx(10.0)


def test_overage_cost_grows_with_cover_for_perishables():
    perishable = inventory.CostInputs(unit_margin=2.0, unit_cost=3.0, shelf_life_days=5,
                                      is_perishable=True)
    assert perishable.overage_cost(4) > perishable.overage_cost(1)


# ---------------------------------------------------------------------------
# Promotion economics
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_inputs():
    """One store, one SKU, no category neighbours - so cannibalisation is zero."""
    pairs = pd.DataFrame([{
        "store_id": "S1", "sku_id": "K1", "category": "Test",
        "price_elasticity_segment": "Medium",
        "mean_daily_units": 10.0,
        "regular_unit_price_gbp": 5.0, "unit_cost_gbp": 3.0,
    }])
    # Category total equals this pair's own volume, so other_units is zero.
    category_baseline = pd.DataFrame([{
        "store_id": "S1", "category": "Test", "baseline_units_per_day": 10.0,
    }])
    return pairs, category_baseline


def test_incremental_profit_matches_hand_calculation(simple_inputs):
    """
    Regular £5, cost £3 -> full margin £2. At 20% off: price £4, margin £1.
    Baseline over 8 days = 80 units.

    With a 100% uplift: 160 units x £1 = £160 against 80 x £2 = £160, so the
    promotion exactly breaks even. Anything less than a doubling loses money.
    """
    pairs, category_baseline = simple_inputs
    curve = {"Medium": {20: float(np.log(2.0))}}  # log uplift of 1.0 -> +100%

    candidates = promo_lp.build_candidates(pairs, curve, category_baseline, duration_days=8)
    row = candidates[candidates["discount_pct"] == 20].iloc[0]

    assert row["baseline_units"] == pytest.approx(80.0)
    assert row["promoted_units"] == pytest.approx(160.0)
    assert row["full_margin"] == pytest.approx(2.0)
    assert row["promo_margin"] == pytest.approx(1.0)
    assert row["incremental_profit"] == pytest.approx(0.0, abs=1e-9)


def test_discount_is_not_deducted_twice(simple_inputs):
    """
    Regression test for the shipped-and-fixed bug.

    The promotional give-away is already inside `promo_margin`. If it were also
    subtracted as `promotion_cost`, the break-even case above would come out
    strongly negative instead of zero.
    """
    pairs, category_baseline = simple_inputs
    curve = {"Medium": {20: float(np.log(2.0))}}
    row = promo_lp.build_candidates(pairs, curve, category_baseline, duration_days=8).iloc[0]

    assert row["promotion_cost"] > 0, "budget consumption should still be tracked"
    assert row["incremental_profit"] == pytest.approx(0.0, abs=1e-9)


def test_deeper_discounts_need_more_uplift_to_break_even(simple_inputs):
    """At a fixed uplift, profit must fall as the discount deepens."""
    pairs, category_baseline = simple_inputs
    curve = {"Medium": {depth: float(np.log(1.5)) for depth in promo_lp.DISCOUNT_DEPTHS}}
    candidates = promo_lp.build_candidates(
        pairs, curve, category_baseline
    ).sort_values("discount_pct")
    assert candidates["incremental_profit"].is_monotonic_decreasing


def test_cannibalisation_reduces_profit(simple_inputs):
    """A category with neighbours must price the harm done to them."""
    pairs, _ = simple_inputs
    with_neighbours = pd.DataFrame([{
        "store_id": "S1", "category": "Test", "baseline_units_per_day": 100.0,
    }])
    curve = {"Medium": {20: float(np.log(2.0))}}

    alone = promo_lp.build_candidates(
        pairs, curve,
        pd.DataFrame([{"store_id": "S1", "category": "Test", "baseline_units_per_day": 10.0}]),
    ).iloc[0]
    crowded = promo_lp.build_candidates(pairs, curve, with_neighbours).iloc[0]

    assert crowded["cannibalisation_loss"] > 0
    assert crowded["incremental_profit"] < alone["incremental_profit"]


def test_zero_cannibalisation_rate_removes_the_charge(simple_inputs):
    pairs, _ = simple_inputs
    neighbours = pd.DataFrame([{
        "store_id": "S1", "category": "Test", "baseline_units_per_day": 100.0,
    }])
    curve = {"Medium": {20: float(np.log(2.0))}}
    row = promo_lp.build_candidates(
        pairs, curve, neighbours, cannibalisation_rate=0.0
    ).iloc[0]
    assert row["cannibalisation_loss"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The integer program
# ---------------------------------------------------------------------------

# Turning cannibalisation off makes every candidate profitable, which turns the
# budget constraint into a genuine knapsack. CBC is slow on those, so the test
# fixture is kept small and every solve is time-boxed - these tests check that
# the constraints hold, not that the optimum is proven.
SOLVE_LIMIT = 10


@pytest.fixture(scope="module")
def many_candidates():
    rng = np.random.default_rng(0)
    rows = []
    for store in range(2):
        for sku in range(6):
            rows.append({
                "store_id": f"S{store}", "sku_id": f"K{sku}",
                "category": f"C{sku % 3}",
                "price_elasticity_segment": "Medium",
                "mean_daily_units": float(rng.uniform(5, 30)),
                "regular_unit_price_gbp": 5.0, "unit_cost_gbp": 2.0,
            })
    pairs = pd.DataFrame(rows)
    category_baseline = (
        pairs.groupby(["store_id", "category"])["mean_daily_units"].sum()
        .reset_index().rename(columns={"mean_daily_units": "baseline_units_per_day"})
    )
    curve = {"Medium": {depth: float(np.log(1 + depth / 12)) for depth in promo_lp.DISCOUNT_DEPTHS}}
    return promo_lp.build_candidates(pairs, curve, category_baseline, cannibalisation_rate=0.0)


def test_solution_respects_the_budget(many_candidates):
    budget = 500.0
    solution = promo_lp.solve(many_candidates, budget=budget, time_limit_seconds=SOLVE_LIMIT)
    assert solution["total_cost"] <= budget + 1e-6


def test_at_most_one_depth_per_store_sku(many_candidates):
    solution = promo_lp.solve(many_candidates, budget=5000.0, time_limit_seconds=SOLVE_LIMIT)
    plan = solution["plan"]
    assert not plan.duplicated(["store_id", "sku_id"]).any()


def test_store_category_cap_is_enforced(many_candidates):
    solution = promo_lp.solve(
        many_candidates,
        budget=100_000.0,
        max_per_store_category=2,
        time_limit_seconds=SOLVE_LIMIT,
    )
    counts = solution["plan"].groupby(["store_id", "category"]).size()
    assert (counts <= 2).all()


def test_only_profitable_candidates_are_selected(many_candidates):
    solution = promo_lp.solve(many_candidates, budget=100_000.0, time_limit_seconds=SOLVE_LIMIT)
    assert (solution["plan"]["incremental_profit"] > 0).all()


def test_a_bigger_budget_never_reduces_profit(many_candidates):
    small = promo_lp.solve(many_candidates, budget=200.0, time_limit_seconds=SOLVE_LIMIT)
    large = promo_lp.solve(many_candidates, budget=2000.0, time_limit_seconds=SOLVE_LIMIT)
    assert large["total_profit"] >= small["total_profit"] - 1e-6


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def test_simulation_produces_a_distribution(many_candidates):
    plan = promo_lp.solve(many_candidates, budget=5000.0, time_limit_seconds=SOLVE_LIMIT)["plan"]
    draws = monte_carlo.simulate_plan(plan, n_draws=400, seed=1)
    assert len(draws) == 400
    assert draws["profit"].std() > 0


def test_summary_percentiles_are_ordered(many_candidates):
    plan = promo_lp.solve(many_candidates, budget=5000.0, time_limit_seconds=SOLVE_LIMIT)["plan"]
    draws = monte_carlo.simulate_plan(plan, n_draws=600, seed=1)
    summary = monte_carlo.summarise(
        draws, deterministic_profit=float(plan["incremental_profit"].sum())
    )
    assert summary["p10"] <= summary["p50"] <= summary["p90"]


def test_downward_effect_bias_lowers_expected_profit(many_candidates):
    """
    Phase 4 concluded the promotional estimate is an upper bound, so the default
    prior shades it down. A smaller multiplier must produce less profit.
    """
    plan = promo_lp.solve(many_candidates, budget=5000.0, time_limit_seconds=SOLVE_LIMIT)["plan"]
    pessimistic = monte_carlo.simulate_plan(plan, n_draws=600, effect_bias=0.6, seed=1)
    optimistic = monte_carlo.simulate_plan(plan, n_draws=600, effect_bias=1.2, seed=1)
    assert pessimistic["profit"].mean() < optimistic["profit"].mean()


def test_simulation_is_reproducible(many_candidates):
    plan = promo_lp.solve(many_candidates, budget=5000.0, time_limit_seconds=SOLVE_LIMIT)["plan"]
    first = monte_carlo.simulate_plan(plan, n_draws=300, seed=42)
    second = monte_carlo.simulate_plan(plan, n_draws=300, seed=42)
    assert np.allclose(first["profit"], second["profit"])
