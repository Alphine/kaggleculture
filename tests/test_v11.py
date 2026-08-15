"""Tests for v11: direct market-inventory trend signals + opponent-
expansion mirroring (see NOTES.md "v11" for why these two were chosen over
a shallow-MCTS third proposal that doesn't fit this game/budget)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggriculture.state import StateParser
from kaggriculture.strategy import StrategyAgent
from tests.fixtures.obs_basic import make_obs, empty_farm, plant_tile


def _obs_with_market_inventory(my_farm, day, wheat_inventory, seeds=None):
    obs = make_obs(my_farm=my_farm, day=day, seeds=seeds)
    obs["market"]["inventory"]["WHEAT"] = wheat_inventory
    return obs


def test_market_surge_further_discounts_a_crop_with_rising_inventory():
    """WHEAT inventory climbing fast (well above I0=10000, still rising)
    signals an incoming price crash regardless of who's causing it —
    should discount WHEAT's allocation relative to a flat-inventory
    baseline, even with zero opponent presence in either case."""
    my_farm_a = empty_farm(money=3000)
    strat_flat = StrategyAgent()
    for day in range(6):
        obs = _obs_with_market_inventory(my_farm_a, day, wheat_inventory=10000)
        plan_flat = strat_flat.plan(StateParser.parse(obs))

    my_farm_b = empty_farm(money=3000)
    strat_surging = StrategyAgent()
    for day in range(6):
        # Inventory climbs from 10000 to 10000 + day*300 (well above I0,
        # rising fast — normalized momentum = 300/T(=400) = 0.75/day).
        obs = _obs_with_market_inventory(my_farm_b, day, wheat_inventory=10000 + day * 300)
        plan_surging = strat_surging.plan(StateParser.parse(obs))

    assert plan_surging.crop_targets["WHEAT"] < plan_flat.crop_targets["WHEAT"]


def test_market_surge_shrinks_hold_floor_for_that_resource():
    my_farm = empty_farm(money=3000)
    strat = StrategyAgent()
    for day in range(6):
        obs = _obs_with_market_inventory(my_farm, day, wheat_inventory=10000 + day * 300)
        plan = strat.plan(StateParser.parse(obs))

    strat_flat = StrategyAgent()
    for day in range(6):
        obs = _obs_with_market_inventory(empty_farm(money=3000), day, wheat_inventory=10000)
        plan_flat = strat_flat.plan(StateParser.parse(obs))

    assert plan.sell_thresholds["WHEAT"] < plan_flat.sell_thresholds["WHEAT"]


def test_no_market_history_degrades_gracefully():
    """A freshly constructed StrategyAgent has no accumulated market
    history — momentum lookups must return 0.0, not raise."""
    obs = make_obs(my_farm=empty_farm(money=3000), day=0)
    plan = StrategyAgent().plan(StateParser.parse(obs))
    assert plan.crop_targets  # just must not crash


def test_expansion_requires_stricter_utilization_when_opponent_hasnt_expanded():
    """A farm that would clear the normal 70% expansion bar but not the
    stricter 85% opponent-mirror bar should be blocked from expanding
    while the opponent is still single-quadrant, and allowed once the
    opponent expands too."""
    def farm_at_utilization(pct, money=5000):
        farm = empty_farm(money=money)
        n = round(25 * pct)
        planted = 0
        for y in range(5):
            for x in range(5):
                if planted >= n:
                    break
                farm["tiles"][y][x] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
                planted += 1
        return farm

    my_farm = farm_at_utilization(0.75)  # clears 70%, not 85%

    opp_single_quadrant = empty_farm(money=3000)  # unlocked_quadrants defaults to ["NW"]
    obs_blocked = make_obs(my_farm=my_farm, opp_farm=opp_single_quadrant, day=8)
    plan_blocked = StrategyAgent().plan(StateParser.parse(obs_blocked))
    assert plan_blocked.buy_land is False

    opp_expanded = empty_farm(money=3000)
    opp_expanded["unlocked_quadrants"] = ["NW", "NE"]
    obs_allowed = make_obs(my_farm=farm_at_utilization(0.75), opp_farm=opp_expanded, day=8)
    plan_allowed = StrategyAgent().plan(StateParser.parse(obs_allowed))
    assert plan_allowed.buy_land is True
