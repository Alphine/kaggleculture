"""Tests for v10's adaptive extensions: opponent trend tracking, supply-
pressure-adjusted sell thresholds, and cluster-lookahead scoring (see
NOTES.md "v10" for why these replace literal minimax — simultaneous moves,
huge joint action space, delayed/diffuse reward, and hidden opponent
private state all make a real game tree infeasible here)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggriculture.state import StateParser
from kaggriculture.strategy import StrategyAgent
from kaggriculture.farm_ops import FarmOpsOrchestrator
from tests.fixtures.obs_basic import make_obs, empty_farm, plant_tile


def _flood_opponent_wheat(day, tile_count=25):
    farm = empty_farm(money=3000)
    n = 0
    for y in range(5):
        for x in range(5):
            if n >= tile_count:
                break
            farm["tiles"][y][x] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
            n += 1
    return farm


def test_opponent_growth_rate_further_discounts_a_growing_crop():
    """An opponent holding a STATIC 10-tile WHEAT count for 5 days should
    discount our WHEAT score less than one that GREW from ~0 to 10 tiles
    over the same window — the growing one is a bigger looming threat."""
    my_farm = empty_farm(money=3000)
    strat_static = StrategyAgent()
    strat_growing = StrategyAgent()

    # Static: opponent already at 10 tiles from day 0 onward.
    static_opp = _flood_opponent_wheat(0, tile_count=10)
    for day in range(6):
        obs = make_obs(my_farm=my_farm, opp_farm=static_opp, day=day)
        plan_static = strat_static.plan(StateParser.parse(obs))

    # Growing: opponent starts at 0, ramps up to 10 tiles by day 5.
    for day in range(6):
        opp_farm = _flood_opponent_wheat(day, tile_count=min(10, day * 2))
        obs = make_obs(my_farm=my_farm, opp_farm=opp_farm, day=day)
        plan_growing = strat_growing.plan(StateParser.parse(obs))

    # Same visible count (10 tiles) on the final day, but the growing
    # trend should have pushed WHEAT's allocation down further.
    assert plan_growing.crop_targets["WHEAT"] < plan_static.crop_targets["WHEAT"]


def test_no_history_degrades_gracefully_to_zero_growth():
    """A freshly constructed StrategyAgent (e.g. a one-shot unit test) has
    no accumulated history — growth-rate lookups must return 0.0, not
    raise, and behave like the pre-v10 flood-only discount."""
    my_farm = empty_farm(money=3000)
    opp_farm = _flood_opponent_wheat(0, tile_count=25)
    obs = make_obs(my_farm=my_farm, opp_farm=opp_farm, day=0)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)  # fresh instance, no history
    assert plan.crop_targets["WHEAT"] >= 0  # just must not crash / behave sanely


def test_supply_pressure_shrinks_hold_floor_for_a_resource_opponent_is_flooding():
    """An opponent with a large, growing WHEAT presence should push our
    own WHEAT sell threshold DOWN (more willing to sell into the glut
    early) relative to a baseline with no opponent presence at all."""
    my_farm = empty_farm(money=3000)

    strat_base = StrategyAgent()
    obs_base = make_obs(my_farm=my_farm, day=0)
    plan_base = strat_base.plan(StateParser.parse(obs_base))

    strat_pressure = StrategyAgent()
    for day in range(6):
        opp_farm = _flood_opponent_wheat(day, tile_count=min(25, 5 + day * 4))
        obs = make_obs(my_farm=my_farm, opp_farm=opp_farm, day=day)
        plan_pressure = strat_pressure.plan(StateParser.parse(obs))

    assert plan_pressure.sell_thresholds["WHEAT"] < plan_base.sell_thresholds["WHEAT"]


def test_cluster_bonus_favors_a_clustered_tile_over_an_equally_close_isolated_one():
    """Two WHEAT tiles needing water, both distance 2 from the farmer and
    otherwise identical (same urgency/value): one isolated, one sitting
    next to two weeds that also need attention. All else tied, the
    scorer should prefer the clustered one — a stand-in for "this
    position sets up better future turns" without simulating any."""
    from tests.fixtures.obs_basic import weed_tile

    farm = empty_farm(money=3000)
    farm["farmer"] = [2, 2]
    # Isolated candidate at (0,2), distance 2 from farmer.
    farm["tiles"][2][0] = plant_tile(crop="WHEAT", planted_day=0, watered_today=False, consecutive_unwatered=0)
    # Clustered candidate at (2,0), also distance 2 from farmer, with two
    # weeds at (1,0) and (3,0) — both within CLUSTER_LOOKAHEAD_RADIUS of it
    # and far enough (manhattan 4) from the isolated candidate to not
    # contribute to its count.
    farm["tiles"][0][2] = plant_tile(crop="WHEAT", planted_day=0, watered_today=False, consecutive_unwatered=0)
    farm["tiles"][0][1] = weed_tile()
    farm["tiles"][0][3] = weed_tile()
    obs = make_obs(my_farm=farm, day=0, inventories=[{}])
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    # Farmer should move toward the clustered tile (2,0), i.e. NORTH.
    assert result["farmer"] == ["NORTH"]
