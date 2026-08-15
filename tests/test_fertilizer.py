import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggriculture.state import StateParser
from kaggriculture.strategy import StrategyAgent
from kaggriculture.farm_ops import FarmOpsOrchestrator
from kaggriculture.config import FERTILIZER_RESERVE_TARGET
from tests.fixtures.obs_basic import make_obs, empty_farm, plant_tile


def test_fetch_fertilizer_when_eligible_crop_and_shed_has_fertilizer():
    farm = empty_farm(money=3000)
    farm["farmer"] = [4, 4]
    # CARROT: max_yield_day=3, bonus window starts at ceil(3/2)=2; age=2 is
    # in-window. yield_units=0 keeps HARVEST from also being a candidate
    # here, isolating the fertilize-fetch behavior.
    farm["tiles"][1][1] = plant_tile(crop="CARROT", planted_day=0, watered_today=True, yield_units=0)
    obs = make_obs(my_farm=farm, shed={"FERTILIZER": 2}, day=2)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["PICKUP", "FERTILIZER", 1]


def test_unit_carrying_fertilizer_fertilizes_at_eligible_tile():
    farm = empty_farm()
    farm["farmer"] = [1, 1]
    farm["tiles"][1][1] = plant_tile(crop="CARROT", planted_day=0, watered_today=True, yield_units=0)
    obs = make_obs(my_farm=farm, day=2, inventories=[{"FERTILIZER": 1}])
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["FERTILIZE"]


def test_unit_carrying_fertilizer_walks_toward_eligible_tile():
    farm = empty_farm()
    farm["farmer"] = [0, 0]
    farm["tiles"][3][3] = plant_tile(crop="CARROT", planted_day=0, watered_today=True, yield_units=0)
    obs = make_obs(my_farm=farm, day=2, inventories=[{"FERTILIZER": 1}])
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] in (["EAST"], ["SOUTH"])


def test_unit_carrying_fertilizer_returns_to_shed_when_nothing_eligible():
    farm = empty_farm()
    farm["farmer"] = [0, 0]
    obs = make_obs(my_farm=farm, day=2, inventories=[{"FERTILIZER": 1}])
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] in (["EAST"], ["SOUTH"])  # heading toward nearest shed tile


def test_one_time_crop_not_eligible_before_bonus_window():
    farm = empty_farm()
    farm["farmer"] = [4, 4]
    # WHEAT bonus window starts at ceil(4/2)=2; age=0 is before it.
    farm["tiles"][1][1] = plant_tile(crop="WHEAT", planted_day=2, watered_today=True, yield_units=0)
    obs = make_obs(my_farm=farm, shed={"FERTILIZER": 2}, day=2)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] != ["PICKUP", "FERTILIZER", 1]


def test_already_fertilized_tile_not_eligible_again():
    farm = empty_farm(money=3000)
    farm["farmer"] = [4, 4]
    farm["tiles"][1][1] = plant_tile(
        crop="CARROT", planted_day=0, watered_today=True, yield_units=0, fertilized_until_day=5,
    )
    obs = make_obs(my_farm=farm, shed={"FERTILIZER": 2}, day=2)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] != ["PICKUP", "FERTILIZER", 1]


def test_ongoing_crop_always_eligible_when_unfertilized():
    farm = empty_farm(money=3000)
    farm["farmer"] = [4, 4]
    farm["tiles"][1][1] = plant_tile(crop="TOMATO", planted_day=0, watered_today=True, yield_units=0)
    obs = make_obs(my_farm=farm, shed={"FERTILIZER": 2}, day=1)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["PICKUP", "FERTILIZER", 1]


def test_wheat_sell_order_reserves_fertilizer_stock_for_fertilizing():
    farm = empty_farm(money=3000)
    obs = make_obs(my_farm=farm, shed={"FERTILIZER": 10}, day=2)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    sell_orders = [o for o in result["market"] if o[0] == "SELL" and o[1] == "FERTILIZER"]
    assert len(sell_orders) == 1
    assert sell_orders[0][2] == 10 - FERTILIZER_RESERVE_TARGET


def test_market_buys_fertilizer_when_below_reserve_and_crops_active():
    farm = empty_farm(money=3000)
    obs = make_obs(my_farm=farm, day=2)  # early phase always has active crop_targets
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    buy_orders = [o for o in result["market"] if o[0] == "BUY_PRODUCT" and o[1] == "FERTILIZER"]
    assert len(buy_orders) == 1
    assert buy_orders[0][2] == FERTILIZER_RESERVE_TARGET
