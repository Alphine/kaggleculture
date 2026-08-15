import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggriculture.state import StateParser
from kaggriculture.strategy import StrategyAgent, DayPlan
from kaggriculture.farm_ops import FarmOpsOrchestrator, step_toward, manhattan
from tests.fixtures.obs_basic import make_obs, empty_farm, plant_tile


def test_step_toward_moves_closer():
    assert step_toward((0, 0), (3, 0)) == "EAST"
    assert step_toward((3, 0), (0, 0)) == "WEST"
    assert step_toward((0, 0), (0, 3)) == "SOUTH"
    assert step_toward((0, 3), (0, 0)) == "NORTH"
    assert step_toward((2, 2), (2, 2)) == "PASS"


def test_manhattan():
    assert manhattan((0, 0), (3, 4)) == 7


def test_farmer_buys_and_plants_wheat_on_first_turn():
    obs = make_obs(seeds={"WHEAT": 1})
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["PLANT", "WHEAT"]
    # v13: MELON is now also eligible in the early phase and competes for
    # the same seed budget, so the exact WHEAT quantity shifts with
    # tuning; assert the buy happens without hardcoding the amount.
    wheat_buys = [o for o in result["market"] if o[0] == "BUY_SEED" and o[1] == "WHEAT"]
    assert len(wheat_buys) == 1
    assert wheat_buys[0][2] > 0


def test_farmer_waters_unwatered_wheat():
    farm = empty_farm()
    farm["tiles"][4][4] = plant_tile(crop="WHEAT", planted_day=0, watered_today=False)
    obs = make_obs(my_farm=farm, day=1)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["WATER"]


def test_farmer_harvests_ready_wheat():
    farm = empty_farm()
    farm["tiles"][4][4] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True, yield_units=2)
    obs = make_obs(my_farm=farm, day=2)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["HARVEST"]


def test_farmer_sells_shed_wheat():
    obs = make_obs(shed={"WHEAT": 3})
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert ["SELL", "WHEAT", 3] in result["market"]


def test_critical_unwatered_plant_gets_dedicated_rescuer_over_big_harvests():
    """Root-caused via a real leaderboard replay + a fixed-seed local
    trace: 3 ready melon harvests (value ~1500 each -> score ~6000) would
    consume every idle unit in the generic scorer, since a routine WATER's
    score (~500, even with its urgency bonus) never comes close. A wheat
    tile sitting at consecutive_unwatered=1 could then go completely
    unattended for a full day despite units being available. The dedicated
    rescue stage (bypassing the scorer) must guarantee it gets a unit
    regardless of how many high-value harvests are competing."""
    farm = empty_farm(money=3000)
    farm["farmer"] = [0, 0]
    farm["hands"] = [[1, 0], [2, 0]]
    # MELON first_yield_day=10, so age (day - planted_day) must be >= 10
    # for HARVEST to be a candidate at all.
    farm["tiles"][0][0] = plant_tile(crop="MELON", planted_day=5, watered_today=True, yield_units=6)
    farm["tiles"][0][1] = plant_tile(crop="MELON", planted_day=5, watered_today=True, yield_units=6)
    farm["tiles"][0][2] = plant_tile(crop="MELON", planted_day=5, watered_today=True, yield_units=6)
    farm["tiles"][4][4] = plant_tile(crop="WHEAT", planted_day=10, watered_today=False, consecutive_unwatered=1)
    obs = make_obs(my_farm=farm, day=15, inventories=[{}, {}, {}])
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)

    all_actions = [result["farmer"]] + result["hands"]
    harvest_count = sum(1 for a in all_actions if a == ["HARVEST"])
    # Exactly 2 of the 3 units should harvest (the closest-to-wheat unit
    # gets dedicated to the rescue instead of its own adjacent melon).
    assert harvest_count == 2
    assert result["hands"][1] == ["SOUTH"]  # unit at (2,0), nearest to (4,4), heads there


def test_multi_crop_planting_lands_different_crops_in_the_same_turn():
    """v13: with ENABLE_MULTI_CROP_PLANTING, seed-capped WHEAT (only 1 in
    stock) shouldn't block OTHER idle units from planting a different
    crop (CARROT, plenty in stock) on the same turn — each unit standing
    on its own empty tile with 0 distance for either crop. Pre-v13, a
    single best-by-ratio crop was chosen for the WHOLE turn, so the 2
    units whose WHEAT candidate got rejected by the seed cap would have
    fallen through to PASS instead of planting CARROT."""
    farm = empty_farm(money=3000)
    farm["farmer"] = [0, 0]
    farm["hands"] = [[1, 0], [2, 0]]
    obs = make_obs(my_farm=farm, seeds={"WHEAT": 1, "CARROT": 5}, day=0, inventories=[{}, {}, {}])
    state = StateParser.parse(obs)
    plan = DayPlan(
        day=0, phase="early",
        crop_targets={"WHEAT": 10, "CARROT": 10},
        animal_targets={}, structure_targets={"COOP": 0, "PASTURE": 0},
        hire_count=0, buy_land=False,
        sell_thresholds={},
    )
    result = FarmOpsOrchestrator().run(state, plan)
    all_actions = [result["farmer"]] + result["hands"]
    planted_crops = [a[1] for a in all_actions if a[0] == "PLANT"]
    assert "WHEAT" in planted_crops
    assert "CARROT" in planted_crops
    assert len(planted_crops) == 3  # all 3 units got a planting action, none fell through to PASS


def test_carrying_unit_returns_to_shed():
    farm = empty_farm()
    farm["farmer"] = [0, 0]
    obs = make_obs(my_farm=farm, inventories=[{"WHEAT": 2}])
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["EAST"]
