import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggriculture.state import StateParser
from kaggriculture.strategy import StrategyAgent
from kaggriculture.farm_ops import FarmOpsOrchestrator
from kaggriculture.config import PHASE_ANIMAL_CAP
from tests.fixtures.obs_basic import make_obs, empty_farm
from tests.fixtures.obs_animals import structure_tile


def test_unit_carrying_animal_places_on_matching_empty_structure():
    farm = empty_farm()
    farm["farmer"] = [2, 2]
    farm["tiles"][2][2] = structure_tile(kind="COOP", animal=None)
    obs = make_obs(my_farm=farm, inventories=[{"GOOSE": 1}])
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["PLACE", "GOOSE"]


def test_unit_carrying_animal_walks_toward_empty_structure():
    farm = empty_farm()
    farm["farmer"] = [0, 0]
    farm["tiles"][2][2] = structure_tile(kind="COOP", animal=None)
    obs = make_obs(my_farm=farm, inventories=[{"GOOSE": 1}])
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] in (["EAST"], ["SOUTH"])


def test_fetch_animal_pickup_when_shed_has_animal_and_empty_structure():
    farm = empty_farm(money=3000)
    farm["farmer"] = [4, 4]
    farm["tiles"][1][1] = structure_tile(kind="COOP", animal=None)
    obs = make_obs(my_farm=farm, shed={"GOOSE": 1}, day=6)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["PICKUP", "GOOSE", 1]


def test_fetch_wheat_pickup_when_animal_unfed_and_shed_has_wheat():
    farm = empty_farm(money=3000)
    farm["farmer"] = [4, 4]
    farm["tiles"][1][1] = structure_tile(kind="COOP", animal="GOOSE", fed_today=False, consecutive_unfed=1)
    obs = make_obs(my_farm=farm, shed={"WHEAT": 5}, day=6)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["PICKUP", "WHEAT", 1]


def test_unit_carrying_wheat_feeds_unfed_animal_at_its_tile():
    farm = empty_farm()
    farm["farmer"] = [1, 1]
    farm["tiles"][1][1] = structure_tile(kind="COOP", animal="GOOSE", fed_today=False, consecutive_unfed=1)
    obs = make_obs(my_farm=farm, inventories=[{"WHEAT": 1}])
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["FEED"]


def test_unit_carrying_wheat_walks_toward_unfed_animal():
    farm = empty_farm()
    farm["farmer"] = [0, 0]
    farm["tiles"][3][3] = structure_tile(kind="COOP", animal="GOOSE", fed_today=False, consecutive_unfed=1)
    obs = make_obs(my_farm=farm, inventories=[{"WHEAT": 1}])
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] in (["EAST"], ["SOUTH"])


def test_farmer_harvests_animal_product():
    farm = empty_farm()
    farm["farmer"] = [1, 1]
    farm["tiles"][1][1] = structure_tile(kind="COOP", animal="GOOSE", fed_today=True, cared_today=True, yield_units=2)
    obs = make_obs(my_farm=farm)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["HARVEST"]


def test_farmer_cares_for_animal_when_no_harvest_pending():
    farm = empty_farm()
    farm["farmer"] = [1, 1]
    farm["tiles"][1][1] = structure_tile(kind="COOP", animal="GOOSE", fed_today=True, cared_today=False, yield_units=0)
    obs = make_obs(my_farm=farm)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["CARE"]


def test_farmer_collects_fertilizer_when_available():
    farm = empty_farm()
    farm["farmer"] = [1, 1]
    farm["tiles"][1][1] = structure_tile(
        kind="COOP", animal="GOOSE", fed_today=True, cared_today=True,
        yield_units=0, fertilizer_available=True,
    )
    obs = make_obs(my_farm=farm)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["COLLECT_FERTILIZER"]


def test_wheat_sell_order_reserves_feed_stock_for_owned_animals():
    # v21: farm filled with already-watered plants (not left empty) so
    # this realistically represents a day-6 farm, not an artificial
    # all-empty edge case — an all-empty farm at day 6 spikes hire_count
    # high enough (tighter buildout-phase capacity, NOTES.md "v21") to
    # crowd this test's SELL order out of MAX_MARKET_ORDERS_PER_TURN
    # entirely, which isn't the scenario this test is actually about.
    from tests.fixtures.obs_basic import plant_tile

    farm = empty_farm(money=3000)
    for y in range(5):
        for x in range(5):
            if (x, y) != (1, 1):
                farm["tiles"][y][x] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    farm["tiles"][1][1] = structure_tile(kind="COOP", animal="GOOSE", fed_today=True)
    obs = make_obs(my_farm=farm, shed={"WHEAT": 10}, day=6)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    sell_orders = [o for o in result["market"] if o[0] == "SELL" and o[1] == "WHEAT"]
    assert len(sell_orders) == 1
    assert sell_orders[0][2] == 8  # 10 in shed - 2 reserved (1 animal x2-day buffer)


def test_wheat_sell_order_holds_entirely_when_stock_at_or_below_reserve():
    farm = empty_farm(money=3000)
    farm["tiles"][1][1] = structure_tile(kind="COOP", animal="GOOSE", fed_today=True)
    obs = make_obs(my_farm=farm, shed={"WHEAT": 2}, day=6)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    sell_orders = [o for o in result["market"] if o[0] == "SELL" and o[1] == "WHEAT"]
    assert sell_orders == []


def test_animal_targets_never_exceed_phase_cap_regardless_of_cash():
    farm = empty_farm(money=1_000_000)
    obs = make_obs(my_farm=farm, day=20)  # scale phase
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    for animal, target in plan.animal_targets.items():
        assert target <= PHASE_ANIMAL_CAP["scale"][animal]


def test_critical_unfed_animal_gets_dedicated_rescuer_over_big_harvest():
    """A melon harvest (base_price 250 x yield) would out-score a routine
    FETCH_WHEAT in the generic candidate pool — the dedicated rescue stage
    (M6 hardening) must grab a unit for feeding regardless, since losing the
    animal (sunk cost + future production) outweighs one turn's harvest."""
    from tests.fixtures.obs_basic import plant_tile

    farm = empty_farm(money=3000)
    farm["farmer"] = [4, 4]
    farm["hands"] = [[0, 0]]
    farm["tiles"][1][1] = structure_tile(kind="COOP", animal="GOOSE", fed_today=False, consecutive_unfed=1)
    farm["tiles"][0][0] = plant_tile(crop="MELON", planted_day=0, watered_today=True, yield_units=6)
    obs = make_obs(my_farm=farm, shed={"WHEAT": 5}, day=11, inventories=[{}, {}])
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    assert result["farmer"] == ["PICKUP", "WHEAT", 1]


def test_endgame_liquidation_ignores_hold_floor_near_season_end():
    farm = empty_farm(money=3000)
    obs = make_obs(my_farm=farm, shed={"STRAWBERRY": 5}, day=29)  # remaining_days=1
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    assert all(pct == 0.0 for pct in plan.sell_thresholds.values())
    result = FarmOpsOrchestrator().run(state, plan)
    assert ["SELL", "STRAWBERRY", 5] in result["market"]


def test_no_hiring_near_season_end_on_a_fully_empty_farm():
    """v15: hiring is no longer blanket-forced to 0 during liquidation
    (see test_hiring_still_happens_during_liquidation_for_existing_plants
    below for why) — but a genuinely empty farm with nothing left to
    plant (remaining_days too few for any crop to fit) still correctly
    computes 0 hire need on its own, since there's nothing for a hand to
    do either way."""
    farm = empty_farm(money=3000)
    obs = make_obs(my_farm=farm, day=29)  # remaining_days=1, no crop fits
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    assert plan.hire_count == 0


def test_hiring_still_happens_during_liquidation_for_existing_plants():
    """v15: found via a real-match replay (NOTES.md "v15") — the OLD
    blanket `hire_count = 0` during LIQUIDATION_DAYS_REMAINING dropped a
    wide, multi-quadrant farm to 0 hands right as the season ended, and
    weed events exploded in exactly those final days, destroying crops
    that could otherwise still have been harvested and sold. A farm with
    existing unwatered plants (not new planting — crop_targets is empty
    this late) should still want hands to save them."""
    from tests.fixtures.obs_basic import plant_tile

    farm = empty_farm(money=3000)
    for y in range(5):
        for x in range(5):
            farm["tiles"][y][x] = plant_tile(crop="WHEAT", planted_day=27, watered_today=False)
    obs = make_obs(my_farm=farm, day=29)  # remaining_days=1, liquidation window
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    assert plan.crop_targets == {}  # confirms this is really the "nothing new to plant" case
    assert plan.hire_count > 0
