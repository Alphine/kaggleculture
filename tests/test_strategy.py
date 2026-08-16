import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggriculture.config import PHASE_DAY_BOUNDARIES
from kaggriculture.state import StateParser
from kaggriculture.strategy import StrategyAgent, phase_for_day, fib_cost
from tests.fixtures.obs_basic import make_obs, empty_farm, plant_tile


def test_phase_boundaries():
    # Reads live from config.PHASE_DAY_BOUNDARIES (an M5 tuning target, §8.6)
    # rather than hardcoding day numbers, so this test asserts the boundary
    # LOGIC and survives tune.py updating the actual cutoff days.
    early_end = PHASE_DAY_BOUNDARIES["early_end"]
    buildout_end = PHASE_DAY_BOUNDARIES["buildout_end"]
    scale_end = PHASE_DAY_BOUNDARIES["scale_end"]

    assert phase_for_day(0) == "early"
    assert phase_for_day(early_end) == "early"
    assert phase_for_day(early_end + 1) == "buildout"
    assert phase_for_day(buildout_end) == "buildout"
    assert phase_for_day(buildout_end + 1) == "scale"
    assert phase_for_day(scale_end) == "scale"
    assert phase_for_day(scale_end + 1) == "endgame"
    assert phase_for_day(29) == "endgame"


def test_fib_cost_sequence():
    assert [fib_cost(i) for i in range(7)] == [1, 1, 2, 3, 5, 8, 13]


def test_early_phase_targets_only_the_early_phase_crop_pool():
    """v13: MELON was moved into PHASE_CROP_POOL["early"] (NOTES.md "v13")
    after a scripted-opponent benchmark showed delaying MELON entry to
    "buildout" cost the first-mover pricing advantage window (0/8 -> 8/8)."""
    obs = make_obs(day=0)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    assert plan.phase == "early"
    assert set(plan.crop_targets) <= {"WHEAT", "CARROT", "MELON"}
    assert sum(plan.crop_targets.values()) == 25  # all 25 unlocked NW tiles allocated


def test_endgame_skips_crops_that_cant_mature():
    farm = empty_farm(money=3000)
    obs = make_obs(day=28, my_farm=farm)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    assert plan.phase == "endgame"
    # remaining_days = 30-28 = 2, only WHEAT/CARROT (first_yield_day=2) qualify
    assert set(plan.crop_targets) <= {"WHEAT", "CARROT"}


def test_endgame_near_season_end_plants_nothing():
    farm = empty_farm(money=3000)
    obs = make_obs(day=29, my_farm=farm)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    # remaining_days = 1, no crop has first_yield_day <= 1
    assert plan.crop_targets == {}


def test_land_expansion_triggers_when_utilized_and_affordable():
    farm = empty_farm(money=5000)
    for y in range(5):
        for x in range(5):
            farm["tiles"][y][x] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    obs = make_obs(day=7, my_farm=farm)  # day >= MIN_DAY_FOR_FIRST_EXPANSION
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    assert plan.buy_land is True


def test_land_expansion_skipped_before_min_day_regardless_of_utilization():
    """A real leaderboard replay traced a cash crash straight to a
    day-4-5 BUY_LAND landing before any harvest revenue had banked —
    utilization alone can hit 70% within days on a wide multi-crop
    portfolio. This is a flat floor independent of utilization/money."""
    farm = empty_farm(money=5000)
    for y in range(5):
        for x in range(5):
            farm["tiles"][y][x] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    obs = make_obs(day=5, my_farm=farm)  # fully utilized, affordable, but too early
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    assert plan.buy_land is False


def test_land_expansion_skipped_when_underutilized():
    farm = empty_farm(money=5000)  # all 25 tiles empty -> 0% utilization
    obs = make_obs(day=7, my_farm=farm)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    assert plan.buy_land is False


def test_land_expansion_skipped_near_season_end():
    farm = empty_farm(money=5000)
    for y in range(5):
        for x in range(5):
            farm["tiles"][y][x] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    obs = make_obs(day=27, my_farm=farm)  # remaining_days=3 < MIN_DAYS_REMAINING_FOR_EXPANSION
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    assert plan.buy_land is False


def test_land_expansion_skipped_when_weed_infested():
    """A real leaderboard replay caught this: our farm hit 31% weeds after
    over-expanding, but the utilization metric (which used to count weeds
    as 'utilized') stayed high enough to keep triggering the next BUY_LAND.
    Weeds must block further expansion, not encourage it."""
    from tests.fixtures.obs_basic import weed_tile

    farm = empty_farm(money=5000)
    for y in range(5):
        for x in range(5):
            if x == 0:
                farm["tiles"][y][x] = weed_tile()  # 5/25 = 20% weeds
            else:
                farm["tiles"][y][x] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    obs = make_obs(day=6, my_farm=farm)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    assert plan.buy_land is False


def test_opponent_flooding_a_crop_reduces_its_relative_allocation():
    """CROP_TARGET_WEIGHT now biases WHEAT so heavily over CARROT (v9:
    1.5 vs 0.3, a 5x gap — see NOTES.md) that a single full-quadrant
    opponent flood can no longer fully flip the ranking the way it could
    at v8's gentler 2x gap. Test the mechanism directionally instead of by
    absolute ranking, so it stays valid regardless of how
    CROP_TARGET_WEIGHT gets tuned later: flooding WHEAT should still
    measurably shrink WHEAT's share and grow CARROT's, relative to an
    unflooded baseline (SDD §5.5).

    v13: MELON joined PHASE_CROP_POOL["early"] and, thanks to its much
    higher CROP_TARGET_WEIGHT, now absorbs almost all of the allocation
    WHEAT loses (CARROT stays pinned at its floor of 1 in both cases), so
    the "grows" side of the assertion checks MELON instead of CARROT.

    v15: run at day=10 (buildout, not early) — EARLY_PHASE_MELON_TILE_CAP
    now caps MELON's early-phase share and redirects the overflow to
    WHEAT unconditionally (NOTES.md "v15": uncapped early MELON crowded
    WHEAT down to a handful of tiles and crashed 7/10 seeds into a
    cash-crunch/weed spiral even vs a passive "pass" opponent), which
    would otherwise mask the flood-penalty mechanism this test targets."""
    my_farm = empty_farm(money=3000)

    obs_base = make_obs(my_farm=my_farm, day=10)
    plan_base = StrategyAgent().plan(StateParser.parse(obs_base))

    opp_farm = empty_farm(money=3000)
    for y in range(5):
        for x in range(5):
            opp_farm["tiles"][y][x] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    obs_flood = make_obs(my_farm=my_farm, opp_farm=opp_farm, day=10)
    plan_flood = StrategyAgent().plan(StateParser.parse(obs_flood))

    assert plan_flood.crop_targets["WHEAT"] < plan_base.crop_targets["WHEAT"]
    assert plan_flood.crop_targets["MELON"] > plan_base.crop_targets["MELON"]


def test_no_opponent_flooding_allocates_by_crop_target_weight():
    """WHEAT and CARROT's raw profit-per-day scores are tied (7.5 each),
    but CROP_TARGET_WEIGHT intentionally biases WHEAT over CARROT
    (config.py v31: 2.0 vs 0.3, a ~6.7x weight gap) — v9's original 5x
    bias was recalibrated in v31 against the real #1-leaderboard team's
    replays (NOTES.md "v31"), which showed a heavier sustained WHEAT
    presence than our old weight gave it. Without any opponent-flood
    signal, allocation should follow that weight, not split evenly.

    v15: run at day=10 (buildout, not early) so EARLY_PHASE_MELON_TILE_CAP's
    unconditional WHEAT top-up (NOTES.md "v15") doesn't mask the
    weight-driven ratio this test targets."""
    my_farm = empty_farm(money=3000)
    obs = make_obs(my_farm=my_farm, day=10)  # opponent farm is empty by default
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    ratio = plan.crop_targets["WHEAT"] / plan.crop_targets["CARROT"]
    assert 6.0 <= ratio <= 11.0


def test_hire_count_zero_on_fully_watered_small_farm():
    farm = empty_farm(money=3000)
    for y in range(5):
        for x in range(5):
            farm["tiles"][y][x] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    obs = make_obs(day=1, my_farm=farm)
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    assert plan.hire_count == 0
