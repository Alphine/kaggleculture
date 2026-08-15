"""Tests for v14's multi-expert regime switching (NOTES.md "v14"): the
orchestrator classifies our visible-footprint standing each day
(ahead/neutral/behind) and biases the economist's risk posture accordingly,
since the competition's rating only cares about win/lose/tie, not margin
(kaggriculture-agent-SDD.md) — a moderate loss costs the same as a huge
one, so it's worth taking more variance to escape a likely loss."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import kaggriculture.strategy as strategy
from kaggriculture.config import HOLD_FLOOR_PCT
from kaggriculture.state import StateParser
from kaggriculture.strategy import StrategyAgent
from tests.fixtures.obs_basic import make_obs, empty_farm, plant_tile


def _flood_melon(farm, n_tiles=20):
    planted = 0
    for y in range(5):
        for x in range(5):
            if planted >= n_tiles:
                return
            farm["tiles"][y][x] = plant_tile(crop="MELON", planted_day=0, watered_today=True)
            planted += 1


def test_regime_classified_behind_when_opponent_footprint_much_larger():
    my_farm = empty_farm(money=3000)
    my_farm["tiles"][0][0] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    opp_farm = empty_farm(money=3000)
    _flood_melon(opp_farm, 20)
    obs = make_obs(my_farm=my_farm, opp_farm=opp_farm, day=0)
    plan = StrategyAgent().plan(StateParser.parse(obs))
    assert plan.regime == "behind"


def test_regime_classified_ahead_when_our_footprint_much_larger():
    my_farm = empty_farm(money=3000)
    _flood_melon(my_farm, 20)
    opp_farm = empty_farm(money=3000)
    opp_farm["tiles"][0][0] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    obs = make_obs(my_farm=my_farm, opp_farm=opp_farm, day=0)
    plan = StrategyAgent().plan(StateParser.parse(obs))
    assert plan.regime == "ahead"


def test_regime_neutral_when_footprints_are_close():
    my_farm = empty_farm(money=3000)
    my_farm["tiles"][0][0] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    opp_farm = empty_farm(money=3000)
    opp_farm["tiles"][0][0] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    obs = make_obs(my_farm=my_farm, opp_farm=opp_farm, day=0)
    plan = StrategyAgent().plan(StateParser.parse(obs))
    assert plan.regime == "neutral"


def test_regime_neutral_when_either_side_has_zero_footprint():
    """Opponent literally hasn't planted anything yet (common in early game
    or a fresh test fixture) shouldn't be read as automatic "ahead" — it's
    a data-insufficiency case, not real evidence of a lead."""
    my_farm = empty_farm(money=3000)  # fully empty -> our footprint is 0
    opp_farm = empty_farm(money=3000)
    opp_farm["tiles"][0][0] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    obs = make_obs(my_farm=my_farm, opp_farm=opp_farm, day=0)
    plan = StrategyAgent().plan(StateParser.parse(obs))
    assert plan.regime == "neutral"


def test_behind_regime_lowers_sell_hold_floor():
    my_farm = empty_farm(money=3000)
    my_farm["tiles"][0][0] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    opp_farm = empty_farm(money=3000)
    _flood_melon(opp_farm, 20)
    obs = make_obs(my_farm=my_farm, opp_farm=opp_farm, day=0)
    plan = StrategyAgent().plan(StateParser.parse(obs))
    assert plan.regime == "behind"
    assert plan.sell_thresholds["WHEAT"] < HOLD_FLOOR_PCT["WHEAT"]


def test_ahead_regime_raises_sell_hold_floor():
    my_farm = empty_farm(money=3000)
    _flood_melon(my_farm, 20)
    opp_farm = empty_farm(money=3000)
    opp_farm["tiles"][0][0] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    obs = make_obs(my_farm=my_farm, opp_farm=opp_farm, day=0)
    plan = StrategyAgent().plan(StateParser.parse(obs))
    assert plan.regime == "ahead"
    assert plan.sell_thresholds["WHEAT"] > HOLD_FLOOR_PCT["WHEAT"]


def test_regime_switching_disabled_falls_back_to_neutral(monkeypatch):
    monkeypatch.setattr(strategy, "ENABLE_REGIME_SWITCHING", False)
    my_farm = empty_farm(money=3000)
    my_farm["tiles"][0][0] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    opp_farm = empty_farm(money=3000)
    _flood_melon(opp_farm, 20)  # would classify "behind" if the flag were on
    obs = make_obs(my_farm=my_farm, opp_farm=opp_farm, day=0)
    plan = StrategyAgent().plan(StateParser.parse(obs))
    assert plan.regime == "neutral"


def test_concentration_mult_sharpens_top_crop_share_without_changing_rank():
    """concentration_mult (>1, "behind" preset) should widen the gap between
    the top crop's tile share and the rest, relative to concentration_mult=1
    (neutral), without flipping which crop ranks first — isolated at the
    _decide_crop_targets level so only concentration_mult varies between
    the two calls, everything else (opponent counts, phase, remaining days)
    held identical.

    v15: run at day=10 (buildout, not early) — EARLY_PHASE_MELON_TILE_CAP's
    unconditional WHEAT top-up (NOTES.md "v15") would otherwise apply
    identically regardless of concentration_mult and mask its effect."""
    my_farm = empty_farm(money=3000)
    obs = make_obs(my_farm=my_farm, day=10)
    state = StateParser.parse(obs)
    phase = strategy.phase_for_day(state.day)
    remaining_days = 30 - state.day

    targets_neutral = strategy._decide_crop_targets(state, phase, remaining_days, {}, {}, concentration_mult=1.0)
    targets_behind = strategy._decide_crop_targets(state, phase, remaining_days, {}, {}, concentration_mult=1.20)

    top_crop = max(targets_neutral, key=targets_neutral.get)
    assert max(targets_behind, key=targets_behind.get) == top_crop
    neutral_share = targets_neutral[top_crop] / sum(targets_neutral.values())
    behind_share = targets_behind[top_crop] / sum(targets_behind.values())
    assert behind_share > neutral_share
