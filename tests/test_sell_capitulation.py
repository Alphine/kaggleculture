"""Tests for v12's sell-capitulation scaling (NOTES.md "v12"): found via a
scripted single-crop opponent that HOLD_FLOOR_PCT's "wait for a recovery"
logic breaks down completely when both players concentrate on the same
crop, since the price then has no structural reason to recover. The more
of a resource already piled up unsold, the more willing we should be to
sell it anyway."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggriculture.state import StateParser
from kaggriculture.strategy import StrategyAgent
from kaggriculture.farm_ops import FarmOpsOrchestrator
from kaggriculture.config import SELL_CAPITULATION_QTY
from tests.fixtures.obs_basic import make_obs, empty_farm


def test_small_shed_pile_still_respects_hold_floor():
    """A small MELON pile (well under SELL_CAPITULATION_QTY) with price
    below the hold floor should still be held, same as pre-v12."""
    farm = empty_farm(money=3000)
    obs = make_obs(my_farm=farm, shed={"MELON": 3}, day=0)
    obs["market"]["prices"]["MELON"] = 1  # deeply below any reasonable hold floor
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    sell_orders = [o for o in result["market"] if o[0] == "SELL" and o[1] == "MELON"]
    assert sell_orders == []


def test_large_shed_pile_capitulates_and_sells_despite_depressed_price():
    """A MELON pile AT SELL_CAPITULATION_QTY should sell at ANY price —
    full capitulation, floor multiplier reaches 0."""
    farm = empty_farm(money=3000)
    obs = make_obs(my_farm=farm, shed={"MELON": SELL_CAPITULATION_QTY}, day=0)
    obs["market"]["prices"]["MELON"] = 1
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    sell_orders = [o for o in result["market"] if o[0] == "SELL" and o[1] == "MELON"]
    assert len(sell_orders) == 1
    assert sell_orders[0][2] == SELL_CAPITULATION_QTY


def test_capitulation_is_gradual_not_a_cliff():
    """Halfway to SELL_CAPITULATION_QTY, the effective hold floor should
    be roughly half the base floor — a partially-depressed price (below
    the full floor but above the halved one) should now sell, whereas a
    small pile at the same price would have held (previous test)."""
    farm = empty_farm(money=3000)
    half_qty = SELL_CAPITULATION_QTY // 2
    obs = make_obs(my_farm=farm, shed={"MELON": half_qty}, day=0)
    # MELON base=250, hold_pct~0.41 (v9 tuned) -> full floor ~$103;
    # halved (capitulation=0.5) -> ~$51. Price of $70 clears the halved
    # floor but not the full one.
    obs["market"]["prices"]["MELON"] = 70
    state = StateParser.parse(obs)
    plan = StrategyAgent().plan(state)
    result = FarmOpsOrchestrator().run(state, plan)
    sell_orders = [o for o in result["market"] if o[0] == "SELL" and o[1] == "MELON"]
    assert len(sell_orders) == 1
