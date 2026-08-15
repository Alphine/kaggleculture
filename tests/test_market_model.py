import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggriculture.market_model import price_at, max_sell_before_floor
from kaggriculture.config import MARKET_PARAMS


def test_price_at_i0_equals_base():
    for resource, params in MARKET_PARAMS.items():
        assert price_at(resource, params["I0"]) == params["base"]


def test_wheat_checkpoints_match_readme_table():
    i0, t = 10000, 400
    assert price_at("WHEAT", i0 - t) == 45
    assert price_at("WHEAT", i0 + t) == 20
    assert price_at("WHEAT", i0 + 2 * t) == 19


def test_melon_checkpoints_match_readme_table():
    i0, t = 10000, 300
    assert price_at("MELON", i0 - t) == 300
    assert price_at("MELON", i0 + t) == 1
    assert price_at("MELON", i0 + 2 * t) == 1


def test_carrot_checkpoints_match_readme_table():
    i0, t = 10000, 450
    assert price_at("CARROT", i0 - t) == 42
    assert price_at("CARROT", i0 + t) == 10
    assert price_at("CARROT", i0 + 2 * t) == 1


def test_fertilizer_checkpoints_match_readme_table():
    i0, t = 10000, 200
    assert price_at("FERTILIZER", i0 - t) == 140
    assert price_at("FERTILIZER", i0 + t) == 60
    assert price_at("FERTILIZER", i0 + 2 * t) == 20


def test_price_floored_at_one_dollar():
    assert price_at("MELON", 10000 + 100000) == 1


def test_max_sell_before_floor_shrinks_as_floor_rises():
    i0 = MARKET_PARAMS["WHEAT"]["I0"]
    generous = max_sell_before_floor("WHEAT", i0, floor_price=1)
    strict = max_sell_before_floor("WHEAT", i0, floor_price=24)
    assert generous > strict
    assert strict >= 0


def test_max_sell_before_floor_zero_when_already_below_floor():
    i0, t = 10000, 300
    assert max_sell_before_floor("MELON", i0 + t, floor_price=100) == 0
