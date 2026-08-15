import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggriculture.state import StateParser, PlantTile, WeedTile
from tests.fixtures.obs_basic import make_obs, empty_farm, plant_tile, weed_tile


def test_parse_empty_farm():
    obs = make_obs()
    state = StateParser.parse(obs)
    assert state.my_farm.money == 3000
    assert state.my_farm.farmer == (4, 4)
    assert state.my_farm.tiles[4][4] is None
    assert state.my_farm.tiles[0][5] == "LOCKED"


def test_parse_plant_tile():
    farm = empty_farm()
    farm["tiles"][2][2] = plant_tile(crop="WHEAT", planted_day=0, watered_today=True)
    obs = make_obs(my_farm=farm)
    state = StateParser.parse(obs)
    tile = state.my_farm.tiles[2][2]
    assert isinstance(tile, PlantTile)
    assert tile.crop == "WHEAT"
    assert tile.watered_today is True


def test_parse_weed_tile():
    farm = empty_farm()
    farm["tiles"][3][3] = weed_tile()
    obs = make_obs(my_farm=farm)
    state = StateParser.parse(obs)
    assert isinstance(state.my_farm.tiles[3][3], WeedTile)


def test_shed_adjacent():
    obs = make_obs()
    state = StateParser.parse(obs)
    assert state.is_shed_adjacent((4, 4)) is True
    assert state.is_shed_adjacent((0, 0)) is False


def test_player_index_swaps_farms():
    farm_a = empty_farm(money=100)
    farm_b = empty_farm(money=200)
    obs = make_obs(player=1, my_farm=farm_a, opp_farm=farm_b)
    state = StateParser.parse(obs)
    assert state.my_farm.money == 100
    assert state.opp_farm.money == 200
