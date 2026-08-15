"""Synthetic obs fixtures for unit tests (SDD §12: pure-function tests first)."""


def empty_farm(money=3000):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    for y in range(10):
        for x in range(10):
            if x >= 5 or y >= 5:
                tiles[y][x] = "LOCKED"
    return {
        "money": money,
        "tiles": tiles,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }


def make_obs(player=0, day=0, hour=0, step=0, my_farm=None, opp_farm=None,
             shed=None, seeds=None, inventories=None):
    my_farm = my_farm or empty_farm()
    opp_farm = opp_farm or empty_farm()
    farms = [my_farm, opp_farm] if player == 0 else [opp_farm, my_farm]
    return {
        "player": player,
        "step": step,
        "day": day,
        "hour": hour,
        "farms": farms,
        "market": {
            "inventory": {"WHEAT": 10000, "CARROT": 10000},
            "prices": {"WHEAT": 25, "CARROT": 35},
        },
        "town": {"unlocked_shops": []},
        "private": {
            "shed": shed or {},
            "seeds": seeds or {},
            "inventories": inventories or [{}],
        },
    }


def plant_tile(crop="WHEAT", planted_day=0, watered_today=False, consecutive_unwatered=1,
               yield_units=1, max_lifespan_step=200, fertilized_until_day=-1):
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": planted_day,
        "watered_today": watered_today,
        "consecutive_unwatered": consecutive_unwatered,
        "yield_units": yield_units,
        "max_lifespan_step": max_lifespan_step,
        "fertilized_until_day": fertilized_until_day,
    }


def weed_tile():
    return {"kind": "WEED"}
