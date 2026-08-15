"""StateParser: converts raw `obs` dict into typed GameState. Single source of
truth — no other module should touch raw obs directly (SDD §3.1)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlantTile:
    crop: str
    planted_day: int
    watered_today: bool
    consecutive_unwatered: int
    yield_units: int
    max_lifespan_step: int
    fertilized_until_day: int


@dataclass
class WeedTile:
    pass


@dataclass
class StructureTile:
    kind: str  # "COOP" | "PASTURE"
    animal: Optional[str]
    placed_day: int
    yield_units: int
    fed_today: bool
    consecutive_unfed: int
    cared_today: bool
    fertilizer_available: bool
    pending_care_bonus: int


Tile = Optional[object]  # None (empty), "LOCKED", PlantTile, WeedTile, StructureTile


def parse_tile(raw) -> Tile:
    if raw is None or raw == "LOCKED":
        return raw
    kind = raw.get("kind")
    if kind == "PLANT":
        return PlantTile(
            crop=raw["crop"],
            planted_day=raw["planted_day"],
            watered_today=raw["watered_today"],
            consecutive_unwatered=raw["consecutive_unwatered"],
            yield_units=raw["yield_units"],
            max_lifespan_step=raw["max_lifespan_step"],
            fertilized_until_day=raw["fertilized_until_day"],
        )
    if kind == "WEED":
        return WeedTile()
    if kind in ("COOP", "PASTURE"):
        return StructureTile(
            kind=kind,
            animal=raw.get("animal"),
            placed_day=raw.get("placed_day", 0),
            yield_units=raw.get("yield_units", 0),
            fed_today=raw.get("fed_today", False),
            consecutive_unfed=raw.get("consecutive_unfed", 0),
            cared_today=raw.get("cared_today", False),
            fertilizer_available=raw.get("fertilizer_available", False),
            pending_care_bonus=raw.get("pending_care_bonus", 0),
        )
    return raw


@dataclass
class Farm:
    money: float
    tiles: list  # tiles[y][x], parsed
    farmer: tuple
    hands: list
    unlocked_quadrants: list
    hires_today: int


@dataclass
class GameState:
    player: int
    step: int
    day: int
    hour: int
    my_farm: Farm
    opp_farm: Farm
    shed: dict
    seeds: dict
    inventories: list
    market_inventory: dict
    market_prices: dict
    unlocked_shops: list

    def tile_at(self, x: int, y: int):
        return self.my_farm.tiles[y][x]

    def is_shed_adjacent(self, pos: tuple) -> bool:
        from .config import SHED_CENTER_TILES
        return tuple(pos) in SHED_CENTER_TILES


def _parse_farm(raw_farm) -> Farm:
    tiles = [[parse_tile(cell) for cell in row] for row in raw_farm["tiles"]]
    return Farm(
        money=raw_farm["money"],
        tiles=tiles,
        farmer=tuple(raw_farm["farmer"]),
        hands=[tuple(h) for h in raw_farm.get("hands", [])],
        unlocked_quadrants=list(raw_farm.get("unlocked_quadrants", [])),
        hires_today=raw_farm.get("hires_today", 0),
    )


class StateParser:
    @staticmethod
    def parse(obs: dict) -> GameState:
        player = obs["player"]
        farms = obs["farms"]
        my_farm = _parse_farm(farms[player])
        opp_farm = _parse_farm(farms[1 - player])
        private = obs.get("private", {})
        market = obs.get("market", {})
        town = obs.get("town", {})
        return GameState(
            player=player,
            step=obs.get("step", 0),
            day=obs["day"],
            hour=obs["hour"],
            my_farm=my_farm,
            opp_farm=opp_farm,
            shed=dict(private.get("shed", {})),
            seeds=dict(private.get("seeds", {})),
            inventories=[dict(i) for i in private.get("inventories", [])],
            market_inventory=dict(market.get("inventory", {})),
            market_prices=dict(market.get("prices", {})),
            unlocked_shops=list(town.get("unlocked_shops", [])),
        )
