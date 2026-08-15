"""Standalone scripted opponents, deliberately independent of the
kaggriculture/ package (no shared code, no shared config) — built to give
tools/tournament.py a genuinely DIFFERENT "species" of opponent to test
against, after v11's A/B test came back with byte-identical results
(NOTES.md "v11"): our own tournament variants are all similar enough to
each other (same portfolio weights, same day-7 expansion floor) that
opponent-divergence-triggered features like market-surge detection and
expansion-mirroring never actually fire in self-play. These opponents
mimic the pattern repeatedly observed beating us in real Kaggle matches
(NOTES.md "v9"/"v11" replay analysis): concentrate on MELON (+ WHEAT for
cash/seed flow), stay in the starting NW quadrant forever, water/harvest
greedily every turn so weeds never build up, hire a small fixed crew.

Two variants:
    melon_focus_agent  — crops only, no animals (mirrors FieldOps, Disleve
                          Kanku, VaraPrabha's opponent — the cleanest,
                          most repeated pattern in the real-match analysis)
    melon_animal_agent — same crop discipline, adds a modest COW/SHEEP
                          herd (mirrors woorym/The Ceris, who ran animals
                          without ever sacrificing the single-quadrant/
                          low-weed discipline)

Usage (as a library):
    from tools.scripted_opponents import melon_focus_agent
    env.run([our_agent, melon_focus_agent])

Usage (smoke test / quick benchmark against our current agent):
    python tools/scripted_opponents.py --episodes 5 --opponent melon
    python tools/scripted_opponents.py --episodes 5 --opponent melon_animal
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(cur, target):
    dx = target[0] - cur[0]
    dy = target[1] - cur[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return "EAST" if dx > 0 else "WEST"
    if dy != 0:
        return "SOUTH" if dy > 0 else "NORTH"
    return "PASS"


SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]


def _nearest(pos, candidates):
    return min(candidates, key=lambda c: manhattan(pos, c)) if candidates else None


class _ScriptedAgent:
    """Shared scaffolding for both scripted opponents — crop-only
    watering/harvesting/planting discipline plus optional animal
    husbandry, all via simple nearest-task greedy assignment (no scoring
    formula, no rescue stages, no lookahead — deliberately simpler than
    our own kaggriculture agent, matching the plain, disciplined style
    observed in the real opponents this is meant to stand in for)."""

    def __init__(self, crops, seed_costs, first_yield_days, target_hands, with_animals):
        self.crops = crops  # priority order, first = most preferred to plant
        self.seed_costs = seed_costs
        self.first_yield_days = first_yield_days
        self.target_hands = target_hands
        self.with_animals = with_animals

    def act(self, obs):
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs.get("private", {})
        day = obs["day"]
        hour = obs["hour"]
        tiles = farm["tiles"]
        farmer = tuple(farm["farmer"])
        hands = [tuple(h) for h in farm["hands"]]
        units = [farmer] + hands
        money = farm["money"]
        seeds = private.get("seeds", {})
        shed = private.get("shed", {})
        inventories = private.get("inventories", [{}] * len(units))

        market = self._market_orders(obs, farm, seeds, shed, money, hour, day)

        empty, unwatered, harvestable, coops_empty, pastures_empty, unfed = self._scan_tiles(tiles, day)

        actions = []
        claimed_tiles = set()
        claimed_animal_target = False

        for i, pos in enumerate(units):
            inv = inventories[i] if i < len(inventories) else {}
            carrying = sum(inv.values()) if inv else 0

            if self.with_animals and any(k in ("GOOSE", "COW", "SHEEP") and v > 0 for k, v in inv.items()):
                animal = next(k for k, v in inv.items() if k in ("GOOSE", "COW", "SHEEP") and v > 0)
                structure_kind = "COOP" if animal == "GOOSE" else "PASTURE"
                pool = coops_empty if structure_kind == "COOP" else pastures_empty
                target = _nearest(pos, [p for p in pool if p not in claimed_tiles])
                if target is None:
                    actions.append(["DROP"] if pos in SHED_TILES else [step_toward(pos, _nearest(pos, SHED_TILES))])
                elif pos == target:
                    actions.append(["PLACE", animal])
                    claimed_tiles.add(target)
                else:
                    actions.append([step_toward(pos, target)])
                continue

            if carrying > 0:
                actions.append(["DROP"] if pos in SHED_TILES else [step_toward(pos, _nearest(pos, SHED_TILES))])
                continue

            if self.with_animals and unfed and inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                target = _nearest(pos, [t for t in SHED_TILES if t not in claimed_tiles]) if pos not in SHED_TILES else pos
                if pos in SHED_TILES:
                    actions.append(["PICKUP", "WHEAT", 1])
                else:
                    actions.append([step_toward(pos, target)])
                continue
            if self.with_animals and unfed and inv.get("WHEAT", 0) > 0:
                target = _nearest(pos, unfed)
                actions.append(["FEED"] if pos == target else [step_toward(pos, target)])
                continue

            candidates = [t for t in unwatered if t not in claimed_tiles]
            if candidates:
                target = _nearest(pos, candidates)
                if pos == target:
                    actions.append(["WATER"])
                else:
                    actions.append([step_toward(pos, target)])
                claimed_tiles.add(target)
                continue

            candidates = [t for t in harvestable if t not in claimed_tiles]
            if candidates:
                target = _nearest(pos, candidates)
                if pos == target:
                    actions.append(["HARVEST"])
                else:
                    actions.append([step_toward(pos, target)])
                claimed_tiles.add(target)
                continue

            if self.with_animals and not claimed_animal_target:
                if shed.get("GOOSE", 0) > 0 and coops_empty:
                    target = _nearest(pos, SHED_TILES)
                    if pos in SHED_TILES:
                        actions.append(["PICKUP", "GOOSE", 1])
                        claimed_animal_target = True
                        continue
                for animal, pool in (("COW", pastures_empty), ("SHEEP", pastures_empty)):
                    if shed.get(animal, 0) > 0 and pool:
                        if pos in SHED_TILES:
                            actions.append(["PICKUP", animal, 1])
                            claimed_animal_target = True
                        else:
                            actions.append([step_toward(pos, _nearest(pos, SHED_TILES))])
                        break
                else:
                    pass
                if len(actions) > i:
                    continue

            plant_crop = next((c for c in self.crops if seeds.get(c, 0) > 0), None)
            candidates = [t for t in empty if t not in claimed_tiles]
            if plant_crop and candidates:
                target = _nearest(pos, candidates)
                if pos == target:
                    actions.append(["PLANT", plant_crop])
                else:
                    actions.append([step_toward(pos, target)])
                claimed_tiles.add(target)
                continue

            actions.append(["PASS"])

        return {"farmer": actions[0], "hands": actions[1:], "market": market}

    def _scan_tiles(self, tiles, day):
        empty, unwatered, harvestable = [], [], []
        coops_empty, pastures_empty, unfed = [], [], []
        for y, row in enumerate(tiles):
            for x, cell in enumerate(row):
                if cell == "LOCKED":
                    continue
                pos = (x, y)
                if cell is None:
                    empty.append(pos)
                elif isinstance(cell, dict) and cell.get("kind") == "PLANT":
                    if not cell["watered_today"]:
                        unwatered.append(pos)
                    crop = cell["crop"]
                    fyd = self.first_yield_days.get(crop, 999)
                    if day - cell["planted_day"] >= fyd and cell["yield_units"] > 0:
                        harvestable.append(pos)
                elif isinstance(cell, dict) and cell.get("kind") in ("COOP", "PASTURE"):
                    if cell.get("animal") is None:
                        (coops_empty if cell["kind"] == "COOP" else pastures_empty).append(pos)
                    elif not cell.get("fed_today"):
                        unfed.append(pos)
        return empty, unwatered, harvestable, coops_empty, pastures_empty, unfed

    def _market_orders(self, obs, farm, seeds, shed, money, hour, day):
        orders = []
        if hour == 0:
            hires_today = farm.get("hires_today", 0)
            for _ in range(max(0, self.target_hands - hires_today)):
                orders.append(["HIRE"])
            if self.with_animals and day >= 3:
                if shed.get("COW", 0) + self._placed_count(farm, "COW") < 3 and money > 1000:
                    orders.append(["BUY_ANIMAL", "COW", 1])
                elif shed.get("SHEEP", 0) + self._placed_count(farm, "SHEEP") < 2 and money > 1200:
                    orders.append(["BUY_ANIMAL", "SHEEP", 1])
        for crop in self.crops:
            target_stock = 5
            held = seeds.get(crop, 0)
            if held < target_stock:
                cost = self.seed_costs[crop]
                affordable = int(money // cost)
                to_buy = min(target_stock - held, affordable)
                if to_buy > 0:
                    orders.append(["BUY_SEED", crop, to_buy])
        for item, qty in shed.items():
            if qty > 0:
                orders.append(["SELL", item, qty])
        return orders[:10]

    def _placed_count(self, farm, animal):
        count = 0
        for row in farm["tiles"]:
            for cell in row:
                if isinstance(cell, dict) and cell.get("animal") == animal:
                    count += 1
        return count


# Exposed as plain functions, not the callable class instances directly —
# kaggle_environments inspects agent callables' argument count via
# reflection to decide whether to pass (obs) or (obs, config), and that
# inspection gets confused by a class instance's __call__ (sees the
# unbound `(self, obs)` signature and passes 2 args instead of 1). Plain
# top-level functions with exactly one parameter sidestep that entirely.
_melon_focus = _ScriptedAgent(
    crops=["MELON", "WHEAT"],
    seed_costs={"MELON": 80, "WHEAT": 10},
    first_yield_days={"MELON": 10, "WHEAT": 2},
    target_hands=3,
    with_animals=False,
)


def melon_focus_agent(obs):
    return _melon_focus.act(obs)


_melon_animal = _ScriptedAgent(
    crops=["MELON", "WHEAT"],
    seed_costs={"MELON": 80, "WHEAT": 10},
    first_yield_days={"MELON": 10, "WHEAT": 2},
    target_hands=3,
    with_animals=True,
)


def melon_animal_agent(obs):
    return _melon_animal.act(obs)


def main():
    from kaggle_environments import make
    from kaggriculture.agent import agent as our_agent

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--opponent", choices=["melon", "melon_animal"], default="melon")
    args = parser.parse_args()

    opponent = melon_focus_agent if args.opponent == "melon" else melon_animal_agent
    wins = 0
    for i in range(args.episodes):
        env = make("kaggriculture", configuration={"episodeSteps": args.steps})
        env.run([our_agent, opponent])
        final = env.steps[-1]
        r0, r1 = final[0].reward, final[1].reward
        outcome = "WIN" if r0 > r1 else ("LOSE" if r0 < r1 else "TIE")
        if r0 > r1:
            wins += 1
        print(f"[{i+1}/{args.episodes}] us={r0:.0f} scripted={r1:.0f}  {outcome}")
    print(f"\nOur agent: {wins}/{args.episodes} wins vs {args.opponent}_agent")


if __name__ == "__main__":
    main()
