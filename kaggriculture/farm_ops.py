"""FarmOpsOrchestrator: tactical layer, runs every turn (SDD §3.3, §6).

M3 adds market-timing to MarketExecutor: sell orders are sized against
MarketModel's hold floor instead of dumping the whole shed every turn (§5.4).
M4 adds animal husbandry to UnitScheduler: building coops/pastures, placing
purchased animals, feeding, caring, collecting fertilizer, and harvesting
animal product — layered onto the same greedy-scored candidate pool used for
crops (§6), plus a small pre-stage for the "carrying something, go deal with
it" cases that don't fit the tile-exclusive candidate model (return-to-shed,
place-animal, fetch-feed-wheat).
"""

import math

from .config import (
    ANIMAL_DATA,
    CLUSTER_LOOKAHEAD_BONUS_PER_NEIGHBOR,
    CLUSTER_LOOKAHEAD_RADIUS,
    CROP_DATA,
    ENABLE_CLUSTER_LOOKAHEAD,
    ENABLE_MULTI_CROP_PLANTING,
    FERTILIZER_BUY_CEILING_MULT,
    PRE_HARVEST_CASH_RESERVE,
    PRE_HARVEST_DAY_THRESHOLD,
    WHEAT_FEED_BUY_CEILING_MULT,
    FERTILIZER_RESERVE_TARGET,
    PRODUCT_BASE_PRICE,
    SELL_CAPITULATION_QTY,
    SHED_CENTER_TILES,
    SHED_OVERFLOW_GUARD_THRESHOLD,
    TASK_SCORE_WEIGHTS,
)
from .market_model import current_price, max_sell_before_floor
from .state import GameState, PlantTile, StructureTile, WeedTile
from .strategy import DayPlan, fib_cost

STRUCTURE_OP = {"COOP": "BUILD_COOP", "PASTURE": "BUILD_PASTURE"}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_shed_tile(pos):
    return min(SHED_CENTER_TILES, key=lambda t: manhattan(pos, t))


def step_toward(cur, target):
    dx = target[0] - cur[0]
    dy = target[1] - cur[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return "EAST" if dx > 0 else "WEST"
    if dy != 0:
        return "SOUTH" if dy > 0 else "NORTH"
    return "PASS"


class UnitScheduler:
    """Greedy priority-scored task assignment across farmer + hands (SDD §6)."""

    def __init__(self):
        # v20: persists which unit is currently walking toward which
        # critical-unwatered tile, across turns within an episode (this
        # instance lives for the whole episode, see agent.py) — see the
        # critical-unwatered rescue block in assign() for why.
        self._committed_rescues = {}  # unit_index -> (x, y) tile position

    def assign(self, state: GameState, plan: DayPlan) -> list:
        unit_positions = [state.my_farm.farmer] + list(state.my_farm.hands)
        inventories = state.inventories if state.inventories else [{}] * len(unit_positions)
        unfed_positions = self._find_unfed_animal_tiles(state)
        fertilizable_positions = self._find_fertilizable_tiles(state)

        actions = [None] * len(unit_positions)
        idle_units = []
        # Tracks which unfed animals already have a wheat-carrying unit en
        # route this turn, so N different carriers converge on N different
        # animals instead of all racing to whichever is nearest (which left
        # every animal but one un-rescued when several were critical at
        # once — a real cause of the high escape rate observed via
        # tools/replay_analyze.py's §8.5 breakdown).
        claimed_feed_targets = set()

        for i, pos in enumerate(unit_positions):
            inv = inventories[i] if i < len(inventories) else {}
            action = self._carrying_action(state, pos, inv, unfed_positions, fertilizable_positions, claimed_feed_targets)
            if action is not None:
                actions[i] = action
            else:
                idle_units.append(i)

        if not idle_units:
            return actions

        # An animal that missed yesterday's feeding turns to weeds/escapes
        # tonight if not fed today (README: 2 consecutive misses). A big
        # harvest can out-score a routine FETCH_WHEAT in the generic scorer
        # below, so dedicate one idle unit PER critical animal directly —
        # losing an animal (sunk cost + future production) outweighs a
        # single turn's harvest/plant delay for that many units.
        critical_unfed = [pos for pos, unfed_days in unfed_positions if unfed_days >= 1]
        if critical_unfed and state.shed.get("WHEAT", 0) > 0 and idle_units:
            num_rescuers = min(len(critical_unfed), len(idle_units))
            rescuers = sorted(
                idle_units, key=lambda i: min(manhattan(unit_positions[i], sp) for sp in SHED_CENTER_TILES)
            )[:num_rescuers]
            for rescuer in rescuers:
                pos = unit_positions[rescuer]
                if pos in SHED_CENTER_TILES:
                    actions[rescuer] = ["PICKUP", "WHEAT", 1]
                else:
                    actions[rescuer] = [step_toward(pos, nearest_shed_tile(pos))]
                idle_units.remove(rescuer)

        if not idle_units:
            return actions

        # Same problem, same fix, for plants: a high-value HARVEST candidate
        # (value = base_price x yield_units, uncapped — a ready melon alone
        # scores ~6000) routinely outscores a routine WATER (~500) in the
        # generic scorer below, even though WATER already carries an
        # urgency bonus. Root-caused via a real leaderboard replay + a
        # fixed-seed local trace: a WHEAT tile sat at consecutive_unwatered
        # == 1 for a FULL day with 6 units available and the tile only 2
        # steps from the nearest one — every unit was getting pulled to
        # harvest tasks instead, all day, every turn. Dedicating rescuers
        # here (bypassing the scorer entirely, same as the feed-rescue
        # above) guarantees a plant on the brink of becoming a weed always
        # gets a unit working toward it.
        critical_unwatered = self._find_critical_unwatered_tiles(state)
        # v20: drop any commitment whose tile is no longer critical (either
        # successfully watered, or already lost to a weed) before doing
        # anything else this turn — keeps the dict from accumulating stale
        # entries and ensures only genuinely still-critical commitments
        # get honored below.
        critical_set = set(critical_unwatered)
        self._committed_rescues = {i: t for i, t in self._committed_rescues.items() if t in critical_set}
        if critical_unwatered and idle_units:
            remaining_tiles = list(critical_unwatered)
            rescuer_pool = list(idle_units)

            # Honor existing commitments FIRST, before any fresh distance
            # optimization — found via a real-replay spatial analysis
            # (NOTES.md "v20") that weeds cluster disproportionately in
            # far corners, and a code audit traced why: this rescue used
            # to re-solve the nearest-pair assignment from scratch EVERY
            # turn, so a unit already 2 turns into walking toward tile A
            # could get reassigned to a newly-appeared, marginally closer
            # tile C — abandoning A's progress and potentially never
            # actually reaching either in time. Committing once a
            # rescue starts (and only dropping it when the tile resolves)
            # guarantees forward progress instead of dithering between
            # targets turn to turn.
            for i in list(rescuer_pool):
                committed_tile = self._committed_rescues.get(i)
                if committed_tile is not None and committed_tile in remaining_tiles:
                    pos = unit_positions[i]
                    actions[i] = ["WATER"] if pos == committed_tile else [step_toward(pos, committed_tile)]
                    rescuer_pool.remove(i)
                    remaining_tiles.remove(committed_tile)
                    idle_units.remove(i)

            num_rescuers = min(len(remaining_tiles), len(rescuer_pool))
            for _ in range(num_rescuers):
                best = min(
                    ((manhattan(unit_positions[i], t), i, t) for i in rescuer_pool for t in remaining_tiles),
                    key=lambda triple: triple[0],
                )
                _, i, t = best
                pos = unit_positions[i]
                actions[i] = ["WATER"] if pos == t else [step_toward(pos, t)]
                self._committed_rescues[i] = t
                rescuer_pool.remove(i)
                remaining_tiles.remove(t)
                idle_units.remove(i)

        if not idle_units:
            return actions

        # v20: TRIED a dedicated rescue here for animals already bought but
        # still sitting in the shed (real-replay analysis kept finding
        # owned-animal counts plateau at 1-4 despite generous caps/cash).
        # REVERTED after a deep-dive trace found a genuine negative side
        # effect: once a unit PICKS UP an animal, `_carrying_action`
        # commits it to a multi-turn carry-and-place trip with NO way to
        # interrupt for a more urgent need — including a critical-unfed
        # animal that turns up mid-carry. Same-seed comparison showed
        # placed-animal count actually DROPPING late-game (4->3->1->1
        # over days 26-29) in a farm that stayed flat at 4 the whole game
        # without this stage — not a placement failure, animals that were
        # ALREADY placed were lost to starvation-escape because a unit
        # that would have been available for feed-rescue was mid-carry-trip
        # instead. Bypassing the scorer for a multi-turn COMMITTED action
        # (unlike the water-rescue stage above, which only takes one
        # WATER/step per call and re-evaluates every turn) is the wrong
        # shape for a dedicated rescue. See NOTES.md "v20" for the full
        # investigation — the animal-execution gap (opponents reach 4-6+,
        # we plateau at 1-4) is still real and unsolved.

        if not idle_units:
            return actions

        candidates, cluster_bonus_by_pos = self._build_candidates(state, plan, unfed_positions, fertilizable_positions)

        pairs = []
        for i in idle_units:
            pos = unit_positions[i]
            for tile_pos, task, urgency, value in candidates:
                dist = manhattan(pos, tile_pos)
                score = (
                    urgency * TASK_SCORE_WEIGHTS["urgency"]
                    + value * TASK_SCORE_WEIGHTS["value"]
                    - dist * TASK_SCORE_WEIGHTS["distance"]
                    + cluster_bonus_by_pos.get(tile_pos, 0.0)
                )
                pairs.append((score, i, tile_pos, task))
        pairs.sort(key=lambda p: p[0], reverse=True)

        assigned_units = set()
        assigned_tiles = set()
        # README: if too many units PLANT the same crop in one turn without
        # enough seed for all of them, NONE of those plants land — not just
        # the excess. Cap same-turn immediate PLANT actions (pos == tile_pos)
        # per crop at the seed stock actually on hand; movement-only
        # assignments toward a planting tile don't consume a seed this turn,
        # so they're re-evaluated fresh next turn once seed stock is known.
        planted_this_turn = {}
        for score, i, tile_pos, task in pairs:
            if i in assigned_units or tile_pos in assigned_tiles:
                continue
            pos = unit_positions[i]
            if pos == tile_pos and task[0] == "PLANT":
                crop = task[1]
                seed_stock = state.seeds.get(crop, 0)
                if planted_this_turn.get(crop, 0) >= seed_stock:
                    continue
                planted_this_turn[crop] = planted_this_turn.get(crop, 0) + 1
            if pos == tile_pos:
                actions[i] = self._task_to_action(task)
            else:
                actions[i] = [step_toward(pos, tile_pos)]
            assigned_units.add(i)
            assigned_tiles.add(tile_pos)

        for i in idle_units:
            if actions[i] is None:
                actions[i] = ["PASS"]

        return actions

    def _carrying_action(self, state: GameState, pos, inv: dict, unfed_positions: list, fertilizable_positions: list, claimed_feed_targets: set):
        """Handles the "unit is carrying something, go deal with it" cases
        that don't fit the tile-exclusive candidate model below: an animal
        needs placing on a matching empty structure, a harvested product
        needs dropping at the shed, feed wheat needs delivering to the
        nearest hungry animal, or fertilizer needs delivering to the
        nearest eligible plant. Returns None if the unit is carrying
        nothing and should fall through to the generic candidate pool."""
        if not inv:
            return None

        animal_carried = next((item for item in inv if item in ANIMAL_DATA and inv[item] > 0), None)
        if animal_carried:
            structure_kind = ANIMAL_DATA[animal_carried]["structure"]
            target = self._nearest_empty_structure(state, pos, structure_kind)
            if target is None:
                return ["DROP"] if pos in SHED_CENTER_TILES else [step_toward(pos, nearest_shed_tile(pos))]
            if pos == target:
                return ["PLACE", animal_carried]
            return [step_toward(pos, target)]

        fertilizer_qty = inv.get("FERTILIZER", 0)
        if fertilizer_qty > 0:
            if fertilizable_positions:
                target = min(fertilizable_positions, key=lambda t: manhattan(pos, t))
                if pos == target:
                    return ["FERTILIZE"]
                return [step_toward(pos, target)]
            return ["DROP"] if pos in SHED_CENTER_TILES else [step_toward(pos, nearest_shed_tile(pos))]

        non_wheat_qty = sum(v for k, v in inv.items() if k not in ("WHEAT", "FERTILIZER") and k not in ANIMAL_DATA)
        if non_wheat_qty > 0:
            return ["DROP"] if pos in SHED_CENTER_TILES else [step_toward(pos, nearest_shed_tile(pos))]

        wheat_qty = inv.get("WHEAT", 0)
        if wheat_qty > 0:
            available = [t for t in unfed_positions if t[0] not in claimed_feed_targets]
            if available:
                target = min(available, key=lambda t: manhattan(pos, t[0]))[0]
                claimed_feed_targets.add(target)
                if pos == target:
                    return ["FEED"]
                return [step_toward(pos, target)]
            if unfed_positions:
                # Every unfed animal already has a carrier headed its way
                # this turn — fall back to the nearest one anyway (a second
                # carrier arriving is harmless, unlike leaving one animal
                # completely unclaimed) rather than wastefully returning
                # this wheat to the shed.
                target = min(unfed_positions, key=lambda t: manhattan(pos, t[0]))[0]
                if pos == target:
                    return ["FEED"]
                return [step_toward(pos, target)]
            return ["DROP"] if pos in SHED_CENTER_TILES else [step_toward(pos, nearest_shed_tile(pos))]

        return None

    def _find_unfed_animal_tiles(self, state: GameState) -> list:
        """Returns [(pos, consecutive_unfed), ...] for every occupied,
        unfed structure — consecutive_unfed drives urgency scaling for the
        FETCH_WHEAT candidate (an animal already at risk of escaping should
        outrank routine crop tasks)."""
        positions = []
        for y, row in enumerate(state.my_farm.tiles):
            for x, cell in enumerate(row):
                if isinstance(cell, StructureTile) and cell.animal is not None and not cell.fed_today:
                    positions.append(((x, y), cell.consecutive_unfed))
        return positions

    def _find_critical_unwatered_tiles(self, state: GameState) -> list:
        """Positions of plants that already missed a watering
        (consecutive_unwatered >= 1) and haven't been watered yet today —
        one more miss turns them into a weed tonight (README)."""
        positions = []
        for y, row in enumerate(state.my_farm.tiles):
            for x, cell in enumerate(row):
                if isinstance(cell, PlantTile) and cell.consecutive_unwatered >= 1 and not cell.watered_today:
                    positions.append((x, y))
        return positions

    def _find_fertilizable_tiles(self, state: GameState) -> list:
        """Positions of existing plants that would benefit from FERTILIZE
        right now (README §Harvest Yields): one-time crops inside their
        watering bonus window (starts at ceil(max_yield_day/2)) not already
        fertilized, or any active ongoing crop not already fertilized
        (fertilize+water doubles its scheduled yield, any day)."""
        positions = []
        for y, row in enumerate(state.my_farm.tiles):
            for x, cell in enumerate(row):
                if not isinstance(cell, PlantTile):
                    continue
                if cell.fertilized_until_day >= state.day:
                    continue  # bonus already active
                crop_info = CROP_DATA.get(cell.crop)
                if crop_info is None:
                    continue
                age = state.day - cell.planted_day
                if crop_info["one_time"]:
                    bonus_start = math.ceil(crop_info["max_yield_day"] / 2)
                    if bonus_start <= age <= crop_info["max_yield_day"]:
                        positions.append((x, y))
                else:
                    positions.append((x, y))
        return positions

    def _nearest_empty_structure(self, state: GameState, pos, structure_kind: str):
        candidates = [
            (x, y)
            for y, row in enumerate(state.my_farm.tiles)
            for x, cell in enumerate(row)
            if isinstance(cell, StructureTile) and cell.kind == structure_kind and cell.animal is None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda t: manhattan(pos, t))

    def _build_candidates(self, state: GameState, plan: DayPlan, unfed_positions: list, fertilizable_positions: list):
        """Watering/harvesting/animal-care apply to any existing plant or
        animal regardless of the current phase's active pool (an old
        planting or animal must still be tended). Only NEW plantings/
        structures are restricted to plan.crop_targets/structure_targets."""
        candidates = []
        empty_tiles = []
        active_crops = [c for c, target in plan.crop_targets.items() if target > 0]
        planted_counts = {c: 0 for c in active_crops}
        structure_counts = {"COOP": 0, "PASTURE": 0}
        empty_structure_counts = {"COOP": 0, "PASTURE": 0}

        tiles = state.my_farm.tiles
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if tile == "LOCKED":
                    continue
                pos = (x, y)
                if tile is None:
                    empty_tiles.append(pos)
                elif isinstance(tile, PlantTile):
                    crop_info = CROP_DATA.get(tile.crop)
                    if crop_info is None:
                        continue
                    if tile.crop in planted_counts:
                        planted_counts[tile.crop] += 1
                    if not tile.watered_today:
                        urgency = 3 if tile.consecutive_unwatered >= 1 else 1
                        value = crop_info["base_price"] * 0.5
                        candidates.append((pos, ("WATER",), urgency, value))
                    age = state.day - tile.planted_day
                    if age >= crop_info["first_yield_day"] and tile.yield_units > 0:
                        value = crop_info["base_price"] * tile.yield_units
                        candidates.append((pos, ("HARVEST",), 2, value))
                elif isinstance(tile, WeedTile):
                    # v18 tried raising this (urgency 0->1, value 5->40)
                    # after a real-replay trace suggested weeds were
                    # under-prioritized — tournament-style validation
                    # against melon_focus_agent immediately caught the
                    # opposite problem: win rate collapsed 15/15 -> 0/15.
                    # Diverting units to DIG (which only clears space,
                    # earns nothing directly) away from HARVEST/WATER
                    # (which do) is a bad trade against a benchmark that
                    # rewards raw output. Reverted — DIG stays deliberately
                    # low priority; see NOTES.md "v18" for the full
                    # investigation and why this lever doesn't work.
                    candidates.append((pos, ("DIG",), 0, 5))
                elif isinstance(tile, StructureTile):
                    structure_counts[tile.kind] = structure_counts.get(tile.kind, 0) + 1
                    if tile.animal is None:
                        empty_structure_counts[tile.kind] = empty_structure_counts.get(tile.kind, 0) + 1
                    if tile.animal is not None:
                        animal_info = ANIMAL_DATA[tile.animal]
                        if tile.yield_units > 0:
                            value = animal_info["base_price"] * tile.yield_units
                            candidates.append((pos, ("HARVEST",), 2, value))
                        if not tile.cared_today:
                            value = animal_info["base_price"] * 0.3
                            candidates.append((pos, ("CARE",), 1, value))
                        if tile.fertilizer_available:
                            # v15: bumped from urgency=1/0.3x to urgency=2/1.0x
                            # after a real-match replay analysis (NOTES.md
                            # "v15") found a top real opponent earning nearly
                            # HALF their total money from fertilizer sales
                            # ($38,700 of ~$79,500) while we earned $229 in
                            # the same matchup. README: "uncollected
                            # fertilizer does not accumulate" — every animal
                            # makes exactly 1 unit available per day whether
                            # collected or not, so a missed collection is a
                            # PERMANENT loss, not a deferred one like a
                            # routine HARVEST/WATER that just waits for the
                            # next turn. The old 0.3x/urgency=1 weighting
                            # treated it like a low-stakes routine task and
                            # let it lose the generic scorer's tie-break to
                            # nearly everything else on a busy farm.
                            candidates.append((pos, ("COLLECT_FERTILIZER",), 2, PRODUCT_BASE_PRICE["FERTILIZER"]))

        # Purchased animals sit in the shed until a unit fetches them
        # (README: BUY_ANIMAL delivers to the shed, not directly onto a
        # structure) — inject a PICKUP task whenever an empty matching
        # structure is ready, mirroring FETCH_WHEAT below. Once a unit
        # picks one up, _carrying_action (assign()) takes over and walks it
        # to the structure to PLACE.
        for animal, info in ANIMAL_DATA.items():
            if state.shed.get(animal, 0) > 0 and empty_structure_counts.get(info["structure"], 0) > 0:
                # v26: kept comfortably above BUILD_STRUCTURE's value (now
                # scaled to up to the highest housed animal's base_price,
                # 200 for SHEEP/PASTURE) — placing an animal ALREADY bought
                # into a structure that ALREADY exists must always outrank
                # starting a brand new structure for one not yet bought,
                # the same "protect committed spend before chasing more"
                # priority already applied to wheat feed top-ups (v20).
                value = info["base_price"] * 5
                for shed_pos in SHED_CENTER_TILES:
                    candidates.append((shed_pos, ("FETCH_ANIMAL", animal), 1, value))

        # An unfed animal doesn't feed itself: inject a PICKUP WHEAT task at
        # the shed whenever wheat is in stock and something needs feeding,
        # so an empty-handed idle unit has a reason to go get it. Once
        # carried, _carrying_action routes the unit to the animal to FEED.
        if unfed_positions and state.shed.get("WHEAT", 0) > 0:
            urgency = 3 if any(c >= 1 for _, c in unfed_positions) else 1
            value = PRODUCT_BASE_PRICE.get("EGG", 50) * 2
            for shed_pos in SHED_CENTER_TILES:
                candidates.append((shed_pos, ("FETCH_WHEAT",), urgency, value))

        # Same fetch-and-deliver pattern for fertilizer, but kept deliberately
        # LOW priority: missing a fertilize only forfeits a bonus (still get
        # the base yield), unlike missing a WATER (loses the whole plant) or
        # a FEED (loses the whole animal) — those carry urgency multipliers
        # for exactly that reason. An earlier version scaled this value the
        # same way as HARVEST (base_price x yield), which made a melon's
        # ~$750 estimate dominate the scorer outright and starve watering
        # (observed directly: weed events jumped to ~20/episode, win rate
        # vs starter collapsed from 100% to 37.5%). Match WATER's value
        # scale instead so fertilizing competes fairly but never wins over
        # anything that risks losing the plant/animal outright.
        if fertilizable_positions and state.shed.get("FERTILIZER", 0) > 0:
            best_value = 0
            for fx, fy in fertilizable_positions:
                crop_info = CROP_DATA.get(state.my_farm.tiles[fy][fx].crop)
                if crop_info:
                    best_value = max(best_value, crop_info["base_price"] * 0.3)
            for shed_pos in SHED_CENTER_TILES:
                candidates.append((shed_pos, ("FETCH_FERTILIZER",), 1, best_value))

        # New plantings: shortfall computed once from actual planted counts
        # (not decremented per tile) — candidates rebuild fresh every turn
        # from live tile state, so the scorer (§6), not iteration order,
        # decides which unit plants where this turn.
        if ENABLE_MULTI_CROP_PLANTING:
            # v13: every eligible crop gets a PLANT candidate at every empty
            # tile, not just the single best-by-ratio one — SDD §6's known
            # "only one crop plants per turn" limitation otherwise forces a
            # multi-crop portfolio to spend several turns in a row all on
            # the SAME crop before another gets a turn at all. Safe because
            # the seed-overcommit cap in assign() is keyed per-crop.
            for crop in active_crops:
                target = plan.crop_targets[crop]
                if target <= 0 or state.seeds.get(crop, 0) <= 0:
                    continue
                if planted_counts.get(crop, 0) >= target:
                    continue
                crop_info = CROP_DATA[crop]
                value = crop_info["base_price"] - crop_info["seed_cost"]
                for pos in empty_tiles:
                    candidates.append((pos, ("PLANT", crop), 1, value))
        else:
            # Pre-v13 behavior, kept for direct A/B comparison via
            # tools/tournament.py. Ranked by shortfall RATIO
            # (shortfall/target), not raw shortfall count: a crop with a
            # much bigger target (e.g. MELON at 26 tiles) will almost
            # always have a bigger absolute shortfall than one with a
            # smaller target (e.g. STRAWBERRY at 8), so raw-count
            # comparison let MELON win the "which crop plants this turn"
            # slot on nearly every turn regardless of how proportionally
            # starved STRAWBERRY was. Ratio comparison treats "0%
            # fulfilled" the same regardless of the target's absolute size.
            best_crop, best_ratio = None, 0.0
            for crop in active_crops:
                target = plan.crop_targets[crop]
                if target <= 0 or state.seeds.get(crop, 0) <= 0:
                    continue
                shortfall = target - planted_counts.get(crop, 0)
                ratio = shortfall / target
                if ratio > best_ratio:
                    best_ratio, best_crop = ratio, crop
            if best_crop:
                crop_info = CROP_DATA[best_crop]
                value = crop_info["base_price"] - crop_info["seed_cost"]
                for pos in empty_tiles:
                    candidates.append((pos, ("PLANT", best_crop), 1, value))

        # New structures: same shortfall-vs-target pattern, one structure
        # kind per turn so it competes fairly against planting for the same
        # empty tiles via the score, rather than being force-prioritized.
        best_structure, best_structure_shortfall = None, 0
        for kind, target in plan.structure_targets.items():
            shortfall = target - structure_counts.get(kind, 0)
            if shortfall > best_structure_shortfall:
                best_structure_shortfall, best_structure = shortfall, kind

        if best_structure:
            # v26: real-match trace found 8 COW bought (structure_targets
            # correctly set PASTURE:8) but only 2 PASTURE ever got built —
            # the old flat value=80 lost the tile-allocation contest to
            # crop candidates almost every time a tile briefly opened up
            # (e.g. MELON's value is base_price(250)-seed_cost(80)=170,
            # more than double), so 7 of the 8 purchased COW sat unplaced
            # in the shed for the rest of the game, dead cash. Scale value
            # to the structure's actual revenue potential (the highest
            # base_price among animals it houses) instead of an arbitrary
            # flat number, so it competes fairly for scarce empty tiles
            # the same way COLLECT_FERTILIZER/PLANT were re-weighted (v15).
            housed_animals = [a for a, info in ANIMAL_DATA.items() if info["structure"] == best_structure]
            value = max((ANIMAL_DATA[a]["base_price"] for a in housed_animals), default=80)
            op = STRUCTURE_OP[best_structure]
            for pos in empty_tiles:
                candidates.append((pos, (op,), 1, value))

        cluster_bonus_by_pos = self._cluster_bonus(candidates) if ENABLE_CLUSTER_LOOKAHEAD else {}
        return candidates, cluster_bonus_by_pos

    def _cluster_bonus(self, candidates: list) -> dict:
        """Bounded stand-in for real multi-turn search (discussed as an
        alternative to literal minimax, which doesn't fit this game — see
        NOTES.md "v10"): a candidate tile surrounded by other actionable
        tiles lets a unit sent there chain efficiently into the next task
        afterward, instead of finishing an isolated tile and needing a long
        return trip. Returned as a separate {pos: bonus} map added directly
        to the final SCORE in assign() — not baked into the candidate's
        `value` field, which gets multiplied by TASK_SCORE_WEIGHTS["value"]
        (3.8x) and made the bonus wildly overpowering even a single tile of
        distance cost (caught by a unit test: a farmer standing exactly on
        a plantable tile moved away from it just to reach a marginally
        better-clustered one one tile over). Counts unique nearby positions
        that already have their own candidate, not other tasks at the SAME
        position (a tile with both a WATER and HARVEST candidate isn't "2
        neighbors" of itself)."""
        unique_positions = list({pos for pos, _, _, _ in candidates})
        bonus_by_pos = {}
        for pos in unique_positions:
            neighbors = sum(
                1 for other in unique_positions
                if other != pos and manhattan(pos, other) <= CLUSTER_LOOKAHEAD_RADIUS
            )
            bonus_by_pos[pos] = neighbors * CLUSTER_LOOKAHEAD_BONUS_PER_NEIGHBOR
        return bonus_by_pos

    def _task_to_action(self, task):
        op = task[0]
        if op == "PLANT":
            return ["PLANT", task[1]]
        if op == "FETCH_ANIMAL":
            return ["PICKUP", task[1], 1]
        if op == "FETCH_WHEAT":
            return ["PICKUP", "WHEAT", 1]
        if op == "FETCH_FERTILIZER":
            return ["PICKUP", "FERTILIZER", 1]
        return [op]


MAX_MARKET_ORDERS_PER_TURN = 10


class MarketExecutor:
    """Builds the market order list for this turn (SDD §3.3)."""

    def build_orders(self, state: GameState, plan: DayPlan) -> list:
        """Every sub-builder below shares ONE running `budget` (starts at
        `state.my_farm.money`, decremented as each builder commits spend) —
        NOT each independently re-checking the full start-of-turn money.
        Without this, two builders can each see "$3000 available" and both
        request as much as that implies, but the game processes orders in
        list order and actually deducts sequentially: whichever order type
        is queued LAST silently gets starved once real money runs out
        mid-list (README: "If a player runs out of money mid-order, the
        order is stopped"). This is exactly what happened when the animal
        budget (`ANIMAL_CASH_FRACTION`) and crop-seed budget were computed
        independently: animals (listed first) looked affordable on their
        own, crop seeds (checked independently, also against the
        untouched full balance) looked affordable too, and together they
        overcommitted — strawberry/melon seed stock stayed starved for
        days despite `crop_targets` correctly asking for far more.

        Priority order (each stage spends from what the previous left):
        HIRE > crop seeds > wheat feed top-up > land/new animals >
        fertilizer top-up. HIRE is first because it's the foundational
        capacity everything else depends on, and it's cheap (Fibonacci:
        1,1,2,3,5,...) — a real leaderboard replay caught hire_count
        silently computing to 0 for 9 straight days once an animal-heavy
        cash crunch left nothing in the budget for even a single $1 hire,
        while weeds compounded because there was no one left to water
        them. Crop seeds come next: the whole tile-target machinery in
        UnitScheduler is dead weight without seed stock to plant.

        v20: wheat feed top-up moved ahead of land/new-animal purchases
        — a real-replay + local-trace investigation (NOTES.md "v20")
        found the OLD order (land/animals budgeted before wheat top-up)
        left the shed's WHEAT reserve at 0 in 93% of turns where a
        critical-unfed animal existed (204 of 220 sampled turns), because
        buying MORE animals routinely exhausted the budget before
        protecting the ones already owned. Feeding animals we already
        have takes priority over acquiring more; land/new-animal
        purchases can wait a turn without losing anything already paid
        for, an unfed animal cannot.
        """
        orders = []
        budget = state.my_farm.money

        # v25: a small cash floor, untouched by ANY purchase this turn
        # (HIRE, seeds, land, animals, fertilizer alike), during the
        # pre-first-harvest window — see PRE_HARVEST_CASH_RESERVE's
        # config.py comment for the real-match death spiral this guards
        # against (v23 guarded only HIRE's own spend, which v25 real-match
        # data showed still let OTHER orders drain cash to $0 in 28% of
        # games). Subtracting it up front from the working budget makes
        # every affordability check below automatically respect the
        # floor, the same way an already-tight budget does.
        if state.day < PRE_HARVEST_DAY_THRESHOLD:
            budget -= PRE_HARVEST_CASH_RESERVE

        # HIRE, BUY_LAND, and BUY_ANIMAL are day-level decisions (§3.2) whose
        # target counts are cached for the whole day — only submit once, on
        # the first turn of the day, or they'd repeat (and overspend) every
        # turn for the rest of the day.
        if state.hour == 0:
            hires_so_far = state.my_farm.hires_today
            for _ in range(plan.hire_count):
                cost = fib_cost(hires_so_far)
                if cost > budget:
                    break
                orders.append(["HIRE"])
                budget -= cost
                hires_so_far += 1

        for crop, target in plan.crop_targets.items():
            if target <= 0:
                continue
            crop_info = CROP_DATA[crop]
            seeds_held = state.seeds.get(crop, 0)
            target_seed_stock = min(target, 5)
            if seeds_held < target_seed_stock:
                affordable = int(budget // crop_info["seed_cost"])
                to_buy = min(target_seed_stock - seeds_held, affordable)
                if to_buy > 0:
                    orders.append(["BUY_SEED", crop, to_buy])
                    budget -= to_buy * crop_info["seed_cost"]

        # v20: protect animals ALREADY owned (feed top-up) before spending
        # on land or MORE animals — see the priority-order docstring above.
        wheat_orders, budget = self._build_wheat_feed_topup_orders(state, budget)
        orders.extend(wheat_orders)

        if state.hour == 0:
            if plan.buy_land:
                # v16: land_cost (resolved in strategy.py, which already
                # knows which quadrant tier is next) is deducted from the
                # SAME shared budget every other order draws from — found
                # via a real-match replay (NOTES.md "v16") that skipping
                # this let same-turn animal orders get computed against a
                # stale, pre-purchase budget and silently truncated by the
                # game engine once real money ran out mid-order-list
                # (README: "If a player runs out of money mid-order, the
                # order is stopped") — observed as 5 PASTURE structures
                # built but only 2 COW ever bought to fill them.
                orders.append(["BUY_LAND"])
                budget -= plan.land_cost
            animal_orders, budget = self._build_animal_orders(state, plan, budget)
            orders.extend(animal_orders)

        fertilizer_orders, budget = self._build_fertilizer_orders(state, plan, budget)
        orders.extend(fertilizer_orders)
        orders.extend(self._build_sell_orders(state, plan))  # sells earn money, no budget to track

        return orders[:MAX_MARKET_ORDERS_PER_TURN]

    def _build_wheat_feed_topup_orders(self, state: GameState, budget: float) -> tuple:
        """Owned animals need WHEAT every day regardless of how the crop
        portfolio's own WHEAT tile share happens to be doing that day —
        planting/harvest timing (only one crop gets planted per turn per
        the shortfall-based scheduler in UnitScheduler) can leave WHEAT
        production too thin to cover feed demand on top of sales. Top up
        via BUY_PRODUCT (only WHEAT/FERTILIZER can be bought back, README)
        whenever the shed is short of the feed reserve — this closes a real
        gap found via tools/replay_analyze.py's escape-event tracking."""
        animal_count = self._animal_count(state)
        if animal_count <= 0:
            return [], budget
        reserve_target = animal_count * 2
        held = state.shed.get("WHEAT", 0)
        if held >= reserve_target:
            return [], budget
        price = current_price("WHEAT", state)
        if not price:
            return [], budget
        # v20: uses its OWN, more generous ceiling (WHEAT_FEED_BUY_CEILING_MULT)
        # — see its config.py comment. Sharing FERTILIZER_BUY_CEILING_MULT
        # (1.5x) was a real bug: a real-scripted-opponent trace found shed
        # WHEAT stuck at 0 for a full day, healthy cash the whole time,
        # because price had drifted to 1.56x base — just over the old
        # shared ceiling — silently blocking every top-up attempt while
        # animals starved. Missing a wheat top-up risks losing a whole
        # animal ($300-500), a much higher stake than fertilizer's minor
        # forfeited yield bonus, so it's worth paying a steeper premium.
        if price > WHEAT_FEED_BUY_CEILING_MULT * PRODUCT_BASE_PRICE["WHEAT"]:
            return [], budget
        affordable = int(budget // price)
        to_buy = min(reserve_target - held, affordable)
        if to_buy > 0:
            return [["BUY_PRODUCT", "WHEAT", to_buy]], budget - to_buy * price
        return [], budget

    def _build_fertilizer_orders(self, state: GameState, plan: DayPlan, budget: float) -> tuple:
        """Opportunistic top-up via BUY_PRODUCT (only WHEAT/FERTILIZER can be
        bought back, README) whenever there's an active crop portfolio that
        could use the yield boost and the shed is below the small reserve
        target. Collected-from-animals fertilizer counts toward the same
        reserve, so this only fires to fill the gap."""
        if not plan.crop_targets:
            return [], budget
        held = state.shed.get("FERTILIZER", 0)
        if held >= FERTILIZER_RESERVE_TARGET:
            return [], budget
        price = current_price("FERTILIZER", state)
        if not price:
            return [], budget
        # v15: the actual bug found via a "pass"-baseline seed sweep
        # (NOTES.md "v15") — this re-check runs every turn (fertilizer
        # gets consumed mid-day by FERTILIZE), and with no price ceiling
        # it kept re-buying through a self-inflated price spike (FERTILIZER's
        # small T=200 means repeated small buys ratchet price up fast),
        # once observed paying $202 for a single unit of a $100-base item
        # during the exact pre-first-harvest cash-tight window where that
        # money was needed for HIRE instead. A missed fertilize bonus costs
        # far less than chasing an inflated price for one.
        if price > FERTILIZER_BUY_CEILING_MULT * PRODUCT_BASE_PRICE["FERTILIZER"]:
            return [], budget
        affordable = int(budget // price)
        to_buy = min(FERTILIZER_RESERVE_TARGET - held, affordable)
        if to_buy > 0:
            return [["BUY_PRODUCT", "FERTILIZER", to_buy]], budget - to_buy * price
        return [], budget

    def _build_animal_orders(self, state: GameState, plan: DayPlan, budget: float) -> tuple:
        orders = []
        for animal, target in plan.animal_targets.items():
            if target <= 0:
                continue
            info = ANIMAL_DATA[animal]
            owned = state.shed.get(animal, 0)
            for inv in state.inventories:
                owned += inv.get(animal, 0)
            for row in state.my_farm.tiles:
                for cell in row:
                    if isinstance(cell, StructureTile) and cell.animal == animal:
                        owned += 1
            shortfall = target - owned
            if shortfall <= 0:
                continue
            affordable = int(budget // info["cost"])
            to_buy = min(shortfall, affordable)
            if to_buy > 0:
                orders.append(["BUY_ANIMAL", animal, to_buy])
                budget -= to_buy * info["cost"]
        return orders, budget

    def _build_sell_orders(self, state: GameState, plan: DayPlan) -> list:
        """Size each SELL order against MarketModel's hold floor (§5.4)
        instead of dumping the whole shed: hold if the current price is
        already below the floor, otherwise sell only as many units as keep
        the price at or above the floor (spreading large sells across
        turns automatically, since the leftover stays in the shed for the
        next turn's re-evaluation).

        WHEAT is also the only feed source (README: "must be fed wheat
        daily") — reserve one unit per owned animal so a FETCH_WHEAT pickup
        (farm_ops.UnitScheduler) never loses the race against this same-turn
        SELL order and starves an animal into escaping. FERTILIZER has the
        same race against the FERTILIZE mechanic, reserved the same way."""
        orders = []
        # x2 buffer: a unit fetching feed takes at least 2 turns round-trip
        # (pickup, then walk/feed), so a 1-day reserve can still run dry
        # mid-transit on a bad day.
        wheat_reserve = self._animal_count(state) * 2 if plan.sell_thresholds.get("WHEAT") is not None else 0
        fertilizer_reserve = FERTILIZER_RESERVE_TARGET if plan.sell_thresholds.get("FERTILIZER") is not None else 0

        # M6 hardening: shed overflow discards items silently at end-of-day
        # (README, shedCapacity default 100) — once close to full, bypass
        # every hold floor so nothing sitting in the shed gets thrown away
        # for free next end-of-day.
        overflow_risk = sum(state.shed.values()) >= SHED_OVERFLOW_GUARD_THRESHOLD

        for resource, hold_pct in plan.sell_thresholds.items():
            qty = state.shed.get(resource, 0)
            if resource == "WHEAT":
                qty = max(0, qty - wheat_reserve)
            elif resource == "FERTILIZER":
                qty = max(0, qty - fertilizer_reserve)
            if qty <= 0:
                continue
            base_price = PRODUCT_BASE_PRICE.get(resource)
            if base_price is None:
                continue

            # v12: the more of THIS resource already sitting unsold, the
            # less we insist on waiting for a price recovery — see
            # SELL_CAPITULATION_QTY's config comment for why. Reaches full
            # capitulation (floor multiplier 0) at that quantity, well
            # before SHED_OVERFLOW_GUARD_THRESHOLD's total-shed check.
            capitulation = max(0.0, 1 - qty / SELL_CAPITULATION_QTY)
            effective_hold_pct = hold_pct * capitulation

            price_now = current_price(resource, state)
            floor_price = 0.0 if overflow_risk else effective_hold_pct * base_price
            if price_now < floor_price:
                continue  # hold — price already too depressed to sell into
            inventory = state.market_inventory.get(resource, 10000)
            sell_qty = min(qty, max_sell_before_floor(resource, inventory, floor_price))
            if sell_qty > 0:
                orders.append(["SELL", resource, sell_qty])
        return orders

    def _animal_count(self, state: GameState) -> int:
        return sum(
            1 for row in state.my_farm.tiles for cell in row
            if isinstance(cell, StructureTile) and cell.animal is not None
        )


class FarmOpsOrchestrator:
    def __init__(self):
        self.unit_scheduler = UnitScheduler()
        self.market_executor = MarketExecutor()

    def run(self, state: GameState, plan: DayPlan) -> dict:
        unit_actions = self.unit_scheduler.assign(state, plan)
        market_orders = self.market_executor.build_orders(state, plan)
        farmer_action = unit_actions[0] if unit_actions else ["PASS"]
        hand_actions = unit_actions[1:] if len(unit_actions) > 1 else []
        return {
            "farmer": farmer_action,
            "hands": hand_actions,
            "market": market_orders,
        }
