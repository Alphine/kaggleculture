"""StrategyAgent: economic/strategic layer (SDD §3.2, §5).

M3 adds real sell-policy hold thresholds (§5.4, sourced from MarketModel via
farm_ops) — this module just supplies the per-resource floors from config.
M4 adds animal portfolio decisions (§5.5): which animals to buy this phase
and how many coop/pasture structures are needed to house them.
"""

import math
from dataclasses import dataclass, field

from .config import (
    ANIMAL_CAPACITY_PER_UNIT,
    ANIMAL_CASH_FRACTION,
    ANIMAL_CASH_RESERVE,
    ANIMAL_DATA,
    CROP_DATA,
    CROP_PRIORITY_ORDER,
    CROP_TARGET_WEIGHT,
    ENABLE_MARKET_TREND_SIGNALS,
    ENABLE_OPPONENT_EXPANSION_MIRROR,
    ENABLE_OPPONENT_TREND_TRACKING,
    ENABLE_SUPPLY_PRESSURE_ADJUSTMENT,
    EARLY_PHASE_MELON_CASH_RESERVE,
    EARLY_PHASE_MELON_TILE_CAP,
    EXPANSION_CAPACITY_BUFFER,
    EXPANSION_UTILIZATION_THRESHOLD,
    HIRE_TRIGGER_LOGIC,
    HOLD_FLOOR_PCT,
    LAND_COSTS,
    LAND_PURCHASE_ORDER,
    LIQUIDATION_DAYS_REMAINING,
    MARKET_PARAMS,
    MARKET_TREND_SELL_DISCOUNT,
    MARKET_TREND_SURGE_PENALTY,
    MARKET_TREND_SURGE_THRESHOLD,
    MARKET_TREND_WINDOW_DAYS,
    MAX_WEED_RATIO_FOR_EXPANSION,
    MIN_DAY_FOR_FIRST_EXPANSION,
    MIN_DAYS_BETWEEN_EXPANSIONS,
    MIN_DAYS_REMAINING_FOR_EXPANSION,
    OPPONENT_EXPANSION_MIRROR_THRESHOLD,
    OPPONENT_FLOOD_PENALTY,
    OPPONENT_GROWTH_PENALTY,
    OPPONENT_SUPPLY_PRESSURE_HOLD_DISCOUNT,
    OPPONENT_SUPPLY_PRESSURE_THRESHOLD,
    OPPONENT_TREND_WINDOW_DAYS,
    PHASE_ANIMAL_CAP,
    PHASE_CROP_POOL,
    PHASE_DAY_BOUNDARIES,
    PRODUCT_BASE_PRICE,
    REGIME_AHEAD_RATIO,
    REGIME_BEHIND_RATIO,
    REGIME_PRESETS,
    ENABLE_REGIME_SWITCHING,
    TIGHT_HIRE_CAPACITY_PER_UNIT,
    TOTAL_DAYS,
    YIELD_PER_TILE_PER_DAY,
)
from .state import GameState, PlantTile, StructureTile, WeedTile


@dataclass
class DayPlan:
    day: int
    phase: str
    crop_targets: dict          # {crop: desired_tile_count}, only crops with target > 0 get new plantings
    animal_targets: dict        # {animal: desired_total_owned}, only animals with target > 0 get bought
    structure_targets: dict     # {"COOP": n, "PASTURE": m} derived from animal_targets
    hire_count: int
    buy_land: bool
    sell_thresholds: dict       # {resource: hold_floor_pct}
    regime: str = "neutral"     # v14 orchestrator's standing classification for this day
    land_cost: float = 0.0      # v16: cost of the pending BUY_LAND, so farm_ops can budget it


def phase_for_day(day: int) -> str:
    if day <= PHASE_DAY_BOUNDARIES["early_end"]:
        return "early"
    if day <= PHASE_DAY_BOUNDARIES["buildout_end"]:
        return "buildout"
    if day <= PHASE_DAY_BOUNDARIES["scale_end"]:
        return "scale"
    return "endgame"


def fib_cost(n: int) -> int:
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _count_unlocked_tiles(farm) -> int:
    return sum(1 for row in farm.tiles for cell in row if cell != "LOCKED")


def _count_utilized_tiles(farm) -> int:
    # A weed is wasted space, not utilization — counting it as "utilized"
    # would make an already-overwhelmed, weed-infested farm look MORE
    # eligible for expansion instead of less (found via a real leaderboard
    # replay: our farm hit 23/75 weeds after expanding while "utilization"
    # stayed high enough to keep triggering the next BUY_LAND).
    return sum(
        1 for row in farm.tiles for cell in row
        if isinstance(cell, (PlantTile, StructureTile))
    )


def _count_weeds(farm) -> int:
    return sum(1 for row in farm.tiles for cell in row if isinstance(cell, WeedTile))


def _next_land_to_buy(farm):
    for quadrant in LAND_PURCHASE_ORDER:
        if quadrant not in farm.unlocked_quadrants:
            return quadrant
    return None


def _eligible_crops_for_phase(phase: str, remaining_days: int) -> list:
    pool = PHASE_CROP_POOL[phase]
    if phase != "endgame":
        # Endgame check applies at every phase too: don't start a crop that
        # can't reach first yield before the season ends (SDD §5.1).
        return [c for c in pool if remaining_days >= CROP_DATA[c]["first_yield_day"]]
    # Explicit endgame pool is empty by config; recover any crop that can
    # still fit an entire quick cycle in the remaining days.
    return [
        c for c in CROP_PRIORITY_ORDER
        if remaining_days >= CROP_DATA[c]["first_yield_day"]
    ]


# === SPECULATOR EXPERT: opponent & market trend signals (v10/v11), plus
# the v14 orchestrator's standing classifier below. Reads what's visible
# about the opponent and the shared market and turns it into risk signals
# for the economist to react to — never decides WHAT to plant/buy/expand
# itself, only how cautiously or aggressively to do it. ===================


def _opponent_crop_counts(state: GameState) -> dict:
    """Opponent's farm is fully visible (README: "can see the state of the
    opponent's farm", only their shed/inventory is hidden) — count what
    they've planted so crop scoring can react to it."""
    counts = {}
    for row in state.opp_farm.tiles:
        for cell in row:
            if isinstance(cell, PlantTile):
                counts[cell.crop] = counts.get(cell.crop, 0) + 1
    return counts


def _opponent_resource_counts(state: GameState) -> dict:
    """Crop counts plus animal counts, both keyed by PRODUCT name (matching
    PRODUCT_BASE_PRICE/YIELD_PER_TILE_PER_DAY) rather than crop/animal
    name, so crop scoring, growth tracking, and market-supply forecasting
    can all share one opponent-visibility snapshot. Crop names already ARE
    their own product name; animals map through ANIMAL_DATA."""
    counts = dict(_opponent_crop_counts(state))
    for row in state.opp_farm.tiles:
        for cell in row:
            if isinstance(cell, StructureTile) and cell.animal is not None:
                product = ANIMAL_DATA[cell.animal]["product"]
                counts[product] = counts.get(product, 0) + 1
    return counts


def _opponent_growth_rate(history: dict, day: int, resource: str, window: int) -> float:
    """Tiles/day change in the opponent's visible resource count over the
    last `window` days, from a {day: {resource: count}} history dict this
    StrategyAgent instance has been accumulating (see plan()). Returns 0.0
    if there's no data far back enough yet (early game, or a fresh
    instance in a unit test with no accumulated history) — growth-rate
    scoring degrades gracefully to the pre-v10 behavior in that case."""
    past_day = day - window
    if past_day not in history or day not in history:
        return 0.0
    past_count = history[past_day].get(resource, 0)
    now_count = history[day].get(resource, 0)
    return (now_count - past_count) / window


def _market_trend_momentum(history: dict, day: int, resource: str, window: int) -> float:
    """Day-over-day change in the market's ACTUAL inventory for `resource`,
    normalized to units of that resource's own MARKET_PARAMS "T" per day
    (so "big" means the same thing whether it's wheat, T=400, or
    strawberry, T=100). Ground truth, not a proxy: reflects both players'
    selling AND town consumption in one number — catches things opponent-
    tile tracking misses entirely, e.g. an opponent who already
    harvested-and-dumped and moved their tiles on to something else, where
    the price-crash from that dump is still working through the market."""
    past_day = day - window
    if past_day not in history or day not in history:
        return 0.0
    past_inv = history[past_day].get(resource)
    now_inv = history[day].get(resource)
    if past_inv is None or now_inv is None:
        return 0.0
    t = MARKET_PARAMS.get(resource, {}).get("T", 1)
    return (now_inv - past_inv) / window / t


def _economic_footprint(state: GameState, side: str) -> float:
    """Visible daily production-VALUE estimate for `side` ('my' or 'opp'):
    sum over every visible crop tile / occupied structure of its own
    YIELD_PER_TILE_PER_DAY x PRODUCT_BASE_PRICE. Same units for both sides
    even though neither player's cash is visible to the other (README: the
    opponent's farm IS fully visible, only their shed/inventory is hidden)
    — this is the v14 orchestrator's only available "who's ahead" proxy."""
    farm = state.my_farm if side == "my" else state.opp_farm
    total = 0.0
    for row in farm.tiles:
        for cell in row:
            if isinstance(cell, PlantTile):
                total += YIELD_PER_TILE_PER_DAY.get(cell.crop, 0.0) * PRODUCT_BASE_PRICE.get(cell.crop, 0.0)
            elif isinstance(cell, StructureTile) and cell.animal is not None:
                product = ANIMAL_DATA[cell.animal]["product"]
                total += YIELD_PER_TILE_PER_DAY.get(product, 0.0) * PRODUCT_BASE_PRICE.get(product, 0.0)
    return total


def _classify_regime(state: GameState) -> str:
    """v14 orchestrator signal: classify our standing from the visible
    footprint proxy above. Uses a ratio BAND rather than a hard >/< split
    so the regime doesn't flip-flop on small day-to-day noise near parity
    — only a clearly lopsided footprint moves off "neutral"."""
    our_footprint = _economic_footprint(state, "my")
    opp_footprint = _economic_footprint(state, "opp")
    # Either side at zero footprint is a data-insufficiency case (very
    # early game, or an opponent that genuinely hasn't planted anything
    # yet), not real evidence of a lead — treating "opponent has 0 visible
    # tiles" as automatic "ahead" would bias the very early game on noise
    # alone. Wait for both sides to have SOME visible footprint before
    # trusting the ratio.
    if our_footprint <= 0 or opp_footprint <= 0:
        return "neutral"
    ratio = our_footprint / opp_footprint
    if ratio >= REGIME_AHEAD_RATIO:
        return "ahead"
    if ratio <= REGIME_BEHIND_RATIO:
        return "behind"
    return "neutral"


def _score_crop(crop: str, opponent_counts: dict, opponent_growth: dict, market_momentum: dict) -> float:
    info = CROP_DATA[crop]
    # Expected profit per tile-day per action-cost (SDD §5.5), approximated
    # from the Object Types table: (base_price - seed_cost) amortized over
    # time-to-first-yield, favoring fast capital recycling. Biased by
    # CROP_TARGET_WEIGHT — this raw formula treats every crop as a one-shot
    # payout and badly undervalues ongoing earners like STRAWBERRY (4
    # scheduled cycles, not 1); the weight corrects for that using a real
    # top-opponent's observed allocation as ground truth (see config.py).
    score = (info["base_price"] - info["seed_cost"]) / max(info["first_yield_day"], 1)
    score *= CROP_TARGET_WEIGHT.get(crop, 1.0)

    # SDD §5.5: "if opponent is flooding a resource, expect price crash
    # there — diversify away from it." Both players sell into the SAME
    # shared market (README "Market" — one dynamic market, not per-player),
    # so an opponent already planting heavily into a crop is a leading
    # indicator of oversupply by the time our own harvest lands. Each
    # opponent tile of that crop discounts our score for it a bit.
    opp_n = opponent_counts.get(crop, 0)
    if opp_n:
        score /= 1 + opp_n * OPPONENT_FLOOD_PENALTY

    # v10: an opponent rapidly SCALING UP a crop is a bigger threat than
    # one sitting on a large but static count — their growth has more room
    # to compound before our harvest lands. Only ever discounts further
    # (a shrinking/negative rate contributes nothing), and only engages
    # once there's enough accumulated history to measure a rate at all.
    growth = opponent_growth.get(crop, 0.0)
    if growth > 0:
        score /= 1 + growth * OPPONENT_GROWTH_PENALTY

    # v11: the market's OWN realized inventory trend, independent of (and
    # complementary to) the opponent-tile-based signals above — inventory
    # rising fast means the price is already heading toward the floor
    # regardless of who's causing it, so redirecting new seed budget away
    # from it is justified either way (SDD §5.5-adjacent, but grounded in
    # actual market state rather than an opponent-visibility proxy).
    momentum = market_momentum.get(crop, 0.0)
    if momentum > 0:
        score /= 1 + momentum * MARKET_TREND_SURGE_PENALTY
    return score


# === ECONOMIST EXPERT: portfolio planning — crop/animal targets, land
# expansion, hiring. Decides WHAT to grow and how much capacity to run;
# takes the speculator's risk signals as inputs but never computes them
# itself. Every function below optionally takes a regime multiplier (v14)
# from the orchestrator's REGIME_PRESETS, defaulting to 1.0 (no-op) so
# regime-switching is purely additive over the pre-v14 behavior. ==========


def _decide_crop_targets(state: GameState, phase: str, remaining_days: int, opponent_growth: dict, market_momentum: dict, concentration_mult: float = 1.0) -> dict:
    eligible = _eligible_crops_for_phase(phase, remaining_days)
    unlocked_tiles = _count_unlocked_tiles(state.my_farm)
    if not eligible or unlocked_tiles == 0:
        return {}

    opponent_counts = _opponent_crop_counts(state)
    # Proportional allocation by score, not equal-shares-by-rank: a crop
    # scoring 2x another should get roughly 2x the tiles, not just "picked
    # first" for a +1 remainder tile. Every score is clamped to a small
    # positive floor so a crop that scores 0 (or negative, from a heavy
    # opponent-flood discount) still gets some minimal presence rather than
    # vanishing outright.
    raw_scores = {c: max(0.01, _score_crop(c, opponent_counts, opponent_growth, market_momentum)) for c in eligible}
    # v14: concentration_mult sharpens (>1, "behind" preset) or flattens
    # (<1, "ahead" preset) the allocation spread via a monotonic power
    # transform — preserves crop RANKING exactly, only changes how
    # lopsided the tile split is, so this composes cleanly with every
    # existing scoring signal above instead of overriding it. Normalized
    # by the max raw score first so the transform is scale-invariant —
    # _score_crop's absolute units vary a lot by context (a heavily
    # opponent-flood-discounted score can be a tiny fraction vs a healthy
    # one), and exponentiating un-normalized values would distort small
    # scores far more aggressively than large ones for no strategic reason.
    if concentration_mult != 1.0:
        max_score = max(raw_scores.values())
        scores = {c: (v / max_score) ** concentration_mult for c, v in raw_scores.items()}
    else:
        scores = raw_scores
    total_score = sum(scores.values())
    ranked = sorted(eligible, key=lambda c: scores[c], reverse=True)

    targets = {}
    allocated = 0
    for i, crop in enumerate(ranked):
        if i == len(ranked) - 1:
            targets[crop] = max(0, unlocked_tiles - allocated)  # remainder goes to the last (lowest-scoring)
        else:
            share = round(unlocked_tiles * scores[crop] / total_score)
            targets[crop] = share
            allocated += share

    # v15: throttle MELON's EARLY-phase tile share, but ONLY while cash is
    # actually tight — found via a "pass"-baseline seed sweep (NOTES.md
    # "v15") that v13's fix (adding MELON to PHASE_CROP_POOL["early"] for
    # a first-mover pricing edge against melon-rushing opponents) had a
    # severe, previously-uncaught side effect: MELON's CROP_TARGET_WEIGHT
    # =3.0 dominance claimed ~80% of early tile allocation (20/25),
    # crowding WHEAT/CARROT down to a handful of tiles each — starving the
    # ONLY fast-cash income (first yield day 2) available before melon's
    # own day-10 payoff. 7/10 seeds crashed into a permanent cash-crunch/
    # weed spiral even against "pass" (a totally passive, non-competing
    # opponent). A FLAT cap (regardless of cash) fixed that completely but
    # also cost the entire v13 first-mover win against `melon_focus_agent`
    # (10/10 -> 0/10) — that matchup's cash stays healthy throughout, so
    # an unconditional cap was solving a problem that specific matchup
    # never had. Gating on `EARLY_PHASE_MELON_CASH_RESERVE` instead: full
    # melon commitment proceeds whenever cash is healthy (preserving the
    # first-mover win), and only throttles back — freeing tiles to WHEAT,
    # not proportionally to every early crop, since WHEAT is specifically
    # the fast-cash crop this is protecting — when a real crunch is
    # actually developing.
    if (
        phase == "early"
        and state.my_farm.money < EARLY_PHASE_MELON_CASH_RESERVE
        and "MELON" in targets
        and targets["MELON"] > EARLY_PHASE_MELON_TILE_CAP
    ):
        freed = targets["MELON"] - EARLY_PHASE_MELON_TILE_CAP
        targets["MELON"] = EARLY_PHASE_MELON_TILE_CAP
        targets["WHEAT"] = targets.get("WHEAT", 0) + freed
    return targets


def _decide_land_expansion(state: GameState, remaining_days: int, expansion_mult: float = 1.0, last_expansion_day: int = None) -> bool:
    farm = state.my_farm
    next_land = _next_land_to_buy(farm)
    if next_land is None or remaining_days < MIN_DAYS_REMAINING_FOR_EXPANSION:
        return False
    if next_land == "NE" and state.day < MIN_DAY_FOR_FIRST_EXPANSION:
        return False
    # v16: cooldown since the LAST expansion (any quadrant, not just the
    # first) — see MIN_DAYS_BETWEEN_EXPANSIONS's config.py comment for
    # the real-match failure this fixes (buying 3 quadrants in ~5 days
    # left most of the new land empty for the rest of the game).
    if last_expansion_day is not None and state.day - last_expansion_day < MIN_DAYS_BETWEEN_EXPANSIONS:
        return False
    # v18: don't buy land we could never realistically staff. Max
    # sustainable coverage = (1 farmer + max_hires) x capacity_per_unit,
    # buffered — see EXPANSION_CAPACITY_BUFFER's config.py comment for
    # the real-match evidence (55-85% of land sitting empty at game end
    # even with the pacing cooldown alone).
    max_coverable_tiles = (1 + HIRE_TRIGGER_LOGIC["max_hires"]) * HIRE_TRIGGER_LOGIC["capacity_per_unit"] * EXPANSION_CAPACITY_BUFFER
    projected_tiles = _count_unlocked_tiles(farm) + 25  # next quadrant is always 25 tiles
    if projected_tiles > max_coverable_tiles:
        return False
    unlocked = _count_unlocked_tiles(farm)
    utilized = _count_utilized_tiles(farm)
    utilization = utilized / unlocked if unlocked else 0.0
    cost = LAND_COSTS[next_land]

    # Weeds are direct evidence we can't keep up with the land we already
    # have — expanding further would only spread unit-turns thinner and
    # compound the backlog. Found via a real leaderboard replay where our
    # farm hit 23/75 (31%) weeds after over-expanding while a
    # single-quadrant opponent with zero weeds tripled our money.
    weed_ratio = _count_weeds(farm) / unlocked if unlocked else 0.0
    if weed_ratio > MAX_WEED_RATIO_FOR_EXPANSION:
        return False

    # v11: real leaderboard replays repeatedly showed opponents who NEVER
    # expand past their starting quadrant beating us decisively — a signal
    # our own utilization-only trigger never looked at. If the opponent
    # hasn't committed to expanding either, hold ourselves to a stricter
    # bar; once they HAVE, their own decision already validated that
    # expanding is viable in this particular matchup, so the normal bar
    # applies with no extra caution.
    required_utilization = EXPANSION_UTILIZATION_THRESHOLD
    if ENABLE_OPPONENT_EXPANSION_MIRROR and len(state.opp_farm.unlocked_quadrants) <= 1:
        required_utilization = OPPONENT_EXPANSION_MIRROR_THRESHOLD
    # v14: >1.0 (ahead) raises the bar — no reason to gamble a likely win
    # on more land to expand a margin the rating won't reward. <1.0
    # (behind) lowers it — worth growing capacity faster to catch up.
    required_utilization *= expansion_mult

    return utilization >= required_utilization and farm.money >= cost * 1.2


def _eligible_animals_for_phase(phase: str, remaining_days: int) -> dict:
    caps = PHASE_ANIMAL_CAP[phase]
    # +2 buffer days: 1 to build the structure, 1 to actually collect the
    # first yield rather than have it land right on the last turn.
    return {a: cap for a, cap in caps.items() if remaining_days >= ANIMAL_DATA[a]["first_yield_day"] + 2}


def _existing_animal_count(state: GameState, animal: str) -> int:
    count = state.shed.get(animal, 0)
    for inv in state.inventories:
        count += inv.get(animal, 0)
    for row in state.my_farm.tiles:
        for cell in row:
            if isinstance(cell, StructureTile) and cell.animal == animal:
                count += 1
    return count


def _decide_animal_targets(state: GameState, phase: str, remaining_days: int, cash_mult: float = 1.0, expected_units: int = None) -> dict:
    caps = _eligible_animals_for_phase(phase, remaining_days)
    if not caps:
        return {}

    # Same guard as land expansion, same reason: weeds are direct evidence
    # the farm is already falling behind, and buying MORE animals (more
    # feed demand, more structures competing for tile space) while it's
    # struggling only compounds the problem. A 10-seed sweep caught a real
    # death spiral where continued animal spending during a weed crisis
    # starved the cash needed to recover — this freezes new animal growth
    # (existing owned animals still get fed/maintained normally) until the
    # farm is back under control.
    farm = state.my_farm
    unlocked = _count_unlocked_tiles(farm)
    weed_ratio = _count_weeds(farm) / unlocked if unlocked else 0.0
    if weed_ratio > MAX_WEED_RATIO_FOR_EXPANSION:
        return {a: _existing_animal_count(state, a) for a in caps if _existing_animal_count(state, a) > 0}

    # `target` is a CAP on total ever-owned, not a per-day increment — once
    # reached, shortfall drops to 0 and MarketExecutor stops buying
    # regardless of how much cash is sitting in the bank. Highest $/day
    # product value gets first claim on the day's animal budget — NOT
    # fastest-to-first-yield, which would rank SHEEP (6 days) ahead of COW
    # (8 days) despite COW being the better earner (base_price/interval_days:
    # COW 160/2=$80/day, SHEEP 200/3=$66.7/day) — matches the real
    # top-opponent's observed ~2:1 COW:SHEEP ratio (see config.py).
    # v23: cap total herd growth by unit CARE capacity too, not just cash
    # — see ANIMAL_CAPACITY_PER_UNIT's config.py comment for the real-match
    # loss this fixes (8 COW + 7 SHEEP bought, cash-affordable each day,
    # but only 4 SHEEP ever stayed fed long enough to matter).
    # `expected_units` is threaded in by plan() as the post-hire headcount
    # (hire_count is decided before this call now) since state.my_farm.hands
    # is ALWAYS empty at this point in the turn (hands reset at day boundary,
    # plan() runs at hour==0 before that day's HIRE orders execute) — reading
    # len(state.my_farm.hands) here silently capped every real game's herd at
    # ~1 animal forever, regardless of actual hire_count (found via a v24
    # real-match trace: hands=5 all game, but animal_targets stuck at {'COW': 1}).
    current_units = expected_units if expected_units is not None else 1 + len(state.my_farm.hands)
    total_owned_all_animals = sum(_existing_animal_count(state, a) for a in ANIMAL_DATA)
    herd_room = max(0, current_units * ANIMAL_CAPACITY_PER_UNIT - total_owned_all_animals)

    spendable = max(0.0, state.my_farm.money - ANIMAL_CASH_RESERVE) * ANIMAL_CASH_FRACTION * cash_mult
    ranked = sorted(caps, key=lambda a: ANIMAL_DATA[a]["base_price"] / ANIMAL_DATA[a]["interval_days"], reverse=True)
    targets = {}
    for animal in ranked:
        cost = ANIMAL_DATA[animal]["cost"]
        cap = caps[animal]
        owned = _existing_animal_count(state, animal)
        room_wanted = max(0, cap - owned)
        affordable = int(spendable // cost) if cost else 0
        room = min(room_wanted, affordable, math.floor(herd_room))
        herd_room -= room
        spendable -= room * cost
        target = owned + room
        if target > 0:
            targets[animal] = target
    return targets


def _decide_structure_targets(animal_targets: dict) -> dict:
    structure_targets = {"COOP": 0, "PASTURE": 0}
    for animal, target in animal_targets.items():
        structure_targets[ANIMAL_DATA[animal]["structure"]] += target
    return structure_targets


def _opponent_supply_pressure(opponent_counts: dict, opponent_growth: dict) -> dict:
    """Projects each resource's opponent-driven daily market-supply add a
    week out (current visible tile/structure count, plus a week of their
    measured growth trend, times the README yield/tile/day figure) — a
    predictive stand-in for "what will the market look like soon" rather
    than reacting only after the glut has already landed."""
    pressure = {}
    for resource, count in opponent_counts.items():
        yield_rate = YIELD_PER_TILE_PER_DAY.get(resource)
        if yield_rate is None:
            continue
        growth = opponent_growth.get(resource, 0.0)
        projected_count = max(0.0, count + growth * 7)
        pressure[resource] = projected_count * yield_rate
    return pressure


def _decide_hires(state: GameState, crop_targets: dict, phase_capacity_per_unit: float = None) -> int:
    """Returns the desired hire count based on actionable-tile need ALONE —
    NOT pre-limited by a cash-fraction budget here. A real leaderboard
    replay caught why that used to matter: gating the desired count by
    `money x fraction` meant that once an animal-heavy cash crunch drove
    money down to a couple dollars, the fraction of that was below even
    the first hire's $1 cost, and hire_count computed to 0 for 9 STRAIGHT
    days despite dozens of tiles needing attention every one of them —
    hiring was cut off exactly when hands were needed most to dig out.
    Actual affordability is now checked once, against the real live
    per-turn budget, in farm_ops.MarketExecutor.build_orders (which also
    budgets HIRE first, before crop seeds/animals, since hands are the
    foundational capacity everything else depends on).

    v15: an EMPTY tile only counts as actionable if `crop_targets` still
    wants something planted there — an empty tile with nothing left to
    plant (endgame, no crop fits the remaining days) isn't real hiring
    need. An EXISTING unwatered plant always counts regardless of
    crop_targets — it needs saving whether or not we're still planting
    new ones, since losing it to a weed forfeits a harvest we could
    otherwise still sell before the season ends (found via a real-match
    replay, NOTES.md "v15": forcing hire_count to 0 during the old
    liquidation window dropped a 4-quadrant farm to 0 hands and weed
    events exploded 27->44->53 tiles in the final 2 days alone).

    v18 tried counting WeedTiles as actionable too, after a real-replay
    trace found hands silently dropping 4->2 and never recovering despite
    a healthy bank ($18k-26k) — a tile that turns into a weed simply
    disappears from this count, which looked like a perverse incentive
    (more weeds -> formula thinks fewer hands are needed). Reverted after
    validation against melon_focus_agent caught a hard regression (win
    rate 15/15 -> 9/15): counting weeds inflated desired hire counts
    enough to compete with the early seed-buying budget (HIRE is budgeted
    first in farm_ops.MarketExecutor), the same failure mode already
    diagnosed once this session for a different parameter
    (`capacity_per_unit`, see NOTES.md "v17"). The original symptom (hands
    not recovering despite healthy cash) is real and still unexplained —
    just not fixed by this particular lever. See NOTES.md "v18" for the
    full investigation.

    v18 (2nd attempt): a day-by-day trace of that same close loss showed
    the formula ISN'T actually stuck — it recomputes fresh every day and
    consistently lands on the SAME plateau (e.g. ceil(18 tiles / 6
    capacity) = 3 total units for an 18-tile farm), which is a fraction
    short of what's needed to fully prevent the occasional missed
    watering that eventually becomes a weed. This is chronic marginal
    UNDER-provisioning from `capacity_per_unit=6` being a bit optimistic,
    not a stuck-value bug — but tightening it globally (tested in
    isolation, `capacity_per_unit=5` alone) regressed vs
    `melon_focus_agent` (15/15 -> 10/15) for the same reason as before:
    more hires desired means more Fibonacci-cost spend competing with the
    early seed-buying budget, even without touching `max_hires`. Since
    the actual understaffing symptom only showed up well into
    buildout/scale (day 14+, long past the vulnerable early-cash window),
    `phase_capacity_per_unit` lets the caller tighten staffing ONLY once
    that window has passed, leaving early/buildout at the current,
    validated value."""
    farm = state.my_farm
    still_planting = sum(crop_targets.values()) > 0
    actionable_tiles = 0
    for row in farm.tiles:
        for cell in row:
            if cell == "LOCKED":
                continue
            if cell is None:
                if still_planting:
                    actionable_tiles += 1
            elif isinstance(cell, PlantTile) and not cell.watered_today:
                actionable_tiles += 1
            elif isinstance(cell, StructureTile) and cell.animal is not None:
                # v27 (continued): a placed animal needs CARE every day,
                # the same daily-reset cadence as an unwatered plant needing
                # WATER — but this formula only ever counted plant-tile
                # workload, so it was blind to animal-care demand entirely.
                # A real-match trace of 3 blowout losses (0.38x-0.44x, all
                # right after v27's structure fix started actually getting
                # 8-9 animals placed) found hands dropping 5->2 around day
                # 13-16 and staying there for the rest of the game: once the
                # crop portfolio was fully planted and well-watered, this
                # formula's actionable count shrank, so it kept computing a
                # LOWER hire need even though 8-9 newly-placed animals were
                # now competing with crops for the same 2 hands' turns every
                # day — the exact "hands not recovering despite healthy
                # cash" symptom flagged as real-but-unexplained back in v18
                # (see this function's docstring). Counting each placed
                # animal as one actionable unit closes that blind spot.
                actionable_tiles += 1
    current_units = 1 + len(farm.hands)
    capacity = phase_capacity_per_unit if phase_capacity_per_unit is not None else HIRE_TRIGGER_LOGIC["capacity_per_unit"]
    needed_units = max(1, math.ceil(actionable_tiles / capacity)) if actionable_tiles else 1
    to_hire = max(0, needed_units - current_units)
    return min(to_hire, HIRE_TRIGGER_LOGIC["max_hires"])


# === ORCHESTRATOR: StrategyAgent.plan() below combines the economist's
# portfolio targets with the speculator's opponent/market signals AND (v14)
# the standing classification into one DayPlan for the turn. The third
# "expert" role, agriculture (tactical unit scheduling — who walks where and
# does what this turn), lives in farm_ops.FarmOpsOrchestrator; it only
# consumes the DayPlan this orchestrator produces and never feeds back into
# it, so it isn't wired in here. ============================================


class StrategyAgent:
    def __init__(self):
        # v10 opponent-trend tracking: {day: {resource: count}}, accumulated
        # across this instance's lifetime. agent.py keeps one StrategyAgent
        # per episode (module-level, persists across turns — SDD §7), so
        # this naturally builds up a real history over a game. A freshly
        # constructed instance (e.g. in a unit test calling
        # StrategyAgent().plan(state) once) simply has no history yet, and
        # every growth-rate lookup degrades to 0.0 (pre-v10 behavior).
        self._opp_history = {}
        # v11: same pattern, but for the market's own realized inventory
        # per resource — {day: {resource: inventory}}.
        self._market_history = {}
        # v16: tracks the day of our own most recent land purchase, so
        # _decide_land_expansion can enforce MIN_DAYS_BETWEEN_EXPANSIONS.
        # `unlocked_quadrants` alone doesn't carry a purchase day, so this
        # is detected by watching the count change turn to turn.
        self._last_expansion_day = None
        self._last_quadrant_count = None

    def plan(self, state: GameState) -> DayPlan:
        phase = phase_for_day(state.day)
        remaining_days = TOTAL_DAYS - state.day

        quadrant_count = len(state.my_farm.unlocked_quadrants)
        if self._last_quadrant_count is not None and quadrant_count > self._last_quadrant_count:
            self._last_expansion_day = state.day
        self._last_quadrant_count = quadrant_count

        opponent_growth = {}
        supply_pressure = {}
        if ENABLE_OPPONENT_TREND_TRACKING:
            self._opp_history[state.day] = _opponent_resource_counts(state)
            resources = self._opp_history[state.day]
            opponent_growth = {
                r: _opponent_growth_rate(self._opp_history, state.day, r, OPPONENT_TREND_WINDOW_DAYS)
                for r in resources
            }
            if ENABLE_SUPPLY_PRESSURE_ADJUSTMENT:
                supply_pressure = _opponent_supply_pressure(resources, opponent_growth)

        market_momentum = {}
        if ENABLE_MARKET_TREND_SIGNALS:
            self._market_history[state.day] = dict(state.market_inventory)
            market_momentum = {
                r: _market_trend_momentum(self._market_history, state.day, r, MARKET_TREND_WINDOW_DAYS)
                for r in self._market_history[state.day]
            }

        # v14 orchestrator step: classify standing, pick the matching preset.
        # ENABLE_REGIME_SWITCHING=False (or an unrecognized regime) falls
        # back to "neutral", whose multipliers are all no-ops — the entire
        # feature degrades cleanly to pre-v14 behavior when disabled, same
        # pattern as every other flag in config.py.
        regime = _classify_regime(state) if ENABLE_REGIME_SWITCHING else "neutral"
        preset = REGIME_PRESETS.get(regime, REGIME_PRESETS["neutral"])

        crop_targets = _decide_crop_targets(
            state, phase, remaining_days, opponent_growth, market_momentum,
            concentration_mult=preset["concentration_mult"],
        )
        buy_land = False if phase == "endgame" else _decide_land_expansion(
            state, remaining_days, expansion_mult=preset["expansion_mult"],
            last_expansion_day=self._last_expansion_day,
        )
        # v16: found via a real-match replay analysis (NOTES.md "v16") —
        # farm_ops.MarketExecutor's shared budget never deducted BUY_LAND's
        # real cost, so same-turn animal orders queued right after it were
        # computed against a stale, pre-purchase budget. The game processes
        # orders sequentially and stops once money actually runs out
        # (README), so an overcommitted animal order could get silently
        # truncated by the ENGINE rather than by our own budget check —
        # observed directly: 5 PASTURE structures built but only 2 COW
        # ever purchased to fill them. land_cost lets farm_ops deduct it
        # like every other spend.
        land_cost = LAND_COSTS[_next_land_to_buy(state.my_farm)] if buy_land else 0.0

        # v15: hiring stays need-based even during liquidation. The
        # original "no time to recoup a new hand's cost" reasoning only
        # applies to hiring as a NEW INVESTMENT (fair to skip — the
        # LAND/SEED/ANIMAL purchases below still stop), but it doesn't
        # apply to maintaining coverage on a farm that ALREADY exists.
        # Found via a real-match replay (NOTES.md "v15"): hire_count
        # forced to 0 on days 28-29 dropped a 4-quadrant, 100-tile farm to
        # 0 hands (just the farmer), and weed events exploded 27 -> 44 ->
        # 53 tiles in exactly those 2 final days — destroying still-
        # unharvested crops that could otherwise have been sold before the
        # season ended, the opposite of what liquidation is supposed to
        # achieve.
        # v18: a stricter capacity_per_unit only once well past the
        # vulnerable early seed-buying window (see _decide_hires'
        # docstring) — None for early/buildout uses the normal
        # HIRE_TRIGGER_LOGIC value unchanged.
        phase_capacity = TIGHT_HIRE_CAPACITY_PER_UNIT if phase in ("buildout", "scale", "endgame") else None
        hire_count = _decide_hires(state, crop_targets, phase_capacity_per_unit=phase_capacity)

        # v24: animal_targets must be computed AFTER hire_count and given the
        # expected post-hire headcount — state.my_farm.hands is always empty
        # here (hands reset at day boundary, this runs at hour==0 before that
        # day's HIRE orders execute), so reading len(state.my_farm.hands)
        # permanently capped herd growth at ~1 animal regardless of the real
        # hire_count that executes later this same turn (see NOTES.md "v24").
        expected_units = 1 + len(state.my_farm.hands) + hire_count
        animal_targets = _decide_animal_targets(
            state, phase, remaining_days, cash_mult=preset["animal_cash_mult"],
            expected_units=expected_units,
        )
        structure_targets = _decide_structure_targets(animal_targets)

        if remaining_days <= LIQUIDATION_DAYS_REMAINING:
            # Liquidate: ignore hold floors, everything in the shed becomes
            # cash before the season ends rather than being stranded stock.
            sell_thresholds = {resource: 0.0 for resource in PRODUCT_BASE_PRICE}
        else:
            sell_thresholds = {resource: HOLD_FLOOR_PCT.get(resource, 0.0) for resource in PRODUCT_BASE_PRICE}
            # v10: an opponent's visible production trend projecting a glut
            # soon means holding for a price recovery is less likely to pay
            # off — sell into it sooner rather than waiting. Only ever
            # shrinks the floor (more willing to sell), never raises it.
            for resource, pressure in supply_pressure.items():
                if pressure > OPPONENT_SUPPLY_PRESSURE_THRESHOLD and resource in sell_thresholds:
                    sell_thresholds[resource] *= OPPONENT_SUPPLY_PRESSURE_HOLD_DISCOUNT
            # v11: same idea, but from the market's own realized trend
            # rather than a projection off the opponent's visible tiles —
            # independent evidence, applied independently (can compound
            # with the above, which is directionally fine: more evidence
            # of an incoming glut only ever means selling sooner is right).
            for resource, momentum in market_momentum.items():
                if momentum > MARKET_TREND_SURGE_THRESHOLD and resource in sell_thresholds:
                    sell_thresholds[resource] *= MARKET_TREND_SELL_DISCOUNT
            # v14: regime-wide hold-floor bias, layered on top of the two
            # evidence-driven discounts above (>1.0 ahead -> hold a bit
            # longer for fuller value; <1.0 behind -> sell faster to
            # reinvest into catching up).
            for resource in sell_thresholds:
                sell_thresholds[resource] *= preset["hold_floor_mult"]

        return DayPlan(
            day=state.day,
            phase=phase,
            crop_targets=crop_targets,
            animal_targets=animal_targets,
            structure_targets=structure_targets,
            hire_count=hire_count,
            buy_land=buy_land,
            sell_thresholds=sell_thresholds,
            regime=regime,
            land_cost=land_cost,
        )
