# --- STRUCTURAL (manual, reasoning-driven — do not feed to optuna) ---

# Object Types table (README §Object Types). one_time crops have no subsequent
# yields; ongoing crops produce on a fixed schedule until max_yield fires.
CROP_DATA = {
    "WHEAT": {
        "seed_cost": 10, "base_price": 25, "one_time": True,
        "first_yield_day": 2, "max_yield_day": 4, "max_yield": 6,
    },
    "CARROT": {
        "seed_cost": 20, "base_price": 35, "one_time": True,
        "first_yield_day": 2, "max_yield_day": 3, "max_yield": 4,
    },
    "TOMATO": {
        "seed_cost": 50, "base_price": 60, "one_time": False,
        "first_yield_day": 8, "max_yield_day": 11, "max_yield": 4, "interval_days": 1,
    },
    "STRAWBERRY": {
        "seed_cost": 100, "base_price": 120, "one_time": False,
        "first_yield_day": 10, "max_yield_day": 16, "max_yield": 4, "interval_days": 2,
    },
    "MELON": {
        "seed_cost": 80, "base_price": 250, "one_time": True,
        "first_yield_day": 10, "max_yield_day": 10, "max_yield": 6,
    },
}

ANIMAL_DATA = {
    "GOOSE": {
        "cost": 300, "product": "EGG", "base_price": 50, "structure": "COOP",
        "first_yield_day": 4, "interval_days": 1, "max_held": 4,
    },
    "COW": {
        "cost": 400, "product": "MILK", "base_price": 160, "structure": "PASTURE",
        "first_yield_day": 8, "interval_days": 2, "max_held": 6,
    },
    "SHEEP": {
        "cost": 500, "product": "WOOL", "base_price": 200, "structure": "PASTURE",
        "first_yield_day": 6, "interval_days": 3, "max_held": 6,
    },
}

# Unified base-price lookup for every sellable item (crops + animal products +
# fertilizer) — used by MarketModel and by value scoring in farm_ops so we
# don't duplicate the Price Function table's `base` column across modules.
PRODUCT_BASE_PRICE = {crop: info["base_price"] for crop, info in CROP_DATA.items()}
PRODUCT_BASE_PRICE.update({info["product"]: info["base_price"] for info in ANIMAL_DATA.values()})
PRODUCT_BASE_PRICE["FERTILIZER"] = 100

# README §Object Types "Yield / tile / day" column, keyed by PRODUCT (not
# crop/animal name) to match PRODUCT_BASE_PRICE — used to convert an
# opponent's VISIBLE tile/structure count into an estimated daily market
# supply contribution (v10 opponent-supply-pressure forecasting).
YIELD_PER_TILE_PER_DAY = {
    "WHEAT": 0.80, "CARROT": 0.75, "TOMATO": 0.33, "STRAWBERRY": 0.24, "MELON": 0.55,
    "EGG": 1.00, "MILK": 0.50, "WOOL": 0.33,
}

# Price Function table (README §"The Price Function"). I0 is the same
# 10,000 for every resource by default. amp is derived at call time, not
# stored (SDD §3.4 / MarketModel).
MARKET_PARAMS = {
    "WHEAT":      {"base": 25,  "I0": 10000, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log",  "above_target": 0.20},
    "CARROT":     {"base": 35,  "I0": 10000, "T": 450, "below_func": "log",  "below_target": 0.20, "above_func": "sqrt", "above_target": 0.70},
    "TOMATO":     {"base": 60,  "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "sqrt", "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": 10000, "T": 300, "below_func": "log",  "below_target": 0.20, "above_func": "sq",    "above_target": 3.60},
    "EGG":        {"base": 50,  "I0": 10000, "T": 332, "below_func": "linear", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK":       {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": 10000, "T": 105, "below_func": "log",  "below_target": 0.20, "above_func": "sq",    "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

LAND_COSTS = {"NE": 1000, "SW": 2000, "SE": 4000}
LAND_PURCHASE_ORDER = ["NE", "SW", "SE"]

CROP_PRIORITY_ORDER = ["WHEAT", "CARROT", "TOMATO", "MELON", "STRAWBERRY"]

# Structural bias multiplier on top of _score_crop's profit estimate, sizing
# how much of the tile allocation each crop gets (proportional, not just
# rank order — see _decide_crop_targets).
#
# v8's weights (WHEAT 1.0/CARROT 0.5/TOMATO 0.3/MELON 1.0/STRAWBERRY 2.5)
# were calibrated against a real top-leaderboard opponent's wide,
# multi-quadrant replay ("THUNDER THUNDER"). A 10-variant local round-robin
# tournament (tools/tournament.py, see NOTES.md "v9") then tested that
# baseline against 9 heuristic perturbations and found a MELON-concentrated
# portfolio won every single match (9/9, then 9/10 in a follow-up round) —
# echoing the real single-quadrant, MELON-only opponents that had already
# beaten v7 in actual matches (NOTES.md "v8"). Re-weighted here accordingly;
# a follow-up tournament combining this with a value-chasing scorer tweak
# made things WORSE (lost head-to-head to the melon-only version directly),
# so only the portfolio weight changed, not TASK_SCORE_WEIGHTS.
CROP_TARGET_WEIGHT = {
    "WHEAT": 1.5,
    "CARROT": 0.3,
    "TOMATO": 0.2,
    "MELON": 3.0,
    "STRAWBERRY": 1.0,
}

# SDD §5.5: discount a crop's score per opponent tile already planted in it
# (both players sell into the same shared market, so their planting is a
# leading indicator of oversupply by the time our harvest lands). Applied
# only to crop selection, not animals — animal portfolios are capped small
# enough (PHASE_ANIMAL_CAP) that flooding risk from either player is low.
OPPONENT_FLOOD_PENALTY = 0.15

EXPANSION_UTILIZATION_THRESHOLD = 0.70
MIN_DAYS_REMAINING_FOR_EXPANSION = 8
# Utilization alone can hit 70% within the first few days on a wide
# multi-crop portfolio (5 crop types seeding into 25 tiles fills up fast)
# — long before any harvest revenue has actually banked. A real leaderboard
# replay traced a cash crash straight to a day-4-5 BUY_LAND ($1000) landing
# on top of ongoing seed costs with zero harvests banked yet. This is a
# flat day-number floor independent of utilization: no expansion before
# this day no matter how "utilized" the starting quadrant looks.
MIN_DAY_FOR_FIRST_EXPANSION = 7
# v16: a real-match replay analysis (NOTES.md "v16") found a recurring
# "underutilized land" loss pattern — buying NE->SW->SE within just 4-5
# days of each other (only the FIRST expansion's timing was gated; once
# utilization briefly clears the bar again on a partially-filled farm,
# nothing stopped an immediate second/third purchase). A newly-unlocked
# quadrant starts fully empty and needs real time (seed purchases capped
# at 5-in-stock at a time, unit-turns to walk out and plant, days to
# mature) to actually get used — one decisive loss showed 60 of 75
# unlocked tiles still EMPTY at game end after buying 3 quadrants that
# fast, while an opponent who stayed at 2 quadrants but planted them
# densely (mixed crops+animals) out-earned us 5x. This is a flat
# cooldown since the LAST land purchase (not just the first), giving each
# quadrant real time to fill before committing to the next.
MIN_DAYS_BETWEEN_EXPANSIONS = 5
# v18: gates expansion directly on projected UNIT CAPACITY, not just
# pacing — real replays kept showing 55-85% of unlocked land still EMPTY
# at game end (NOTES.md "v17") even with the pacing cooldown above,
# because pacing alone doesn't stop buying more land than we could ever
# staff. Our max sustainable coverage is (1 + max_hires) units x
# capacity_per_unit tiles/unit (HIRE_TRIGGER_LOGIC, config.py) — hands
# reset every day and must be re-hired from scratch (a known gotcha, see
# CLAUDE.md), so this is a genuine ceiling, not just today's headcount.
# A 1.5x buffer allows expanding a bit past the "fully reliable" line
# (some slack for a fast-growing early hand count / good weed luck) but
# blocks buying land that would leave the farm structurally understaffed
# regardless of execution quality.
EXPANSION_CAPACITY_BUFFER = 1.5
# Block further BUY_LAND once weeds cover more than this share of the
# unlocked farm — a direct signal current land isn't being kept up with
# (found via a real leaderboard replay, not local self-play: our farm hit
# 31% weeds after over-expanding to 3 quadrants).
MAX_WEED_RATIO_FOR_EXPANSION = 0.10
# v20: `_decide_animal_targets` shares this threshold to freeze new
# animal purchases too. TRIED decoupling it to a more lenient 0.30
# (reasoned that a WeedTile only ever spawns from an EMPTY unlocked tile
# per README, never an occupied PASTURE/COOP, so animals don't directly
# compound the weed problem the way more LAND does) — REVERTED after a
# same-seed controlled A/B (20 seeds vs melon_animal_agent) refuted it:
# avg money $14,722->$13,962, avg margin $11,431->$10,675, and weed ratio
# actually INCREASED 20.5%->23.3%. The direct-spawn reasoning was correct
# but missed the systemic effect — more owned animals means more daily
# FEED/CARE demand competing for the same unit-turns that would
# otherwise go to watering, indirectly increasing weed risk despite
# animals never causing it directly. See NOTES.md "v20" for the full
# investigation; don't re-attempt this exact decoupling without new
# evidence.
HIRE_TRIGGER_LOGIC = {
    "capacity_per_unit": 6,       # rough actionable-tiles one unit can cover per day
    "max_hires": 5,
    # No cash-fraction cap here (there was one — see NOTES.md "v8" for why
    # it got removed: it created a vicious cycle where an animal-heavy cash
    # crunch cut off hiring exactly when hands were needed most to recover).
    # Actual affordability is checked directly against the real live budget
    # in farm_ops.MarketExecutor.build_orders, which also budgets HIRE
    # first, before crop seeds/animals.
}
# v22: a real-match trace found a severe death-spiral: extending the
# tighter buildout-phase hire capacity (below) could spike desired hire
# count right at the early->buildout boundary (day 5), and an unlucky
# combination with buildout's own new crop-seed costs (STRAWBERRY/TOMATO
# entering the pool) drained cash to exactly $0 a full 9 days before any
# melon harvest could land — hands then stayed at 0 for 9+ days, weeds
# destroyed the whole planting, and the game was lost 14x before a
# single harvest ever happened. v23 first guarded only HIRE's own spend
# (a 40-seed local sweep found 0/40 against the scripted-opponent proxy —
# looked rare). v25 real-match data proved that wrong: 5/18 games (28%)
# still hit near-zero cash before day 10, because the guard never
# protected the reserve from BUY_SEED/BUY_LAND/BUY_ANIMAL/BUY_PRODUCT —
# any of them could still drain the last dollars, and once at $0 even
# the cheapest $1 HIRE was blocked, so hands stayed at 0 for days. Now
# applied as a blanket floor on the whole turn's spendable budget in
# farm_ops.MarketExecutor.build_orders, not just HIRE's own check.
PRE_HARVEST_CASH_RESERVE = 50   # never let ANY purchase drain cash below this...
PRE_HARVEST_DAY_THRESHOLD = 10  # ...before this day (MELON's first_yield_day, the earliest big payoff)

# v23: real-match trace found a genuine 15.4x blowout loss where we bought
# 8 COW + 7 SHEEP across days 14-21 (cash-affordable each day, correctly
# budgeted) but ended the game with 0 COW and only 4 SHEEP ever placed —
# the rest were lost to starvation-escape. `ANIMAL_CASH_FRACTION` only
# gates spending by CASH; nothing gated it by whether current unit count
# can physically walk to, PICKUP wheat for, and FEED that many animals
# every single day (each feed trip costs several turns of a limited
# 24-turns/day, N-units budget already shared with watering/harvesting).
# Growing the herd faster than real feeding throughput can keep up starves
# animals even with plenty of wheat in the shed (the "v21" fix solved
# PRICE-driven wheat shortages, not raw unit-capacity ones) — the same
# class of mistake `EXPANSION_CAPACITY_BUFFER` already fixed for land.
ANIMAL_CAPACITY_PER_UNIT = 1.5   # rough max animals one unit can reliably feed/care for per day, on top of its other duties
# v18: a real-replay trace found `capacity_per_unit=6` chronically
# UNDER-provisions by a small margin — a farm that exactly matches the
# formula's own "needed" headcount still sees weeds slowly creep in
# (found via a close loss: hands plateaued at exactly the formula's
# computed minimum for 15+ straight days, healthy cash the whole time,
# weeds still climbing ~1 every 5 days). Tightening capacity_per_unit
# GLOBALLY regressed hard vs melon_focus_agent (15/15 -> 10/15, tested in
# isolation) — more hires desired means more Fibonacci-cost spend
# competing with the early seed-buying budget (HIRE is budgeted first).
# Scoped to only "scale"/"endgame" phases instead — well past the
# vulnerable early-cash window, where a stricter headcount can actually
# help without re-triggering that regression.
TIGHT_HIRE_CAPACITY_PER_UNIT = 4

SHED_CENTER_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
BOARD_SIZE = 10
TOTAL_DAYS = 30  # SDD §5.1 assumption: default season length (episodeSteps / turnsPerDay)

# Phase -> crops eligible for new planting (SDD §5.1). Existing plants of any
# crop are always watered/harvested regardless of phase; this only gates NEW
# plantings.
#
# v13: MELON added to "early" (was WHEAT/CARROT-only) after a scripted-
# opponent benchmark (NOTES.md "v13") found this was the single biggest
# lever tested — bigger than multi-crop planting, fixed hiring, or a
# zero hold-floor combined. Root cause: a MELON-focused opponent who
# plants it from day 0 gets a temporary "first mover" monopoly on melon
# pricing for our first ~4-6 days of being locked out of it by phase —
# by the time we arrive, they've already been selling into a still-fresh
# price. Matching their day-0 timing crashes MELON's price for BOTH sides
# earlier (its `above_target=3.6` curve punishes oversupply hard either
# way), but we come out ahead because OPPONENT_FLOOD_PENALTY/market-trend
# signals let us react and diversify once that happens, while a static
# script never adjusts what it plants. Flipped a scripted-opponent
# benchmark from 0/8 (margin -14,000) to 8/8 (margin +11,000) with this
# one change alone; held on melon_animal_agent too (0/6 -> 6/6) and caused
# no regression vs starter/random/pass (unaffected, still 100%).
PHASE_CROP_POOL = {
    "early": ["WHEAT", "CARROT", "MELON"],
    "buildout": ["WHEAT", "CARROT", "TOMATO", "MELON", "STRAWBERRY"],
    "scale": ["WHEAT", "CARROT", "TOMATO", "MELON", "STRAWBERRY"],
    "endgame": [],  # gated further per-crop by remaining-days-to-mature check
}
# v15: caps how many of the early phase's tiles MELON's CROP_TARGET_WEIGHT
# dominance can claim, but ONLY while cash is actually tight — see the
# _decide_crop_targets comment in strategy.py for the full mechanism (a
# "pass"-baseline seed sweep found uncapped early MELON crowding WHEAT
# down to a handful of tiles, starving the only fast-cash income before
# melon's own day-10 payoff, and crashing 7/10 seeds into a permanent
# weed spiral). Gating on cash rather than applying a flat cap
# unconditionally matters: a flat cap also cost the ENTIRE v13
# first-mover win against `melon_focus_agent` (10/10 -> 0/10), because
# that matchup's cash stays healthy the whole time — the cap should only
# ever kick in when the crash risk it's protecting against is real. The
# freed tiles go to WHEAT, not proportionally to every early crop, since
# WHEAT is specifically the fast-cash crop this is protecting.
EARLY_PHASE_MELON_CASH_RESERVE = 1000   # below this money, throttle MELON's early tile share
EARLY_PHASE_MELON_TILE_CAP = 5          # ...down to this many tiles

# Phase -> {animal: total portfolio cap} (SDD §5.1/§5.5 — M4). This is a
# CAP on total ever-owned, not a per-day increment — once reached, no more
# get bought regardless of cash on hand.
#
# GOOSE is deliberately absent: at $/day of product value
# (base_price/interval_days), COW = 160/2 = $80/day and SHEEP = 200/3 =
# $66.7/day both beat GOOSE's 50/1 = $50/day outright, and GOOSE never
# appeared even once in the reverse-engineered top-opponent replays (see
# CROP_TARGET_WEIGHT) — they went straight to COW+SHEEP from day ~0-5 and
# had 5 animals placed by day 5, ~12 by day 10, ~14-17 by day 25. Caps
# below are sized to roughly match that pace.
PHASE_ANIMAL_CAP = {
    "early": {},
    "buildout": {"COW": 8, "SHEEP": 4},
    "scale": {"COW": 10, "SHEEP": 7},
    "endgame": {},  # never buy new animals this late — no runway to amortize
}
# NOTE: an initial pass set these to (200, 0.80) to match the observed
# top-opponent's aggression directly. A 10-seed sweep against the do-
# nothing "pass" baseline caught a real death spiral in 2/10 seeds: an
# unlucky early weed run drains cash, and at 0.80 the animal budget kept
# consuming the cash needed to recover (rehire, rebuy seed) faster than the
# farm could dig out, compounding to 27+ weeds and a final loss to an
# agent that never spends a cent. Dialed back to a safer level — still well
# above the pre-pivot 0.25, but with headroom for recovery spending.
# v15: raised from (800, 0.30) after a real-match replay (NOTES.md "v15")
# showed a top opponent earning 5x+ our money via a much wider
# animal+fertilizer-heavy portfolio while our own execution was clean (no
# weeds, no cash crashes) — this was flagged as a strategic fork, not
# changed unilaterally, pending the same seed-sweep death-spiral testing
# v7/v8 required before touching this exact knob. 20-seed sweep vs `pass`
# at (600, 0.45): 20/20 wins, no crash (min $15,423 vs baseline's $16,960
# — a small, expected margin cost since animals don't pay off against a
# non-competing opponent, not a death spiral). Deliberately well short of
# v7's failed 0.80 (which caused 2/10 seeds to lose to `pass` outright).
# Local scripted-opponent benchmarks show no measurable difference either
# way (melon_focus_agent/melon_animal_agent don't run wide enough
# portfolios themselves to reward more animal investment on our side) —
# this change is evidenced SAFE locally, not evidenced BENEFICIAL locally;
# the actual benefit this is chasing can only be confirmed via real-match
# data over time, the same local-validation-is-necessary-but-not-sufficient
# limit this whole project keeps re-discovering.
ANIMAL_CASH_RESERVE = 600        # keep this much cash untouched when sizing animal purchases
ANIMAL_CASH_FRACTION = 0.45      # never commit more than this share of spendable cash to animals in one day

# --- NUMERIC WEIGHTS (automated search target, §8.6) ---
# M5/v14: tuned via tools/tune.py (optuna). M5's original run (15 trials x
# 4 episodes vs "starter") is superseded — v14 re-ran the same search
# against `melon_focus_agent` instead (NOTES.md "v14"): the built-in
# baselines have been saturated at 100% win rate since well before v13, so
# they can no longer discriminate between candidate configs at all (only
# margin, via the 1e-6 tie-break, moves at all) — the scripted opponent is
# now the only LOCAL benchmark with any remaining resolving power. 15
# trials x 4 episodes, same win_rate + 1e-6*avg_margin objective. Verified
# the winning trial generalizes (not overfit to the one tuned-against
# opponent) before adopting: 10/10 vs melon_focus_agent (the tuned-against
# opponent), 10/10 vs melon_animal_agent (a DIFFERENT scripted opponent,
# never seen during tuning), 6/6 each vs starter/random/pass — no
# regression anywhere. Previous (M5) values, for reference:
#   TASK_SCORE_WEIGHTS = {"urgency": 155.12, "value": 3.80, "distance": 6.37}
#   HOLD_FLOOR_PCT = {"WHEAT": 0.51, "CARROT": 0.11, "TOMATO": 0.40,
#                      "STRAWBERRY": 0.23, "MELON": 0.41, "EGG": 0.19,
#                      "MILK": 0.26, "WOOL": 0.44, "FERTILIZER": 0.26}
#   PHASE_DAY_BOUNDARIES = {"early_end": 3, "buildout_end": 17, "scale_end": 25}
TASK_SCORE_WEIGHTS = {
    "urgency": 23.54745363071848,
    "value": 0.3559324123632561,
    "distance": 4.9360245884817004,
}

# M3: hold floor as a fraction of base_price — below this, hold rather than
# dump (SDD §5.4).
HOLD_FLOOR_PCT = {
    "WHEAT": 0.07009831424802637, "CARROT": 0.592236743821915, "TOMATO": 0.2364097314045229,
    "STRAWBERRY": 0.2839619019897315, "MELON": 0.00821424471410085,
    "EGG": 0.052511390096969285, "MILK": 0.011121232135362591, "WOOL": 0.008429312336436734,
    "FERTILIZER": 0.46651116120049074,
}

# Spread large sells across turns instead of one giant SELL order (SDD §5.4):
# cap how many units of price-impact simulation we walk per turn when sizing
# a sell order against MarketModel's hold floor.
MAX_SELL_SIMULATION_UNITS = 200

# README: shed caps non-seed items at 100; anything added past the cap at
# end-of-day drop is silently discarded. M6 hardening: once the shed gets
# this full, bypass hold floors entirely and sell aggressively — a
# depressed-price sale still beats losing the unit to overflow discard.
SHED_CAPACITY = 100
SHED_OVERFLOW_GUARD_THRESHOLD = 85

# v12: found via a scripted-opponent test (NOTES.md "v12") that
# SHED_OVERFLOW_GUARD_THRESHOLD alone was far too late a backstop.
# HOLD_FLOOR_PCT's "wait for a recovery" logic implicitly assumes a
# TRANSIENT glut — it breaks down completely when TWO players both
# concentrate on the same crop, since the price then has no reason to ever
# recover while both keep producing it. Observed directly: MELON piled up
# to 64 units in the shed (price crashed to $1, hold floor still ~$103)
# while cash crashed to double digits, only forced to sell once the shed
# neared its hard 100-item cap — by which point most of that inventory's
# value was already lost to holding it at $1 instead of realizing SOME
# cash steadily. This scales the effective hold floor down as held
# quantity grows, reaching 0 (full capitulation, sell at any price) well
# before the shed is anywhere near full — a large accumulated pile is
# itself evidence the hoped-for recovery isn't coming, regardless of the
# exact price trajectory.
SELL_CAPITULATION_QTY = 30

# v13: SDD §6 flagged this from the start as a known limitation of the
# greedy-per-turn scheduler — only ONE crop's shortfall "wins" the
# planting slot each turn (by ratio, since M2's raw-count version starved
# whichever crop had the smaller target). A scripted single-crop opponent
# benchmark (NOTES.md "v12") made the cost of that concrete: our portfolio
# needs several turns in a row all landing on the SAME crop before another
# one gets a turn at all, while a script with only one crop to plant never
# pays that tax. Setting this True generates a PLANT candidate for every
# eligible crop (with seed stock and remaining shortfall) at every empty
# tile, instead of only the single best-by-ratio crop — the seed-overcommit
# cap in UnitScheduler.assign() is already keyed per-crop, so multiple
# crops landing in the same turn (via different idle units) is safe.
ENABLE_MULTI_CROP_PLANTING = True

# M6: keep a small opportunistic fertilizer reserve on hand (bought via
# BUY_PRODUCT, or from animal COLLECT_FERTILIZER) so FERTILIZE has something
# to consume — a whole yield-boost mechanic (README §Harvest Yields: doubles
# the per-day watering bonus for 3 days) that earlier milestones never used.
FERTILIZER_RESERVE_TARGET = 3

# v15: found via a "pass"-baseline seed sweep (NOTES.md "v15", the classic
# death-spiral test from v7/v8) — 7/10 seeds crashed to a permanent
# cash-crunch/weed spiral that v8 was supposed to have already fixed.
# Root cause: BOTH top-up builders below re-check "are we short of the
# reserve?" every single turn (unavoidable — feed/fertilizer gets consumed
# mid-day), but neither ever capped how much they'd PAY. FERTILIZER's
# small T=200 (README Price Function table) means repeated small buys
# across just a handful of turns ratchet its price up fast — one seed
# traced a SINGLE unit costing $202 (2x base) by the third re-buy that
# day, right in the pre-first-harvest cash-tight window (day 5-8) where
# every dollar spent here was a dollar not available to HIRE. Neither
# top-up is worth paying a steep premium for — missing a day's fertilize
# bonus or a feed top-up (there's still the untouched WHEAT tile
# allocation as backup) costs far less than the cash spent chasing an
# inflated price.
FERTILIZER_BUY_CEILING_MULT = 1.5   # never pay more than this x base_price for a top-up buy

# v20: WHEAT feed top-up used to share FERTILIZER_BUY_CEILING_MULT (1.5x)
# with fertilizer — found via a real-scripted-opponent trace to be a real
# bug, not a safe reuse: a day-by-day trace showed shed WHEAT sitting at
# 0 for a FULL day (hour 2 through 23) with healthy cash ($15k+) the
# whole time, and zero BUY_PRODUCT WHEAT orders issued despite
# `_build_wheat_feed_topup_orders` wanting to top up — because wheat's
# price had drifted to $39 (1.56x base $25), just over the shared 1.5x
# ceiling. The stakes are wildly asymmetric: missing a FERTILIZER top-up
# forfeits a minor yield bonus (the reasoning that justified 1.5x
# originally), but missing a WHEAT top-up risks losing an entire animal
# ($300-500 already spent) to a starvation escape — a much higher price
# premium is worth paying to prevent that. A 20-turn/220-critical-unfed-
# turn sample found shed WHEAT was 0 in 93% of turns with a
# critical-unfed animal present, and this ceiling collision is the
# confirmed mechanism.
WHEAT_FEED_BUY_CEILING_MULT = 4.0   # separate, much more generous ceiling — losing an animal costs far more than a pricey wheat top-up
# v14: re-tuned alongside TASK_SCORE_WEIGHTS/HOLD_FLOOR_PCT above — see the
# comment there for the full optuna run and generalization check.
PHASE_DAY_BOUNDARIES = {"early_end": 4, "buildout_end": 20, "scale_end": 27}

# SDD §5.1 endgame liquidation: with this few days left, dump everything in
# the shed regardless of hold floor — unsold inventory doesn't count toward
# the win condition (README "Reward"), so even a price-floor $1 sale beats
# holding stock that never gets sold.
LIQUIDATION_DAYS_REMAINING = 2

# --- v10: adaptive/opponent-aware extensions -----------------------------
# Discussed explicitly as an alternative to literal chess/Othello-style
# minimax, which doesn't fit this game (simultaneous moves not alternating
# turns, astronomical per-turn joint action space, reward resolves 240+
# turns after the decision that caused it, and the opponent's private
# state — cash, seed stock — is invisible so their legal-action set can't
# be enumerated for a real game tree). These three pieces instead extend
# the existing reactive opponent-awareness (OPPONENT_FLOOD_PENALTY) into
# something trend-predictive, and add a bounded local-cluster heuristic in
# place of true multi-turn search. All three are flag-gated so
# tools/tournament.py can A/B them against the pre-v10 baseline directly.

ENABLE_OPPONENT_TREND_TRACKING = True
# How many days back to compare against when estimating an opponent
# resource's growth rate (tiles/day). Shorter windows react faster to
# opponent behavior changes but are noisier turn to turn.
OPPONENT_TREND_WINDOW_DAYS = 5
# Multiplies the flood-penalty divisor per unit of measured daily growth
# rate on top of the existing per-tile OPPONENT_FLOOD_PENALTY — an
# opponent rapidly SCALING UP a crop is treated as a bigger threat than
# one holding a large but static tile count, since our own harvest lands
# after their growth has had more time to compound.
OPPONENT_GROWTH_PENALTY = 0.35

ENABLE_SUPPLY_PRESSURE_ADJUSTMENT = True
# Projected opponent daily production (visible tile/structure count x
# YIELD_PER_TILE_PER_DAY, scaled by their measured growth trend) beyond
# which we treat a resource's price as headed for a glut regardless of
# our own behavior, and shrink our own HOLD_FLOOR_PCT for it accordingly
# — better to sell into the crash early than hold for a recovery that
# incoming opponent supply makes less likely.
OPPONENT_SUPPLY_PRESSURE_THRESHOLD = 8.0  # units/day
OPPONENT_SUPPLY_PRESSURE_HOLD_DISCOUNT = 0.5  # multiplies HOLD_FLOOR_PCT down when pressure exceeds the threshold

ENABLE_CLUSTER_LOOKAHEAD = True
# Bounded stand-in for real multi-turn search (SDD §6's "v1.5 shallow
# lookahead" suggestion): rather than simulate future turns, score a
# candidate tile higher if OTHER actionable tiles already sit near it —
# a unit sent there this turn is well-positioned to chain into the next
# task efficiently, instead of finishing an isolated tile and needing a
# long return trip. Approximates the benefit of a real lookahead without
# reimplementing the game engine's transition function to simulate one.
CLUSTER_LOOKAHEAD_RADIUS = 2
# NOTE: an initial value of 8.0 was far too strong once actually measured —
# on a fully-empty 5x5 quadrant, a corner tile (~5 same-radius neighbors)
# vs a center tile (~12) differ by ~56 points at that setting, equivalent
# to ~9 tiles of distance cost. That overpowered even a unit standing
# directly ON a plantable tile (0 distance cost) into walking away from it
# for a marginally better-clustered neighbor (caught by a unit test).
# Dialed down so the bonus nudges ties, not overrides immediate zero-cost
# actions.
CLUSTER_LOOKAHEAD_BONUS_PER_NEIGHBOR = 1.0

# --- v11: direct market-trend signal + opponent-expansion mirroring ------
# Two lightweight (O(1)/turn) reactive heuristics, chosen over v11's third
# candidate proposal (a shallow MCTS with day-scale rollouts) after review:
# that would need a from-scratch forward simulator of the whole game engine
# (yield formulas, weed-spawn RNG, market/town updates — none of which we
# reimplement today), doesn't map cleanly onto a SIMULTANEOUS-move game
# without an opponent-response model we don't have, and "thousands of
# rollouts per decision" is measured against a free-standing laptop, not
# the actual 1.6 vCPU Kaggle evaluation budget our own turns run in
# (0.12ms/turn today — a single 48-72 turn rollout alone would already cost
# tens of ms, before "thousands" of them).

ENABLE_MARKET_TREND_SIGNALS = True
# How many days back to compare when computing market inventory momentum.
# Shorter than OPPONENT_TREND_WINDOW_DAYS on purpose — market price impact
# from a sell-off is immediate, not something that needs 5 days to trust.
MARKET_TREND_WINDOW_DAYS = 3
# Momentum is normalized to units of each resource's own MARKET_PARAMS "T"
# per day (T = that resource's calibrated production-capacity throughput,
# README "Price Function") so a "big move" means the same thing across
# differently-scaled resources (e.g. wheat's T=400 vs strawberry's T=100).
# This coefficient converts normalized momentum into a crop-score discount
# divisor, same mechanism as OPPONENT_GROWTH_PENALTY but driven by the
# market's OWN realized inventory trend (ground truth: captures both
# players' selling AND town consumption in one number) rather than
# inferring future supply from the opponent's visible tile count alone —
# catches cases tile-tracking misses entirely, e.g. an opponent who already
# harvested-and-dumped and moved their tiles on to a new crop, where the
# price-crash from that dump is still working through the market.
MARKET_TREND_SURGE_PENALTY = 3.0
# Same normalized-momentum threshold, but gates an ADDITIONAL sell-side
# reaction: once a resource's inventory is rising this fast, shrink our own
# hold floor for it too — the crash is already underway, waiting for a
# recovery is a worse bet than usual.
MARKET_TREND_SURGE_THRESHOLD = 0.05
MARKET_TREND_SELL_DISCOUNT = 0.6

# Real leaderboard replays (NOTES.md "v9") repeatedly showed opponents who
# NEVER expand past their starting quadrant beating us decisively — our own
# expansion trigger only ever looked at OUR utilization, never at whether
# the strategy the opponent is visibly committing to (stay lean, single-
# quadrant) is working for them. When the opponent hasn't expanded yet,
# require a stricter utilization bar before we do either; once they HAVE
# expanded, the normal EXPANSION_UTILIZATION_THRESHOLD applies (no reason
# for extra caution if the opponent's own decision already validated that
# expanding is viable in this particular matchup).
ENABLE_OPPONENT_EXPANSION_MIRROR = True
OPPONENT_EXPANSION_MIRROR_THRESHOLD = 0.85

# --- v14: multi-expert regime switching (orchestrator) --------------------
# The "economist" (portfolio targets/expansion/hire, _decide_* functions
# below) and "speculator" (opponent/market trend signals, v10/v11 above)
# roles already existed as separate concerns inside strategy.py. This adds
# an explicit ORCHESTRATOR decision on top of both: classify our standing
# each day (ahead/neutral/behind) and let that bias risk posture, instead of
# running one fixed posture the whole game.
#
# Rationale (SDD §1.1, confirmed via kaggle-agent-SDD.md line ~32): the
# competition's actual scoring is an Elo-like skill rating where "coin
# margin does not affect rating change at all — only win/lose/tie matters."
# That means the whole prior tuning history (CROP_TARGET_WEIGHT, etc.) was
# implicitly optimizing money margin, which correlates with winning but
# isn't quite the right target. Rating theory says the optimal posture
# shifts with standing: protect a likely win with LOWER variance (ahead),
# accept HIGHER variance to escape a likely loss (behind) — a moderate loss
# costs the same rating as a huge one, so there's no penalty for swinging
# for it once behind.
ENABLE_REGIME_SWITCHING = True
# Opponent cash is hidden (README), so "ahead/behind" can't compare bank
# balances directly. Proxy: visible economic footprint = sum over each
# side's visible tiles/structures of (YIELD_PER_TILE_PER_DAY x
# PRODUCT_BASE_PRICE) — same units for both sides, uses only what's
# actually visible (README: opponent's farm is fully visible, only their
# shed/inventory is hidden).
REGIME_AHEAD_RATIO = 1.15    # our footprint > opponent's by this ratio -> "ahead"
REGIME_BEHIND_RATIO = 0.87  # our footprint < opponent's by this ratio (~1/1.15) -> "behind"
REGIME_PRESETS = {
    # Ahead: protect the lead. Diversify a bit more (lower concentration
    # variance), hold slightly longer for fuller value, raise the
    # expansion/animal bar rather than lower it — no reason to gamble a
    # likely win to chase a bigger margin the rating won't reward anyway.
    "ahead":   {"concentration_mult": 0.85, "hold_floor_mult": 1.10, "expansion_mult": 1.10, "animal_cash_mult": 0.85},
    # Neutral: exactly today's tuned baseline — every multiplier a no-op.
    "neutral": {"concentration_mult": 1.00, "hold_floor_mult": 1.00, "expansion_mult": 1.00, "animal_cash_mult": 1.00},
    # Behind: worth taking more variance to catch up. Concentrate harder on
    # the top-scoring crop, sell faster for cash to reinvest, lower the
    # expansion/animal bar to grow capacity more eagerly.
    "behind":  {"concentration_mult": 1.20, "hold_floor_mult": 0.80, "expansion_mult": 0.90, "animal_cash_mult": 1.20},
}
