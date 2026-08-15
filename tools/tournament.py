"""Round-robin tournament among N heuristic-parameter variants of our own
agent (SDD §8.4's iteration loop, extended past pairwise head-to-head).

Each variant is a full independent copy of the kaggriculture package
(via importlib, same trick as tools/head_to_head.py) with a config
override dict applied on load — so all 10 variants can be genuinely live
and distinct in the SAME process, unlike tools/tune.py's single shared
config singleton.

Scoring: win=3, draw=1, loss=0 (as specified), tie-broken by total money
margin across all of a variant's matches.

Usage:
    python tools/tournament.py --steps 720
    python tools/tournament.py --steps 720 --variants a,b,c   # subset
"""

import argparse
import importlib
import importlib.util
import itertools
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggle_environments import make

PKG_DIR = pathlib.Path(__file__).parent.parent / "kaggriculture"

# --- Variant definitions -----------------------------------------------
# Each is a dict of {config_attr_name: {key: value}} overrides applied on
# top of the current kaggriculture/config.py (v8) baseline. Chosen to each
# probe one coherent axis, not random noise, so the tournament result is
# actually interpretable afterward.
VARIANTS = {
    "a": {},  # control: current v11 baseline (melon-concentrated portfolio +
              # v10's opponent-trend/supply-pressure/cluster-lookahead +
              # v11's market-inventory-trend signals + opponent-expansion
              # mirroring — everything ON by default)

    "l": {  # v10 features OFF — isolates what the three v10 adaptive
            # systems (opponent trend tracking, supply-pressure hold-floor
            # adjustment, cluster-lookahead scoring) are actually worth on
            # top of the already-adopted melon-concentrated portfolio
        "ENABLE_OPPONENT_TREND_TRACKING": False,
        "ENABLE_SUPPLY_PRESSURE_ADJUSTMENT": False,
        "ENABLE_CLUSTER_LOOKAHEAD": False,
    },
    "m": {  # v11 features OFF only (v10 stays ON) — isolates what the two
            # v11 additions (direct market-inventory-trend signals,
            # opponent-expansion mirroring) are worth on their own
        "ENABLE_MARKET_TREND_SIGNALS": False,
        "ENABLE_OPPONENT_EXPANSION_MIRROR": False,
    },

    "b": {  # aggressive animal investment
        "ANIMAL_CASH_FRACTION": 0.55,
        "ANIMAL_CASH_RESERVE": 400,
    },
    "c": {  # conservative animal investment
        "ANIMAL_CASH_FRACTION": 0.15,
        "ANIMAL_CASH_RESERVE": 1200,
    },

    "d": {  # early, low-bar land expansion
        "MIN_DAY_FOR_FIRST_EXPANSION": 4,
        "EXPANSION_UTILIZATION_THRESHOLD": 0.55,
    },
    "e": {  # very late / reluctant land expansion
        "MIN_DAY_FOR_FIRST_EXPANSION": 20,
        "EXPANSION_UTILIZATION_THRESHOLD": 0.90,
    },

    "f": {  # hire aggressively, more hands
        "HIRE_TRIGGER_LOGIC": {"capacity_per_unit": 3, "max_hires": 8},
    },
    "g": {  # hire sparingly, fewer hands
        "HIRE_TRIGGER_LOGIC": {"capacity_per_unit": 10, "max_hires": 3},
    },

    "h": {  # tactical scorer: distance-averse (prefer nearby tasks strongly)
        "TASK_SCORE_WEIGHTS": {"urgency": 200.0, "value": 3.8, "distance": 12.0},
    },
    "i": {  # tactical scorer: value-chasing (willing to travel for a big payoff)
        "TASK_SCORE_WEIGHTS": {"urgency": 155.0, "value": 8.0, "distance": 3.0},
    },

    "j": {  # concentrated portfolio: melon-heavy, echoing the real single-crop
            # opponents that beat v7 (NOTES.md v8) instead of the wide
            # THUNDER-THUNDER-style spread
        "CROP_TARGET_WEIGHT": {"WHEAT": 1.5, "CARROT": 0.3, "TOMATO": 0.2, "MELON": 3.0, "STRAWBERRY": 1.0},
    },

    "k": {  # combined: round-1 winner (j, melon-concentrated portfolio) +
            # round-1 runner-up (i, value-chasing scorer) — tests whether
            # the two independent improvements stack.
        "CROP_TARGET_WEIGHT": {"WHEAT": 1.5, "CARROT": 0.3, "TOMATO": 0.2, "MELON": 3.0, "STRAWBERRY": 1.0},
        "TASK_SCORE_WEIGHTS": {"urgency": 155.0, "value": 8.0, "distance": 3.0},
    },

    "n": {  # v17 brainstorm: raise hire capacity so more land can actually
            # be worked. Real replay analysis (NOTES.md "v17") found our
            # current baseline (capacity_per_unit=6, max_hires=5, so at
            # most 6 total units) leaves 55-85% of a 3-quadrant (75-tile)
            # farm EMPTY at game end — the land-expansion pacing fix alone
            # doesn't help if there simply aren't enough hands to work the
            # land once it's unlocked. Raising max_hires only raises the
            # CAP on desired hires (still budget-gated for real in
            # farm_ops.MarketExecutor), so this can't force overspending
            # on its own — just removes the artificial ceiling.
        # capacity_per_unit deliberately left at baseline (6) — an earlier
        # pass also lowered it to 5, which regressed hard vs
        # melon_focus_agent (15/15 -> 3-5/10): a lower capacity_per_unit
        # inflates the DESIRED hire count for the same tile count,
        # triggering extra hiring that competes with the early seed-buying
        # budget (HIRE is budgeted first in farm_ops.MarketExecutor).
        # Isolated: raising ONLY max_hires (this cap alone) kept 10/10 vs
        # melon_focus_agent with no regression — the fix is removing the
        # CEILING, not making the trigger more eager.
        "HIRE_TRIGGER_LOGIC": {"max_hires": 10},
    },

    "o": {  # v17 brainstorm: "animal-pure" — two real top-opponent
            # replays (NOTES.md "v17") showed opponents who nearly ignore
            # crop diversification (3-19 crop tiles, most land left
            # deliberately empty) and go all-in on animals (12-14 owned)
            # instead, vastly outscoring us ($72k/$72k vs our $3.6k/$4.8k)
            # even while tolerating high weed rates (9-43%) — suggesting
            # animal income (steady, no daily watering, no crop-cycle
            # risk) may have a fundamentally higher ceiling than our
            # crop-heavy portfolio once cash allows scaling into it.
            # WHEAT stays dominant (cheap, fast cash-flow AND the feed
            # source animals need) while every other crop's weight drops
            # near zero; animal caps/cash go up aggressively.
        "CROP_TARGET_WEIGHT": {"WHEAT": 3.0, "CARROT": 0.1, "TOMATO": 0.05, "MELON": 0.3, "STRAWBERRY": 0.1},
        "PHASE_ANIMAL_CAP": {"early": {}, "buildout": {"COW": 12, "SHEEP": 8}, "scale": {"COW": 20, "SHEEP": 15}, "endgame": {}},
        "ANIMAL_CASH_FRACTION": 0.70,
        "ANIMAL_CASH_RESERVE": 300,
    },

    "p": {  # n + o combined — round-1 (a/n/o) found "o" (animal-pure)
            # underperformed despite real-match evidence favoring heavy
            # animal investment, plausibly because it's STILL capacity-
            # constrained (6 units can't fetch/place/feed/care for 20+
            # animals regardless of cash available). Tests whether raising
            # hire capacity (n) unlocks what animal-pure (o) alone couldn't
            # execute on.
        "HIRE_TRIGGER_LOGIC": {"max_hires": 10},
        "CROP_TARGET_WEIGHT": {"WHEAT": 3.0, "CARROT": 0.1, "TOMATO": 0.05, "MELON": 0.3, "STRAWBERRY": 0.1},
        "PHASE_ANIMAL_CAP": {"early": {}, "buildout": {"COW": 12, "SHEEP": 8}, "scale": {"COW": 20, "SHEEP": 15}, "endgame": {}},
        "ANIMAL_CASH_FRACTION": 0.70,
        "ANIMAL_CASH_RESERVE": 300,
    },

    # --- Round 1 (v19 follow-up): weeds are still 60-67% of real losses
    # even after the phase-scoped hire fix (NOTES.md "v19") — these probe
    # different angles at reducing weeds specifically, config-tunable only
    # (no code changes, unlike the DIG-priority/weed-counting attempts
    # that already failed validation this session).
    "q": {  # tighter late-game hire capacity than v19's TIGHT_HIRE_CAPACITY_PER_UNIT=4
        "TIGHT_HIRE_CAPACITY_PER_UNIT": 3,
    },
    "r": {  # smaller max farm size — less land to keep watered at all
        "EXPANSION_CAPACITY_BUFFER": 1.2,
    },
    "s": {  # stricter expansion weed-gate — stop expanding sooner once
            # weeds appear, instead of tolerating up to 10%
        "MAX_WEED_RATIO_FOR_EXPANSION": 0.05,
    },
    "t": {  # prioritize WATERING over chasing big harvests — raise
            # urgency's weight relative to value/distance so a routine
            # WATER competes better against a big HARVEST in the generic
            # scorer, preventing weeds before they form instead of
            # reacting after (unlike the failed DIG-priority attempt,
            # this makes PREVENTION more competitive, not cleanup)
        "TASK_SCORE_WEIGHTS": {"urgency": 280.0, "value": 3.8, "distance": 6.4},
    },
    "u": {  # slower expansion pacing than v18's MIN_DAYS_BETWEEN_EXPANSIONS=5
        "MIN_DAYS_BETWEEN_EXPANSIONS": 8,
    },
    "v": {  # revert ANIMAL_CASH_FRACTION to pre-v16 (0.30) — tests whether
            # the raised animal investment is itself competing for cash
            # that would otherwise stabilize hiring/watering
        "ANIMAL_CASH_FRACTION": 0.30,
        "ANIMAL_CASH_RESERVE": 800,
    },
    "w": {  # combined: q (tighter late hire) + t (urgency-weighted scorer)
        "TIGHT_HIRE_CAPACITY_PER_UNIT": 3,
        "TASK_SCORE_WEIGHTS": {"urgency": 280.0, "value": 3.8, "distance": 6.4},
    },
    "x": {  # combined: r (smaller max farm) + t (urgency-weighted scorer)
        "EXPANSION_CAPACITY_BUFFER": 1.2,
        "TASK_SCORE_WEIGHTS": {"urgency": 280.0, "value": 3.8, "distance": 6.4},
    },
    "y": {  # combined: s (stricter weed gate) + q (tighter late hire)
        "MAX_WEED_RATIO_FOR_EXPANSION": 0.05,
        "TIGHT_HIRE_CAPACITY_PER_UNIT": 3,
    },

    # --- Round 2: refining around round-1's winners (q: 58pts/19W-1D-7L,
    # v: 52pts/17W-1D-9L). "t" (urgency-weighted scorer) and every
    # combination touching it (w, x) all finished last — confirms
    # over-prioritizing routine WATER over big HARVEST is a bad trade,
    # the same lesson as the earlier failed DIG-priority attempt. Not
    # explored further this round.
    "z1": {  # push q's tightening even further
        "TIGHT_HIRE_CAPACITY_PER_UNIT": 2,
    },
    "z2": {  # q + v stacked (round-1's #1 and #2)
        "TIGHT_HIRE_CAPACITY_PER_UNIT": 3,
        "ANIMAL_CASH_FRACTION": 0.30,
        "ANIMAL_CASH_RESERVE": 800,
    },
    "z3": {  # q + v + s stacked (round-1's #1, #2, #4)
        "TIGHT_HIRE_CAPACITY_PER_UNIT": 3,
        "ANIMAL_CASH_FRACTION": 0.30,
        "ANIMAL_CASH_RESERVE": 800,
        "MAX_WEED_RATIO_FOR_EXPANSION": 0.05,
    },
    "z4": {  # q + a moderate (not v17's failed 10) max_hires bump —
             # tests whether q's tighter trigger wants more headroom
        "TIGHT_HIRE_CAPACITY_PER_UNIT": 3,
        "HIRE_TRIGGER_LOGIC": {"max_hires": 7},
    },
    "z5": {  # v + s stacked, no hire-capacity change at all
        "ANIMAL_CASH_FRACTION": 0.30,
        "ANIMAL_CASH_RESERVE": 800,
        "MAX_WEED_RATIO_FOR_EXPANSION": 0.05,
    },
    "z6": {  # q + slower expansion pacing
        "TIGHT_HIRE_CAPACITY_PER_UNIT": 3,
        "MIN_DAYS_BETWEEN_EXPANSIONS": 7,
    },
    "z7": {  # even more conservative animal investment than v
        "ANIMAL_CASH_FRACTION": 0.30,
        "ANIMAL_CASH_RESERVE": 1000,
    },
}

POINTS = {"win": 3, "draw": 1, "loss": 0}


def load_package_copy(alias: str, overrides: dict):
    """Loads kaggriculture/ under a fresh top-level module name so it gets
    its own independent config/state/strategy/farm_ops/agent module
    objects, then applies `overrides` to its config module in place."""
    spec = importlib.util.spec_from_file_location(alias, PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    for submodule in ("config", "state", "market_model", "strategy", "farm_ops", "agent"):
        importlib.import_module(f"{alias}.{submodule}")

    if overrides:
        config = importlib.import_module(f"{alias}.config")
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(getattr(config, key, None), dict):
                getattr(config, key).update(value)
            else:
                setattr(config, key, value)
    return module


def build_agent(alias: str, overrides: dict):
    load_package_copy(alias, overrides)
    agent_mod = importlib.import_module(f"{alias}.agent")
    strategy_mod = importlib.import_module(f"{alias}.strategy")
    farm_ops_mod = importlib.import_module(f"{alias}.farm_ops")
    state_mod = importlib.import_module(f"{alias}.state")

    strat = strategy_mod.StrategyAgent()
    ops = farm_ops_mod.FarmOpsOrchestrator()
    mem = {"plan": None, "day": None}

    def agent(obs):
        s = state_mod.StateParser.parse(obs)
        if mem["day"] != s.day:
            mem["plan"] = strat.plan(s)
            mem["day"] = s.day
        return ops.run(s, mem["plan"])

    return agent


def run_match(name_a: str, name_b: str, match_idx: int, steps: int):
    agent_a = build_agent(f"kt_{name_a}_{match_idx}", VARIANTS[name_a])
    agent_b = build_agent(f"kt_{name_b}_{match_idx}", VARIANTS[name_b])
    env = make("kaggriculture", configuration={"episodeSteps": steps})
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    r_a, r_b = final[0].reward, final[1].reward
    return r_a, r_b


def run_tournament(variant_names: list, steps: int, repeats: int = 1):
    """Each pair plays `repeats` games (no fixed seed — kaggle_environments
    randomizes per env.run() call), points/margin summed across repeats.
    A single game per pair has high variance (confirmed empirically: two
    variants with byte-identical config still landed 7 points apart across
    a 55-match round robin, purely from which random episodes they each
    happened to draw) — averaging over repeats is how tools/tune.py and
    tools/head_to_head.py already handle this same issue elsewhere."""
    standings = {v: {"points": 0, "wins": 0, "draws": 0, "losses": 0, "money_margin": 0.0} for v in variant_names}
    matchups = {}  # (a, b) -> list of (reward_a, reward_b) across repeats

    pairs = list(itertools.combinations(variant_names, 2))
    total_pairs = len(pairs)
    match_counter = 0
    total_matches = total_pairs * repeats

    for pair_idx, (a, b) in enumerate(pairs, start=1):
        matchups[(a, b)] = []
        for rep in range(repeats):
            match_counter += 1
            t0 = time.time()
            r_a, r_b = run_match(a, b, match_counter, steps)
            elapsed = time.time() - t0
            matchups[(a, b)].append((r_a, r_b))

            standings[a]["money_margin"] += r_a - r_b
            standings[b]["money_margin"] += r_b - r_a
            if r_a > r_b:
                standings[a]["points"] += POINTS["win"]
                standings[a]["wins"] += 1
                standings[b]["points"] += POINTS["loss"]
                standings[b]["losses"] += 1
                outcome = f"{a} beats {b}"
            elif r_b > r_a:
                standings[b]["points"] += POINTS["win"]
                standings[b]["wins"] += 1
                standings[a]["points"] += POINTS["loss"]
                standings[a]["losses"] += 1
                outcome = f"{b} beats {a}"
            else:
                standings[a]["points"] += POINTS["draw"]
                standings[b]["points"] += POINTS["draw"]
                standings[a]["draws"] += 1
                standings[b]["draws"] += 1
                outcome = f"{a} draws {b}"

            print(f"[{match_counter}/{total_matches}] {a} vs {b} (rep {rep+1}/{repeats}): {r_a:.0f} - {r_b:.0f}  ({outcome}, {elapsed:.1f}s)")

    return standings, matchups


def print_standings(standings: dict):
    print("\n--- Standings (win=3, draw=1, loss=0) ---")
    ranked = sorted(standings.items(), key=lambda kv: (-kv[1]["points"], -kv[1]["money_margin"]))
    print(f"{'rank':>4} {'variant':>7} {'points':>6} {'W':>3} {'D':>3} {'L':>3} {'money_margin':>14}")
    for rank, (name, s) in enumerate(ranked, start=1):
        print(f"{rank:>4} {name:>7} {s['points']:>6} {s['wins']:>3} {s['draws']:>3} {s['losses']:>3} {s['money_margin']:>14.0f}")


def print_matchup_grid(variant_names: list, matchups: dict):
    """Cell = row's AVERAGE reward vs column across all repeats."""
    print("\n--- Matchup grid (row's avg reward vs column, across repeats) ---")
    header = "     " + "".join(f"{v:>8}" for v in variant_names)
    print(header)
    for row in variant_names:
        cells = []
        for col in variant_names:
            if row == col:
                cells.append(f"{'--':>8}")
            elif (row, col) in matchups:
                vals = [ra for ra, rb in matchups[(row, col)]]
                cells.append(f"{sum(vals)/len(vals):>8.0f}")
            elif (col, row) in matchups:
                vals = [rb for ra, rb in matchups[(col, row)]]
                cells.append(f"{sum(vals)/len(vals):>8.0f}")
            else:
                cells.append(f"{'?':>8}")
        print(f"{row:>4} " + "".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--repeats", type=int, default=1, help="games per pair, summed into standings (averages out per-episode randomness)")
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS.keys()))
    args = parser.parse_args()

    variant_names = args.variants.split(",")
    for v in variant_names:
        if v not in VARIANTS:
            raise ValueError(f"unknown variant '{v}', choices: {list(VARIANTS.keys())}")

    n = len(variant_names)
    total_pairs = n * (n - 1) // 2
    print(f"Round robin: {n} variants, {total_pairs} pairs x {args.repeats} repeats = {total_pairs*args.repeats} matches, {args.steps} steps each")
    for name in variant_names:
        print(f"  {name}: {VARIANTS[name] or '(baseline, no overrides)'}")
    print()

    standings, matchups = run_tournament(variant_names, args.steps, args.repeats)
    print_standings(standings)
    print_matchup_grid(variant_names, matchups)


if __name__ == "__main__":
    main()
