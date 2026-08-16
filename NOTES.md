# Iteration Notes

## v0 (M0-M1) — wheat-only greedy loop

**Scope:** `state.py` parsing, minimal `strategy.py` (single-crop DayPlan, no
portfolio/expansion/hiring logic yet), `farm_ops.py` UnitScheduler with
urgency-only greedy scoring (plant/water/harvest wheat, dig weeds, return
carried inventory to shed), `MarketExecutor` (buy seed top-up, sell all shed
wheat). Movement is single-step-toward-target (no BFS needed — board has no
obstacles, locked tiles are passable).

**Environment setup note:** `pip install kaggle-environments` pulls a large
transitive dependency chain (jax/flax/orbax-checkpoint/transformers/gymnax/
open_spiel/pettingzoo/pokerkit — bundled because the package hosts dozens of
unrelated Kaggle simulation environments). On Windows, installing the full
chain failed with `OSError: [WinError 206] filename or extension too long`
while unpacking `orbax-checkpoint`'s test fixtures (Windows path length
limit). Kaggriculture itself does not need any of that — installed with
`pip install --no-deps kaggle-environments` plus the small set of core deps
it actually imports at startup (`jsonschema`, `numpy`, `requests`, `Flask`,
`pydantic`). Confirmed `make("kaggriculture")` and `env.run()` work fine
without jax/flax/etc. installed.

**Results (local `env.run()`, no submission-quota cost):**
- vs `random`, 200 steps: 3029 vs 1610 (win)
- vs `starter`, 720 steps: 5329 vs 3455 (win)
- vs `starter`, 10×720-step batch (`tools/run_selfplay.py`): **10/10 wins**, 0 errors

**Confirmed assumption (§7 open question):** the same agent module/process
persists across all turns of a local `env.run()` episode — module-level
`_memory` dict in `agent.py` correctly caches `DayPlan` across turns within
one episode. Still worth double-checking under the actual Kaggle evaluation
harness before relying on it further (e.g. for order-history tracking in M3+).

## v1 (M2) — phase-based multi-crop portfolio + land expansion + hiring

**Scope:** `strategy.py` now computes a real `DayPlan` — game phase from day
(§5.1), crop portfolio split across active-phase-eligible crops weighted by
`(base_price - seed_cost) / first_yield_day` (§5.5 profit-per-tile-day proxy),
land-expansion trigger on utilization + affordability + runway (§5.2), and a
Fibonacci-cost-aware hiring heuristic (§5.3). `farm_ops.py`'s `UnitScheduler`
now tends *any* existing plant regardless of current phase (an old planting
must still be watered/harvested even if the phase has moved on), and only
gates *new* plantings by the active crop pool. `MarketExecutor` adds
`HIRE`/`BUY_LAND` orders, gated to the first turn of each day so day-level
decisions don't repeat every turn (§3.2 cadence). Market-timing hold/dump
thresholds (M3) and animals (M4) are still out of scope — sell-on-harvest
stays immediate for all 5 crops.

**Bug caught during testing:** initial candidate-generation logic decremented
a per-crop shortfall counter while scanning tiles in fixed row-major order,
so on a large empty farm the crop quota could be exhausted by spatially
clustered tiles before ever considering the tile a unit was *standing on* —
observed in a unit test as the farmer walking away from its own plantable
tile. Fixed by computing shortfall once from actual board state and letting
every empty tile carry a candidate, so the score-based greedy scorer (§6),
not the candidate-generation order, decides which unit plants where.

**`build_submission.py` bug caught:** the line-regex import stripper broke
on `strategy.py`'s multi-line parenthesized `from .config import (...)` —
only the first line matched, leaving orphaned continuation lines and an
`IndentationError` in the bundled `main.py`. Rewrote to strip/collect
imports via AST node line-ranges (`node.lineno`..`node.end_lineno`), which
handles multi-line imports correctly regardless of wrapping. Added
`tests/test_build_submission.py` regression coverage for this.

**Results (local `env.run()`):**
- vs `starter`, single 720-step episode: **11,230 vs 3,293** (up from M1's 5,329 vs 3,455)
- vs `starter`, 15×720-step batch (`tools/run_selfplay.py`): **15/15 wins**, 0 errors
- Bundled `main.py` (post-fix) vs `starter`: **12,928 vs 3,468**, compiles and runs correctly

## v2 (M3+M4) — market-timing sell policy + animal husbandry

**Scope:** New [market_model.py](kaggriculture/market_model.py) implements
the README price function exactly (`price(inv) = base + sign*amp*f(|inv-I0|)`
with per-resource shape/target params, §3.4). `MarketExecutor` now holds
instead of dumping when price is below a per-resource floor (§5.4), and
sizes sell orders against `max_sell_before_floor` so large sells spread
across turns automatically (leftover stays in shed for next turn's
re-evaluation). `UnitScheduler` gained full animal husbandry (§5.5/M4):
build COOP/PASTURE, fetch bought animals from the shed and PLACE them, fetch
feed wheat and FEED, CARE, COLLECT_FERTILIZER, harvest animal product —
layered onto the same greedy-scored candidate pool used for crops, plus a
small pre-stage (`_carrying_action`) for the "carrying something, go deal
with it" cases (return-to-shed, place-animal, deliver-feed) that don't fit
the tile-exclusive candidate model.

**Three bugs caught during testing** (all via actually running the agent in
`env.run()`, not just unit tests — the unit tests all passed while the
agent was cash-crashing and losing to `starter` in the real environment):

1. **Unbounded animal spending.** `_decide_animal_targets` originally
   recomputed `target = owned + room` fresh every day with no ceiling, so it
   kept adding up to 2 more animals of each type *every single day* forever
   as long cash allowed — by day 10 the bank had crashed from $3000 to $7
   buying animals that hadn't even produced yet. Fixed by making `target` a
   fixed **portfolio cap** per phase (`PHASE_ANIMAL_CAP`) that stops growing
   once reached, plus a `ANIMAL_CASH_FRACTION` limit on how much of one
   day's spendable cash can go to animals at all.
2. **Animals stuck in the shed forever.** `BUY_ANIMAL` delivers to the shed
   (not directly onto a structure, per README), but no task ever told a
   unit to `PICKUP` the animal and carry it to a coop/pasture — the
   "carrying an animal → PLACE it" logic existed, but nothing triggered the
   pickup in the first place. Fixed by injecting a `FETCH_ANIMAL` candidate
   into the tile-candidate pool whenever the shed holds an animal with a
   matching empty structure available. The same gap existed for feed wheat
   (`FETCH_WHEAT`) and was fixed identically.
3. **Feeding lost the race against selling.** Once wheat *did* reach the
   shed, `MarketExecutor` sold all of it every turn regardless of whether
   animals needed it, racing the `FETCH_WHEAT` pickup and starving animals
   into escaping (observed directly: two geese placed at day 10 had both
   escaped by day 15). Fixed by reserving `2 × owned-animal-count` units of
   WHEAT from every sell order before selling the rest.

**Results (local `env.run()`):**
- vs `starter`, single 720-step episode: **12,719 vs 3,501** (up from M2's 11,230 vs 3,293)
- vs `starter`, 15×720-step batch: **15/15 wins**, 0 errors
- Bundled `main.py`: builds clean, vs `starter`: **10,275 vs 3,505**; 10/10 wins in a follow-up batch
- Confirmed via direct tile inspection: coops/pastures get built, purchased
  animals get placed, EGG appears in shed by day 20 — the full husbandry
  loop closes end-to-end, though some animals still occasionally escape
  under heavier portfolios (e.g. cow/sheep at day 20-25 in one trace) —
  feeding reliability under multi-animal load is a known rough edge, not
  fully solved.

**Test coverage added:** [test_market_model.py](tests/test_market_model.py)
(8 tests, price checkpoints validated against the exact README table
values) and [test_animals.py](tests/test_animals.py) (12 tests covering
place/fetch/feed/care/collect/harvest and the animal-cap/wheat-reserve
guards) — 43 tests total, all passing.

## v3 (M6 hardening + M5 tuning)

**M6 hardening — two fixes, both found by actually running the agent, not
by unit tests:**

1. **Feeding lost to bigger tasks under load.** The generic greedy scorer
   let a big harvest (e.g. melon, value ≈ base_price × yield_units) outscore
   a routine `FETCH_WHEAT`, so animals near the 2-consecutive-day escape
   threshold could still lose the race to a farmer busy harvesting. Fixed
   by adding a dedicated rescue stage in `UnitScheduler.assign()`
   (farm_ops.py): whenever any animal has `consecutive_unfed >= 1` (already
   missed once — the next miss is fatal) and the shed has wheat, the
   nearest idle unit is force-assigned to fetch it, bypassing the scorer
   entirely. Non-critical feeding (an animal that hasn't missed yet) still
   goes through the normal scored `FETCH_WHEAT` candidate.
2. **Endgame liquidation.** Per README ("unsold items in inventory do not
   count towards [win]"), added `LIQUIDATION_DAYS_REMAINING = 2`: with that
   few days left, `sell_thresholds` drop to 0.0 for every resource (ignore
   hold floors — even a $1 sale beats stranded stock) and `hire_count`
   forces to 0 (no time to recoup a new hand's cost).

Verified via direct tile inspection across a full episode: geese no longer
escape at all in the traced run (previously escaped by day 15); pasture
animals (cow/sheep) still occasionally escape under heavier portfolios —
noted as a remaining rough edge, not fully solved.

**M5 — optuna tuning** ([tools/tune.py](tools/tune.py)): searches only the
NUMERIC WEIGHTS block per §8.6 (`TASK_SCORE_WEIGHTS`, `HOLD_FLOOR_PCT`,
`PHASE_DAY_BOUNDARIES`) — structural/reasoning-driven parameters
(`CROP_PRIORITY_ORDER`, `HIRE_TRIGGER_LOGIC`, `PHASE_ANIMAL_CAP`, ...) are
never touched, per the SDD's explicit split. Mutates `kaggriculture.config`'s
dicts in place so every already-imported reference in `strategy.py`/
`farm_ops.py` picks up each trial's values without re-importing.

**Objective-design finding:** a first 15-trial run against `starter` hit
**100% win rate on every single trial** — the agent already dominates
`starter` regardless of these weights, so plain win-rate couldn't
discriminate between configs at all (optuna was picking among ties
arbitrarily). Fixed per SDD §8.4.5 ("track average win margin as a
secondary/robustness metric") by changing the objective to
`win_rate + 1e-6 * avg_margin` — small enough that margin can never flip a
win_rate comparison (rating only cares about win/lose/tie, §1.1), but large
enough to break ties between equally-100%-winning configs.

**Tuning run:** 15 trials × 4 episodes vs `starter` at 720 steps (~2.5 min).
Applied the winning trial to `config.py`, then validated with a direct 8-vs-8
episode comparison against the original hand-set values before committing:

| | win rate | avg margin |
|---|---|---|
| Original (hand-set) | 8/8 | 6,267 |
| Tuned | 8/8 | **15,366** |

Same win rate, **2.4x the money margin** — kept the tuned values. Confirmed
no regression across the full baseline pool afterward: **15/15 vs `starter`**,
**8/8 vs `random`**, **5/5 vs `pass`**, all 0 errors. Bundled `main.py`
rebuilt and compiles clean.

**Known limitation:** the tuning validation compares tuned-vs-original both
independently against `starter`, not a true head-to-head (tuned agent vs
original agent directly) — Python's module-level `config` singleton makes
running two different configs in the same process non-trivial without
duplicating the package under a second import path. If deeper validation is
wanted later, that's the way to do it (SDD §8.4 step 7 calls for direct
head-to-head specifically).

**Test coverage added:** 3 new tests in test_animals.py (critical-feed
rescue, endgame liquidation ignoring hold floor, no hiring during
liquidation) — 46 tests total, all passing. Fixed one pre-existing test
(`test_phase_boundaries`) that hardcoded day numbers instead of reading
`PHASE_DAY_BOUNDARIES` from config — it broke the moment tuning changed
those numbers, which is exactly the kind of stale-literal test that a
tuning loop will keep breaking if left as-is.

**Runtime profiling (closes an SDD §11 open item):** `FarmOpsOrchestrator.run`
averages **0.12ms/turn** on a busy 4-unit, fully-planted-NW-quadrant state
(2000-call average). At 720 turns/episode that's ~90ms of actual compute for
the whole season — trivial against the 1.6 vCPU budget (§1.1), confirming
the greedy-scorer architecture choice over anything search-heavy was right.
`StrategyAgent.plan` (runs once/day, not per-turn) averages 0.03ms/call.

**Remaining open items (SDD §10-11, not yet addressed):** confirm whether
`kaggle_environments` persists the same agent process across all 720 turns
under the *actual* Kaggle evaluation harness (assumed yes, verified only
under local `env.run()`); confirm allowed third-party packages in the
submission environment (currently pure stdlib inside `kaggriculture/`, so
likely moot).

## v4 — closing gaps found in a self-audit against the SDD

Prompted by a direct question ("is anything still missing?"), re-read the
SDD end-to-end against the actual code and found 10 real gaps — features
the SDD explicitly calls for that were never built, missing tooling, and
one live correctness bug. Fixed all of them; two required a second pass
after the fix itself introduced a regression (caught by testing in the
real environment, not by unit tests — a recurring pattern across this whole
project: unit tests validate logic in isolation, but only `env.run()` +
actually reading the resulting state catches emergent scheduling failures).

**1. FERTILIZE was never used.** A whole yield-boost mechanic (README
§Harvest Yields: doubles the per-day watering bonus for 3 days) sat unused
since M4 — `COLLECT_FERTILIZER` gathered it, `HOLD_FLOOR_PCT` just sold it
raw. Added the fetch-and-apply flow (`_find_fertilizable_tiles`,
`FETCH_FERTILIZER` candidate, `FERTILIZE` action) mirroring the existing
wheat/feed pattern, plus a `BUY_PRODUCT FERTILIZER` opportunistic top-up and
a sell-reserve so fertilizer doesn't get auto-sold before use — the same
sell-vs-use race already fixed for WHEAT in M6.

  **Caused a real regression, caught by testing:** the first version valued
  `FETCH_FERTILIZER` the same way as `HARVEST` (`base_price × yield`), which
  for melon (~$750) massively outscored routine `WATER` tasks (~$12.5) in
  the tuned scorer and starved watering across the board. Win rate vs
  `starter` collapsed from 100% to 37.5%, weed events jumped to ~20/episode.
  Fixed by scaling `FETCH_FERTILIZER`'s value down to match `WATER`'s scale
  (`base_price × 0.3`) — fertilizing only forfeits a bonus if skipped, unlike
  watering/feeding which lose the whole plant/animal, so it must never
  outrank anything that carries that risk.

**2. BUY_PRODUCT was never called.** Added via the fertilizer top-up above,
and separately for WHEAT: `_build_wheat_feed_topup_orders` buys WHEAT
directly whenever owned animals' feed reserve is short, instead of relying
solely on the crop portfolio's own WHEAT tile share (which can be too thin
day-to-day since only one crop gets planted per turn in the current
shortfall-based scheduler). This closed a second real problem, found via
the new logging (see #5): animal escape events were running ~13/episode
even after the M6 critical-feed-rescue fix — direct tile tracing showed
shed WHEAT hovering near 0 most days despite animals needing to eat.
Escape events dropped to **0.3/episode** after this fix (10-episode
average) — wheat supply, not delivery logistics, was the real bottleneck.

**3. Opponent farm state was parsed but never read.** `state.py` has parsed
`opp_farm` since M0; nothing in `strategy.py`/`farm_ops.py` ever looked at
it. Added `_opponent_crop_counts` + a scoring discount in `_score_crop`
(`OPPONENT_FLOOD_PENALTY = 0.15` per opponent tile of that crop) per SDD
§5.5 ("if opponent is flooding a resource, expect price crash there —
diversify away from it"). Both players sell into the same shared market, so
the opponent's planting is a leading indicator of oversupply by the time
our own harvest lands.

**4-5. `tools/replay_analyze.py` didn't exist; `run_selfplay.py` only
tracked win/lose/tie**, despite SDD §8.5 calling the fuller breakdown
"instrumentation, not optional." Built `replay_analyze.py` to ingest the
`kaggle_environments` replay schema (`steps[t][player]["action"/"observation"]`)
— works identically on a local `env.toJSON()` dump or a real
`kaggle competitions replay <EPISODE_ID>` download, since both use the same
schema, closing the §9 requirement for one code path to cover both. Computes
revenue-per-resource (best-effort: known buy costs are exact table lookups,
sell revenue is inferred from leftover money delta and split across that
turn's SELL orders by weight — documented in the module docstring as an
approximation, not exact accounting), weed events (PLANT→WEED transitions
only, not the random empty-tile spawns), animal escape events, and
idle-unit-turns. Wired into `run_selfplay.py` via a `--detailed` flag. This
tooling is what actually *found* gaps #1's regression and #2's wheat
shortage — built to enable exactly the kind of manual failure-mode judgment
SDD §8.4 step 4 calls for, and it did.

**6/10. No true head-to-head; no varied opponents.** The M5 tuning
validation (`NOTES.md` v3) compared tuned-vs-original by running each
independently against `starter`, not against each other directly — flagged
at the time as a known limitation, since `kaggriculture.config` is a
process-wide singleton and two different configs can't both be live in one
process without extra work. Built `tools/head_to_head.py`, which uses
`importlib` to load the package twice under different top-level names
(`kaggriculture_a`, `kaggriculture_b`), giving each side its own independent
config/state/strategy/farm_ops module objects. Re-ran the M5 validation
properly: **62.5% win rate, +1,226 avg margin over 8 episodes** — still a
real edge, but smaller than the vs-`starter`-only comparison suggested
(that comparison indirectly overstated the effect size since `starter` is a
much weaker, more predictable opponent than an equally-tuned mirror). This
tool also doubles as the `mirror_agent` self-play opponent from §8.4.3 and
stands in for "varied heuristic/adaptive opponents" (Goals §2) — the
competition only ships three static baselines, so different versions of our
own agent are the only other opponent available for that purpose.

**7. Real correctness bug: PLANT seed-overcommit.** README states that if
too many units try to `PLANT` the same crop in one turn without enough seed
for all of them, *none* of those plants land — not just the excess. The
scheduler generated a `PLANT` candidate for every empty tile once a crop was
chosen, with no check that multiple idle units could simultaneously be
assigned an immediate (same-turn) plant on that crop beyond actual seed
stock. Fixed by capping same-turn immediate `PLANT` assignments per crop at
`state.seeds.get(crop, 0)` during the greedy assignment loop; movement-only
assignments (unit not yet on the tile) aren't capped, since they don't
consume a seed this turn and get re-evaluated fresh next turn anyway.

**8. Shed overflow wasn't guarded.** Items past `shedCapacity` (100) are
silently discarded at end-of-day (README). Added `SHED_OVERFLOW_GUARD_THRESHOLD
= 85`: once the shed's total item count crosses it, every hold floor is
bypassed for that turn's sell orders — a depressed-price sale beats losing
the item to overflow discard for free.

**9. Not yet actually submitted to Kaggle** — everything above is validated
via local `env.run()` only, per the SDD's own stated testing cadence (§1.1:
reserve real submissions for genuine leaderboard progress, not routine
debugging). This is a process step for the user to decide on, not something
fixable in code.

**Final validation after all fixes** (local `env.run()`, full baseline pool):
**12/12 vs `starter`, 6/6 vs `random`, 4/4 vs `pass`** — 100% across the
board, 0 errors. Bundled `main.py` rebuilt, compiles clean, verified
separately: $15,101 vs $3,360 vs `starter`. Full test suite: **57 tests**,
all passing (9 new for fertilizer, 2 new for opponent-awareness, plus the
existing 46 from v0-v3).

**Genuinely still open (not fixable without more information or scope):**
weed events (crop unwatering failures, ~20/episode) are still nontrivial —
unlike the wheat-supply fix for animal escapes, this wasn't root-caused in
this pass; whether it's a similar single-resource bottleneck or an
inherent limit of the one-crop-planted-per-turn scheduler under a 5-crop
portfolio is an open question for the next iteration. Confirming
`kaggle_environments` process persistence and allowed packages under the
*actual* Kaggle harness (not just local `env.run()`) still requires an
actual submission to check.

## v5 — first real submission, first real loss, root-caused via replay

Submitted `main.py` (v4) to the actual competition (submission id
55410207). Validation episode passed, then it played one real public
episode against another player ("Roman Svet") — and **lost badly**: us
10,716 vs them 31,896, despite v4 beating every built-in baseline 100% of
the time locally. This is exactly why the SDD's Goals §2 wanted "varied
heuristic/adaptive opponents" and not just the three static baselines —
`starter`/`random`/`pass` never exposed this failure mode.

Downloaded the real episode's replay (`kaggle competitions replay
91685935`) and ran it through `tools/replay_analyze.py` for the first time
against genuine adversarial data instead of self-play. Found the actual
cause via direct tile inspection at day 24: **our farm had 23/75 tiles
(31%) turned to WEED**, while the opponent — who never expanded past their
starting NW quadrant at all — had zero weeds and had tripled our money.

**Root cause, two compounding bugs in `strategy.py`'s land-expansion
logic:**

1. `_count_utilized_tiles` counted `WeedTile` as "utilized" (anything
   non-`None`/non-`"LOCKED"`). A weed is wasted space, not utilization —
   this meant a farm already drowning in weeds looked *more* eligible for
   the next `BUY_LAND`, not less, actively encouraging the death spiral:
   expand -> can't keep up with watering -> weeds spread -> "utilization"
   stays high because weeds count -> expand again.
2. Nothing checked whether we could actually *staff* more land before
   buying it — the utilization/affordability check had no signal for "are
   we already falling behind on the land we have."

Fixed both: `_count_utilized_tiles` now only counts `PlantTile`/
`StructureTile` (weeds excluded), and added `MAX_WEED_RATIO_FOR_EXPANSION =
0.10` — `_decide_land_expansion` refuses to buy more land once weeds cover
more than 10% of the unlocked farm, regardless of how "utilized" or
affordable it looks otherwise.

**Validated locally:** weed events dropped from ~20-22/episode to
**12.80/episode** (10-episode average) — improved, though not eliminated;
the deeper question of whether the one-crop-planted-per-turn scheduler
itself is the remaining bottleneck under a wide multi-crop portfolio is
still open. Win rate vs the baseline pool stayed at 100% throughout. Added
`test_land_expansion_skipped_when_weed_infested` (58 tests total, all
passing) using the exact scenario the real replay exposed. Rebuilt
`main.py` — not yet resubmitted; holding the daily submission quota (4
remaining) until this is validated further, per the SDD's own guidance
(§1.1) that submissions are scarce and should mark genuine proven progress,
not routine debugging.

**Methodological note for future iterations:** this is the second time in
this project that a bug was invisible to both unit tests and self-play
against the three built-in baselines, and only surfaced against a real
opponent (the first was the FERTILIZE value-scale regression, caught by
`env.run()` but not by baseline self-play either, since even a briefly
broken agent still beat `pass`/`random` easily). `starter` in particular
seems to be a weak enough opponent that a wide range of strategy quality
still beats it — local self-play win-rate against it stayed 100% through
this ENTIRE incident, including the version that was actively losing to a
real player. Local self-play is fast/free but is not sufficient evidence
of competitive strength by itself; real replay analysis via
`tools/replay_analyze.py` is what actually caught this, and should be
treated as a required step before trusting a locally-"proven" version, not
an optional extra.

## v6 — root-caused the residual weed events (not yet resubmitted)

Continued digging into the ~12.8/episode weed rate left after v5's
land-expansion fix, per explicit request to investigate further before
using another submission slot. Used a **fixed episode seed**
(`configuration={"seed": 42}`) to make a single episode reproducible, then
correlated exact weed-event tile positions against per-turn unit positions
and tile status leading up to each one — the first attempt at this used two
separate un-seeded `env.run()` calls and produced nonsense correlations
(different random episodes, positions don't mean anything across them);
worth remembering for next time.

**Root cause:** a WHEAT tile at `consecutive_unwatered == 1` (already
critical — one more miss turns it to a weed) sat unwatered for a **full
day**, despite 6 units being available and the tile only 2 steps from the
nearest one. `HARVEST`'s value (`base_price x yield_units`, uncapped — a
ready melon alone scores ~5700 once weighted by the tuned `value` multiplier)
routinely outscores a routine `WATER` (~500 even with its urgency bonus) in
the generic candidate scorer. Whenever multiple high-value harvests were
ready simultaneously (common once melon/tomato mature), **every** idle unit
got pulled to harvest, leaving critical watering completely unattended for
entire days. This is exactly the greedy-scorer limitation the SDD's §6
anticipated ("watering a low-value tile now vs moving toward a high-value
harvest... locally optimal but globally worse") and flagged as a v1.5
stretch item — it turned out not to be optional.

**Fix:** applied the same dedicated-rescue pattern already used for
critical-unfed animals to critical-unwatered plants. A new rescue stage in
`UnitScheduler.assign()` finds every plant with `consecutive_unwatered >= 1`
not yet watered today and greedily pairs the nearest idle units to them
*before* the generic scorer runs at all — bypassing the score comparison
entirely, the same way the feed-rescue stage does. Added
`_find_critical_unwatered_tiles` and a regression test
(`test_critical_unwatered_plant_gets_dedicated_rescuer_over_big_harvests`)
using the exact failure shape: 3 ready melons + 1 critical wheat tile, 3
units — asserts exactly 2 units harvest and the third heads to the wheat
instead of all 3 harvesting.

**Results (same fixed seed=42, so directly comparable to the diagnostic
trace):** weed events **12 -> 7** immediately after the fix. A wider
10-episode batch against `starter` showed the fuller picture: weed events
**~12.8 -> 1.42/episode**, and — unexpectedly — idle-unit-turns also
dropped **72 -> 9/episode** (the rescue stage keeps units productively
occupied instead of losing scorer ties that left them defaulting to PASS).
Win rate held at 100% throughout. 59 tests total, all passing.

**Still not fully zero:** a handful of weed events remain, traced to crops
planted very late in the day (e.g. hour 20-22) — by the time a plant enters
"critical" status (immediately, per README: planting day counts as the
first miss), there may simply not be enough remaining turns that day for
any unit to walk there, even with an immediate dedicated rescuer. This
looks like a genuine timing tail-risk rather than a scheduling bug — worth
revisiting only if it turns out to matter after more real-opponent data
(e.g. deprioritizing brand-new plantings in the last few hours of a day).

**Not yet resubmitted** — `main.py` is rebuilt and locally validated
(same discipline as v5: hold the submission slot until proven, don't burn
quota on routine iteration). Still pending: a true head-to-head
(`tools/head_to_head.py`) of this fix against the pre-fix version wasn't
run, since there's no version control in this repo to diff against and the
mechanism/before-after evidence (the exact seeded trace, the isolated
regression test, and the 12.8->1.42 batch comparison) was already
unambiguous enough not to need it.

## v7 — reverse-engineered a real top-opponent's strategy, submitted

**Discovery.** Used `kaggle competitions team-submissions <team_id>` (the
leaderboard's `teamId` column) + `episodes <submission_id>` + `replay
<episode_id>` to pull 3 replays from "THUNDER THUNDER" (rank #1 as of
2026-08-10). All 3 showed **nearly identical day-by-day tile/money numbers**
regardless of opponent — their strategy is a fixed schedule, not reactive,
which made it directly reverse-engineerable rather than just observable:

| day | WHEAT | STRAWBERRY | MELON | COW | SHEEP | GOOSE | money |
|---|---|---|---|---|---|---|---|
| 5  | 6  | 3     | 5  | 1 | 4 | **0** | ~540 |
| 10 | 13 | 20    | 5  | 8 | 4 | 0 | ~2,150 |
| 15 | 13 | 36    | 14 | 8 | 4 | 0 | ~10,400 |
| 25 | 41-45 | 14-16 | -  | 9-10 | 4-7 | 0 | **43,502-71,432** |

Final money 5-10x anything we'd produced. Two things stood out immediately:
GOOSE never appears even once, and STRAWBERRY gets a far bigger tile share
than our own portfolio ever gave it.

**Why, structurally:** GOOSE's $/day (base_price/interval_days = 50/1 =
$50) loses to COW (160/2 = $80) and SHEEP (200/3 = $66.7) outright — our
own `_decide_animal_targets` was ranking by `first_yield_day` (fastest
payoff) instead, which favors GOOSE for the wrong reason. Our crop scorer
(`(base_price-seed_cost)/first_yield_day`) also structurally undervalues
STRAWBERRY: it treats every crop as a one-shot payout, so STRAWBERRY's
$100 seed cost and 10-day first yield score it near WHEAT despite
STRAWBERRY paying out across 4 scheduled cycles afterward, not once.

**Changes (config.py + strategy.py):**
- Dropped GOOSE from `PHASE_ANIMAL_CAP` entirely; `COW`/`SHEEP` now unlock
  in `buildout` (was `scale`) with caps sized to the observed pace (8/4 ->
  10/7). Animal ranking now sorts by $/day (`base_price/interval_days`,
  descending) instead of `first_yield_day`.
- New `CROP_TARGET_WEIGHT` structural bias (`STRAWBERRY: 2.5`, `WHEAT/MELON:
  1.0`, `CARROT: 0.5`, `TOMATO: 0.3` — TOMATO never appeared in any of the 3
  opponent replays either) multiplies into `_score_crop`. `STRAWBERRY`
  added to the `buildout` crop pool (was `scale`-only).
- `_decide_crop_targets` allocation changed from equal-shares-with-
  remainder-to-top-rank to **proportional-by-score** — a crop scoring 2x
  another now gets ~2x the tile target, not just "picked first for a +1
  tile."
- Raised `ANIMAL_CASH_FRACTION` 0.25 -> 0.45 (see death-spiral note below
  for why not all the way to the opponent's apparent ~0.8).

**Two execution bugs found and fixed while validating this** (targets were
computed correctly; execution wasn't reaching them):
1. **Shared-budget market ordering.** Every `MarketExecutor` sub-builder
   (seeds, hire, land, animals, wheat top-up, fertilizer) independently
   checked affordability against the SAME untouched `state.my_farm.money`
   snapshot, each blind to what the others were about to spend in the same
   turn. The game processes the order list sequentially and deducts as it
   goes (README: "If a player runs out of money mid-order, the order is
   stopped") — animal orders (listed first) looked affordable on their own
   and so did crop-seed orders (checked independently), but combined they
   overcommitted, and whichever was processed later got starved. Observed
   directly: `STRAWBERRY` seed stock stuck at 1 for 10+ days despite a
   target of 8 and nominally "affordable" money. Fixed by threading one
   shared, decrementing `budget` through every sub-builder, and reordering
   crop seeds to be budgeted FIRST (the tile-target machinery is dead
   weight without seed stock; animals/hire/land can wait a turn).
2. **Shortfall-ratio planting selection.** `UnitScheduler`'s "which crop
   plants this turn" choice picked the crop with the largest RAW shortfall
   count, so a crop with a much bigger target (MELON at 26 tiles) almost
   always beat one with a smaller target (STRAWBERRY at 8) even when
   STRAWBERRY was proportionally just as starved (both 100% unfulfilled).
   Fixed by ranking on shortfall RATIO (`shortfall/target`) instead of raw
   count.

**Death spiral caught by a 10-seed sweep against `pass`** (SDD's own
guidance to treat local self-play as required evidence, not a paper win):
at the initial aggressive settings (`ANIMAL_CASH_FRACTION=0.80`,
`ANIMAL_CASH_RESERVE=200`), 2 of 10 seeds **lost to an agent that never
takes a single action** — final money as low as $838. Traced via a
fixed-seed run: an early unlucky weed run drained cash, and continued
animal spending at 80% of remaining cash kept consuming exactly the money
needed to recover (rehire, rebuy seed), compounding to 27+ weeds (over
half the farm) and never recovering. Fixed two ways: dialed
`ANIMAL_CASH_FRACTION` back to 0.45 and `ANIMAL_CASH_RESERVE` up to 500
(still well above the pre-pivot 0.25/500... 25/500, but not reckless), and
added the same weed-ratio gate already used for land expansion to animal
target growth — new animal purchases freeze (existing ones still get fed
normally) once weeds exceed 10% of the unlocked farm. Re-ran the same
10-seed sweep after the fix: **10/10 wins**, range $22,335-$30,179 (vs the
original scattered $838-$11,564) — tighter AND substantially higher across
the board, not just the worst case fixed.

**Final validation:** 59 tests passing (one updated —
`test_no_opponent_flooding_splits_tied_crops_evenly` tested the OLD
equal-split behavior directly and needed to become
`test_no_opponent_flooding_allocates_by_crop_target_weight`, asserting the
now-intentional ~2:1 WHEAT:CARROT ratio instead). Local batches: **12/12 vs
`starter`** (weed events 3.42/ep, up slightly from v6's 1.42 but far below
the original ~20 — the new higher-throughput portfolio asks more of the
same scheduler), **8/8 vs `random`**, **8/8 vs `pass`** (previously 3/4).
Bundled `main.py` verified: $27,227 vs `starter`'s $3,388.

**Submitted** (per explicit instruction to submit right after building):
submission id `55411795`, description "v7 - strategic pivot...". This uses
the 3rd of 5 daily submission slots (2 remaining today) and pushes `v4`
(id `55410207`) out of the active-2 pool once `v7` finishes validation —
`v6` (id `55411353`, already `COMPLETE`) and `v7` are now the two that
matter for matchmaking and the final leaderboard.

**Open questions for next time:** the reverse-engineered blueprint came
from only 3 episodes of ONE opponent (rank #1, but n=1 opponent) — worth
re-checking against 2nd-5th place teams (`team-submissions` works the same
way) to see whether COW/SHEEP-over-GOOSE and heavy STRAWBERRY is a general
pattern among strong players or specific to this one team. `MELON`'s raw
yield/tile/day (per README's own table: 250 x 0.55 = 137.5) is far higher
than `STRAWBERRY`'s (120 x 0.24 = 28.8), yet the opponent visibly caps
MELON modestly relative to STRAWBERRY — likely because MELON's price curve
crashes far harder on oversupply (`above_target=3.6`, sq-shaped, vs
STRAWBERRY's `1.60`, linear) per the Price Function table, which
`CROP_TARGET_WEIGHT`'s flat multiplier doesn't model; a more principled
version would fold `MarketModel`'s saturation curve into the scoring
directly instead of a hand-set weight copied from observed behavior.

## v8 — root-caused v7's real-match losses, tightened variance dramatically

Prompted by "still losing often in real matches" after v7. Pulled 10 recent
real public episodes for v7 (submission `55411795`) via `episodes
<submission_id>` + `replay <episode_id>`: **7 wins, 3 losses** — better than
pre-v7, but the losses were worth digging into.

**Finding: the 2 most decisive losses both came from the SAME opposing
strategy shape** — a single-quadrant, MELON-concentrated opponent with
almost zero weeds (1-2 the whole game), against our multi-quadrant, wide
7-resource portfolio drowning in weeds (20-25/50 tiles, up to half the
farm). Confirmed via day-by-day tile tracing on both losses
(`episode-91713162`, `episode-91732621`): opponents "FieldOps" and "Disleve
Kanku" never touched a second quadrant and still out-earned us 2-3x.

**Root cause, traced with a fixed-seed local run:** `hire_count` computed
to **0 for 9 straight days** despite 24-35 tiles needing attention every
single one of them. `_decide_hires` gated the desired hire count by `money
x max_cash_fraction (0.30)` — once v7's animal-heavy spending crashed
money to a couple dollars, 30% of that landed below even the first hire's
$1 Fibonacci cost. With no hands, the critical-unwatered rescue stage (v6)
can only save one tile per turn; dozens needed it. This is the third
distinct failure mode traced back to the same underlying issue across
v6-v8: our own spending pace outstripping what the scheduler can execute,
just manifesting differently each time (weeds -> land-expansion,
escapes -> wheat supply, now weeds again -> hiring).

**Three fixes, all found via the same fixed-seed trace-then-verify loop:**
1. **Removed the hire cash-fraction cap entirely.** `_decide_hires` no
   longer pre-limits itself by affordability — it returns the tile-need-
   based desired count, full stop. Actual affordability is checked once,
   for real, against the live per-turn budget in
   `MarketExecutor.build_orders` — which now also budgets **HIRE first**,
   ahead of crop seeds/land/animals, since hands are the foundational
   capacity everything else depends on and are cheap enough (Fibonacci)
   that they should never be the thing cash crunches cut off.
2. **`MIN_DAY_FOR_FIRST_EXPANSION = 7`**: a flat day-number floor on the
   first `BUY_LAND`, independent of utilization. Utilization alone can hit
   70% within 4-5 days on a wide 5-crop portfolio (each crop only needs a
   handful of tiles to seed a 25-tile quadrant that far), long before any
   harvest revenue has actually banked — traced a cash crash straight to a
   day-4-5 land purchase landing on top of ongoing seed costs with zero
   income yet.
3. **Dialed `ANIMAL_CASH_FRACTION` back further** (0.45 -> 0.30,
   `ANIMAL_CASH_RESERVE` 500 -> 800) — the v7 death-spiral fix wasn't fully
   sufficient once combined with the wider crop/land spending v7 also
   introduced; needed another notch down.

**Validation — this fixed both the acute failure AND overall variance,**
not just the worst case:
- 10-seed sweep vs `pass` (the same stress test that caught v7's death
  spiral): **$19,783-$23,379** — every seed, no exceptions. For
  comparison: pre-v7 baseline ~$11-15k with some variance, v7 initial
  (broken) $838-$11,564, v7 fixed $22,335-$30,179 (wide swings, good
  average), v8 **now the tightest range of any version while still
  matching v7's improved average**.
- Weed events: v7's 7.75/episode (10-episode avg) -> **0.13/episode**
  (15-episode avg) — better than any prior version including the pre-v7
  peak (1.42/ep).
- Fixed-seed trace (seed=0, the exact one used for diagnosis): money never
  dropped below $261 the entire episode (previously crashed to $0 for 5+
  consecutive days). Units stayed at 3-4 the whole game instead of
  collapsing to 1.
- 60 tests passing (2 updated for the new day-7 expansion floor, one new
  test added asserting expansion is blocked before day 7 even when fully
  utilized and affordable).
- Full baseline pool: 15/15 vs `starter`, 8/8 vs `random`, 10/10 vs `pass`.

**Submitted:** id `55416977` — last of today's 5 submission slots (0
remaining). `v7` (`55411795`) and `v8` are now the active-2 pool; `v6`
drops out of matchmaking once `v8` finishes validation.

**Open question carried forward:** all three of v6/v7/v8's root causes
trace to the same underlying tension — a portfolio ambitious enough to
match top-opponent economics (wide crops, many animals, multi-quadrant
land) keeps outrunning what the current single-crop-per-turn,
greedy-per-turn scheduler can reliably execute under adversarial
real-match conditions (as opposed to the weaker built-in baselines, which
tolerate almost any strategy quality). Each fix so far has been a targeted
patch (budget ordering, day floors, cash fractions) rather than resolving
the structural mismatch. If real-match losses continue, the next
escalation is architectural: either scale ambition down further to match
proven single-crop-concentrated opponent strategies (safer, lower
ceiling), or invest in the SDD §6 "v1.5 shallow lookahead" to genuinely
raise the scheduler's execution ceiling (higher ceiling, more engineering).

## v10 — real-match analysis, minimax feasibility discussion

Pulled 20 more real matches for v8 (submission `55416977`, deployed since
NOTES.md "v8") via `episodes`/`replay`: **9W-11L (45%)**, worse than v7's
70%. The 3 most decisive losses (woorym $21,185 vs $64,598; T. Scharf
$18,962 vs $63,069; The Ceris $16,319 vs $68,443) all showed the SAME
pattern as v9's earlier finding: opponents who commit hard to a lean,
disciplined economy (mostly COW/SHEEP + minimal crops, or pure MELON) beat
our wider portfolio decisively. New finding this round: **our own weed
count climbed steadily throughout EVERY loss traced** (e.g. 4->5->9->12
tiles), even in games where we never expanded past 1 quadrant — meaning
the weed problem isn't just an over-expansion side effect, it's a more
basic scheduling-reliability gap that real opponents expose but
`starter`/`random`/`pass` self-play never did. Important context: v8 (the
deployed version) predates the melon-concentration + hire-first-budget
fixes made afterward — this reinforces that local self-play against the
3 built-in baselines is not sufficient validation on its own (as already
noted after v8's release); real replay analysis keeps finding gaps that
self-play misses entirely.

Discussed literal minimax/game-tree search (chess/Othello-style) as a
way to make the agent "adaptive, able to counter the opponent's moves."
Concluded it doesn't fit this game: Kaggriculture processes both players'
actions SIMULTANEOUSLY each turn (not alternating), a single turn's joint
action space (farmer + hands + up to 10 market orders, each with many
op/param choices) is astronomically larger than a board game's branching
factor, the actual payoff (final bank money) resolves 240+ turns after
whatever decision caused it (most crop cycles are 48-384+ turns), and the
opponent's private state (cash, seed stock) is invisible so their legal
action set can't be enumerated for a real game tree. Proposed instead:
extend the existing reactive opponent-awareness (`OPPONENT_FLOOD_PENALTY`)
into something trend-predictive, and add a bounded local heuristic in
place of true multi-turn search — implemented next as v11... err, "v10"
(kept the version-name collision as-written; see below).

**Changes**, all flag-gated in `config.py` so `tools/tournament.py` can A/B
them against the pre-change baseline directly:
1. **Opponent trend tracking** (`ENABLE_OPPONENT_TREND_TRACKING`):
   `StrategyAgent` now accumulates a `{day: {resource: count}}` history of
   the opponent's visible crop+animal counts across the episode (crop and
   animal counts unified by PRODUCT name via a new `_opponent_resource_counts`).
   `_score_crop` discounts a crop further if the opponent's presence in it
   is actively GROWING (`OPPONENT_GROWTH_PENALTY`), not just large —
   growth has more room to compound into oversupply before our own
   harvest lands.
2. **Supply-pressure sell adjustment** (`ENABLE_SUPPLY_PRESSURE_ADJUSTMENT`):
   projects the opponent's visible tile/structure count + growth trend a
   week out, converts to units/day via the README's own
   "Yield / tile / day" table (`YIELD_PER_TILE_PER_DAY`), and shrinks
   `HOLD_FLOOR_PCT` for any resource where that projection crosses a
   threshold — sell into the glut early rather than hold for a recovery
   incoming supply makes less likely.
3. **Cluster-lookahead scoring** (`ENABLE_CLUSTER_LOOKAHEAD`): a candidate
   tile surrounded by other actionable tiles scores a small bonus, so a
   unit sent there is positioned to chain into the next task efficiently —
   approximates the benefit of real multi-turn search without simulating
   any future turn (no forward-simulator of the game engine needed).
   **Bug caught by a unit test**: baking the bonus into the candidate's
   `value` field (which gets multiplied by `TASK_SCORE_WEIGHTS["value"]`,
   3.8x) made it wildly overpowering — a corner tile on an empty 5x5
   quadrant differed from a center tile by ~56 points at the initial
   `CLUSTER_LOOKAHEAD_BONUS_PER_NEIGHBOR=8.0`, equivalent to ~9 tiles of
   distance cost, enough to make a farmer walk AWAY from a tile it was
   already standing on. Fixed by applying the bonus directly to the final
   score in `UnitScheduler.assign()` instead, and dialing the constant
   down to 1.0.

**Validation:** 64 tests passing (4 new, covering growth-rate discounting,
graceful degradation with no history, supply-pressure hold-floor
shrinkage, and the cluster-bonus preference itself). Local batch: 10/10 vs
`starter`, weed events **0.00/episode** (previously 0.13-3.42 depending on
version) — best result yet on this metric, though see v11 below for why
that's not the whole story.

## v11 — market-inventory-trend signal + opponent-expansion mirroring, and why local tournaments couldn't validate them

User proposed 3 lightweight architectures as minimax alternatives: (1)
market trend indicators (EMA/momentum on market inventory, block buying
into a commodity that's surging), (2) opponent fingerprinting (classify
opponent archetype from early-game behavior, switch to a counter-strategy),
(3) micro-MCTS (2-3 day-deep rollout search, leaf-evaluated by the
existing `_score_crop` heuristic).

Pushed back on (3) specifically: it would need a from-scratch forward
simulator of the whole game engine (yield formulas, weed-spawn RNG,
market/town updates — none of which we reimplement today; a simulator
that drifts from the real engine would actively mislead rather than help),
doesn't map cleanly onto a SIMULTANEOUS-move game without an opponent-
response model we don't have, and "thousands of rollouts per decision on
a standard laptop" understates the real cost against the actual 1.6 vCPU
Kaggle evaluation budget our own turns run in (0.12ms/turn measured — a
single 48-72 turn rollout alone would cost tens of ms before "thousands"
of them). Agreed to implement (1) and a scoped-down version of (2) instead
of (3).

**Implemented:**
1. **Market-inventory-trend signal** (`ENABLE_MARKET_TREND_SIGNALS`):
   tracks `state.market_inventory` (ground truth — captures BOTH players'
   selling and town consumption in one number, unlike the v10 opponent-
   tile proxy) per day, computes day-over-day momentum normalized to each
   resource's own `MARKET_PARAMS["T"]` (so "big" means the same thing
   across differently-scaled resources), and discounts `_score_crop` /
   shrinks `HOLD_FLOOR_PCT` further when a resource's inventory is rising
   fast — independent of, and complementary to, the opponent-tile-based
   v10 signal (catches cases tile-tracking misses entirely, e.g. an
   opponent who already harvested-and-dumped and moved on to a different
   crop, where the price-crash from that dump is still working through
   the market).
2. **Opponent-expansion mirroring** (`ENABLE_OPPONENT_EXPANSION_MIRROR`):
   scoped down from "fingerprint 3-4 opponent archetypes and hard-switch
   strategies" (risk: misclassification, and we only ever see one opponent
   per episode with no repeated-encounter history to build a real
   profile) to one direct, evidence-backed `if`: real replays repeatedly
   showed opponents who never expand past their starting quadrant beating
   us decisively, so `_decide_land_expansion` now requires a stricter 85%
   utilization bar (`OPPONENT_EXPANSION_MIRROR_THRESHOLD`) instead of the
   normal 70% while the opponent is still single-quadrant, reverting to
   the normal bar once they've expanded too (their own decision already
   validated expansion is viable in that matchup).

**A/B test came back a null result — and revealed something more important
than "these features don't help":** `tools/tournament.py` run a: (all v11
ON) vs m: (all v11 OFF) produced **byte-identical scores across all 3
repeats** (e.g. 18198-18198, 18285-18285, 20320-20320). Root cause: our
own tournament variants are all similar enough to each other (same v9
portfolio weights, same `MIN_DAY_FOR_FIRST_EXPANSION=7`) that the
opponent-DIVERGENCE-triggered conditions these features react to
essentially never occur when playing against slight variations of
ourselves — market inventory stays relatively stable when both sides run
similar economies, and both expand around the same day so "opponent
hasn't expanded yet" rarely differentiates the pair. **Self-play among our
own heuristic variants cannot validate opponent-reactive features at
all** — this needs either real Kaggle matches (submission-quota cost) or
a genuinely different local opponent, which led directly to v12 below.

**Validation:** 68 tests passing (4 new). Local batch: 10/10 vs `starter`,
weed events still 0.00/episode.

## v12 — a scripted opponent finally gives real signal, and it's ugly

Built `tools/scripted_opponents.py`: two standalone agents (zero shared
code with `kaggriculture/`, deliberately a different "species") mimicking
the real pattern that kept beating us — `melon_focus_agent` (MELON+WHEAT
only, plain greedy water-first-then-harvest-then-plant, hire 3 hands flat,
NEVER buys land) and `melon_animal_agent` (same, plus a modest COW/SHEEP
herd). Fixed a `kaggle_environments` gotcha along the way: passing a
callable CLASS INSTANCE as the agent broke the environment's arg-count
reflection (it inspected the unbound `__call__(self, obs)` and tried to
pass 2 args) — worked around by exposing plain top-level wrapper functions
instead of the instances directly.

**First real local signal, and it's bad: our current agent (v11) lost
0/8 to `melon_focus_agent`**, margin averaging around -12,000 to -14,000
across several batches. This is the first LOCAL reproduction of the exact
pattern real Kaggle opponents have shown since v8/v9 — a validated
benchmark to iterate against without spending submission quota.

**Tried the obvious next lever — even MORE melon concentration — and it
got WORSE, not better** (0/6, margin -22,322, money crashing to
double-digits). Traced why: `HOLD_FLOOR_PCT`'s "hold for a price recovery"
logic (`_build_sell_orders`) implicitly assumes a TRANSIENT glut. It
breaks down completely when both players concentrate on the SAME crop —
the price has no structural reason to recover while both keep producing
it. Watched it directly: MELON shed inventory climbed to **64 units**
while price crashed from $250 to $1, our hold floor (~$103) never once
cleared, and we only capitulated once the shed neared its 100-item hard
cap — by which point most of that inventory's value was already lost to
holding instead of realizing SOME cash steadily. Bonus finding: this also
exposed a real gap in v11's own momentum signal — README says a
floor-priced sale doesn't add to market inventory, so once price is
pinned at $1, inventory STOPS moving and the momentum-based surge
detector goes quiet even though the situation is at its worst.

**Fix**: `SELL_CAPITULATION_QTY = 30` in `_build_sell_orders` — the more of
a resource already piled up unsold, the more the effective hold floor
scales down (linearly, reaching full capitulation — sell at any price —
at 30 units), independent of price trajectory. No history tracking
needed; a large accumulated pile is itself sufficient evidence the hoped-
for recovery isn't coming. Verified directly: re-traced the same extreme-
concentration stress scenario, MELON shed inventory now stays at 0-7
units instead of climbing to 64 — the specific hoarding bug is fixed.

**Honest result: this fixed a real, verified bug, but did NOT flip our
fate against `melon_focus_agent`** — still 0/8, margin similar to before
the fix (some runs even slightly worse, plausibly noise from small
8-episode batches). Separately traced the PRODUCTION config (not the
artificially extreme 8.0-weight test variant) and confirmed
`OPPONENT_FLOOD_PENALTY` correctly diversifies away from MELON once the
scripted opponent floods it (crop_targets pivot fully to WHEAT/CARROT by
day 21) and money recovers steadily ($1,233 -> $13,497 by day 27) rather
than crashing permanently — so the flood-avoidance mechanism itself works
as designed. The gap is that our recovery, even when "not broken," simply
doesn't compound fast enough in 30 days to catch a simpler opponent that
never has to spend a turn or a dollar on anything but water/harvest/plant/
sell. No regression: 71 tests passing (3 new), still 10/10 vs `starter`
and 6/6 vs `pass`.

**Open, unresolved, and now backed by a reproducible local benchmark**:
why does a nearly context-free greedy script consistently out-earn a
heuristic with real market-timing, opponent-awareness, and portfolio
logic, within the same 30-day/720-turn budget? Candidate hypotheses not
yet tested: (a) our own overhead (hiring cost, seed-buying, land-
expansion evaluation, seed-buying while diversifying) is a real drag that
a zero-frills script doesn't pay; (b) our multi-crop scheduler's
single-crop-plantable-per-turn bottleneck (SDD §6, still present) costs
more tempo than expected once a portfolio pivot is needed mid-game; (c)
`melon_focus_agent`'s constant 3-hand hiring (vs our need-based, budget-
gated hiring) is simply a better fixed policy for this game's economics
than anything need-based. `tools/scripted_opponents.py` now exists
specifically to test each of these without touching submission quota.

## v13 — tested hypotheses #2/#3 in isolation (both ruled out), then found the real lever: MELON entry timing

Per explicit request ("lanjut uji dugaan #2 dan #3"), tested the two
remaining candidate hypotheses from v12 against `melon_focus_agent`,
in isolation against the pre-v13 config/PHASE_CROP_POOL:

- **#3 (fixed vs need-based hiring):** forcing a flat 3-hand hire policy
  (matching the scripted opponent's own policy) gave only a minor
  improvement — margin **-13,794 -> -11,876**, still 0/8. Not the primary
  cause; need-based hiring itself isn't the bottleneck.
- **#2 (single-crop-per-turn planting bottleneck, SDD §6's own flagged
  limitation):** `ENABLE_MULTI_CROP_PLANTING` (letting every eligible
  crop with a shortfall get PLANT candidates every turn, not just one
  best-by-ratio crop) gave **no measurable improvement** — margin -15,860
  vs -14,674, still 0/8. Ruled out as the primary cause too, despite being
  a real, long-standing architectural limitation.

**The actual dominant lever, found by day-by-day tracing (not one of the
3 originally proposed hypotheses):** `PHASE_CROP_POOL["early"]` was
`["WHEAT", "CARROT"]` only — MELON wasn't eligible to plant until the
`buildout` phase kicked in. `melon_focus_agent` commits to MELON from
day 0. Tracing the two runs side-by-side showed this delayed our own
MELON entry by roughly **4-6 days** relative to the opponent, which was
enough to lose the entire first-mover pricing-advantage window on the
shared market (early buyers get the good side of the price curve before
either side's supply has moved it) — by the time we entered, the
opponent's early advantage had already compounded past recoverable.

**Fix:** added `"MELON"` to `PHASE_CROP_POOL["early"]` (now `["WHEAT",
"CARROT", "MELON"]`). This is a config-only change — no new signal,
tracking, or scoring logic needed; the existing portfolio-weighting
machinery (`CROP_TARGET_WEIGHT`, `_score_crop`) already routes tiles to
MELON correctly once it's simply allowed to compete for them from day 0.

**Result — this is the fix that actually flips the outcome:**
- vs `melon_focus_agent`: **0/8 -> 8/8**, margin **-14,000 -> +11,031**
- vs `melon_animal_agent`: **0/6 -> 6/6**, margin **-12,966 -> +13,658**
- No regression against any built-in baseline: still 100% vs `starter`/
  `random`/`pass`.
- Re-verified after rebuilding `main.py`: 8/8 vs `melon_focus_agent`, 6/6
  vs `melon_animal_agent`, matching the pre-rebuild numbers exactly.

**3 tests needed updating** to reflect MELON now being an intentionally
early-phase-eligible crop, not a "buildout+" one — the same established
pattern as prior versions' test updates (`test_phase_boundaries`,
`test_no_opponent_flooding_...`, etc.) for intentional behavior changes:
`test_farmer_buys_and_plants_wheat_on_first_turn` (exact WHEAT seed-buy
quantity shifted since MELON now shares the same seed budget — loosened
to check the buy happens without hardcoding the amount),
`test_early_phase_targets_only_cash_crops` (renamed
`test_early_phase_targets_only_the_early_phase_crop_pool`, allowed set
now includes MELON), and `test_opponent_flooding_a_crop_reduces_its_
relative_allocation` (CARROT no longer visibly grows when WHEAT is
flooded — it's pinned at its floor of 1 either way, since MELON's much
higher `CROP_TARGET_WEIGHT` now absorbs almost all of the freed
allocation instead; the "grows" side of the assertion now checks MELON).
72 tests passing, no regressions.

**Open questions carried forward:** whether combining this fix with #2
(multi-crop planting) and/or #3 (fixed hiring) — both tested in isolation
against the OLD `PHASE_CROP_POOL` — yields further improvement on top of
the v13 baseline is untested; their true marginal value now that the
dominant lever is fixed is unknown. Not yet resubmitted to Kaggle — user
has directed holding all submissions pending further local validation
("wait jangan submit dulu, kuota terbatas").

**Update:** submitted per explicit go-ahead ("submit bro, real battle
testing") — id `55434988`, `COMPLETE`, publicScore **475.6** (up from v8's
396.7). Checked standing: rank 3000/4036 on the public leaderboard, but
`submissionCount=2` for our team vs a competition that's been running
since 2026-07-29 — this reflects how NEW our rating is (few games played
so far), not necessarily a skill gap. Confirmed via
`kaggriculture-agent-SDD.md`: scoring is an **Elo-like skill rating** where
**"coin margin does not affect rating change at all — only win/lose/tie
matters."** This reframes the whole project: every tuning pass so far
(CROP_TARGET_WEIGHT, TASK_SCORE_WEIGHTS, etc.) was implicitly optimizing
money MARGIN, which correlates with winning but isn't quite the target —
rating theory says the optimal risk posture should actually shift with
standing (protect a likely win with lower variance, accept more variance
to escape a likely loss, since a moderate loss costs the rating exactly
the same as a huge one). Directly motivated v14 below.

## v14 — multi-expert light refactor (regime-switching orchestrator) + re-tuning against a real discriminative benchmark

User asked for a "multi-expert" agent (economist/speculator/agriculture +
orchestrator). Established first that this already existed implicitly:
`StrategyAgent`'s portfolio/expansion/hire logic IS the economist, its
opponent/market-trend signals (v10/v11) ARE the speculator, and
`FarmOpsOrchestrator`/`UnitScheduler` IS the agriculture role — just fused
together without explicit names. Recommended (and built) the light
version: label the existing 3 roles clearly via section comments in
`strategy.py`, and use the natural orchestrator seam (`StrategyAgent.plan()`)
to add ONE new behavior — regime-based risk-posture switching — rather
than a heavyweight rebuild into separate agent classes with a voting
protocol (no real payoff: compute budget is trivial at 0.12ms/turn, and we
control all units centrally so there's no genuine multi-party negotiation
to model).

**New (config.py + strategy.py):**
- `_economic_footprint(state, side)`: visible daily production-VALUE
  estimate (`YIELD_PER_TILE_PER_DAY x PRODUCT_BASE_PRICE`, summed over
  every visible tile/structure) for either side — a same-units proxy for
  "who's producing more," usable despite the opponent's CASH being hidden
  (README: only shed/inventory is hidden, the opponent's farm tiles are
  fully visible).
- `_classify_regime(state)`: buckets the footprint ratio into
  ahead/neutral/behind via a band (`REGIME_AHEAD_RATIO=1.15`,
  `REGIME_BEHIND_RATIO=0.87`) rather than a hard split, so it doesn't
  flip-flop on day-to-day noise near parity. Either side at exactly 0
  footprint forces "neutral" — a data-insufficiency case (very early game,
  or an opponent that genuinely hasn't planted yet), not real evidence of
  a lead; the first version of this check treated opponent-footprint-0 as
  automatic "ahead" and it broke 2 pre-existing tests by biasing the very
  early game on pure noise (fixed before this ever reached a real match).
- `REGIME_PRESETS` (ahead/neutral/behind): multipliers applied on top of
  every existing tuned constant, not replacing them — `concentration_mult`
  (sharpens/flattens `_decide_crop_targets`' allocation spread via a
  scale-invariant power transform: normalize by the max raw score first,
  since `_score_crop`'s absolute units vary wildly by context and
  exponentiating un-normalized values would distort small scores far more
  than large ones for no strategic reason), `hold_floor_mult` (scales
  `sell_thresholds`), `expansion_mult` (scales the land-expansion
  utilization bar), `animal_cash_mult` (scales the animal spending
  fraction). "neutral" is all 1.0 — a pure no-op, so
  `ENABLE_REGIME_SWITCHING=False` degrades cleanly to pre-v14 behavior,
  same pattern as every other flag in config.py.
- `DayPlan.regime` field (default `"neutral"`, so the one existing test
  that builds a `DayPlan` directly without it doesn't break).

**Re-tuning finding — local benchmarks are now saturated, even the
scripted opponent:** re-ran `tools/tune.py` (extended to accept the
scripted opponents as callables, not just the 3 built-in baseline names)
against `melon_focus_agent` instead of `starter`, since `starter` has been
100%-saturated since well before v13 and can no longer discriminate
between candidate configs at all. Found the SAME saturation this time:
every one of 15 trials also hit `win_rate=1.0` against the scripted
opponent now (v13's fix + v14's concentration sharpening already close
that gap entirely) — margin (the 1e-6 tie-break term) was the only thing
still moving. Adopted the winning trial anyway since it's free (win rate
unaffected) and DID verify it generalizes rather than overfits to the one
tuned-against opponent: **10/10 vs `melon_focus_agent`** (tuned-against),
**10/10 vs `melon_animal_agent`** (a DIFFERENT scripted opponent, never
seen during tuning), **6/6 each vs `starter`/`random`/`pass`** — no
regression anywhere. `TASK_SCORE_WEIGHTS`/`HOLD_FLOOR_PCT`/
`PHASE_DAY_BOUNDARIES` all updated (previous M5 values kept in a config.py
comment for reference).

**Rare tail-risk found during a wider validation sweep:** one 10-episode
unseeded batch vs `melon_animal_agent` came back 9/10 with a genuine
blowout loss (us $4,163 vs $25,156). A 20-seed sweep (`configuration={"seed":
N}`) came back 20/20, and a follow-up 30-episode unseeded batch came back
30/30 — combined, **1 loss in 60 episodes (~1.7%)**. This is in the same
category the SDD has already flagged elsewhere as "genuine timing
tail-risk" (e.g. the v6 late-day-planting weed residual) rather than a
reproducible systemic bug like v7's 2/10 death spiral (which reliably hit
at the OLD 0.80 animal-cash-fraction setting and was root-caused/fixed
outright) — not chased further this round given the low incidence rate,
but flagged here for anyone revisiting variance in the future; a
fixed-seed trace of the specific losing scenario would be the way to
root-cause it if it recurs at a materially higher rate.

**Test coverage added:** `tests/test_regime.py` (8 new tests: regime
classification in each direction, the zero-footprint neutral guard, the
flag-disabled fallback via `monkeypatch`, hold-floor bias in both
directions, and the concentration-sharpening transform tested in
isolation at the `_decide_crop_targets` level so only `concentration_mult`
varies between the compared calls). 80 tests total, all passing.

**Final validation:** 80 tests passing, 10/10 vs `melon_focus_agent`, 9/10
(then 30/30 and 20/20 in follow-up sweeps) vs `melon_animal_agent`, 6/6
each vs `starter`/`random`/`pass`. `main.py` rebuilt, compiles clean.

**Not yet resubmitted** — this round's changes (regime-switching +
re-tuning) haven't been sent to Kaggle; per the standing policy, hold
until explicit fresh permission is given, and note the real-match evidence
this time will take longer to accumulate meaningfully (Elo rating needs
games + time to converge, not just another local validation pass).

**Update:** submitted per explicit go-ahead ("submit bro") — id
`55459062`, `COMPLETE`, publicScore **433.7** — LOWER than v13's 475.6,
despite v14 passing every local benchmark 100%. This directly motivated
v15's real-match investigation below rather than more local tuning.

## v15 — real-match investigation (rank ~2700-3000, still far from top 100): a genuine structural bug fixed, and a much bigger strategic gap found

Prompted by "masih peringkat 2700-an... target 100 besar masih jauh" —
pulled and analyzed real match replays directly instead of iterating on
local benchmarks further, since v14's real publicScore regression (433.7
vs v13's 475.6) despite 100% local win rates proved local self-play has
exhausted its usefulness as a signal (same conclusion NOTES.md already
reached once before, now confirmed again).

**Computed the actual real-match win rate directly** (18 v13 episodes, 8
v14 episodes, matched against real opponent names via each replay's
`TeamNames`, not assumed player-index): **v13 41% (7W-10L), v14 38%
(3W-5L)**. Both far below the 100% local win rate against every
benchmark — the local/real gap is real and large, consistent with the
project's whole history (v5, v8, v10 all found the same pattern).

**Bug #1 (fixed): liquidation hire-freeze destroying still-harvestable
crops.** Traced a decisive v13 loss (episode 92016628, us $6,268 vs
opponent $58,260) day-by-day: expanded to all 4 quadrants in a rapid
4-day window (day 13→17), then weeds climbed slowly (2→9 tiles, day
15→21) — survivable — until `hire_count` hit its old blanket-forced 0
during the last `LIQUIDATION_DAYS_REMAINING=2` days, dropping a
100-tile, 4-quadrant farm to 0 hands (just the farmer). Weeds then
exploded **27→44→53 tiles in exactly those final 2 days**, destroying
crops that could otherwise still have been harvested and sold before the
season ended — the opposite of what liquidation is supposed to achieve.
Root cause: `_decide_hires`' "no time to recoup a new hand's cost"
reasoning is valid for hiring as a NEW investment, but the liquidation
override applied it to ALL hiring, including hands needed just to
maintain (not grow) existing coverage.

**Fix (`strategy.py`):** removed the blanket liquidation override;
`_decide_hires` now also only counts an EMPTY tile as "actionable" if
`crop_targets` still wants something planted there (a genuinely-empty
farm on the last day correctly still computes 0 need), but an EXISTING
unwatered plant always counts regardless of remaining days — maintaining
what's already there stays worthwhile even when starting something new
isn't.

**Bug #2 (found, not yet fixed — flagged for a decision): fertilizer
massively under-collected.** The same replay's opponent-side breakdown
showed **$38,700 of their ~$79,500 total (nearly half) came from selling
FERTILIZER** — confirmed via README: `COLLECT_FERTILIZER` gives every
surviving animal owner 1 free unit/day regardless of feeding/care, and
critically "uncollected fertilizer does not accumulate" — a missed
collection is a PERMANENT loss, not deferred like a routine harvest. Our
own side of the SAME match: **$229** (5 units). Root cause: the
`COLLECT_FERTILIZER` candidate was scored like a routine low-stakes task
(urgency=1, value=0.3x base) alongside HARVEST candidates that routinely
score in the thousands — on a busy farm it loses the generic scorer's
priority race almost every day, and each loss is unrecoverable. Bumped
urgency 1→2 and value 0.3x→1.0x base (still far below HARVEST/critical
rescue stages, but no longer certain to lose to nearly everything).

**Bigger, NOT-yet-actioned finding — a strategic fork, flagged for
explicit decision rather than changed unilaterally:** both traced losses
showed real top opponents running much WIDER, more animal-heavy
portfolios (the v14 loss opponent's breakdown: FERTILIZER $38,700,
STRAWBERRY $15,613, MILK $11,222, TOMATO $9,774, MELON only $8,032 — MELON
wasn't even their main earner) that out-earned us 5x+ **despite our own
execution being clean in that match** (0-7 weeds, no cash crashes) — this
wasn't a bug, just insufficient scale. Our `ANIMAL_CASH_FRACTION`/
`ANIMAL_CASH_RESERVE` have been deliberately conservative since v7/v8's
death-spiral incidents (NOTES.md "v7"/"v8": 2/10 seeds lost to `pass` at
more aggressive settings). Raising animal investment now, informed by
real data instead of a hand-copied replay this time, could close a
meaningful part of the gap — but deserves the same seed-sweep death-spiral
testing v7/v8 required before touching it, and is a bigger strategic call
(portfolio ambition level) than this round's two bug fixes. Not changed
this round; flagged here for a follow-up decision.

**Validation:** 81 tests passing (2 new/renamed in test_animals.py:
liquidation no longer forces 0 hires when existing plants need saving,
still correctly computes 0 when there's genuinely nothing to do). Local
batches: 10/10 vs `melon_focus_agent`, 10/10 vs `melon_animal_agent` (no
recurrence of the earlier 1.7% tail-risk loss in this batch), 8/8 vs
`starter`, 20/20-seed sweep vs `melon_animal_agent` (checking the
liquidation-hire fix didn't open a NEW death-spiral risk by hiring more
late-game) — no regressions found. `main.py` rebuilt, compiles clean.

**Not yet resubmitted** — holding per standing policy pending explicit
fresh permission, and given real-match evidence takes real time (games +
Elo convergence) to read, not just another quick local pass.

**Update:** submitted per explicit go-ahead — id `55461061`, `PENDING`
at time of writing.

## v15 (continued) — asked to raise animal investment; found something much bigger first: a severe, previously-hidden cash-crunch regression from v13's own melon fix

User asked to move on to the flagged strategic fork (raising
`ANIMAL_CASH_FRACTION`) with seed-sweep testing, per the same discipline
v7/v8 required. Before touching that knob, ran the CLASSIC v7/v8
death-spiral test (10-seed sweep vs the do-nothing `pass` baseline) as a
sanity baseline first — and it failed badly: **7 of 10 seeds crashed to a
permanent cash-crunch/weed spiral**, final money $5,027-$5,229 (vs v8's
historical $19,783-$23,379 baseline for this exact test). This had never
been re-run since v13 — every post-v13 validation checked WIN RATE against
`pass` (still 100%, since `pass` never accumulates money either way and
can't lose even to a crippled version of us), which is exactly why this
sat undetected for 3 versions.

**Root-caused via day-by-day tracing (seed=0):** money crashed to $0-101
by day 6-8 (normal — matches the established pre-first-harvest dip
pattern) but then, unlike the historical recovery, **never came back**:
weeds exploded from 0 to 22/25 tiles right around day 10-13, exactly when
the first MELON harvest should have landed and funded the recovery.
Bisected by testing config variants against the same seed sweep one
change at a time (ruled out `FERTILIZER` buy-price inflation and the v14
`HOLD_FLOOR_PCT` retune as red herrings — both produced byte-identical
results when reverted) until isolating the actual cause: **removing
MELON from `PHASE_CROP_POOL["early"]` alone fixed all 10 seeds
completely** (money jumped to $23,641-$31,007, weeds down to 3-12).

**The real mechanism (opportunity cost, not direct overspend):**
MELON's `CROP_TARGET_WEIGHT=3.0` so dominates the proportional tile-share
formula that it claimed ~80% of early allocation (20/25 tiles) the moment
v13 made it early-eligible — crowding WHEAT's share down from all 25
tiles (pre-v13) to just 4-5. WHEAT is the ONLY fast-cash crop available
in the pre-melon-harvest window (first yield day 2 vs MELON's day 10);
starving its tile share starves the trickle income that used to fund
hiring/recovery through that whole window. It's not that melon seed cost
directly drained the bank — it's that intent to plant melon on 20 tiles
left WHEAT with almost no harvest volume to sell. This never showed up
against `melon_focus_agent` specifically because that matchup's cash
happens to stay healthy regardless (still 10/10 there throughout); it's
plausibly a real, silent contributor to the 38-41% real-match win rate
though, since unlike `pass`, a real opponent WOULD punish a crippled
$5,200 economy with an outright loss, not just a smaller margin.

**First fix attempt (flat cap) overcorrected:** capping MELON's early
tile share at a flat 5 tiles unconditionally fixed the `pass` sweep
completely (20/20, $23,641-$31,007) but ALSO collapsed the v13
`melon_focus_agent` win outright (10/10 -> **0/10**) — that specific
matchup never needed protecting since its cash never crashes, so an
unconditional cap was trading away a real win to fix a problem that
matchup didn't have.

**Final fix — cash-aware throttle, not a flat cap:** `EARLY_PHASE_MELON_CASH_RESERVE
= 1000` gates the cap — full melon commitment proceeds whenever
`state.my_farm.money` is healthy (preserving the melon-rush matchup
untouched), and only throttles MELON down to `EARLY_PHASE_MELON_TILE_CAP
= 5` (freeing the difference to WHEAT specifically) when cash is
actually tight. This is the same pattern already used for
`ANIMAL_CASH_FRACTION`/`ANIMAL_CASH_RESERVE` — gate the risky behavior on
the actual risk signal (cash health) rather than a blanket structural
rule. Validated both benchmarks together: **10/10 vs `melon_focus_agent`**
(recovered from the flat-cap's 0/10), **20/20 vs `pass`** with money now
consistently $16,960-$27,609 (no seed anywhere near the old $5,000
crash), **10/10 vs `melon_animal_agent`**, **10/10 vs `starter`**, **6/6
vs `random`**.

**3 tests updated** (`test_opponent_flooding_a_crop_reduces_its_relative_allocation`,
`test_no_opponent_flooding_allocates_by_crop_target_weight`,
`test_concentration_mult_sharpens_top_crop_share_without_changing_rank`)
to run at day=10 (buildout) instead of day=0 (early) — all three were
testing mechanisms (flood penalty, weight ratio, concentration transform)
that the new early-phase throttle would otherwise mask or confound at
day=0, same established pattern as every other intentional-behavior-change
test update this project has made. 81 tests passing throughout, no
regressions.

**Animal-investment work (the original ask), now that the baseline is
trustworthy again:** raised `ANIMAL_CASH_FRACTION` 0.30->0.45 and
lowered `ANIMAL_CASH_RESERVE` 800->600 (well short of v7's failed 0.80).
20-seed sweep vs `pass`: 20/20 wins, no crash (min $15,423 vs the
now-fixed baseline's $16,960 — a small, expected margin cost since
animals don't pay off against a non-competing opponent, not a death
spiral). 8-episode checks vs both scripted opponents: win rate and money
both statistically indistinguishable from the 0.30/800 baseline (16,691
vs 16,916 avg vs `melon_focus_agent`; 16,004 vs 15,916 vs
`melon_animal_agent`) — **local benchmarks show this change is safe, but
can't show it's beneficial**, since neither scripted opponent runs a wide
enough portfolio itself to reward more animal investment on our side.
This is the same local-validation-is-necessary-but-not-sufficient limit
the project keeps re-discovering; the actual benefit this is chasing
(matching real opponents' animal+fertilizer-heavy income) can only be
confirmed via real-match data over time, not another local pass.

One 9/10 batch vs `melon_animal_agent` surfaced a blowout loss (us
$3,900 vs $25,236) — re-ran as a 30-episode sweep (30/30) to confirm this
is the SAME ~1-2% tail-risk already documented in "v14" (a rare, severe-
but-infrequent loss category, not a reproducible systemic bug like v7's
2/10 death spiral), not a new regression from this change.

**Final validation for this whole v15 round:** 81 tests passing, 10/10 vs
`melon_focus_agent`, 9/10 then 30/30 vs `melon_animal_agent`, 10/10 vs
`starter`, 6/6 vs `random`, 20/20-seed sweep vs `pass` (no crashes). Two
distinct real bugs fixed (liquidation hire-freeze, early-phase MELON cash
crunch) plus one evidenced-safe strategic change (animal investment).
`main.py` rebuilt, compiles clean.

## v16 — real-match validation of the v15 (deployed) fixes, before-the-fact

Asked to "analyze the current agent's real matches" — pulled all 20
public episodes for the DEPLOYED v15 submission (`55461061`, which only
had the liquidation-hire fix + fertilizer-priority bump, NOT yet the
early-melon cash-crunch fix or the animal-investment increase from the
v15 section above — those were still sitting unsubmitted locally).

**v15 real win rate: 40% (8W-12L)**, in line with v13's 41%/v14's 38% —
no improvement yet, as expected since the bigger fix hadn't shipped.
Classified all 12 losses by failure signature:
- **8/12 (67%): CATASTROPHIC-WEEDS** (final weed ratio >15% of unlocked
  farm) — the exact signature of the early-phase MELON cash-crunch bug
  root-caused earlier in "v15" above, confirming it's not a local-only
  artifact but a real, dominant driver of actual match losses.
- **4/12: UNDERUTILIZED-LAND** (>40% of unlocked tiles still empty at
  game end) — a NEW pattern, not yet fixed. Clearest case (episode
  `92390128`, us $14,320 vs opponent $77,858): we expanded to 3 quadrants
  (75 tiles) but ended with 60 EMPTY — only 2 MELON + 2 SHEEP actually
  planted — while the opponent stayed at just 2 quadrants (50 tiles) but
  ran a much denser mixed portfolio (5 COW + 5 STRAWBERRY + 4 SHEEP) with
  no cash crash and low weeds, reaching 5x our money. We paid the
  `BUY_LAND` cost for tiles we never had the capacity (cash, seed
  purchases, or unit-turns) to actually fill — `EXPANSION_UTILIZATION_
  THRESHOLD` only gates the FIRST expansion's timing, nothing throttles
  how fast SUBSEQUENT expansions can follow once utilization briefly
  clears the bar. Flagged as the next candidate fix, not yet actioned —
  same "3 quadrants bought in 4 days" pacing issue noted in passing
  during the earlier v13 replay trace (episode `92016628`), now confirmed
  recurring across multiple real losses.

Given 67% of real losses matched a bug already root-caused and fixed
locally (just not yet shipped), submitted immediately rather than
delaying further for the land-utilization investigation — id `55468115`
("v16"), `PENDING`, combining the early-melon cash-aware throttle +
ANIMAL_CASH_FRACTION increase. 2 submissions remaining today.

**Next candidate investigation (not started):** the underutilized-land
pattern — likely fix direction is pacing subsequent `BUY_LAND` purchases
against actual fill-rate capacity (recent seed-purchase/hire trends), not
just a one-time utilization snapshot at decision time.

## v16 (continued) — fixed the underutilized-land pattern

**Root cause confirmed:** `MIN_DAY_FOR_FIRST_EXPANSION` only gates the
FIRST `BUY_LAND` (the `next_land == "NE"` check specifically) — nothing
throttled how fast SUBSEQUENT purchases (NE->SW->SE) could follow once
utilization briefly cleared `EXPANSION_UTILIZATION_THRESHOLD` again on a
still-mostly-empty farm. Real replays showed 3 quadrants bought within
4-5 days of each other twice now (episode `92016628` in "v13" section
above, episode `92390128` in "v16" above) — a newly-unlocked quadrant
starts fully empty and needs real time (seed purchases capped at 5-in-
stock at a time, unit-turns to walk out and plant, days to mature) to
actually get used, so buying the next one before the last is filled just
compounds empty tiles rather than growing usable capacity.

**Fix:** `MIN_DAYS_BETWEEN_EXPANSIONS = 5` — a flat cooldown since the
LAST land purchase (any quadrant, not just the first), same pattern as
`MIN_DAY_FOR_FIRST_EXPANSION` generalized to every expansion.
`StrategyAgent` now tracks `_last_expansion_day` by watching
`unlocked_quadrants` count change turn to turn (no purchase-day field
exists in the observation itself) — degrades gracefully to "never
blocks" for a fresh instance/single `plan()` call, same as the v10/v11
history-dependent signals.

**Validation:** 81 tests passing (no test changes needed — every existing
land-expansion test only calls `plan()` once per fresh `StrategyAgent`
instance, where `_last_expansion_day` is still `None` and the cooldown
never engages). 15/15 vs `melon_focus_agent`, 15/15 vs
`melon_animal_agent`, 8/8 vs `starter`, 20-seed sweep vs `pass`: 20/20
wins, no crash (min $15,471 vs the pre-cooldown $16,960 — a small,
expected margin cost from pacing expansion more conservatively, not a
regression). `main.py` rebuilt, compiles clean. Not yet resubmitted —
`v16` (id `55468115`) is already `PENDING` from the earlier fixes this
session; this cooldown fix would be the next submission once that one's
real-match results are in, or bundled sooner with explicit permission.

## v16 (continued 2) — real-match check on the DEPLOYED v16, plus one more real bug found

Asked to re-analyze real matches and keep pushing toward top-100. Pulled
all 16 public episodes for the now-`COMPLETE` v16 submission (`55468115`
— shipped the early-melon cash-aware throttle + `ANIMAL_CASH_FRACTION`
increase, but NOT yet the expansion cooldown or the fix below — both were
still sitting local at the time).

**v16 real win rate: 44% (7W-9L)** — up from v15's 40%, v14's 38%, v13's
41%. Modest but real, and the LOSS BREAKDOWN is the more useful signal:
classified the 9 losses the same way as before —
**CATASTROPHIC-WEEDS dropped from 8/12 (67%) in v15 to 4/9 (44%)** in
v16, while **UNDERUTILIZED-LAND rose to 5/9 (56%)** — exactly the
priority shift predicted: fixing the dominant failure mode (melon cash
crunch) let the NEXT one (land expansion outpacing fill capacity, fixed
earlier in this "v16" section) show through more clearly as the new
majority pattern. Margins also narrowed: most remaining losses are CLOSE
(opponent ahead by 1.1x-2.2x), only 2 are outright blowouts (3.3x-31x) —
a materially different picture than the wide 5x+ blowouts that dominated
earlier analysis, suggesting the remaining gap is closing rather than
being a totally separate unsolved problem.

**One more concrete bug found from a close loss (episode `92454082`, us
$16,946 vs opponent $18,518, 1.1x):** tile comparison showed we'd built
**5 PASTURE structures but only ever bought 2 COW** to fill them, while
the opponent (who also stayed at 2 quadrants) had 8 COW + 1 SHEEP
actually filling their pastures. Root cause: `farm_ops.MarketExecutor`'s
shared per-turn `budget` never deducted `BUY_LAND`'s actual cost (a
long-standing, explicitly-documented gap — the comment reasoned
`plan.buy_land` was only ever True after strategy.py's own 1.2x-buffered
affordability check, so "no additional budget deduction is tracked for
it here"). But that reasoning missed that SAME-TURN orders queued right
after `BUY_LAND` (animal purchases specifically) are computed against
that same shared `budget` variable — if it's not decremented, those
orders get sized against a budget that's stale by the exact land cost,
and the game engine (not our own check) silently truncates whatever
doesn't fit once real money runs out processing the list in order
(README: "If a player runs out of money mid-order, the order is
stopped"). This is the same CLASS of bug as v7's original "shared-budget
market ordering" fix (NOTES.md "v7") — just reintroduced for one specific
order type (`BUY_LAND`) that was deliberately exempted from budget
tracking at the time.

**Fix:** `DayPlan` gained a `land_cost` field (computed in strategy.py,
which already resolves which quadrant tier is next); `farm_ops.py`
deducts it from the shared `budget` right after queuing `BUY_LAND`, same
as every other spend.

**Validation:** 81 tests passing (no test changes needed). 15/15 vs
`melon_focus_agent`, 15/15 vs `melon_animal_agent`, 8/8 vs `starter`,
20-seed sweep vs `pass` (20/20, no crash, same range as the
expansion-cooldown fix above since this change doesn't affect `pass`
games — `pass` never buys land to trigger the overlapping-budget
scenario in the first place). `main.py` rebuilt, compiles clean.

**Rank/rating reality check, given the "target top 100" ask:** pulled
the leaderboard directly — current rank ~3128/4133, score 461.5
(Elo-like), **top-100 cutoff is ~2886** — a gap of roughly 2,400 rating
points. This is not a gap a few more bug fixes alone can close: at a 44%
real win rate, more games played will converge OUR rating toward what a
sub-50%-win-rate agent should sit at, not toward top-100 (which likely
requires a sustained 60-70%+ win rate against the field). The encouraging
part is the TREND — win rate climbing (38% -> 40% -> 44%) and loss
margins narrowing as each concrete bug gets fixed — not yet enough data
to know if this round (expansion cooldown + land-budget fix) crosses
50%, but the mechanism (fixing real, replay-evidenced execution bugs one
at a time, each validated against the SAME real-match data before moving
to the next) is working as designed. Not yet resubmitted — 2 fixes this
session (expansion cooldown, land-cost budgeting) are ready to ship
together once the user gives the go-ahead.

**Update:** submitted as `v17` (id `55469051`) — `COMPLETE`, publicScore
peaked at 494.7 (best of any version so far) then DECLINED over
subsequent games to 417.1 — confirmed via a fresh leaderboard pull:
**rank dropped 3128->3433 (of 4247)**, and real-match win rate on the 9
public v17 episodes was still 44% (4W-5L) — several severe blowouts
(9.7x-20x margin). Root-caused two of those blowouts by tile comparison:
both showed the SAME underutilized-land signature persisting even with
the cooldown fix (one had 64/75 tiles EMPTY, another 55/75) — the pacing
fix alone didn't address the deeper issue: nothing capped how much land
we bought relative to how much our unit count could ever realistically
work.

## v17 (continued) — brainstormed and tournament-tested two fixes; one had a hidden bug, the other didn't pan out

Compared two hypotheses head-to-head in `tools/tournament.py` (extended
with 2 new variants) before committing to either, given the ANIMAL
lever's mixed history this session (evidenced-safe but NOT evidenced-
beneficial locally):
- **"n" — raise hire capacity** (`HIRE_TRIGGER_LOGIC.max_hires`
  5->10), reasoning that more units could work the wider land.
- **"o" — "animal-pure"** (near-zero crop diversification weights,
  much higher `PHASE_ANIMAL_CAP`/`ANIMAL_CASH_FRACTION`), reasoning
  directly from 2 real blowout-loss replays where opponents nearly
  ignored crops and won huge on animals alone (one tolerated 43% weeds
  and still won 20x).

**First round-robin (a/n/o, 3 repeats) found "n" a clear winner (5W-1L)
and "o" dead last (1W-5L) — but this was misleading.** The "n" variant
accidentally ALSO lowered `capacity_per_unit` 6->5 alongside raising
`max_hires` (a copy-paste artifact). Validating "n" against the
established scripted-opponent benchmarks (a closer proxy to real
opponents than self-play) exposed the bug: **win rate vs
`melon_focus_agent` collapsed from 15/15 to 3-5/10.** Isolated the cause
by testing each parameter separately: `capacity_per_unit=5` ALONE
regressed hard (lowering it inflates the desired-hire-count formula for
the same tile count, triggering extra hiring that competes with the
early seed-buying budget — `HIRE` is budgeted first in
`farm_ops.MarketExecutor`); `max_hires=10` ALONE (capacity_per_unit left
at baseline) kept the full 10/10 with a small money gain.

**Re-ran the round-robin with the bug fixed (max_hires only).** Result
reversed: the CURRENT baseline ("a") won convincingly (7W-2L,
+11,592 margin), "n" fell to 2nd (5W-4L, -4,432 margin — barely better
than a coin flip), "o" and the n+o combination ("p") stayed near the
bottom. Cross-checked "n" once more directly against both scripted
opponents in isolation: essentially a wash (+7% money vs
`melon_focus_agent`, -4% vs `melon_animal_agent`, win rate unchanged at
10/10 either way). **Verdict: neither hypothesis is worth shipping as
tested.** "Animal-pure" in particular is a good example of over-fitting
a fix to 2 anecdotal replays without controlled validation — the real
opponents' FINAL numbers (heavy animal counts) don't prove that copying
their CONFIG (higher caps/cash fraction) reproduces the mechanism that
got them there; something else (execution efficiency, a completely
different action pattern, or — per the fix below — enough unit capacity
to actually act on that ambition) was doing the real work.

## v18 — tied land expansion directly to hire capacity, instead of raising hire capacity

The tournament detour above answered "should we just hire more?" with a
clear no. Went with the more surgical alternative flagged at the time:
**don't buy more land than the CURRENT hire ceiling could ever
realistically staff**, rather than raising the ceiling itself (which
tournament-tested worse) or leaving land expansion uncapped by capacity
(which real replays showed leaves 55-85% of a 3-4-quadrant farm
permanently empty).

**Fix (`config.py` + `strategy.py`):** `EXPANSION_CAPACITY_BUFFER = 1.5`
— `_decide_land_expansion` now computes `max_coverable_tiles = (1 +
max_hires) x capacity_per_unit x buffer` from the EXISTING
`HIRE_TRIGGER_LOGIC` (no capacity change) and refuses to buy land that
would push total unlocked tiles past that ceiling. At baseline settings
(`max_hires=5`, `capacity_per_unit=6`) this works out to 54 tiles max —
enough for NW+NE (50) but blocks SW/SE, capping expansion at 2 quadrants
by design rather than by accident. The 1.5x buffer intentionally allows
a LITTLE overreach (a fast-growing early hand count, or good weed luck,
can cover somewhat more than the strict ceiling) without permitting the
3-4-quadrant overreach real replays kept showing.

**Validation:** 81 tests passing (no test changes needed). 15/15 vs
`melon_focus_agent`, 15/15 vs `melon_animal_agent`, 8/8 vs `starter`,
20-seed sweep vs `pass`: 20/20 wins, and this time an actual IMPROVEMENT,
not just "no regression" — min money $17,251 (up from the pre-fix
$15,471), avg $20,496 (up from $19,545). Direct tile check on a
scripted-opponent match confirmed the mechanism: expansion now naturally
stops at 2 quadrants (NW+NE) instead of reaching 3-4, with a healthier
(though still imperfect — 52% empty, an inherent consequence of the
6-unit ceiling itself, not a bug) utilization ratio.

**Honest reality check, given the "target top 10" ask:** current rank
~3433/4247, score 417 and declining — top-10 cutoff is ~3077, roughly
2,660 rating points away. This is an extremely large gap that concrete
bug-fixing alone is very unlikely to close in the remaining time; it
would require a sustained 70-80%+ win rate against the full field, far
above anything measured so far (38-44% across v13-v17). The realistic
framing: every fix this session has been real, replay-evidenced, and
locally validated (not guesswork) — but rating is currently DECLINING
game-over-game even as local execution measurably improves, which most
likely means our peak local win rate against the CURRENT config still
sits under 50% against the actual field, not that the fixes are wrong.
Top-100 remains the more realistic near-term target discussed earlier;
top-10 would need either many more rounds of this same real-match-
evidenced iteration loop, or a genuinely different strategic tier (not
yet identified) beyond what local tournament-testing has surfaced so
far. Not yet resubmitted — pending explicit permission.

**Update:** submitted as `v18` (id `55482653`) — `COMPLETE`, publicScore
fluctuated 492.3 -> 529.4 across successive checks (confirms the
"declining score" pattern users see is normal Elo convergence noise from
a small early sample, not a monotonic decline — see the meta-discussion
below). 7 public episodes: 3W-4L (43%), consistent with v13-v17's 38-44%
band.

## v18 (continued) — used the FULL local replay archive (108 files, 6 versions) for a much larger-sample failure analysis, tried two more fixes, both failed validation (and that's useful)

User asked directly why scores start high then decline, and whether the
agent can be built right once instead of repeated trial-and-error.
**Answer given:** the decline is normal Elo-rating convergence (new
submissions start on a volatile provisional rating; a few early wins can
spike it, but it settles toward whatever a ~40%-real-win-rate agent
should sit at as more games accumulate — the "489 -> 417 -> 529" swings
ARE that convergence, not a bug). On "why not build it right once":
genuinely not possible for this class of problem — the opponent's
strategy/code is never visible, only outcomes after the fact via
replays, against 4000+ unknown and varied opponents; every real weakness
found this whole project (9+ distinct root causes across v5-v18) was
found empirically, never by reasoning alone, and local tools have a
proven, repeated ceiling on how well they predict real-match performance.
Iteration IS the process, not a workaround for insufficient upfront design.

**Acted on the "use everything" prompt directly:** instead of reading
another 7-9-episode batch for the newest version alone, combined ALL
locally cached real-match losses across v15-v18 (30 losses, all sharing
the modern post-refactor architecture — v13/v14 excluded as predating
several structural fixes) into one classification pass. This corrected
an earlier conclusion: **CATASTROPHIC-WEEDS is still the dominant
failure signature at 60% (18/30)**, not UNDERUTILIZED-LAND (37%,
11/30) as the smaller single-version v16 sample suggested. Also found
v18 specifically (post the capacity-tied expansion fix) had ZERO
underutilized-land losses in its small sample (4/4 were weeds) —
consistent with that fix working, and weeds being the next real
priority.

**Traced a close v18 weed loss (episode `92645961`, margin 1.4x) day by
day and found a striking, reproducible pattern:** hands dropped 4->3->2
over days 11-14 and NEVER recovered for the rest of the game, DESPITE
money staying healthy the whole time ($18k-26k, never below $14k).
Root-caused to `_decide_hires`: a `WeedTile` contributes to NEITHER the
"empty tile" nor the "unwatered PlantTile" branches of the
actionable-tile count, so once a tile turns into a weed it silently
DISAPPEARS from the hire-need formula entirely — a perverse incentive
where accumulating weeds makes the formula think FEWER hands are needed,
not more.

**Two fix attempts, BOTH failed validation against `melon_focus_agent`
(our most reliable local proxy) and were reverted:**
1. Counting `WeedTile`s as actionable (fixing the formula gap directly):
   win rate 15/15 -> 9/15. Inflating the desired-hire-count competes with
   the early seed-buying budget once weeds appear — the SAME failure
   mode already diagnosed once this session for a different parameter
   (lowering `capacity_per_unit`, NOTES.md "v17").
2. Raising `DIG`'s priority (urgency 0->1, value 5->40) so hired
   capacity would actually target weed removal: win rate 15/15 ->
   **0/15**, even worse. Diverting units to DIG (which only clears space,
   earns nothing directly) away from HARVEST/WATER (which do) is a bad
   trade against a benchmark that rewards raw output — `DIG`'s
   deliberately low priority turns out to be correct, not an oversight.

**Both reverted** back to the validated v18 state after each failed its
own validation pass — the same discipline that caught the "hire cap" and
"animal-pure" dead ends earlier this session. 81 tests passing, 15/15 vs
`melon_focus_agent` confirmed restored after both reverts. **The original
symptom (hands not recovering despite healthy cash, in a game we
narrowly lost) remains real and unexplained** — this specific lever
(weed-awareness in the hire/DIG pipeline) just isn't the right fix for
it. Worth revisiting with a different angle (e.g., why does
`_decide_hires` compute a LOW desired count at all on a farm with 18/25
tiles planted and only 2 units, independent of weeds — is 18 planted
tiles' worth of daily watering genuinely fitting inside `actionable_tiles`
staying low, or is something else suppressing the count) rather than
adding weed-awareness on top of a formula that may have a different bug.
main.py unchanged from the validated v18 build (both attempted changes
were reverted before rebuilding for submission).

## v18 (continued 2) — dug from a different angle, found a real (if partial) fix

Re-examined the same close-loss trace (episode `92645961`) more
carefully instead of reaching for another blunt lever. Key correction to
the earlier read: `_decide_hires` ISN'T stuck on a stale value — it
recomputes fresh every day and consistently lands on the exact SAME
plateau (`ceil(actionable_tiles / capacity_per_unit)`, e.g. 3 total
units for an 18-tile farm at `capacity_per_unit=6`). That's chronic
marginal UNDER-provisioning, not a stuck-value bug: the formula's own
"needed" headcount is a hair short of what's actually required to fully
prevent the occasional missed watering that becomes a weed.

Tested tightening `capacity_per_unit` in isolation (5, `max_hires`
untouched this time — ruling out the max_hires interaction cleanly):
still regressed vs `melon_focus_agent` (15/15 -> 10/15, avg money
$14,622 -> $11,412). Confirms the mechanism is capacity_per_unit itself,
not just its interaction with max_hires — ANY increase in desired hire
count competes with the early seed-buying budget, since `HIRE` is
budgeted first every turn regardless of game phase.

**The real fix: scope it by phase.** The understaffing symptom only ever
showed up well into buildout/scale (day 14+ in the traced loss) — long
past the vulnerable early-cash window a tighter `capacity_per_unit`
keeps colliding with. Added `TIGHT_HIRE_CAPACITY_PER_UNIT = 4`, applied
via a new `_decide_hires(..., phase_capacity_per_unit=...)` parameter
that `StrategyAgent.plan()` only passes for `phase in ("scale",
"endgame")` — early/buildout keep the unchanged, validated
`capacity_per_unit=6`.

**Validation:** 81 tests passing. 15/15 vs `melon_focus_agent`
(unchanged — confirms the early-game protection holds), 14/15 vs
`melon_animal_agent` (1 loss matches the ~1-2% tail-risk already
documented since "v14", not a new regression), 8/8 vs `starter`,
20-seed sweep vs `pass`: 20/20, min $17,261 / avg $20,557 (both a touch
ABOVE the pre-fix baseline, not just "no regression"). Direct trace
confirmed the mechanism firing as intended: hands rose 3->5 right at the
scale-phase boundary (day 20-21) instead of staying capped at 3 — though
weeds still crept in later in that same trace (day 22 onward, reaching
12 by day 29), so this is a real, evidenced IMPROVEMENT, not a complete
elimination of the weed problem. Consistent with "CATASTROPHIC-WEEDS at
60% of all losses" likely having more than one contributing cause; this
fixes one concrete piece of it (the phase-boundary understaffing) without
regressing anything else. `main.py` rebuilt, compiles clean. Not yet
resubmitted — pending explicit permission.

**Update:** submitted as `v19` (id `55486270`) — `COMPLETE`, publicScore
519.8 (best of any version so far). Pulled all 12 public real episodes:
**50% win rate (6W-6L)** — the first version to cross 50% with a
reasonable sample size, up from the 38-44% band every prior version
(v13-v18) sat in. Land utilization is visibly healthier across the
board (5 of 6 losses stayed at 1-2 quadrants, confirming "v18"'s
capacity-tied expansion fix is holding up in real matches, not just
local tests). Loss breakdown: 4/6 (67%) still `CATASTROPHIC-WEEDS`, 2/6
`UNDERUTILIZED-LAND` — weeds remain the dominant unsolved pattern, as
expected (this fix was explicitly scoped as a partial improvement, not a
full solution, when it shipped). Next investigation should stay on
weeds specifically, now from a fresh angle given `_decide_hires`'
capacity assumption and phase scoping are already addressed.

## v20 — two-round local tournament (inconclusive) + a real weed fix from spatial + code-path analysis

**Tournament round 1** (10 variants: baseline + 9 new config-only probes
at reducing weeds — tighter late hire capacity, smaller max farm size,
stricter expansion weed-gate, urgency-weighted scorer, slower expansion
pacing, reverted animal fraction, and combinations): `q`
(`TIGHT_HIRE_CAPACITY_PER_UNIT=3`, even tighter than v19's 4) won
decisively — 58pts, 19W-1D-7L, +40,545 margin. `t` (raise urgency's
scorer weight so WATER competes better against HARVEST) and every
combination touching it finished dead last — a third confirmation this
session (after the DIG-priority and weed-counting attempts) that
tilting the generic scorer toward routine tasks over big harvests is
consistently the wrong lever.

**Tournament round 2** (refining around round-1's winners `q` and `v`
[reverted `ANIMAL_CASH_FRACTION` to 0.30], playing against 7 NEW
combinations instead of round 1's field): reversed sharply — `q` fell
to 9th/10 (31pts), `v` won instead (50pts). **This is the headline
finding of the two rounds**: local tournament rankings are highly
sensitive to which OTHER variants are in the pool, not a stable measure
of a config's real strength — consistent with, and now more starkly
demonstrated than, the variance warning already in `tools/tournament.py`'s
own docstring. Validated round-2's winner (`v`) against the project's
actual gold-standard local benchmarks anyway: mixed results (+6% vs
`melon_focus_agent`, -12% vs `melon_animal_agent`, +3% vs `pass`) that
don't clear the bar to override the real-match evidence favoring MORE
animal investment, not less (NOTES.md "v17"). **Neither round's findings
were applied** — the tournament served its purpose by ruling out bad
ideas (any `t`-touching variant) rather than producing a ship-ready
winner this time.

**Found a real, validated fix via a different angle: spatial + code-path
analysis instead of more config knob-turning.** Combined all 108 locally
cached real-match replays (v15-v19) and computed weed tile positions
relative to the shed center: weeds at the farthest possible distance
(the extreme corner) occur at roughly 2x their expected rate by tile
count (8% actual vs 4% expected) — a real but moderate spatial skew, not
the dominant driver on its own. Went further and re-audited
`UnitScheduler`'s critical-unwatered rescue stage (`assign()` in
farm_ops.py) line by line instead of tuning around it, and found a
genuine structural bug: the rescue pairing re-solves the GLOBAL
nearest-unit-to-nearest-critical-tile assignment FROM SCRATCH every
single turn. A unit already 2 turns into walking toward tile A (not yet
arrived) is still counted as "idle" and can get reassigned to a newly-
appeared, marginally closer tile C the moment C shows up — abandoning
A's progress. Since new critical tiles appear continuously in a busy
game, this creates real dithering risk: a tile can have "a rescuer
assigned" every single turn without ever having the SAME rescuer stick
around long enough to actually arrive.

**Fix:** `UnitScheduler` gained persistent instance state
(`self._committed_rescues: {unit_index: tile_pos}`, safe since
`FarmOpsOrchestrator`/`UnitScheduler` are instantiated once per episode
and live across all turns — same pattern `StrategyAgent`'s v10/v11/v16
history dicts already rely on). Each turn, existing commitments are
honored FIRST (continue toward the same tile) before any fresh distance
optimization runs on the leftover idle units and still-critical tiles; a
commitment is only dropped once its tile is no longer critical (watered,
or already lost).

**Validation:** 81 tests passing (no changes needed — existing tests
build a fresh `UnitScheduler` per call, which doesn't exercise cross-
turn behavior). 15/15 vs `melon_focus_agent`, 14/15 vs
`melon_animal_agent` (the 1 loss matches the ~1-2% tail-risk already
documented since "v14"), 8/8 vs `starter`, 20-seed sweep vs `pass`:
20/20, min $17,772 / avg $20,652 (both up from the pre-fix $17,261 /
$20,557 — a real improvement, not just "no regression"). Ran a
same-seed controlled A/B specifically measuring weed ratio (15 seeds,
`melon_animal_agent`, otherwise identical): **weed ratio 23.1% (old,
non-sticky) -> 18.1% (new, sticky)** — about a 22% relative reduction,
clean evidence the mechanism works as reasoned, not a coincidence (an
earlier UNSEEDED comparison had been noisy/inconclusive on this specific
metric — the seed-controlled version is the one to trust). `main.py`
rebuilt, compiles clean. Not yet resubmitted — pending explicit
permission.

**Update:** submitted as `v20` (id `55487794`) — `COMPLETE`, publicScore
643.4, the highest of any version. Leaderboard rank jumped **3433 ->
2490** (of 4288). Only 3 public episodes so far (too small to trust a
win-rate number): 1W-2L, but both losses' tile comparisons showed the
SAME real-opponent pattern documented repeatedly this project — one
opponent won with 46/75 (61%!) weeds AND 6 COW + 4 SHEEP, the other with
minimal crops and 4 COW + 2 SHEEP. Our own animal counts in both losses
stayed at 1-2 — the single most consistently recurring gap across
essentially every large real-match loss this whole session.

## v20 (continued) — local-only iteration (no submission spent): a code-audit hypothesis tested and correctly rejected

Asked directly for local-only next steps (save submission quota) to keep
chasing the animal-execution gap above. A second code audit (same
method that found the sticky-rescue bug) found `_decide_animal_targets`
reuses `MAX_WEED_RATIO_FOR_EXPANSION` (0.10) to freeze ALL new animal
purchases once weeds exceed 10% — and real losses regularly sit at
15-40% weeds, so this looked like a strong candidate for why owned
animal counts keep plateauing at 2-4 despite generous caps (17-20) and
cash fraction (0.45). Reasoning: a `WeedTile` can only ever spawn from
an EMPTY unlocked tile (README), never an occupied PASTURE/COOP, so
buying more animals doesn't directly compound the weed problem the way
buying more land does — the shared threshold looked miscalibrated.

**Tested by decoupling to a separate, more lenient
`MAX_WEED_RATIO_FOR_ANIMAL_FREEZE = 0.30` — and a same-seed controlled
A/B (20 seeds vs `melon_animal_agent`, identical seeds both runs) clearly
REFUTED the hypothesis:** avg money $14,722 -> $13,962 (worse), avg
margin $11,431 -> $10,675 (worse), and — the real surprise — avg weed
ratio actually INCREASED 20.5% -> 23.3%, the opposite of what "more
animals, since they don't directly weed" would predict. The direct-spawn
reasoning was correct in isolation but missed the systemic effect: more
owned animals means more daily FEED/CARE demand, competing for the same
limited unit-turns that would otherwise go to watering crops —
indirectly increasing weed risk even though no animal ever causes a weed
itself. **Reverted cleanly** (config comment documents the finding so
this exact experiment isn't blindly re-run later); 81 tests passing,
`main.py` rebuilt back to the validated v20 state.

**Why this negative result still has value:** it rules out a plausible-
sounding, code-evidenced hypothesis WITHOUT spending submission quota —
exactly the local-iteration workflow that was asked for. It also
sharpens the open question: real opponents clearly DO execute heavy
animal counts successfully even at high weed tolerance (46/75 weeds and
6 COW + 4 SHEEP in the same v20 loss above) — so the gap isn't simply
"our freeze threshold is too strict," it's something about HOW those
opponents balance animal-care unit-turns against watering that our
current architecture doesn't replicate by adjusting this one guard alone.
Worth revisiting from yet another angle (e.g., dedicated animal-care
rescue stage analogous to the critical-unwatered/unfed stages, instead of
a blanket cash/weed-gated target) rather than more threshold tuning on
this same knob.

## v20 (continued 2) — built the dedicated animal-placement rescue, found a subtler bug via deep-dive tracing, reverted

Built a dedicated rescue stage for animals already bought but still
sitting in the shed (`_find_unplaced_animals` + a new stage in
`assign()`, mirroring the feed/water critical-rescue pattern): every
unplaced animal is $300-500 already spent producing nothing until
carried to a structure, and a generic-scored `FETCH_ANIMAL` candidate
(value ~$300-400) routinely loses to a big HARVEST (thousands).

**First-pass validation looked mixed, not clearly good:** same-seed A/B
(20 seeds vs `melon_animal_agent`) showed avg money UP ($14,722 ->
$15,201) and `pass`-sweep also up (min $17,772 -> $18,869, avg $20,652
-> $21,033) — no regression, win rate unchanged (20/20 everywhere) — but
the metric the fix specifically targeted, average PLACED animal count,
went DOWN (3.2 -> 2.2), the opposite of the goal.

**Root-caused via a day-by-day trace comparing OLD vs NEW on the same
seed (1) side by side:** structure count stayed flat at 4 the whole game
in BOTH conditions (so it was never a "can't build structures" issue).
The real divergence: `placed` (animals actually occupying a structure)
stayed at a stable 4 through day 29 in the OLD run, but in the NEW run
it dropped **4 -> 3 -> 1 -> 1 over days 26-29** — animals that were
ALREADY successfully placed earlier in the game were being LOST later,
not failing to place in the first place.

**Mechanism:** once a unit executes `PICKUP` on an animal, it falls into
`_carrying_action`'s domain (walk to structure, `PLACE`) — a multi-turn
COMMITTED trip with no way to interrupt, unlike the water-rescue stage
(which only ever commits one `WATER`-or-one-step per call and
re-evaluates fresh next turn). If a critical-unfed animal turns up
mid-carry, the carrying unit can't be redirected to it even though the
dedicated feed-rescue stage runs with HIGHER priority in the same
`assign()` call — the unit simply isn't in the idle pool to be
considered at all once it's carrying something. Late-game, when more
animals are owned and daily feed pressure is higher, this quietly
starves the feed-rescue mechanism during the exact multi-turn windows a
carry trip takes, causing REAL escapes that wouldn't have happened
otherwise. Net effect: rushing to place each new animal faster
individually made the total owned-and-fed population smaller over the
whole game.

**Reverted** (`_find_unplaced_animals` and the dedicated stage removed
entirely — dead code cleanup, not just disabled). 81 tests passing,
10/10 vs `melon_animal_agent` confirmed back to the stable v20 baseline.

**Design lesson for any future dedicated-rescue stage:** the pattern only
works safely for actions that resolve in a SINGLE bypass-the-scorer step
per call (like `WATER` once adjacent, or one movement step toward the
target) that gets freshly re-evaluated every turn — never for an action
that hands control to a multi-turn, uninterruptible commitment
(`_carrying_action`'s domain) once started.

## v20 (continued 3) — followed the trail to the actual root cause: a price ceiling silently starving animals

Asked to keep digging into what's starving feed-rescue from OTHER
carrying commitments. Measured the CURRENT (already-reverted, stable)
baseline first: `tools/replay_analyze.py`'s escape-event counter showed
**avg 11.7 escape events per episode** (10 seeds vs `melon_animal_agent`)
— confirming this is a large, pre-existing problem, not something either
of today's two reverted experiments introduced.

**Checked how often the dedicated critical-unfed rescue stage's own gate
condition (`state.shed.get("WHEAT", 0) > 0`) was actually satisfied:**
sampled every turn with at least one critical-unfed animal (220 turns
across one 10-seed run) and found shed WHEAT was sitting at **0 in 204 of
them (93%)** — the rescue stage was silently disabled almost every time
it was actually needed.

**First hypothesis (budget priority) tested and found NOT to be the
cause:** `MarketExecutor.build_orders` budgeted wheat feed top-up AFTER
land/new-animal purchases — reordered it to run first (protect animals
already owned before spending on more). Re-ran the exact same 10 seeds:
**zero change whatsoever** — byte-identical escape counts, money, and
animal counts. This ruled out budget ordering cleanly rather than
leaving it as an untested assumption.

**Traced hour-by-hour instead of guessing again:** shed WHEAT sat at 0
for a FULL day (hour 2 through 23) with healthy cash ($15k+) the whole
time, and — the key observation — **zero `BUY_PRODUCT WHEAT` orders were
issued at all** despite the top-up builder wanting to fire. Checked the
market price at that exact moment: **$39, against a $37.50 ceiling**
(`FERTILIZER_BUY_CEILING_MULT=1.5x` applied to wheat's $25 base). Found
it: `_build_wheat_feed_topup_orders` was sharing the SAME price ceiling
introduced in "v15" for fertilizer — a ceiling reasoned around
fertilizer's low stakes (missing it only forfeits a minor yield bonus).
Wheat's stakes are wildly different: missing a feed top-up risks losing
an entire animal already worth $300-500. A perfectly ordinary, mild
price fluctuation (1.56x base — not even a real spike) was silently
blocking every single feed rescue attempt.

**Fix:** gave wheat feed top-up its own, much more generous
`WHEAT_FEED_BUY_CEILING_MULT = 4.0`, separate from
`FERTILIZER_BUY_CEILING_MULT` (kept at 1.5 for fertilizer, unchanged).

**Validation — this is the largest single improvement measured all
session:**
- Escape events (10 seeds vs `melon_animal_agent`): **11.7 -> 0.0** (completely eliminated)
- Money avg (same 10 seeds): $15,185 -> $18,012 (+18.6%)
- Animal count avg (same 10 seeds): 3.0 -> 3.9, consistently near the achievable max (4 in 9/10 seeds)
- 15/15 vs `melon_focus_agent` (money noticeably higher throughout: ~$20-23k vs the old ~$14-19k range)
- 14/15 vs `melon_animal_agent` (the 1 loss matches the pre-existing ~1-2% tail-risk, unrelated to this fix)
- `pass` seed sweep: 20/20, min $17,772 -> **$21,694**, avg $20,652 -> **$22,860**
- 8/8 vs `starter`
- 81 tests passing throughout

**Methodology note:** the "budget ordering" hypothesis looked equally
plausible on paper and was tested FIRST — it would have been easy to
assume it was the fix without the identical-results check that ruled it
out. The actual cause (a price ceiling shared across two very
differently-staked use cases) was only found by tracing actual hour-by-
hour execution instead of stopping at the first plausible-sounding
theory. `main.py` rebuilt, compiles clean.

**Update:** submitted as `v21` (id `55493709`) — `COMPLETE`, publicScore
519.6. Pulled 22 public real episodes, the largest single-version sample
this whole project: **50% win rate (11W-11L)** — matches v19, but the
QUALITY of play improved noticeably: owned-animal counts now 2-4 in
nearly every game (vs 0-2 before this fix), and loss margins are mostly
CLOSE (1.2x-1.9x in 6 of 11 losses) rather than the wide blowouts
dominating earlier analysis. 9/11 losses still tag as weed-heavy, but
several turned out to be moderate (20-36%) on closer inspection, not
catastrophic — genuinely close, competitive games rather than execution
collapses. One standout: our closest loss (1.2x) was against an
opponent running only 3 GOOSE and leaving 22/25 tiles empty — an
ultra-minimal, near-zero-risk strategy that edges out our more complex
one purely by having nothing to execute wrong.

## v21 (continued) — extended the tighter hire capacity into buildout phase

Traced the closest v21 loss day-by-day and found weeds first appearing
at day 12-13 — squarely inside "buildout" phase (days 5-20 per
`PHASE_DAY_BOUNDARIES`), a full 8-9 days BEFORE "v19"'s
`TIGHT_HIRE_CAPACITY_PER_UNIT` fix engages (scoped to `scale`/`endgame`
only, day 21+). Hands stayed flat at 3 the whole buildout window in that
trace, weeds crept 1->2 slowly, then the moment scale phase kicked in
(day 21, hands jumped 3->5), weed growth nearly stopped.

**Hypothesis:** the ORIGINAL reason for scoping the tighter capacity away
from early game was that it regressed vs `melon_focus_agent` by
inflating hire costs during the fragile pre-first-harvest cash window
(NOTES.md "v19"). But "buildout" (days 5-20) is NOT that fragile window
— by day 10-11 the first harvest has landed and cash is typically
healthy ($14k-19k in every traced example this session — the fragile
window is specifically `early`, days 0-4). Extending the tighter
capacity to `buildout` too should catch the weed-creep window without
re-triggering the original regression, since buildout doesn't share
early's cash constraints.

**Fix:** `phase_capacity = TIGHT_HIRE_CAPACITY_PER_UNIT if phase in
("buildout", "scale", "endgame") else None` (was `("scale", "endgame")`
only).

**One test fixture broke and was fixed, not just patched around:**
`test_wheat_sell_order_reserves_feed_stock_for_owned_animals` used a
day-6 farm left COMPLETELY empty except for one animal structure — an
unrealistic edge case that, under the tighter buildout capacity, spiked
`hire_count` enough to fill `MAX_MARKET_ORDERS_PER_TURN` (10) with HIRE
orders alone, crowding the test's target SELL order out of the list
entirely. Fixed by filling the fixture farm with already-watered WHEAT
plants (representative of a real day-6 farm) instead of loosening the
assertion — the underlying `MAX_MARKET_ORDERS_PER_TURN` interaction is a
real, worth-remembering constraint (a big enough hire burst COULD crowd
out other important orders in a real game too), just not what this
specific test is about.

**Validation:** 81 tests passing. 15/15 vs `melon_focus_agent`
(confirms no regression of the original v19 concern), 15/15 vs
`melon_animal_agent` (no tail-risk loss in this batch), 8/8 vs
`starter`, `pass` seed sweep: 20/20, avg $22,860 -> $23,789 (further
improvement, not just "no regression"). Direct trace on a fresh seed
confirmed the mechanism: hands reached 5 already by day 10-11 (well
into buildout, not waiting for scale), and weeds stayed stable at 3-5
for the rest of that game instead of climbing. `main.py` rebuilt,
compiles clean. Not yet resubmitted — pending explicit permission.

**Update:** submitted as `v22` (id `55506717`) — `COMPLETE`, publicScore
476.1. Pulled 12 public real episodes: **50% win rate (6W-6L)**,
consistent with v21 (confirms the buildout extension isn't a regression
on the whole — combined v21+v22 sample: 13W-14L of 27 games, 48%).

## v22 (continued) — the buildout extension reintroduced a rare-but-severe death spiral; found and fixed with a targeted reserve

One of v22's 6 real losses was a genuine 14x blowout with a clear death-
spiral signature: money hit exactly $0 at day 5 and stayed there through
day 14 (9+ consecutive days), hands dropped to 0 for the same window,
weeds exploded to 23/25 (92%) by day 13, and the game was lost before a
single harvest ever landed. Traced the exact market orders around day 5
and found the mechanism: the early->buildout phase boundary (day 5)
triggered a burst of 5 simultaneous HIRE orders (this session's
"buildout" extension raising desired hire count right at that
transition) landing in the SAME turn as buildout's new crop pool
entries (BUY_SEED STRAWBERRY, BUY_SEED TOMATO) — draining the very last
of the pre-first-harvest cash cushion to exactly $0, nine days before
MELON's first possible yield (day 10).

**Measured the true incidence rate before reacting:** a 40-seed sweep vs
melon_focus_agent checking for the same signature (money still under
$500 at day 14) found 0/40 — this specific real-match loss didn't
reproduce at all locally, confirming it's a genuinely rare tail risk
(consistent with the ~1-2% class of risk already documented multiple
times this session), not a systemic regression the earlier 20-seed
validation should have caught. Chose NOT to revert the whole buildout
extension given its measured net-positive effect across many more games
(weed reduction, money improvement) — a severe-but-rare edge case calls
for a targeted guard, not throwing away a broadly-validated fix.

**Fix:** HIRE_CASH_RESERVE_PRE_HARVEST = 50, gated to state.day <
PRE_HARVEST_DAY_THRESHOLD (10, matching MELON's first_yield_day — the
earliest plausible payoff). MarketExecutor now refuses to let a HIRE
order drop money below this small reserve during the pre-first-harvest
window specifically, leaving hiring unconstrained everywhere else.

**Validation:** 81 tests passing. 14/15 vs melon_focus_agent, 15/15 vs
melon_animal_agent, 8/8 vs starter, pass seed sweep: 20/20, min $21,274
-> $22,428, avg $23,789 -> $24,606 (further improvement, not a
tradeoff). Re-ran the exact 40-seed death-spiral check that motivated
this fix: 0/40 death spirals, 0/40 total losses — at least as safe as
before, with the buildout benefit still intact. main.py rebuilt,
compiles clean.

**Update:** submitted as `v23` (id `55508043`) — `COMPLETE`, publicScore
574.5, highest of any version so far. Pulled 4 public real episodes: 2W-2L
(50%, small sample). One loss was a genuine new finding, investigated
below.

## v23 (continued) — herd size outpacing real feeding capacity, found via a 15.4x blowout

Traced a 15.4x real-match loss: money declined steadily for 7 straight
days (day 14-21, $12k -> $3.2k) DESPITE healthy hands (5) and low weeds
(8-14%) — not a weed-driven collapse, something else. Market-order trace
showed the cause: `BUY_ANIMAL COW 8` and three separate `BUY_ANIMAL
SHEEP` orders across days 14-21 (15 animal purchases total), each
individually cash-affordable and correctly budgeted. Final tile state:
**0 COW and only 4 SHEEP ever placed** — the rest were bought, presumably
placed, and lost to starvation-escape, plus 39/50 (78%) of unlocked land
sat completely empty the whole time.

**Root cause:** `_decide_animal_targets` only gated growth by CASH
(`ANIMAL_CASH_FRACTION`) — nothing capped it by whether the CURRENT unit
count could physically walk to, fetch wheat for, and FEED that many
animals every single day. Each feed trip costs several of a strictly
limited 24-turns/day, N-units budget already shared with
watering/harvesting/planting; growing the herd faster than real feeding
throughput starves animals even with plenty of wheat available (the
"v21" fix solved PRICE-driven wheat shortages specifically, not this
raw unit-capacity ceiling) — the same class of mistake
`EXPANSION_CAPACITY_BUFFER` already fixed for land ("v18").

**Fix:** `ANIMAL_CAPACITY_PER_UNIT = 1.5` — `_decide_animal_targets` now
also caps total herd growth at `current_units x ANIMAL_CAPACITY_PER_UNIT`
(summed across ALL animal types, since feeding capacity is a shared
resource, not per-species), on top of the existing cash and phase-cap
gates.

**Validation:** 81 tests passing. 15/15 vs `melon_focus_agent`, 15/15 vs
`melon_animal_agent`, 8/8 vs `starter`. `pass` seed sweep: 20/20, min
$22,428 -> **$28,384**, avg $24,606 -> **$29,692** — the single largest
jump measured on this benchmark all session, consistent with the
mechanism (cash that used to be wasted on animals that starved before
paying off now stays productive). `main.py` rebuilt, compiles clean. Not
yet resubmitted — pending explicit permission.

**Update:** submitted as `v24`. Pulled real match data: 16 episodes,
7W-9L (44%). ALL 9 losses showed `animals=1` in final tile state — a
suspiciously uniform number given `ANIMAL_CAPACITY_PER_UNIT=1.5` with
typically 4-5 hands should allow ~6-9 animals. Investigated below.

## v24 (continued) — ANIMAL_CAPACITY_PER_UNIT's herd cap silently pinned at 1 all game, via the hands-reset-at-day-boundary gotcha

Traced a representative loss (episode-93033819-replay.json): only 1
PASTURE/COOP structure was EVER built (day 14), animal count stayed at
exactly 1 for the rest of the game (days 15-29) despite `hands=5`
consistently and money climbing healthily to $28,000+. Only ONE
`BUY_ANIMAL COW 1` order was ever issued (day 14), never repeated. Ruled
out the weed-freeze guard (weed_ratio stayed 0-8%, always under the 10%
threshold) and shed inventory (no unplaced animals sitting idle).

A first repro using a test fixture with `hands` manually set to 5 real
positions showed `_decide_animal_targets` working correctly (target:
`{'COW': 8, 'SHEEP': 1}`) — contradicting the real-match behavior. A
second, more precise repro fed the EXACT real observation dict from the
replay at day16/hour==0 directly into a fresh `StrategyAgent().plan()`
call, and reproduced the bug: `hands=0`, `animal_targets={'COW': 1}`.

**Root cause:** the v23 fix read `current_units = 1 +
len(state.my_farm.hands)` inside `_decide_animal_targets`, but
`state.my_farm.hands` is ALWAYS empty at the moment `plan()` runs
(hour==0, before that day's HIRE orders execute) — this is the exact
"hands reset at day boundary" gotcha already flagged in CLAUDE.md, which
the v23 fix fell into. `current_units` was therefore always effectively
1, permanently capping `herd_room = current_units * 1.5 - owned` at
~0.5-1 for the ENTIRE game, regardless of the real hire_count executed
later that same turn. Confirmed via code read: `plan()` computed
`animal_targets` BEFORE `hire_count`, so there was no way for
`_decide_animal_targets` to know the real post-hire unit count.

**Fix:** reordered `plan()` to compute `hire_count` before
`animal_targets`, then thread the expected post-hire headcount
(`1 + len(state.my_farm.hands) + hire_count`) into
`_decide_animal_targets` via a new `expected_units` parameter, replacing
the stale `len(state.my_farm.hands)` read. `_decide_structure_targets`
moved down to stay adjacent to `animal_targets`.

**Validation:** 81 tests passing. Re-ran the exact day16/hour==0 replay
repro: `animal_targets` now `{'COW': 8, 'SHEEP': 1}` (was `{'COW': 1}`),
matching the earlier hands=5 fixture repro exactly (`expected_units = 1 +
0 + 5 = 6`, `herd_room = 6*1.5 = 9`). 10/10 vs `melon_focus_agent`, 10/10
vs `melon_animal_agent`. `pass` seed sweep: 10/10 wins, min $25,501, avg
$27,516, 0 animal escape events, 1.6 weed events/episode — no regression
vs the v23 numbers. `main.py` rebuilt, compiles clean.

**Update:** submitted as `v25` (id `55515741`), `COMPLETE`, publicScore
516.2 (leaderboard rank 3151/4496 — for scale, rank 100 sits at 2826.7,
a 5.5x gap no single fix closes). Pulled 18 public real episodes: 10W-8L
(55.6%), the best sample yet (was 50%). Investigated below.

## v25 (continued) — pre-harvest cash reserve only protected HIRE's own spend, not the other purchases that could drain it just as easily

Traced the worst v25 loss (26.8x blowout, $1,074 vs $28,787):
`money` hit exactly $0 at day 5 hour 23 (via a `BUY_SEED STRAWBERRY`
order, not HIRE), then `hands` stayed at 0 for 10 straight days (day
6-15) because even the cheapest $1 HIRE was blocked by the v23 reserve
guard once cash was already gone — weeds exploded 0 -> 23 in that
window. Checking all 18 v25 replays for the same pattern (`money` <= $5
before day 10) found **5/18 games (28%)** hit it — not the rare tail
risk the earlier 40-seed local sweep suggested (0/40), a real and
frequent vulnerability in real matches specifically.

**Root cause:** v23's `HIRE_CASH_RESERVE_PRE_HARVEST` guard only checked
`cost > budget - hire_reserve` inside the HIRE order loop — nothing
stopped `BUY_SEED`/`BUY_LAND`/`BUY_ANIMAL`/`BUY_PRODUCT` from draining
the exact same shared `budget` variable straight to $0 first, at which
point HIRE's own guard (correctly) refused to push further below the
reserve — but the reserve had already been spent by something else
before HIRE ever got a turn to protect it.

**Fix:** renamed to `PRE_HARVEST_CASH_RESERVE` and moved it to the top
of `MarketExecutor.build_orders` — subtracted from the working `budget`
once, before ANY order type is built, so every downstream affordability
check (`budget // cost`) automatically respects the floor the same way
an already-tight budget does. Removed the now-redundant HIRE-specific
check.

**Validation:** 81 tests passing. 10/10 vs `melon_focus_agent`, 10/10 vs
`melon_animal_agent`. 20-seed `pass` sweep specifically re-checking the
`money <= $5` before day 10 condition: **0/20** (was 5/18 in real
matches), min $25,088, avg $26,146 — no regression vs prior numbers.
`main.py` rebuilt, compiles clean.

**Update:** submitted as `v26` (id `55518155`), `COMPLETE`, publicScore
532.5. Pulled 10 public real episodes: 5W-5L (50%, small sample — two
losses were near-ties at 0.95x/0.99x margin, not bugs). Two big blowout
losses (0.34x, 0.36x) investigated below — a much bigger find than
expected.

## v26 (continued) — 8 COW bought, only 1-2 ever placed: structures lost the tile-allocation contest to crops

Traced both 0.34x/0.36x blowout losses: CLEAN-BUT-OUTSCALED pattern
(weeds 6-8%, cash healthy $16-26k throughout, no crisis) — matches the
long-documented "opponents run wider portfolios" strategic fork. But
digging into WHY our own animal investment underperformed found something
new: `plan.animal_targets` correctly asked for `{'COW': 8, 'SHEEP': 1}`
from day 11 through day 20 (confirmed via direct `plan()` replay), a
single `BUY_ANIMAL COW 8` order was correctly issued and filled — but
shed inventory showed **7 of those 8 COW sitting unplaced for the entire
rest of the game** (day 15 through day 29), with only 2 PASTURE
structures ever built despite `structure_targets` correctly asking for
8. $2,400+ of purchased animals never produced a single unit of milk.

**Root cause:** `_build_candidates`' structure-building code
(`farm_ops.py`, the "one structure kind per turn" block) used a flat
`value = 80` for ALL `BUILD_PASTURE`/`BUILD_COOP` candidates — while
competing crop-planting candidates use `base_price - seed_cost`, e.g.
MELON's 250-80=170, more than double. Every time a tile briefly opened
up (post-harvest), the scorer handed it to a crop almost every time,
starving structure-building of the empty tiles it needs. This is the
same class of "value too low, gets starved by fair competition" bug the
COLLECT_FERTILIZER re-weighting fixed in v15 — just never applied to
structures.

**Fix:** structure value now scales to the highest `base_price` among
animals that structure houses (200 for PASTURE via SHEEP, 50 for COOP
via GOOSE) instead of the flat 80 — competitive with crop values instead
of automatically losing. This surfaced a second, smaller bug during
validation: `test_fetch_animal_pickup_when_shed_has_animal_and_empty_
structure` failed because the new PASTURE value (200) could now
outrank `FETCH_ANIMAL` (pick up an animal ALREADY bought, from the shed,
into a structure that ALREADY exists) for a *different* animal's GOOSE
FETCH (100) — a real regression (placing paid-for stock should always
beat starting a brand-new structure). Fixed by bumping FETCH_ANIMAL's
value multiplier from `base_price * 2` to `base_price * 5`, keeping it
comfortably above any structure value (GOOSE's floor of 250 vs PASTURE's
ceiling of 200).

**Validation:** 81 tests passing. 10/10 vs `melon_focus_agent`, 10/10 vs
`melon_animal_agent` (money noticeably up in both, ~$26-27k vs prior
~$22-23k in several seeds). 15-seed `pass` sweep specifically tracking
placed-animal count: **8/8 placed animals in every single seed** (was
plateauing at 2-5 in real matches), min money $25,088 -> **$33,109**,
avg $26,146 -> **$35,146** — the single largest validated jump measured
all session, bigger than v21's wheat-ceiling fix. `main.py` rebuilt,
compiles clean.

**Update:** submitted as `v27` (id `55519130`), `COMPLETE`, publicScore
524.4. Pulled 21 public real episodes: 10W-11L (47.6%) — down from v25's
55.6% despite the strongest local validation all session. Investigated
below.

## v27 (continued) — `_decide_hires` never counted animal-care workload, so hands collapsed 5->2 the moment the structure fix started actually placing a full herd

Traced the three worst v27 losses (0.38x-0.44x margin, all
CLEAN-BUT-OUTSCALED — low weeds, healthy cash, no crisis, just decisively
outscaled). All three show the exact same shape: animals correctly reach
8-9 by day 12-14 (v27's structure fix working as intended), but `hands`
drops from 5 to 2 in that same window and NEVER recovers for the rest of
the game, despite money staying healthy ($16k-31k) the whole time.

**Root cause:** `_decide_hires`'s `actionable_tiles` count — the sole
input driving desired hire count — only ever counted crop-related work
(empty tiles still needing a planting, unwatered plants). It never
counted placed animals needing daily CARE, the same daily-reset cadence
as WATER. Once the crop portfolio was fully planted and well-watered
(normal by mid-buildout), this count naturally shrank, so the formula
kept computing a LOWER hire need — completely blind to the fact that
8-9 newly-placed animals (a direct consequence of v27's OWN fix) were
now competing with just 2 hands for daily attention. This is exactly the
"hands not recovering despite healthy cash" symptom the `_decide_hires`
docstring flagged as real-but-unexplained back in v18 — it just needed
enough animals actually placed to become visible, which didn't reliably
happen until v27's structure-value fix.

**Fix:** count each placed animal (`StructureTile` with `.animal is not
None`) as one actionable unit in `_decide_hires`, alongside plant tiles.

**Validation:** 81 tests passing. Replay repro on the exact failing
trajectory: `hire_count` now 4-5 on days 13-20 (was silently landing on
2). 10/10 vs `melon_focus_agent` and `melon_animal_agent`, money up
again in both (several seeds now $32k-35k, vs $26-27k after the
structure fix alone). 15-seed `pass` sweep: hands sustain 4-5 from day 5
through day 28 (only the expected day-29 wind-down dips lower, matching
existing end-of-season liquidation behavior — not a regression), min
money $33,109 -> **$38,149**, avg $35,146 -> **$40,975** — another large
jump stacking directly on top of v27's structure fix. `main.py` rebuilt,
compiles clean.

**Update:** submitted as `v28` (id `55522346`), `PENDING`. 2 submissions
remaining today.

## "Rocket Turtle" Monte Carlo experiment — large-scale numeric-weight search on Kaggle Notebooks, and why its best candidate got rejected

Built a self-contained Kaggle Notebook (`kaggriculture-rocket-turtle-monte-carlo`, 3 parallel copies with different RNG seeds, GPU deliberately OFF since this workload is pure Python game-logic with no tensor math to accelerate — multiprocessing across every CPU core is the real lever) that embeds a FROZEN, read-only copy of the battle-tested agent and Monte Carlo-searches `TASK_SCORE_WEIGHTS`/`HOLD_FLOOR_PCT`/`PHASE_DAY_BOUNDARIES` (same space as `tools/tune.py`), each candidate evaluated head-to-head against the frozen baseline (8 episodes) plus both scripted opponents (3 episodes each). Kernel 1 completed 420 trials in 210 minutes using 4 workers.

Top candidate ("trial 217": `urgency=207.6, value=0.50, distance=5.71`, `early_end=8, buildout_end=17, scale_end=26`, assorted hold-floors) reported a perfect 8/8 (100%) win rate vs the frozen baseline in the notebook, plus 100% vs both scripted opponents — looked like a clean win.

**Re-validated locally with a larger, more decisive sample before touching config.py** (15 head-to-head episodes vs the current `kaggriculture` package, not just the notebook's 8): **20% win rate (3/15), avg margin -3,728 — trial 217 actually LOSES decisively to our own agent**, despite still going 10/10 against both scripted opponents and showing healthy money in a `pass` seed sweep (min $39,500, avg $41,449, no death spirals). It overfits hard to beating the two static single-quadrant scripted bots while trading away robustness against a real adaptive opponent — exactly the "local validation is necessary but not sufficient" lesson this project keeps re-learning, just now caught INSIDE the tuning process itself rather than only at the real-match stage. The notebook's 8-episode sample was too small to see this; 15 did.

**Verdict: trial 217 NOT adopted.** `kaggriculture/config.py` was never touched. Kernels 2 and 3 (different seeds, same search space) still running — any of their top candidates need this same large-sample local re-validation (h2h vs the CURRENT real agent, not just the notebook's own small-sample numbers) before being considered, not a free pass just because the notebook reported a win.

## Rocket Turtle, continued — kernels 2 and 3 finish (1,464 trials total), trial 351 survives large-sample scrutiny and gets adopted

Kernels 2 (592 trials) and 3 (452 trials) completed, bringing the merged
total across all 3 parallel kernels to 1,464 trials. Ranked all trials
with `win_rate_vs_baseline == 1.0` (143 candidates) by margin — a strong
reminder that testing 1,464 candidates against an 8-episode benchmark
will produce a meaningful number of lucky false positives on pure chance
alone (trial 217 was one), so the top-margin candidates still needed the
same large-sample re-validation before trusting them.

Re-tested the next 4 highest-margin candidates with 15 head-to-head
episodes each vs the real v27/v28 code:

| trial | notebook (n=8) | re-check (n=15) | verdict |
|---|---|---|---|
| k2/565 | 100% | 33.3%, margin -464 | REJECTED — overfit, same as 217 |
| k2/513 | 100% | 20.0%, margin -816 | REJECTED — overfit |
| k1/351 | 100% | **100%, margin +5,989** | survives |
| k2/90  | 100% | 86.7%, margin +1,401 | survives, but weaker |

**Trial 351** (`urgency=267.9, value=0.12, distance=8.19`,
`early_end=6, buildout_end=9, scale_end=19`, hold-floors ~0.10-0.47)
went further: a SECOND independent 30-episode h2h batch (different RNG
draws) confirmed it at 29W-1L (96.7%, avg margin +5,332) — combined
44W-1L across 45 total head-to-head episodes vs the real agent. Also
15/15 vs `melon_focus_agent`, 15/15 vs `melon_animal_agent`, and a
20-seed `pass` death-spiral sweep: 0/20 losses, min money
$38,876/avg $43,043 — HIGHER than the baseline's own pass-sweep numbers
(min ~$38k/avg ~$41k). An earlier crude "money touched <= $5 before day
15" flag fired on 6/20 seeds and briefly looked like a death-spiral
signal, but turned out to be a red herring: `hands` never collapsed
(0/20 multi-day-zero streaks) and every seed still finished with
excellent money — a transient, self-recovering pre-harvest cash dip
(more frequent than baseline's own dip pattern, given the more
compressed `early_end`/`buildout_end`), not the v7/v8-style spiral
(cash crash -> hands stuck at 0 -> weeds compound -> real loss) that
standard is meant to catch.

**Applying trial 351's weights broke one existing unit test**
(`test_critical_unwatered_plant_gets_dedicated_rescuer_over_big_harvests`):
only 1 of 3 units harvested instead of 2. Traced the mechanism rather
than just loosening the assertion: the critical-unwatered dedicated
rescue stage (bypasses the scorer, guarantees exactly one rescuer via
`num_rescuers = min(len(remaining_tiles), len(rescuer_pool))`) was
working correctly, but the tile it claimed was never removed from the
candidate pool `_build_candidates` hands to the SCORER for the
remaining idle units — with trial 351's much higher `urgency` weight
(267.9 vs the old 23.5), that same tile's ordinary WATER candidate could
now outscore even a zero-distance ready HARVEST for a second unit too,
pulling it toward an already-handled tile and abandoning an adjacent
harvest for nothing. Old (low-urgency) weights never made this visible
because the WATER candidate's score never got competitive enough to
matter. Fixed by seeding the scorer's `assigned_tiles` exclusion set
with every tile already claimed by a dedicated rescuer
(`kaggriculture/farm_ops.py`, `UnitScheduler.assign()`) before the
generic scoring pass runs, so a second unit can't redundantly re-target
it. All 81 tests pass after the fix (no assertion needed loosening).

**Applied to `kaggriculture/config.py`**: `TASK_SCORE_WEIGHTS`,
`HOLD_FLOOR_PCT`, `PHASE_DAY_BOUNDARIES` replaced with trial 351's
values (old values kept in a comment for reference), plus the
rescue-tile-exclusion fix in `farm_ops.py`.

**Final validation with both changes together:** 81 tests passing.
10/10 vs `melon_focus_agent` (money $34,784-$41,580, up from prior
~$26-27k), 10/10 vs `melon_animal_agent` (money $35,465-$40,238). 20-seed
`pass` sweep: **0/20 zero-cash hits** (down from 6/20 with trial 351's
weights alone — the rescue-exclusion fix also improved cash efficiency),
min money **$34,702**, avg **$42,462**. `main.py` rebuilt, compiles
clean.

**Update:** submitted as `v29` (id `55527409`), `COMPLETE`, publicScore
590.8 — highest all session. Pulled 19 public real episodes: 10W-9L
(52.6%, up from v28's 47.8%). Also backed up the whole project to GitHub
(`github.com/Alphine/kaggleculture`, SSH) as a separate ask this
session — source, tests, tools, NOTES.md pushed; `replays/` (~1GB,
regeneratable via the Kaggle API) intentionally excluded via
`.gitignore`.

## v29 (continued) — investigated a late-game "empty land" pattern; hypothesis tested and REJECTED, no change applied

All 19 v29 replays showed the same shape: empty tiles low (0-8) through
day ~19-20, then growing every single game afterward (up to 34-38/100 by
day 29). Hypothesized cause: trial 351 cut `TASK_SCORE_WEIGHTS["value"]`
~3x while raising `["urgency"]` ~11x, so PLANT (urgency=1, same baseline
as routine WATER/CARE) increasingly loses the scoring competition once
the farm is fully staffed (9 animals needing daily CARE) — worsened by
`scale_end` dropping from 27 to 19, stretching "endgame" to ~10 days
instead of ~2-3.

**Tested the fix (`PLANT_URGENCY = 2`, matching HARVEST's urgency) with
a controlled local A/B before touching anything else** — and it made
things dramatically WORSE, not better: 15-seed `pass` sweep min money
$39,948 -> $27,846, avg $43,011 -> $29,451, avg empty-tiles-at-day-29
5.9 -> 24.0. Likely cause: PLANT now competing evenly with HARVEST
(also urgency=2) let planting occasionally beat out harvesting itself,
delaying revenue. Reverted immediately (`git stash drop`) — never
committed.

**Re-examined the premise**: pure v29 (unmodified) against BOTH scripted
opponents (8 episodes each, competitive, not passive `pass`) shows
empty-tiles averaging only 6.5-6.9 (max 10-11) — nothing like the
34-38 seen in a couple of real matches. Traced the worst real case
(episode-93318792, empty=38) day-by-day: empty was ALREADY 14-21 from
day 10-19, well before "endgame" even started at day 20 — contradicting
the phase-boundary-length theory. Weeds stayed low (0-3) the whole
game, hands stayed at 5, animals filled to 9 — this specific match's
land underuse doesn't fit the hypothesized global mechanism at all.

**Verdict: no fix applied.** The global "value vs urgency" theory failed
its own predicted fix, and the milder version of the pattern doesn't
reproduce locally against either scripted opponent at anywhere near the
severity seen in 1-2 real matches — this looks like an opponent-specific
interaction (a real opponent doing something our scripted proxies
don't) rather than a systemic weight-tuning flaw. v29 as submitted
remains the best validated state; flagging this as an open question for
a future, more targeted investigation (ideally once more real replays
showing the same extreme pattern are available to compare against) —
not a diagnosed bug with a known fix yet.

## Rocket Turtle round 2 — refined local search around trial 351, across 5 parallel platforms, found nothing that survives large-sample re-validation

Instead of another blind wide search, round 2 perturbed trial 351's own
values locally (urgency/value/distance ±50%, hold-floors ±0.15,
phase boundaries ±2-8 days, `scale_end` deliberately widened upward
toward the pre-v29 default of 27 to test whether that eases the
late-game land-utilization pattern flagged above) — baseline frozen as
the CURRENT `kaggriculture/` (v29, trial 351 + rescue-exclusion fix).

Ran across genuinely parallel *grid computing*, not just multiple
Kaggle kernels: 4 Kaggle CPU kernels (batch-session cap is 5 concurrent,
confirmed via Kaggle's own product-feedback page) totaling 2,044 trials,
plus — since `google-colab-cli`'s official headless mode
(`googlecolab/google-colab-cli` on GitHub) turned out to need
POSIX-only `termios` and doesn't run on native Windows (confirmed by
directly installing and hitting `ModuleNotFoundError: No module named
'termios'`; Docker was ruled out too, since Docker Desktop on Windows
just runs Linux containers via a WSL2 backend anyway, no simpler than
installing WSL directly) — a same-source `.ipynb` handed to the user to
run manually in Google Colab (own RNG seed range, zero collision risk:
every trial is self-contained, results just concatenate), adding 286
more trials. **2,330 trials total across 5 independent compute nodes.**

135 candidates hit a perfect win_rate_vs_baseline==1.0 in-notebook
(8 episodes) with >=0.66 vs both scripted opponents. Re-validated the
top 4 by margin with 15 head-to-head episodes each vs the REAL current
v29 agent (not the notebook's small sample) — same discipline that
caught trial 217:

| candidate | notebook (n=8) | re-check (n=15) |
|---|---|---|
| b/645 | 100%, +6,203 | 13.3%, **-2,027** |
| d/325 | 100%, +6,187 | 53.3%, +68 (~coin flip) |
| c/185 | 100%, +5,908 | 13.3%, **-1,350** |
| b/75  | 100%, +5,906 | 6.7%, **-1,938** |

**All 4 failed.** Notably, all of round 2's top candidates clustered at
or near the search's own bounds (`urgency` pinned at 300, `distance`
pinned at 10, `value` pinned near the 0.1 floor) — extreme parameter
combinations are inherently higher-variance strategies, exactly the
kind that can luck into a perfect small-sample score without holding up
under more games.

**Verdict: no change applied.** Round 2 found no candidate that beats
the current agent under real scrutiny — trial 351 looks like it's at or
very near a local optimum for this search radius. `kaggriculture/
config.py` stays as-is (v29, unchanged). Confirms the process is doing
its job: catching overfit "wins" before they ever reach
`kaggriculture/config.py`, even at the cost of a round that produced no
usable improvement.

## v29 (continued) — 30 real episodes pulled, 46.7% win rate, found and fixed a real crop-selection gap: opponent-visibility proxies can fade to zero while a crashed price never recovers

Pulled 11 more real v29 episodes (30 total): 14W-16L (46.7%, down from
the earlier 19-episode sample's 52.6% — normal convergence as more
games land, and expected given the leaderboard rating keeps pulling in
tougher real opponents as it moves up). Two of the newest losses were
by far the most extreme margins seen all project — 0.12x ($15,089 vs
$127,741) and 0.21x ($34,528 vs $164,618).

Traced both day-by-day: our OWN execution was clean in both (hands
stayed at 5, animals filled to 9, weeds stayed low) — but money nearly
flatlined for 15+ days in the worse one ($13,538 -> $15,089, day 14 to
29). First hypothesis (WHEAT stuck at 17-18 units unsold, hoarding due
to trial 351's much higher `HOLD_FLOOR_PCT`) was checked and DISPROVEN
by reading the actual sell-order code: `wheat_reserve =
animal_count * 2` (9 animals -> 18) is a deliberate feed reserve,
excluded from selling on purpose (README: animals need daily WHEAT) —
not a hoarding bug at all, a correct guard already documented at v20.

Checked MELON's live market price instead: it crashed from $99 (day 10)
to $1-22 for the ENTIRE rest of both games (base price $250 — under
10% of base). In the worse loss, the opponent's own visible MELON tile
count dropped to 0 by day 17 (they read the crash and pivoted away
entirely — their money then quadrupled on a completely different
portfolio, $28,855 -> $127,741 in 12 days with zero MELON tiles). We
kept 8 MELON tiles planted through day 22, five days into a market that
had been sub-$22 (91%+ below base) the entire time.

**Root cause:** `_score_crop` never looks at the crop's actual live
market price at all — only three indirect proxies (visible opponent
tile count, opponent growth rate, market inventory momentum). All three
can fade back toward zero once the CAUSE of a crash goes away (the
opponent stops planting, inventory gets sold down) while the price
itself stays crashed. Confirmed directly: `opponent_counts.get('MELON')`
hit 0 the moment the real opponent abandoned MELON, silencing the flood
penalty completely, even though price was still at $1-7.

**Fix:** added a direct realized-price backstop, independent of the
existing proxies. `_decide_crop_targets` now computes
`current_price(crop, state) / CROP_DATA[crop]["base_price"]` per
eligible crop and passes it into `_score_crop`, which discounts the
score continuously once price falls below `MARKET_PRICE_CRASH_THRESHOLD
= 0.5` (a strict no-op at or above the threshold, scaling down toward
zero as price approaches 0).

**Validation:** 81 tests passing. Replay repro on the exact failing
trajectory (episode-93431034): MELON's target now correctly shrinks
with its crashing price — day15 (price $19): target 10; day17 ($7):
target 5; day19 ($1): target 2; day21 ($7): target dropped out of the
plan entirely, reallocated to WHEAT/CARROT instead. 10/10 vs
`melon_focus_agent` (money $32-42k), 10/10 vs `melon_animal_agent`
(money $34-51k, highest seen all session). 20-seed `pass` sweep: 0/20
zero-cash hits, min money $34,702 -> **$39,065**, avg $42,462 ->
$42,273 (flat, no regression). `main.py` rebuilt, compiles clean.

**Update:** submitted as `v30` (id `55539069`), `COMPLETE`, publicScore
558.7. Pulled 30 public real episodes: 14W-16L (46.7%, matches v29's
rate exactly) — but money is far more consistently healthy now (floor
$22,279, was closer to $1k-14k in earlier versions' worst cases).

Traced the newest most-extreme loss (0.35x, $36,710 vs $105,068):
completely clean execution — hands stayed at 5, weeds under 6%, animals
filled to 9, land utilization stayed reasonable (2-11 empty tiles, far
better than the 34-38 seen in earlier extreme losses), money climbed
steadily and healthily the entire game. **No bug found** — this is the
long-standing CLEAN-BUT-OUTSCALED pattern, purely a scale/ambition gap
now that v27-v30's fixes (structure placement, hire formula, price-crash
backstop) have closed off the execution-bug losses that used to hide it.

Leaderboard check: rank 2837/4644, score 608.5 — rank 10 sits at
2972.0, rank 1 at 3221.6. The gap to any single-digit rank is roughly
5x our current score; top-100 (~2800) is the more realistic near-term
target, still needing several more multiples of real-match performance.

## "Rocket Turtle goes Darwin" — evolutionary self-play, testing the long-flagged scale-up hypothesis under real competitive pressure instead of a manual guess

Since v27-v30 closed the execution-bug losses and what remains is
squarely the scale gap, and the Monte Carlo rounds' local-refinement
search around trial 351 found nothing further (round 2, NOTES.md above),
built a genuinely different search: EVOLUTIONARY SELF-PLAY instead of
random search vs a frozen baseline. A population of 12 genomes plays
each other directly every generation (`play_match`, 3 episodes/pairing);
losers get replaced by a mutated copy of that pairing's winner; the
current battle-tested v30 config is re-seeded into the population every
generation as a fixed anchor (measurable against a real, known-good
reference even as everything else evolves).

Genome space deliberately widened beyond tune.py's numeric-weights-only
scope to also include `ANIMAL_CASH_FRACTION`, `ANIMAL_CASH_RESERVE`, and
`CROP_TARGET_WEIGHT` — directly reopening the "scale-up ambition"
question flagged repeatedly since v15 (CLAUDE.md: "needs the same
seed-sweep death-spiral testing v7/v8 required — flagged for an explicit
decision, not changed unilaterally") to real competitive self-play
pressure, rather than another manual guess. The one earlier scale-up
attempt this session failed, but that was BEFORE the animal-capacity,
hire-formula, and structure-placement bugs were fixed — this lets
self-play re-litigate that question on the current, bug-fixed
foundation.

4 parallel Kaggle CPU kernels launched (`kaggriculture-evolve-selfplay-
a/b/c/d`, distinct seeds, population 12, 3.5h budget each, GPU off —
same reasoning as the Monte Carlo rounds, this is still pure Python
game-logic simulation with no tensor math to accelerate). Smoke-tested
locally first (6-genome population, 1 generation) to confirm the
generational loop, mutation, and tournament logic all work correctly
before spending real Kaggle compute. Results still pending — same
validation discipline applies once they land: any winning genome needs
a large-sample head-to-head vs the REAL current v30 agent (not just
other self-play population members) before ever being considered for
`kaggriculture/config.py`.
