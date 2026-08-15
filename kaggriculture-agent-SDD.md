# Software Design Document: Kaggriculture Agent

**Project:** Kaggriculture Nusa agent competitive agent (Kaggle simulation competition)
**Author:** Basyid / Kawalight Studio
**Status:** Draft v1 — ready for Claude Code implementation
**Target:** Python agent submission (`agent()` function) + offline self-play training/tuning harness

---

## 1. Problem Summary

Kaggriculture is a 1v1, turn-based farming economy simulation. Each episode: 720 turns (24 turns/day × 30 days), two players sharing one dynamic market. Winner = most money in bank at episode end. Our submission is a stateless Python function `agent(obs) -> action_dict` invoked every turn by the Kaggle environment (`kaggle_environments`). No internet access, no persistent state across episodes at evaluation time — all "learning" must happen offline before submission.

Full game rules (objects, actions, market pricing formula, turn processing order, observation schema) are in the attached `README.md` (Kaggriculture official rules) — **read this file first**, it is the ground truth for all mechanics, formulas, and the exact observation/action JSON schema. This SDD does not restate every rule; it only restates what's load-bearing for architecture decisions.

---

## 1.1 Confirmed Competition Constraints (from official Kaggle Rules, AGENTS.md, and Overview page)

These are now confirmed from primary sources (Official Competition Rules + AGENTS.md getting-started guide + Competition Overview page), replacing prior assumptions:

- **No ingress/egress at evaluation** — official rule, verbatim: *"During the evaluation of an episode your Submission may not pull in or use any information external to the Submission and Environment and may not send any information out."* This is a hard rule, not just a platform default — confirms no LLM/API calls, no network calls of any kind inside the submitted agent at evaluation time.
- **No Private Leaderboard in Simulation competitions** — official rule, verbatim. Scoring is continuous episode-based aggregation (skill rating), not a train/public/private test-set split like standard Kaggle prediction competitions.
- **Full timeline (confirmed)**:
  - Start Date: **July 29, 2026**
  - Entry Deadline / Team Merger Deadline: **September 23, 2026**
  - **Final Submission Deadline: September 30, 2026**
  - Post-deadline evaluation window: games keep running **Oct 1 – ~Oct 15, 2026** to let the leaderboard converge; a **Bradley-Terry tournament** is run on the accumulated episodes to produce the final leaderboard (a pairwise-comparison statistical model for estimating relative strength — generally more robust than a raw final rating snapshot, since it uses the full body of match results rather than just the latest number).
  - All deadlines 11:59 PM UTC.
- **Submission limits**: max **5 submissions/day**; but **only the latest 2 submissions are "active"** — they're the ones that keep getting matched for new episodes and are the ones used for final leaderboard evaluation. Older submissions stop accumulating new games once bumped out of the active-2. Newer submissions get matched more frequently than older ones (to stabilize their rating faster). Only your best-scoring active bot is shown on the public leaderboard, but all submissions' progress is trackable on the Submissions page.
- **Validation Episode on upload**: every submission automatically plays one episode against a copy of itself before entering the matchmaking pool. If that fails, the submission is marked `Error` and agent logs can be downloaded to debug. This is a correctness gate, not a skill gate.
- **Ranking system (skill rating, Elo-like)**: win increases rating, loss decreases it, ties pull ratings closer together. Rating *change magnitude* depends on the opponent's rating (beating a stronger opponent moves your rating more). **Coin margin does not affect rating change at all — only win/lose/tie matters.** This confirms the earlier strategic conclusion: favor consistent wins over high-variance high-upside plays that occasionally win by a huge margin but sometimes lose.
- **Submission resource limits (confirmed — resolves prior open question on compute budget)**:
  - Max submission size: **100 MiB**
  - Runtime resources: **8 GiB HDD, 6.5 GiB RAM, 1.6 vCPU**
  - Submission files live at `/kaggle_simulations/agent/` at runtime — relevant if the bundled `main.py` needs to reference any co-bundled non-code assets.
  - No explicit per-turn time limit is stated, but the **1.6 vCPU budget is small** — this reinforces the architecture decision (§3–§6) to keep the tactical layer a cheap greedy/scored heuristic rather than any per-turn search or ML inference; there is not much compute headroom for anything heavier at 720 turns/episode.
- **Submission format confirmed**: a `main.py` at the root of the submission, containing an `agent` function. Multi-file agents are supported by bundling into a `.tar.gz` with `main.py` at the root (e.g. `tar -czf submission.tar.gz main.py helper.py model_weights.pkl`).
- **Built-in baseline agents available by name** (no need to build our own from scratch for early testing): `"pass"` (always passes), `"random"` (legal-random actions), `"starter"` (a deterministic baseline provided by the competition itself — its reference implementation, a basic wheat loop, is given in AGENTS.md). Use `"starter"` as the primary early benchmark instead of hand-rolling a greedy baseline.
- **Observation includes a `step` field** (0-indexed global turn counter, supplied by the `kaggle_environments` framework itself) in addition to `day`/`hour` — useful for one-time-only logic (e.g. "on the very first turn, buy starter seeds").
- **Public code sharing is explicitly permitted** (Rules §3.6.b) via Kaggle's own competition forum/notebooks — anything shared that way is implicitly Open Source Initiative-licensed. **Private sharing between individuals/teams outside an official Team merge is prohibited** (Rules §3.5.d). Treat any third-party notebook found outside the official competition forum/notebook listing with caution — verify provenance before using it as a reference, and never submit copied code as original work.
- **Winner obligations**: if a submission wins a prize, the methodology (architecture, tuning, hyperparameters, reproducibility) must be documented and open-sourced under CC-BY 4.0 — worth keeping `NOTES.md` and `config.py` clean and well-commented from the start rather than retrofitting documentation later.
- **CLI tooling available** for the full submission lifecycle: `kaggle competitions submit`, `kaggle competitions submissions`, `kaggle competitions episodes <SUBMISSION_ID>`, `kaggle competitions replay <EPISODE_ID>`, `kaggle competitions logs <EPISODE_ID> <player_index>`, `kaggle competitions leaderboard kaggriculture -s`. This lets us pull real leaderboard replays/logs for post-hoc analysis against actual opponents, not just our own local self-play — fold this into the `replay_analyze.py` scope (§8).
- **Testing workflow — use Kaggle Notebook (or local device) as the primary test ground, not live submissions**: submissions are a scarce, slow-feedback resource (5/day quota, only 2 active at a time, matchmaking-dependent turnaround) and should be reserved for genuine leaderboard progress once an agent version has already proven itself locally. All iteration/debugging (§8's self-play loop) should run via `pip install -U kaggle-environments` and `env.run([agent, "starter"])` in a Kaggle Notebook or on a local machine — free, instant feedback, zero submission-quota cost. Only submit once a version consistently beats `"starter"` in local self-play.

---

## 2. Goals & Non-Goals

**Goals**
- Build a competitive agent that beats baseline/random opponents consistently, and performs well against varied heuristic/adaptive opponents.
- Modular architecture: strategic (economic) decisions separated from tactical (unit movement/action) decisions.
- Offline self-play training harness to iterate strategy without needing to resubmit to Kaggle each time.
- Fast, deterministic, pure-Python execution — no network calls, no LLM calls inside the submitted agent (runtime constraints at evaluation forbid this).

**Non-Goals (v1)**
- No real-time/online learning during Kaggle evaluation (not supported by platform — confirmed architectural constraint).
- No deep RL (PPO/self-play neural nets) in v1. Reserved as a v2 stretch goal if timeline allows — v1 is heuristic + scored planning, tuned via offline self-play/optimization.
- No LLM-in-the-loop decision-making at inference time.

---

## 3. Architecture Overview

Two-layer orchestrator pattern, re-evaluated fresh each call to `agent(obs)` (agent is stateless between calls unless we implement our own turn-counter/memoization inside the returned action encoding — see §7 State Handling).

```
                    ┌─────────────────────────┐
                    │   agent(obs) entrypoint │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   StateParser            │  parses obs into typed GameState
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴───────────────────┐
              │                                       │
   ┌──────────▼──────────┐                ┌───────────▼───────────┐
   │  StrategyAgent        │                │   FarmOpsOrchestrator  │
   │  (economic layer,      │───plan/goals──▶│   (tactical layer)     │
   │  re-evaluated every    │                │   runs every turn      │
   │  N turns or on trigger)│                └───────────┬───────────┘
   └──────────┬──────────┘                                │
              │                                ┌───────────┴───────────┐
              │                        ┌────────▼──────┐      ┌────────▼───────┐
              │                        │ UnitScheduler  │      │ MarketExecutor │
              │                        │ (farmer+hands  │      │ (buy/sell      │
              │                        │  task assign)  │      │  order queue)  │
              │                        └────────────────┘      └────────────────┘
              │
   ┌──────────▼──────────┐
   │  MarketModel          │  price curve math, sell-timing heuristics
   └───────────────────────┘
```

### 3.1 StateParser
Converts raw `obs` dict into a typed internal representation (dataclasses). Single source of truth for the rest of the pipeline — no other module touches raw `obs` directly. Responsible for:
- Flattening `farms[player]` into our-farm / opponent-farm views.
- Computing derived fields not present in raw obs: e.g. per-tile "action needed today" flags (unwatered, unfed, harvestable, decaying), current-day index, turns-remaining-in-day, turns-remaining-in-season.
- Computing opponent farm summary (visible tiles only — opponent shed/inventory is hidden per rules).

### 3.2 StrategyAgent (economic/strategic layer)
Runs a lighter-weight planning pass, **not every turn** — triggered:
- At the start of each day (turn % 24 == 0), and
- On significant state changes (land newly unlocked, big market price swing detected, opponent's visible farm composition changes materially).

Responsibilities:
- Decide **crop/animal portfolio mix** for the current phase of the season (early/mid/late game — see §5).
- Decide **land expansion timing** (BUY_LAND ROI check).
- Decide **hiring plan for the day** (how many hands, given Fibonacci cost curve — see §5.3).
- Set **sell-policy targets** per resource (hold vs. dump thresholds) that MarketExecutor consumes every turn — this reacts to the price-curve math in §5.4.
- Output: a `DayPlan` object (crop targets, tile assignments goal-state, hiring count, sell thresholds) consumed by FarmOpsOrchestrator.

### 3.3 FarmOpsOrchestrator (tactical layer)
Runs **every turn**. Consumes the current `DayPlan` + live GameState and produces the actual per-unit actions + market order list for this turn.

Sub-components:
- **UnitScheduler**: assigns one task each to farmer + all hired hands for this turn (move/water/harvest/plant/fertilize/pickup/drop/feed/care/build/dig). Task assignment is a **priority-scored greedy assignment**, not a full solver (keep it fast — see §6 for scoring rubric). Handles pathfinding (simple BFS/Manhattan-based movement toward target tile, since board is small 10×10).
- **MarketExecutor**: builds the market order list (max `maxMarketOrdersPerTurn`, default 10) — BUY_SEED for next planting per DayPlan, SELL orders for shed inventory above sell-policy thresholds, BUY_PRODUCT for wheat/fertilizer replenishment, HIRE orders at start of day.

### 3.4 MarketModel
Pure-math module implementing the price function from README §"The Price Function" per resource, used for:
- Estimating expected sell price before committing to SELL (helps StrategyAgent decide hold vs. dump).
- Estimating breakeven for fertilizer use per crop.
- Detecting opponent-driven price shifts (if price deviates sharply from our own order history, opponent is likely trading heavily in that resource this turn).

---

## 4. Why This Split (Design Rationale)

- **Real multi-unit control**: farmer + N hired hands each get one independent action per turn — this is genuine tactical multi-agent coordination, not a flat single-decision problem. UnitScheduler owns this complexity in isolation so StrategyAgent stays simple and fast.
- **Different re-evaluation cadence**: strategic decisions (what to plant, when to expand, sell thresholds) don't need to be recomputed every single turn — recomputing every turn wastes budget and adds churn/thrashing risk (e.g. flip-flopping crop plans). Daily cadence matches the game's own day-boundary refresh (README "Turn Processing Order" — day refresh resets watered/fed flags).
- **No LLM / no network at inference**: everything above is deterministic Python + numeric scoring, safe under Kaggle's offline no-internet evaluation constraint.
- **Testability**: each layer can be unit-tested independently against synthetic `GameState` fixtures, and swapped/improved incrementally (e.g. replace UnitScheduler's greedy scorer with a small search later without touching StrategyAgent).

---

## 5. Strategy Layer — Decision Logic Detail

### 5.1 Game Phases
Segment the 30-day season into phases to drive portfolio decisions:
- **Early (days 0–4):** cash-flow priority. Cheap fast crops (wheat, carrot) to build capital and unlock land ASAP. Avoid animals (high upfront cost, long time-to-first-yield with no payoff yet).
- **Build-out (days 5–14):** start higher-value crops (tomato, melon) and first animal (likely goose — cheapest at $300, 4-day time-to-first-yield, ongoing indefinite production) once land + cash allow. Consider first BUY_LAND here if ROI math clears (see §5.2).
- **Scale (days 15–24):** full portfolio diversification across quadrants, second/third BUY_LAND if profitable, animals fully online (cow/sheep if capital allows), fertilizer use ramps up on high-value crops.
- **Endgame (days 25–29):** stop new long-cycle plantings that won't mature before season end (check `time_to_first_yield` / `max_lifespan` against remaining days). Liquidate — sell all shed inventory before final turn, since unsold inventory doesn't count toward the win condition. Avoid new BUY_LAND (no time to amortize).

### 5.2 Land Expansion (BUY_LAND) Decision
Costs $1k / $2k / $4k for 2nd/3rd/4th quadrant. Trigger expansion when:
`projected_marginal_revenue_from_new_quadrant_over_remaining_days > land_cost + opportunity_cost_of_cash_spent`
Simple heuristic: only expand if (a) current quadrant is >70% tile-utilized (planted/structured, not empty/weed), and (b) remaining season days ≥ ~8 (enough runway for at least one crop cycle on the new land, e.g. wheat/carrot).

### 5.3 Hiring Decision
Hand cost follows Fibonacci per additional hire per day (1,1,2,3,5,8,13...). Decide hires/day by comparing marginal hand cost vs. expected value of one extra unit-turn-budget for that day (extra harvest/water/plant actions enabled). Rule of thumb: hire while `fib_cost(n) < expected_marginal_value`; expected marginal value scales with how many "actionable" tiles (needs water/harvest/plant) exceed what farmer+existing hands can cover in the day's 24 turns. Cap hires once marginal cost exceeds ~1 day's expected crop margin.

### 5.4 Sell-Policy / Market Timing
Use MarketModel to avoid dumping into a crashing price:
- Compute expected price for next N units of a SELL order using the price function (since orders are processed one unit at a time, price moves mid-order for large sells).
- Set a **hold threshold** per resource: if current price < `hold_floor_pct * base_price`, hold and re-check next few turns rather than dump (especially for premium/steep resources: strawberry, melon, milk, wool — their `above_target` multipliers are steep, so gluts crash price hard and recovery may lag).
- Spread large sells across multiple turns instead of one giant SELL order, to avoid single-order price collapse.
- Watch town shop unlocks (README "Town Buildings") — demand increases every `townShopUnlockInterval` days; rising demand = better time to sell staples matching newly unlocked shop types.

### 5.5 Crop/Animal Selection Heuristic
Score each candidate crop/animal by **expected profit per tile-day per action-cost**, derived from the Object Types table (yield/tile/day × expected sell price) minus seed/fertilizer/feed cost, adjusted by:
- Time-to-first-yield (shorter = faster capital recycling, valuable early game).
- Market price sensitivity (`above_target`) — deprioritize thin-margin overselling risk for steep-glut resources unless we can pace sells.
- Opponent's visible portfolio (if opponent is flooding a resource, expect price crash there — diversify away from it).

---

## 6. Tactical Layer — Unit Task Scoring

Every turn, for each available unit (farmer, each hand) not already mid-task:
1. Enumerate candidate tasks: {water tile X, harvest tile X, plant crop Y at tile X, fertilize tile X, feed animal at X, care animal at X, collect_fertilizer at X, pickup item, drop at shed, build structure at X, dig weed/plant at X, move toward shed for market handoff}.
2. Score each candidate: `score = urgency_weight + value_weight - distance_cost`
   - **Urgency**: e.g. `consecutive_unwatered == 1` → high urgency (next miss = weed). Same logic for `consecutive_unfed == 1` on animals.
   - **Value**: expected $ impact (harvest value, fertilize bonus value, avoided-loss value of preventing a weed/escape).
   - **Distance cost**: Manhattan distance from unit's current position to target tile (proxy for turns spent moving before the action lands).
3. Assign tasks via greedy max-score matching across all (unit, task) pairs this turn, avoiding double-assignment of the same tile to two units in the same turn (README: over-planting/double actions on one tile per turn can no-op).
4. Convert assigned task into this turn's single action per unit (movement step toward tile if not yet adjacent/on it, or the actual tile action if in position).

This is intentionally a **greedy scorer**, not a full solver — 720 turns × up to several units means we need O(units × candidate_tasks) per turn, not combinatorial search. Revisit only if profiling shows this is a weak point after baseline works.

**v1.5 stretch option — shallow bounded search:** a reference third-party notebook (structure only, not code — see §1.1 provenance note) includes a dedicated `search.py` module alongside scheduler/planner, suggesting a viable middle ground between pure greedy scoring and full RL: a **shallow lookahead** (e.g. 2–5 turns ahead per unit-task decision, not the full 720-turn horizon) to catch cases where the greedy-best immediate task is locally optimal but globally worse (e.g. watering a low-value tile now vs. moving toward a high-value harvest that's one turn further away). Only pursue this after M1–M4 are working and profiled — it's an optimization on top of a working greedy baseline, not a prerequisite.

---

## 7. State Handling Across Turns

The agent function itself is stateless per the platform's calling convention, but we are allowed to maintain **our own process-local memory** across calls within a single running episode (e.g. a module-level dict keyed by nothing, since it's one episode per process) — useful for:
- Tracking our own recent order history (to distinguish our price impact from opponent's).
- Caching the current `DayPlan` so we don't recompute StrategyAgent every turn.
- Tracking in-progress unit tasks (so a unit walking toward a tile doesn't get re-scored into a different task mid-walk unless something more urgent appears).

**Important:** verify with the actual `kaggle_environments` harness whether the same Python process/agent object persists for the full episode (very likely, based on the Quick Start example using a plain function) — confirm this assumption early during implementation and note any surprises.

---

## 8. Offline Self-Play & Tuning Harness

Since real learning must happen pre-submission, build a local harness:

1. **`env.run([agent_a, agent_b])`** loop (per Quick Start in README) run in batches (hundreds–thousands of episodes) with varied seeds (`configuration.seed`).
2. Log per-episode: final money for both players, win/lose/tie, and optionally time-series of money/turn for post-hoc analysis.
3. **Baseline opponents to test against** — use the competition's own built-in agents first (no need to hand-roll a greedy baseline for this):
   - `"pass"` — always passes, sanity/floor baseline.
   - `"random"` — legal-random actions.
   - `"starter"` — deterministic baseline provided by the competition itself. This is the primary early benchmark; treat "consistently beating starter" as the M2–M3 milestone gate before investing further in tuning.
   - `mirror_agent` (optional, later) — an earlier committed version of our own agent, for self-play comparison across iterations once we're past `starter`.
4. **Iteration loop** (explicit ordering — do not skip ahead to step 6 before 1–5 are done for a given change):
   1. Build/modify agent version `v_N` (structural/rule logic changes first, not numeric tuning).
   2. Run self-play: `v_N` vs. baseline pool (`pass`, `random`, `starter`), 500–1000 episodes via local `env.run()`.
   3. Log results with the full breakdown from §8.5 below — not just win/lose/tie.
   4. Analyze failure modes manually: what's actually costing us games or money (idle units, weed losses, animal escapes, bad sell timing, under-expansion)? This step is manual judgment, not automated.
   5. Fix the structural/logic bug behind the dominant failure mode. Re-run steps 2–4 until logic is solid — **do not tune numeric weights on top of broken logic**.
   6. Once logic is solid, run automated tuning (`optuna` random/Bayesian search) over `config.py`'s numeric weights (§8.6) against the same baseline pool.
   7. Re-run self-play with tuned config, compare `v_N` (tuned) vs. `v_N-1` (previous best) head-to-head directly, not just both vs. baselines.
   8. Promote to new best if it wins the head-to-head and doesn't regress vs. baseline pool. Repeat from step 1.
5. **Metric to optimize**: start with simple win-rate vs. the baseline pool (`pass`/`random`/`starter`). Once consistently beating `starter`, consider a weighted metric — wins vs. stronger opponents (later-iteration `mirror_agent`) count more than wins vs. `random`/`pass` — to avoid overfitting to weak opponents. Track average win margin as a secondary/robustness metric only; the actual leaderboard is win/lose/tie-based continuous rating (§1.1), so margin is diagnostic, not the optimization target.
6. **Tunable parameter separation** — split `config.py` into two groups:
   - **Structural/rule parameters** (crop priority order, expansion trigger conditions, hire trigger conditions): set manually based on the economic reasoning in §5, refine via manual iteration (step 4–5 above), not automated search — the space is small and better navigated by reasoning than blind search.
   - **Numeric weights** (task-scoring weights in §6, `hold_floor_pct` per resource, phase day-boundaries): these are the actual target of automated `optuna` search in step 6 above — hard to hand-tune, well-suited to search.

---

### 8.5 Required Per-Episode Logging (instrumentation, not optional)

Every self-play episode logged by `run_selfplay.py` must capture at minimum:
- Final money, both players; win/lose/tie outcome.
- Revenue breakdown per resource (which crops/animals were actually profitable vs. net-negative after seed/fertilizer/feed cost).
- Count of weed events (plants lost to unwatering) and animal escape events (proxy for watering/feeding scheduling efficiency).
- Idle-unit-turns: count of turns where a unit returned `PASS` while at least one actionable task existed (proxy for UnitScheduler inefficiency — should trend toward 0 as FarmOps matures).
- Realized average sell price vs. base price per resource (proxy for market-timing quality — a value persistently well below base signals we're dumping into crashes rather than timing sells).

Without this breakdown, failure-mode analysis in the iteration loop (§8.4 step 4) is guesswork. `replay_analyze.py` should also be able to ingest real leaderboard replay/log data pulled via `kaggle competitions replay <EPISODE_ID>` and `kaggle competitions logs <EPISODE_ID> <player_index>` (§1.1), so the same analysis can run against actual opponents once we're submitting, not just local self-play.

### 8.6 Config Structure

`config.py` is split into two clearly labeled sections reflecting §8.4 step 6:
```python
# --- STRUCTURAL (manual, reasoning-driven — do not feed to optuna) ---
CROP_PRIORITY_ORDER = [...]
EXPANSION_UTILIZATION_THRESHOLD = 0.70
HIRE_TRIGGER_LOGIC = {...}

# --- NUMERIC WEIGHTS (automated search target) ---
TASK_SCORE_WEIGHTS = {"urgency": ..., "value": ..., "distance": ...}
HOLD_FLOOR_PCT = {"WHEAT": ..., "STRAWBERRY": ..., ...}
PHASE_DAY_BOUNDARIES = {"early_end": 4, "buildout_end": 14, "scale_end": 24}
```
`tune.py` only ever mutates the second block.

## 9. File Structure (proposed)

```
kaggriculture-agent/
├── kaggriculture/
│   ├── __init__.py
│   ├── state.py               # StateParser + typed GameState/Farm/Tile dataclasses
│   ├── market_model.py        # price function math, sell-timing helpers
│   ├── strategy.py            # StrategyAgent, DayPlan, phase logic
│   ├── farm_ops.py            # FarmOpsOrchestrator, UnitScheduler, task scoring
│   ├── config.py              # tunable thresholds/weights (single source of truth, §8.6)
│   └── agent.py                # thin wiring: agent(obs) entrypoint, imports the above
├── build_submission.py        # merges kaggriculture/*.py into a single main.py for submission
├── tools/
│   ├── run_selfplay.py        # batch episode runner + logging (§8.5), uses "pass"/"random"/"starter"
│   ├── tune.py                 # optuna/random-search harness over config.py's numeric-weights block
│   └── replay_analyze.py      # post-hoc analysis: local self-play logs AND real leaderboard replay/logs
├── tests/
│   ├── test_state.py
│   ├── test_market_model.py
│   ├── test_farm_ops.py
│   └── fixtures/               # synthetic obs fixtures for unit tests
├── README.md                  # (official Kaggriculture rules — already provided)
├── AGENTS.md                  # (official getting-started guide — already provided)
├── main.py                     # GENERATED — output of build_submission.py, this is what gets submitted
└── NOTES.md                   # running log of iteration results / findings (also doubles as
                                # methodology documentation if a prize is ever won — §1.1)
```

**Confirmed submission format (§1.1):** the actual submission needs `main.py` at the root containing the `agent` function — either directly, or via `kaggle competitions submit -f main.py`, or bundled as `tar.gz` with `main.py` at the root for multi-file setups. Since our package is naturally multi-module (`kaggriculture/`), use a **build step** (`build_submission.py`) that concatenates the package's modules into a single flat `main.py` — this mirrors the exact approach seen in the reference notebook (§1.1 provenance note): it strips internal cross-module imports, de-duplicates external imports, concatenates module bodies in dependency order, writes one `main.py`, then sanity-checks it with `compile()` before treating it as ready to submit. This keeps the actual development code modular and testable while still producing a single-file (or `tar.gz`) submission Kaggle accepts. Verify allowed external packages (beyond stdlib) in the competition's environment before depending on anything like `numpy` inside `kaggriculture/` itself — `optuna` and other tuning-only dependencies stay confined to `tools/` and never get bundled into `main.py`.

---

## 10. Implementation Milestones

1. **M0 — Skeleton & baselines**: `state.py` parsing + `baselines.py` (random, greedy_v0) + `run_selfplay.py` working end-to-end against the Quick Start example. Goal: confirm environment integration works, get first win-rate numbers.
2. **M1 — FarmOps v1**: UnitScheduler with basic urgency-only scoring (water/feed/harvest priority, no fertilize/expansion logic yet), single-crop-type focus (wheat only) to validate movement/action mechanics are correct.
3. **M2 — Strategy v1**: phase-based crop portfolio + basic sell-on-harvest policy, multi-crop support, land expansion heuristic.
4. **M3 — Market timing**: MarketModel price-function implementation + hold/dump thresholds + spread-sell logic.
5. **M4 — Animals**: coop/pasture build, feed/care/collect_fertilizer logic integrated into UnitScheduler and StrategyAgent portfolio scoring.
6. **M5 — Tuning loop**: `tune.py` running optimization over `config.py` against baseline pool, track win-rate improvement over iterations.
7. **M6 — Hardening**: edge cases (shed overflow, locked-tile no-ops, endgame liquidation, opponent-aware price reasoning), profiling for per-turn runtime budget, final packaging into single-file `agent.py`.

---

## 11. Open Questions to Resolve Before/During Build

**Resolved by official Rules + AGENTS.md + Overview page (§1.1) — no longer open:**
- ~~No ingress/egress at evaluation~~ — confirmed hard rule, no network calls of any kind in the submitted agent.
- ~~Submission format~~ — confirmed `main.py` at root, single-file or `tar.gz` bundle, max 100 MiB.
- ~~Private vs. public leaderboard structure~~ — confirmed no private leaderboard for simulation competitions; continuous episode-based skill rating instead, final leaderboard via Bradley-Terry tournament.
- ~~Submission compute/resource budget~~ — confirmed 8 GiB HDD, 6.5 GiB RAM, 1.6 vCPU. No explicit per-turn time limit given, but the small vCPU budget confirms the greedy/heuristic architecture choice over anything search-heavy or ML-inference-heavy.
- ~~Full competition timeline~~ — confirmed: Final Submission Deadline **Sept 30, 2026**, evaluation window runs to ~Oct 15, 2026.
- ~~Rating mechanics~~ — confirmed Elo-like skill rating, win/lose/tie only (coin margin does not affect rating change), final leaderboard via Bradley-Terry tournament over the full episode history.

**Still open — resolve during build (lower priority, none block starting M0):**
- Whether `kaggle_environments` persists the same agent process/object across all 720 turns of one episode (assumed yes based on the Quick Start / notebook examples using a plain stateful-capable function — confirm early via a local `env.run()` test with a module-level counter, since §7 state-handling design depends on it).
- Allowed third-party packages in the final submission environment beyond stdlib (affects whether e.g. `numpy` can be imported inside `kaggriculture/` itself vs. staying confined to `tools/`-only offline tooling per §9). The reference notebook's own submission (961 lines, 5 imports, no ML framework imports visible in the build log) suggests a lean stdlib-heavy approach is the safe default — check the Docker Image referenced on the Overview page's FAQ for the exact preinstalled package list if this becomes a blocker.

---

## 12. Notes for Claude Code

- Start with M0–M1 exactly as scoped — do not attempt full strategy/market layers before movement and basic actions are verified correct against the real environment (`kaggle_environments`import and `env.run(...)`).
- Keep `config.py` as the single place for every magic number/threshold — this is what the tuning harness (M5) will search over.
- Prefer dataclasses + type hints throughout `kaggriculture/` for clarity and testability; keep `agent.py` itself thin (just wiring + the bundling needed for submission).
- Write unit tests against synthetic fixtures (not live env) for `state.py` and `market_model.py` first — these are pure functions and easiest to get right in isolation.
- Reference the attached official `README.md` for exact field names/schema and price-function math, and `AGENTS.md` for the exact observation/action shapes, built-in baseline names (`"pass"`/`"random"`/`"starter"`), CLI workflow, and confirmed submission format — do not guess observation/action field names or re-derive rules already stated in either document.
- Build `build_submission.py` early (even before the full agent logic exists) using the trivial Quick Start agent as a smoke test — validates the module-merge/bundling approach (§9) works end-to-end before it's load-bearing for a real submission deadline.
- Use the confirmed built-in `"starter"` baseline (not a hand-rolled greedy agent) as the primary early benchmark opponent per §8.4 — don't spend time building a custom baseline the competition already provides.
- **Testing cadence**: all iteration (M0–M5) should run via local `env.run()` in a Kaggle Notebook or on-device — zero submission-quota cost, instant feedback. Only touch `kaggle competitions submit` once a version already beats `"starter"` consistently in local self-play; submissions are scarce (5/day, only 2 active at a time — §1.1) and should mark genuine leaderboard progress, not routine debugging.
