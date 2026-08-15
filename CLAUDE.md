# Kaggriculture Agent — Project Brief

Read this first. It's a dense pointer map, not a duplicate of the history —
drill into [NOTES.md](NOTES.md) only for the "why" behind a specific past
decision, don't re-read it end to end.

## What this is

A competitive agent for Kaggle's "Kaggriculture" simulation competition
(2-player farming sim, 30-day/720-turn season, shared dynamic market).
Design doc: [kaggriculture-agent-SDD.md](kaggriculture-agent-SDD.md). Game
rules + CLI onboarding: [README.md](README.md), [AGENTS.md](AGENTS.md).
Full iteration history (chronological, ~70KB, every bug/fix/finding with
evidence): [NOTES.md](NOTES.md).

**Not a git repo.** No branches/commits to check — `main.py` and
`kaggriculture/*.py` ARE the current state, nothing is staged/uncommitted.

## Architecture

```
kaggriculture/
  config.py     STRUCTURAL (hand-reasoned) + NUMERIC WEIGHTS (tune.py target) sections
  state.py      StateParser: raw obs dict -> GameState/PlantTile/StructureTile/WeedTile
  strategy.py   StrategyAgent.plan() -> DayPlan (once/day, cached across turns)
  farm_ops.py   FarmOpsOrchestrator.run() -> per-turn action dict (UnitScheduler + MarketExecutor)
  agent.py      thin glue: caches DayPlan per day, calls FarmOpsOrchestrator every turn
build_submission.py   bundles kaggriculture/*.py -> single main.py (flattens imports)
main.py               the actual Kaggle submission artifact — ALWAYS regenerate via
                       build_submission.py after editing kaggriculture/, never hand-edit
tools/
  run_selfplay.py       batch vs pass/random/starter, --detailed for revenue/weed/idle breakdown
  scripted_opponents.py melon_focus_agent / melon_animal_agent — standalone (no shared code),
                         mimic real winning patterns; the ONLY local benchmark with any
                         remaining discriminating power (starter/random/pass saturated at 100%)
  tournament.py          round-robin among config variants (importlib dual-loading trick to
                         get independently-configured live copies of kaggriculture in one process)
  tune.py                optuna search over the NUMERIC WEIGHTS block only; --opponent accepts
                         starter/random/pass OR melon/melon_animal (scripted, extended in v14)
  head_to_head.py        two configs face each other directly (same dual-loading trick)
  replay_analyze.py      ingests a replay.json (local OR `kaggle competitions replay` download);
                         --player {0,1} — MUST match TeamNames, index is NOT always "us"
tests/          81 tests, pytest tests/ -q — run before every build_submission.py
replays/        downloaded real-match replays, organized v13_matches/ v14_matches/ etc.
```

**strategy.py's internal roles** (labeled via section comments, not separate
classes — see NOTES.md "v14" for why a heavier multi-agent split wasn't
worth it): economist (`_decide_crop_targets`/`_decide_land_expansion`/
`_decide_animal_targets`/`_decide_hires` — WHAT to grow/buy), speculator
(opponent/market trend signals + `_classify_regime` — risk-posture input,
never decides WHAT), orchestrator (`StrategyAgent.plan()` — combines both +
picks the regime preset). farm_ops.py is the agriculture role (tactical,
per-turn unit scheduling) and never feeds back into strategy.py.

## Current status (as of 2026-08-12, v15)

- **Submitted**: v15 (id `55461061`, pending validation) — v14 (`55459062`,
  COMPLETE) and v13 (`55434988`, COMPLETE) are the recent history.
- **Real-match win rate is the metric that matters, and it's currently
  weak**: v13 measured 41% (7W-10L), v14 38% (3W-5L) across real Kaggle
  opponents — computed directly from downloaded replays, matched by each
  replay's own `TeamNames` (see the replay_analyze usage gotcha below).
  Every local benchmark (starter/random/pass/both scripted opponents) has
  been at 100% since v13 — **local self-play validation is necessary but
  NOT sufficient**; it has repeatedly missed real failure modes across
  this whole project (v5, v8, v10, v14 all independently rediscovered
  this). Always pull real replays before trusting a "locally proven"
  version.
- **Scoring is Elo-like, margin-blind**: confirmed in the SDD — "coin
  margin does not affect rating change at all — only win/lose/tie
  matters." A new submission's rating needs games + time to converge, not
  just another local pass; low current leaderboard rank partly reflects
  low games-played count, not certain skill gap.
- **Open strategic fork, NOT yet actioned** (NOTES.md "v15"): real
  opponents who beat us decisively run much wider, animal/fertilizer-heavy
  portfolios that out-earn us 5x+ even when our own execution is clean (no
  weeds, no cash crashes) — this is a scale/ambition gap, not a bug.
  `ANIMAL_CASH_FRACTION`/`ANIMAL_CASH_RESERVE` have been deliberately
  conservative since v7/v8's death-spiral incidents (2/10 seeds lost to a
  do-nothing baseline at more aggressive settings). Raising animal
  investment could close real ground but needs the same seed-sweep
  death-spiral testing v7/v8 required — flagged for an explicit decision,
  not changed unilaterally.

## Standing policies (do not relitigate without the user revisiting them)

- **Never submit to Kaggle without fresh, explicit in-chat permission each
  time.** 5 submissions/day, only 2 stay active for real matchmaking.
  Permission is per-submission, not standing.
- **Never delete `tournament_results*.txt`** — kept as local historical
  data on purpose.
- Full workflow before ANY submission: `python -m pytest tests/ -q` (all
  passing) → `python build_submission.py` → local validation (scripted
  opponents + baseline pool, seed sweeps for anything touching
  hiring/animals/expansion) → update NOTES.md with the finding → confirm
  with the user → `kaggle competitions submit -c kaggriculture -f main.py
  -m "..."`.

## Gotchas that have already cost real debugging time (don't re-discover these)

1. **Hands reset at day boundary.** Checking hand count at `hour==0`
   always shows the pre-rehire state — sample at `hour==23` for real
   end-of-day headcount.
2. **Replay player index is not always 0.** A real match's `TeamNames`
   list (`data['info']['TeamNames']`) tells you which index is "us" — it
   flips between episodes. `replay_analyze.py --player {0,1}` must match,
   or the whole analysis silently describes the wrong side.
3. **`replay_analyze.py`'s revenue-by-resource is inferred, not exact**
   (money-delta split across that turn's SELL orders by weight) — it
   assumes an order pattern similar to OUR OWN agent's. For a real
   opponent with a very different action distribution, the inferred
   total can be off by an order of magnitude from their actual final
   money. Cross-check any surprising revenue number against the raw tile
   composition (`obs['farms'][idx]['tiles']`) before trusting it.
4. **`kaggle_environments` breaks on a callable class instance as agent**
   (reflects the unbound `__call__(self, obs)`, passes 2 args). Expose
   plain top-level functions instead (see `scripted_opponents.py`).
5. **Local self-play/scripted-opponent 100% win rate does not predict
   real-match performance.** Don't treat a locally-saturated benchmark as
   "done" — pull real replays whenever local tuning runs out of signal
   (every trial hitting the same win_rate is the tell).
6. **`from .config import SOME_FLAG` binds a value at import time** —
   mutating `config.SOME_FLAG` later does NOT affect the name already
   bound inside `strategy.py`/`farm_ops.py`. To toggle a flag in a test,
   `monkeypatch.setattr(strategy_module, "SOME_FLAG", False)` on the
   MODULE that imported it, not on `config`. (Mutable dicts like
   `HOLD_FLOOR_PCT` don't have this problem — `tune.py` relies on that.)

## Quick commands

```bash
python -m pytest tests/ -q                                   # 81 tests, run first
python build_submission.py                                   # kaggriculture/*.py -> main.py
python tools/scripted_opponents.py --episodes 10 --opponent melon         # or melon_animal
python tools/run_selfplay.py --episodes 10 --opponent starter             # or random/pass
python tools/tune.py --trials 15 --episodes-per-trial 4 --opponent melon  # numeric re-tune
kaggle competitions submissions -c kaggriculture                          # check status/scores
kaggle competitions episodes <submission_id>                              # list real episodes
kaggle competitions replay <episode_id> -p replays/                       # download for analysis
python tools/replay_analyze.py replays/.../episode-X-replay.json --player {0,1}
```
