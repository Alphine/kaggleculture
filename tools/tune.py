"""Optuna tuning harness over config.py's NUMERIC WEIGHTS block only
(SDD §8.4 step 6, §8.6): TASK_SCORE_WEIGHTS, HOLD_FLOOR_PCT,
PHASE_DAY_BOUNDARIES. Structural parameters (CROP_PRIORITY_ORDER,
EXPANSION_UTILIZATION_THRESHOLD, HIRE_TRIGGER_LOGIC, PHASE_CROP_POOL,
PHASE_ANIMAL_CAP, ...) are reasoning-driven and are never touched here —
per the SDD, don't tune structural/logic parameters until the logic itself
is solid (§8.4 step 5); this harness only searches the numeric dials once
that's true.

Mutates kaggriculture.config's dicts IN PLACE (not reassigned) so every
already-imported reference (strategy.py, farm_ops.py) picks up each trial's
values without re-importing the package.

Usage:
    python tools/tune.py --trials 20 --episodes-per-trial 5 --opponent starter
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import optuna
from kaggle_environments import make

from kaggriculture import config
from kaggriculture.farm_ops import FarmOpsOrchestrator
from kaggriculture.state import StateParser
from kaggriculture.strategy import StrategyAgent
from scripted_opponents import melon_focus_agent, melon_animal_agent

# `starter`/`random`/`pass` are built-in kaggle_environments agent names
# (strings); the scripted opponents are plain callables (NOTES.md "v12") —
# env.run() accepts either. starter/random/pass saturate to 100% win rate
# well before any interesting weight combination (see evaluate()'s
# docstring), so they can only ever break ties on margin; the scripted
# opponents are the only LOCAL benchmark that can actually discriminate
# between candidate configs on win_rate itself.
SCRIPTED_OPPONENTS = {"melon": melon_focus_agent, "melon_animal": melon_animal_agent}


def make_agent():
    strat = StrategyAgent()
    ops = FarmOpsOrchestrator()
    mem = {"plan": None, "day": None}

    def agent(obs):
        state = StateParser.parse(obs)
        if mem["day"] != state.day:
            mem["plan"] = strat.plan(state)
            mem["day"] = state.day
        return ops.run(state, mem["plan"])

    return agent


def evaluate(episodes: int, opponent, steps: int) -> float:
    """win_rate + tiny margin tie-breaker, in that priority order.

    Win/lose/tie is the actual leaderboard signal (§1.1: coin margin does
    not move rating at all), so win_rate must dominate the objective. But
    once an agent already beats a baseline 100% of the time (true for
    `starter` well before any interesting weight combination), win_rate
    alone can't tell configs apart — every trial ties at 1.0 and optuna
    picks among them arbitrarily. Per SDD §8.4.5, fall back to average
    margin as a secondary/robustness metric only: scaled tiny enough
    (1e-6) that it can never flip a win_rate comparison, only break ties
    within the same win_rate.
    """
    wins = 0.0
    margin_total = 0.0
    counted = 0
    for _ in range(episodes):
        agent = make_agent()
        env = make("kaggriculture", configuration={"episodeSteps": steps})
        env.run([agent, opponent])
        final = env.steps[-1]
        r0, r1 = final[0].reward, final[1].reward
        if r0 is None or r1 is None:
            continue
        counted += 1
        margin_total += r0 - r1
        if r0 > r1:
            wins += 1.0
        elif r0 == r1:
            wins += 0.5
    if not counted:
        return 0.0
    win_rate = wins / counted
    avg_margin = margin_total / counted
    return win_rate + 1e-6 * avg_margin


def objective_factory(episodes: int, opponent, steps: int):
    def objective(trial: optuna.Trial) -> float:
        config.TASK_SCORE_WEIGHTS["urgency"] = trial.suggest_float("w_urgency", 10.0, 300.0)
        config.TASK_SCORE_WEIGHTS["value"] = trial.suggest_float("w_value", 0.1, 5.0)
        config.TASK_SCORE_WEIGHTS["distance"] = trial.suggest_float("w_distance", 0.1, 10.0)

        for resource in list(config.HOLD_FLOOR_PCT):
            config.HOLD_FLOOR_PCT[resource] = trial.suggest_float(f"hold_{resource}", 0.0, 0.6)

        early_end = trial.suggest_int("early_end", 2, 8)
        buildout_end = trial.suggest_int("buildout_end", early_end + 3, 20)
        scale_end = trial.suggest_int("scale_end", buildout_end + 2, 27)
        config.PHASE_DAY_BOUNDARIES["early_end"] = early_end
        config.PHASE_DAY_BOUNDARIES["buildout_end"] = buildout_end
        config.PHASE_DAY_BOUNDARIES["scale_end"] = scale_end

        return evaluate(episodes, opponent, steps)

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--episodes-per-trial", type=int, default=5)
    parser.add_argument("--opponent", type=str, default="starter",
                         choices=["pass", "random", "starter", "melon", "melon_animal"])
    parser.add_argument("--steps", type=int, default=720)
    args = parser.parse_args()
    opponent = SCRIPTED_OPPONENTS.get(args.opponent, args.opponent)

    baseline = {
        "TASK_SCORE_WEIGHTS": dict(config.TASK_SCORE_WEIGHTS),
        "HOLD_FLOOR_PCT": dict(config.HOLD_FLOOR_PCT),
        "PHASE_DAY_BOUNDARIES": dict(config.PHASE_DAY_BOUNDARIES),
    }
    print("Baseline config:", baseline)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective_factory(args.episodes_per_trial, opponent, args.steps),
        n_trials=args.trials,
    )

    print(f"\nBest win rate: {study.best_value:.3f}")
    print("Best params:", study.best_params)
    print(
        "\nTo apply: copy the winning w_urgency/w_value/w_distance into "
        "TASK_SCORE_WEIGHTS, hold_<RESOURCE> into HOLD_FLOOR_PCT, and "
        "early_end/buildout_end/scale_end into PHASE_DAY_BOUNDARIES in "
        "kaggriculture/config.py, then re-run tools/run_selfplay.py to "
        "confirm the tuned config doesn't regress vs the baseline pool "
        "(SDD §8.4 step 7-8)."
    )


if __name__ == "__main__":
    main()
