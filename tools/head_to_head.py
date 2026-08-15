"""True head-to-head: two independently-configured copies of the
kaggriculture agent playing directly against each other (SDD §8.4 step 7-8:
"compare v_N (tuned) vs v_N-1 (previous best) head-to-head directly, not
just both vs baselines" — also doubles as the "mirror_agent" self-play
opponent referenced in §8.4.3, and stands in for the "varied
heuristic/adaptive opponents" goal in §2 since the competition only ships
three static baselines).

Why this needs its own loader: kaggriculture.config is a plain module-level
singleton — mutating its dicts in place (as tools/tune.py does) is how
every other tool in this repo applies config changes, but it means only ONE
config state can be live in a process at a time. To run two DIFFERENT
configs against each other simultaneously, each side needs its own
independent copy of the whole package (including its own config module
object), not just the same module with different values. importlib lets us
load the same source files under two different top-level names, giving each
its own independent globals.

Usage:
    python tools/head_to_head.py --episodes 10 --steps 720
    (runs the current kaggriculture/ config against itself as a sanity
    check — pass --config-b-overrides to diff two configs, e.g. an earlier
    committed version's numeric weights vs the current ones)
"""

import argparse
import importlib
import importlib.util
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggle_environments import make

PKG_DIR = pathlib.Path(__file__).parent.parent / "kaggriculture"


def load_package_copy(alias: str):
    """Loads kaggriculture/ under a second top-level module name so it gets
    its own independent config/state/strategy/farm_ops/agent module objects,
    decoupled from whatever's already imported as `kaggriculture`."""
    spec = importlib.util.spec_from_file_location(alias, PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    for submodule in ("config", "state", "market_model", "strategy", "farm_ops", "agent"):
        importlib.import_module(f"{alias}.{submodule}")
    return module


def build_agent(alias: str, config_overrides: dict = None):
    module = load_package_copy(alias)
    if config_overrides:
        config = importlib.import_module(f"{alias}.config")
        for key, updates in config_overrides.items():
            getattr(config, key).update(updates)

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


def run_head_to_head(episodes: int, steps: int, overrides_a: dict = None, overrides_b: dict = None):
    wins_a = 0.0
    margins = []
    for i in range(episodes):
        agent_a = build_agent(f"kaggriculture_a_{i}", overrides_a)
        agent_b = build_agent(f"kaggriculture_b_{i}", overrides_b)
        env = make("kaggriculture", configuration={"episodeSteps": steps})
        env.run([agent_a, agent_b])
        final = env.steps[-1]
        r0, r1 = final[0].reward, final[1].reward
        if r0 is None or r1 is None:
            continue
        margins.append(r0 - r1)
        if r0 > r1:
            wins_a += 1.0
        elif r0 == r1:
            wins_a += 0.5

    counted = len(margins)
    return {
        "episodes": counted,
        "win_rate_a": wins_a / counted if counted else 0.0,
        "avg_margin_a": sum(margins) / counted if counted else 0.0,
        "margins": margins,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument(
        "--config-b-overrides", type=str, default=None,
        help='JSON dict of config overrides for side B, e.g. \'{"TASK_SCORE_WEIGHTS": {"urgency": 100.0}}\' to compare current config (side A) against an earlier hand-set version (side B).',
    )
    args = parser.parse_args()

    overrides_b = json.loads(args.config_b_overrides) if args.config_b_overrides else None

    result = run_head_to_head(args.episodes, args.steps, overrides_a=None, overrides_b=overrides_b)
    print(f"Episodes: {result['episodes']}")
    print(f"A win rate: {result['win_rate_a']:.1%}")
    print(f"A avg margin: {result['avg_margin_a']:.0f}")
    print(f"Per-episode margins (A - B): {result['margins']}")


if __name__ == "__main__":
    main()
