"""Batch episode runner + logging (SDD §8, §8.5).

Usage:
    python tools/run_selfplay.py --episodes 100 --opponent starter
    python tools/run_selfplay.py --episodes 20 --opponent starter --detailed
"""

import argparse
import sys
import pathlib
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggle_environments import make

from kaggriculture.agent import agent
from replay_analyze import analyze_episode


def run_batch(opponent: str, episodes: int, episode_steps: int = 720, detailed: bool = False):
    results = {"win": 0, "lose": 0, "tie": 0, "error": 0}
    money_log = []
    aggregate = {
        "idle_unit_turns": 0,
        "weed_events": 0,
        "escape_events": 0,
        "revenue_by_resource": defaultdict(lambda: {"qty_sold": 0.0, "total_revenue": 0.0}),
    }

    for ep in range(episodes):
        env = make("kaggriculture", configuration={"episodeSteps": episode_steps})
        env.run([agent, opponent])
        final = env.steps[-1]
        statuses = [s.status for s in final]
        if "ERROR" in statuses or "INVALID" in statuses:
            results["error"] += 1
            continue

        rewards = [s.reward for s in final]
        if rewards[0] is None or rewards[1] is None:
            results["error"] += 1
            continue
        if rewards[0] > rewards[1]:
            results["win"] += 1
        elif rewards[0] < rewards[1]:
            results["lose"] += 1
        else:
            results["tie"] += 1
        money_log.append(rewards)

        if detailed:
            metrics = analyze_episode(env.toJSON(), player=0)
            aggregate["idle_unit_turns"] += metrics["idle_unit_turns"]
            aggregate["weed_events"] += metrics["weed_events"]
            aggregate["escape_events"] += metrics["escape_events"]
            for resource, info in metrics["revenue_by_resource"].items():
                agg = aggregate["revenue_by_resource"][resource]
                agg["qty_sold"] += info["qty_sold"]
                agg["total_revenue"] += info["total_revenue"]

    return results, money_log, aggregate


def print_aggregate(aggregate: dict, episodes: int):
    print(f"\n--- Detailed breakdown (SDD §8.5, averaged over {episodes} episodes) ---")
    print(f"Idle-unit-turns/episode: {aggregate['idle_unit_turns'] / episodes:.1f} (should trend toward 0)")
    print(f"Weed events/episode (unwatering failures): {aggregate['weed_events'] / episodes:.2f}")
    print(f"Animal escape events/episode: {aggregate['escape_events'] / episodes:.2f}")
    print("Revenue by resource (approximate — see tools/replay_analyze.py docstring):")
    from kaggriculture.config import PRODUCT_BASE_PRICE

    for resource, info in sorted(aggregate["revenue_by_resource"].items(), key=lambda kv: -kv[1]["total_revenue"]):
        if info["qty_sold"] <= 0:
            continue
        avg_price = info["total_revenue"] / info["qty_sold"]
        base = PRODUCT_BASE_PRICE.get(resource, 0)
        pct = 100 * avg_price / base if base else 0
        print(
            f"  {resource:12s} revenue/ep=${info['total_revenue']/episodes:8.0f}  "
            f"avg_price=${avg_price:6.1f} ({pct:.0f}% of base ${base})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--opponent", type=str, default="starter", choices=["pass", "random", "starter"])
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument(
        "--detailed", action="store_true",
        help="Also compute the §8.5 required breakdown (revenue/resource, weed/escape events, idle-unit-turns) via replay_analyze.py. Slower — walks every turn's action log per episode.",
    )
    args = parser.parse_args()

    results, money_log, aggregate = run_batch(args.opponent, args.episodes, args.steps, args.detailed)

    total = sum(results.values())
    print(f"Opponent: {args.opponent}  Episodes: {total}")
    print(f"  Win:  {results['win']} ({results['win']/total:.1%})")
    print(f"  Lose: {results['lose']} ({results['lose']/total:.1%})")
    print(f"  Tie:  {results['tie']} ({results['tie']/total:.1%})")
    print(f"  Error: {results['error']}")

    if args.detailed and results["win"] + results["lose"] + results["tie"] > 0:
        print_aggregate(aggregate, results["win"] + results["lose"] + results["tie"])


if __name__ == "__main__":
    main()
