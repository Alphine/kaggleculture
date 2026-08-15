"""Post-hoc episode analysis (SDD §8.5, §9).

Ingests either a local self-play episode (via `env.toJSON()`, same shape
`run_selfplay.py` can hand it directly) or a real leaderboard replay
downloaded via `kaggle competitions replay <EPISODE_ID>` — both are the same
kaggle_environments schema, so one code path covers both per §9's stated
scope.

Computes the §8.5 required breakdown: revenue per resource, weed events
(plants lost to unwatering), animal escape events, idle-unit-turns, and
realized average sell price vs base price per resource. Without this,
failure-mode analysis in the iteration loop (SDD §8.4 step 4) is guesswork.

Step-pairing note: in the kaggle_environments schema, `steps[t][player]
["action"]` is the action that player SUBMITTED at turn t, which causes the
transition to `steps[t+1][player]["observation"]` (verified empirically:
BUY_SEED/HIRE orders logged at step t only show up as a money change at
step t+1). All money-delta attribution below pairs on that basis.

Revenue attribution is a best-effort approximation, not exact accounting:
BUY_SEED/BUY_ANIMAL/BUY_LAND/HIRE costs are deterministic table lookups
(exact), BUY_PRODUCT uses the live market price as an estimate, and
whatever money delta remains after subtracting known buy costs is treated
as sell revenue, split across that turn's SELL orders weighted by
base_price x qty. Exact per-unit-turn attribution would require the game
engine's internal per-unit price walk, which isn't exposed to the agent.

Usage:
    python tools/replay_analyze.py path/to/replay.json --player 0
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from kaggriculture.config import ANIMAL_DATA, CROP_DATA, LAND_COSTS, PRODUCT_BASE_PRICE
from kaggriculture.strategy import fib_cost


def _known_buy_cost(market_orders: list, market_prices: dict, hires_today: int) -> float:
    """Deterministic $ cost of this turn's non-SELL market orders. Every
    order type except BUY_PRODUCT has a fixed table cost; BUY_PRODUCT uses
    the live per-unit price at the time as a best-effort stand-in for the
    engine's actual per-unit-walk cost."""
    cost = 0.0
    hires_so_far = hires_today
    for order in market_orders:
        op = order[0]
        if op == "BUY_SEED":
            _, crop, n = order
            cost += CROP_DATA[crop]["seed_cost"] * n
        elif op == "BUY_ANIMAL":
            _, animal, n = order
            cost += ANIMAL_DATA[animal]["cost"] * n
        elif op == "BUY_LAND":
            # Exact tier isn't recoverable from the action alone without
            # tracking unlocked_quadrants across turns; approximate with
            # the cheapest remaining tier (usually correct in practice
            # since land is bought in ascending-cost order).
            cost += min(LAND_COSTS.values())
        elif op == "HIRE":
            cost += fib_cost(hires_so_far)
            hires_so_far += 1
        elif op == "BUY_PRODUCT":
            _, item, n = order
            price = market_prices.get(item, PRODUCT_BASE_PRICE.get(item, 0))
            cost += price * n
    return cost


def analyze_episode(replay: dict, player: int = 0) -> dict:
    steps = replay["steps"]
    opponent = 1 - player

    revenue_by_resource = defaultdict(float)
    qty_sold_by_resource = defaultdict(float)
    idle_unit_turns = 0
    weed_events = 0
    escape_events = 0

    prev_day = None
    prev_tiles = None

    for t in range(len(steps) - 1):
        obs_now = steps[t][player]["observation"]
        obs_next = steps[t + 1][player]["observation"]
        action = steps[t][player].get("action") or {}
        farm_now = obs_now["farms"][player]
        money_now = farm_now["money"]
        money_next = obs_next["farms"][player]["money"]

        farmer_action = action.get("farmer") or ["PASS"]
        if farmer_action[0] == "PASS":
            idle_unit_turns += 1
        for hand_action in action.get("hands", []) or []:
            if hand_action and hand_action[0] == "PASS":
                idle_unit_turns += 1

        market_orders = action.get("market", []) or []
        sell_orders = [o for o in market_orders if o[0] == "SELL"]
        if market_orders:
            buy_cost = _known_buy_cost(market_orders, obs_now["market"]["prices"], farm_now.get("hires_today", 0))
            money_delta = money_next - money_now
            sell_revenue = money_delta + buy_cost
            total_weight = sum(PRODUCT_BASE_PRICE.get(o[1], 1) * o[2] for o in sell_orders)
            if sell_orders and total_weight > 0 and sell_revenue > 0:
                for _, resource, qty in sell_orders:
                    weight = PRODUCT_BASE_PRICE.get(resource, 1) * qty
                    share = sell_revenue * (weight / total_weight)
                    revenue_by_resource[resource] += share
                    qty_sold_by_resource[resource] += qty

        day = obs_now["day"]
        cur_tiles = farm_now["tiles"]
        if prev_day is not None and day != prev_day and prev_tiles is not None:
            weed_events += _count_new_weeds(prev_tiles, cur_tiles)
            escape_events += _count_escapes(prev_tiles, cur_tiles)
        prev_day = day
        prev_tiles = cur_tiles

    final_obs = steps[-1][player]["observation"]
    final_money = final_obs["farms"][player]["money"]
    final_opp_money = final_obs["farms"][opponent]["money"]

    revenue_breakdown = {}
    for resource, revenue in revenue_by_resource.items():
        qty = qty_sold_by_resource[resource]
        if qty:
            revenue_breakdown[resource] = {
                "avg_realized_price": round(revenue / qty, 2),
                "base_price": PRODUCT_BASE_PRICE.get(resource),
                "qty_sold": qty,
                "total_revenue": round(revenue, 2),
            }

    return {
        "final_money": final_money,
        "final_opponent_money": final_opp_money,
        "outcome": "win" if final_money > final_opp_money else ("lose" if final_money < final_opp_money else "tie"),
        "idle_unit_turns": idle_unit_turns,
        "weed_events": weed_events,
        "escape_events": escape_events,
        "revenue_by_resource": revenue_breakdown,
    }


def _count_new_weeds(prev_tiles: list, cur_tiles: list) -> int:
    """Only counts PLANT -> WEED transitions (a watering failure) — NOT
    None -> WEED, which is just the game's random weedSpawnChance on empty
    tiles and isn't something our scheduling caused."""
    count = 0
    for y in range(len(prev_tiles)):
        for x in range(len(prev_tiles[y])):
            prev_cell, cur_cell = prev_tiles[y][x], cur_tiles[y][x]
            if (
                isinstance(prev_cell, dict) and prev_cell.get("kind") == "PLANT"
                and isinstance(cur_cell, dict) and cur_cell.get("kind") == "WEED"
            ):
                count += 1
    return count


def _count_escapes(prev_tiles: list, cur_tiles: list) -> int:
    """An occupied structure whose animal turned to None between day
    boundaries — the only way that happens is a 2-consecutive-day feed
    miss (README); there's no action that otherwise removes a placed
    animal."""
    count = 0
    for y in range(len(prev_tiles)):
        for x in range(len(prev_tiles[y])):
            prev_cell, cur_cell = prev_tiles[y][x], cur_tiles[y][x]
            if (
                isinstance(prev_cell, dict) and prev_cell.get("kind") in ("COOP", "PASTURE")
                and prev_cell.get("animal") is not None
                and isinstance(cur_cell, dict) and cur_cell.get("kind") == prev_cell.get("kind")
                and cur_cell.get("animal") is None
            ):
                count += 1
    return count


def print_report(metrics: dict):
    print(f"Outcome: {metrics['outcome']}  (us: {metrics['final_money']}, opponent: {metrics['final_opponent_money']})")
    print(f"Idle-unit-turns: {metrics['idle_unit_turns']}")
    print(f"Weed events (unwatering failures): {metrics['weed_events']}")
    print(f"Animal escape events: {metrics['escape_events']}")
    print("Revenue by resource (approximate, see module docstring):")
    for resource, info in sorted(metrics["revenue_by_resource"].items(), key=lambda kv: -kv[1]["total_revenue"]):
        pct_of_base = 100 * info["avg_realized_price"] / info["base_price"] if info["base_price"] else 0
        print(
            f"  {resource:12s} qty={info['qty_sold']:6.0f}  "
            f"revenue=${info['total_revenue']:9.0f}  "
            f"avg_price=${info['avg_realized_price']:6.1f} "
            f"({pct_of_base:.0f}% of base ${info['base_price']})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_path", type=str, help="Path to a replay.json (local dump or `kaggle competitions replay` download)")
    parser.add_argument("--player", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    with open(args.replay_path, "r", encoding="utf-8") as f:
        replay = json.load(f)

    metrics = analyze_episode(replay, player=args.player)
    print_report(metrics)


if __name__ == "__main__":
    main()
