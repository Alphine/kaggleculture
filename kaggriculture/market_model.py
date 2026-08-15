"""MarketModel: price-curve math, sell-timing helpers (SDD §3.4, M3).

Implements the README "Price Function":
    price(inv) = base + sign * amp * f(|inv - I0|)
    sign = +1 if inv < I0 (scarcity -> price up), -1 if inv > I0 (glut -> price down)
    amp  = target * base / f(T)   (derived, not stored)
    f    in {linear, sq, sqrt, log}   (log uses ln(1+x))
Floored at $1, rounded to the nearest dollar.
"""

import math

from .config import MARKET_PARAMS, MAX_SELL_SIMULATION_UNITS


def _shape(func_name: str, x: float) -> float:
    if func_name == "linear":
        return x
    if func_name == "sq":
        return x * x
    if func_name == "sqrt":
        return math.sqrt(x)
    if func_name == "log":
        return math.log1p(x)
    if func_name == "log10":
        return math.log10(1 + x)
    raise ValueError(f"unknown shape function: {func_name}")


def price_at(resource: str, inventory: int) -> int:
    """Price for the next unit sold/bought at the given market inventory level."""
    params = MARKET_PARAMS[resource]
    base, i0, t = params["base"], params["I0"], params["T"]
    diff = inventory - i0
    if diff == 0:
        return base

    if diff < 0:
        sign, func, target = 1, params["below_func"], params["below_target"]
    else:
        sign, func, target = -1, params["above_func"], params["above_target"]

    denom = _shape(func, t)
    amp = (target * base / denom) if denom else 0.0
    raw = base + sign * amp * _shape(func, abs(diff))
    return max(1, round(raw))


def current_price(resource: str, state) -> int:
    """Price from live obs if available, else derive it from inventory."""
    if resource in state.market_prices:
        return state.market_prices[resource]
    inv = state.market_inventory.get(resource, MARKET_PARAMS[resource]["I0"])
    return price_at(resource, inv)


def max_sell_before_floor(resource: str, inventory: int, floor_price: float, max_units: int = MAX_SELL_SIMULATION_UNITS) -> int:
    """How many units can be sold one-at-a-time (each sale nudges inventory up
    and price down, per README "Selling inventory to the market") before the
    per-unit price drops below `floor_price`. Caps the walk at max_units —
    plenty for any single-turn order given maxMarketOrdersPerTurn."""
    inv = inventory
    count = 0
    for _ in range(max_units):
        p = price_at(resource, inv)
        if p < floor_price:
            break
        count += 1
        inv += 1
    return count
