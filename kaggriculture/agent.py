"""Thin wiring: agent(obs) entrypoint (SDD §3, §7).

Keeps process-local memory (module-level) so StrategyAgent's DayPlan isn't
recomputed every turn — only at the start of each day, per §3.2's
re-evaluation cadence.
"""

from .state import StateParser
from .strategy import StrategyAgent
from .farm_ops import FarmOpsOrchestrator

_strategy_agent = StrategyAgent()
_farm_ops = FarmOpsOrchestrator()
_memory = {"day_plan": None, "plan_day": None}


def agent(obs):
    state = StateParser.parse(obs)

    if _memory["plan_day"] != state.day:
        _memory["day_plan"] = _strategy_agent.plan(state)
        _memory["plan_day"] = state.day

    return _farm_ops.run(state, _memory["day_plan"])
