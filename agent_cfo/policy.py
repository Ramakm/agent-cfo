"""Policy: given remaining budget and a proposed call, decide what to do.

Denial is a return value, not an exception in the hot path — the caller
needs to react (retry smaller, go local), not unwind a stack
(CLAUDE.md § Rules that matter here).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent_cfo.estimate import estimate_call_cost
from agent_cfo.pricing import DOWNGRADE_LADDER


class Action(Enum):
    ALLOW = "allow"  # run the requested model as-is
    DOWNGRADE = "downgrade"  # run a cheaper model instead
    DENY = "deny"  # nothing affordable — stop


@dataclass(frozen=True)
class Decision:
    action: Action
    model_id: str | None  # the model to actually call; None when denied
    estimated_cost: float | None
    reason: str


def decide(
    *,
    requested_model: str,
    remaining_budget: float,
    input_tokens: int,
    max_tokens: int,
    ladder: tuple[str, ...] = DOWNGRADE_LADDER,
) -> Decision:
    """Choose which model (if any) can afford this call.

    Walks the downgrade ladder starting at `requested_model`'s position
    (or the top, if the requested model isn't on the ladder) and returns
    the first one that fits inside `remaining_budget`. The free/local model
    at the bottom of the ladder always fits when `remaining_budget >= 0`
    and acts as the floor.
    """
    if requested_model in ladder:
        start = ladder.index(requested_model)
    else:
        # Not on the ladder (e.g. a one-off model) — price it standalone,
        # falling back to the ladder only if it doesn't fit.
        cost = estimate_call_cost(
            requested_model, input_tokens=input_tokens, max_tokens=max_tokens
        )
        if cost <= remaining_budget:
            return Decision(Action.ALLOW, requested_model, cost, "requested model affordable")
        start = 0

    for model_id in ladder[start:]:
        cost = estimate_call_cost(model_id, input_tokens=input_tokens, max_tokens=max_tokens)
        if cost <= remaining_budget:
            action = Action.ALLOW if model_id == requested_model else Action.DOWNGRADE
            reason = (
                "requested model affordable"
                if action is Action.ALLOW
                else f"downgraded from {requested_model!r}: ${cost:.4f} <= "
                f"${remaining_budget:.4f} remaining"
            )
            return Decision(action, model_id, cost, reason)

    return Decision(
        Action.DENY,
        None,
        None,
        f"no model in the downgrade ladder fits ${remaining_budget:.4f} remaining",
    )
