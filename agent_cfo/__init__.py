"""Budget enforcement for autonomous agents.

    from agent_cfo import Budget

    with Budget(2.00) as budget:
        planned = budget.plan(model_id="gpt-4o", input_tokens=800, max_tokens=1024)
        # ... call the model named in planned.decision.model_id, then:
        budget.record(planned, actual_cost=result.cost)

See CLAUDE.md for the full design.
"""

from agent_cfo.budget import Budget, BudgetExhaustedError, PlannedCall
from agent_cfo.ledger import InsufficientBudgetError, Ledger, Reservation
from agent_cfo.policy import Action, Decision, decide
from agent_cfo.pricing import (
    DOWNGRADE_LADDER,
    PRICING,
    ModelPricing,
    UnknownModelError,
    estimate_cost,
    get_pricing,
)

__all__ = [
    "Budget",
    "BudgetExhaustedError",
    "PlannedCall",
    "Ledger",
    "InsufficientBudgetError",
    "Reservation",
    "Action",
    "Decision",
    "decide",
    "PRICING",
    "DOWNGRADE_LADDER",
    "ModelPricing",
    "UnknownModelError",
    "estimate_cost",
    "get_pricing",
]
