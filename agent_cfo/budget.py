"""Public surface: wrap an agent run in a budget.

    with Budget(2.00) as budget:
        planned = budget.plan(model_id="gpt-4o", input_tokens=800, max_tokens=1024)
        if planned.decision.action is Action.DENY:
            ...  # stop the run
        result = provider.call(planned.decision.model_id, messages, max_tokens=1024)
        budget.record(planned, actual_cost=result.cost)

`plan` reserves before any call happens; `record` settles the reservation
against the real cost once `usage` comes back; `abandon` releases it if the
call never ran at all (error, skip, retry). See CLAUDE.md § The core loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

from agent_cfo.ledger import Ledger, Reservation
from agent_cfo.policy import Action, Decision, decide
from agent_cfo.pricing import DOWNGRADE_LADDER


@dataclass
class PlannedCall:
    """A policy decision paired with its live reservation, ready to record against."""

    decision: Decision
    reservation: Reservation | None


class BudgetExhaustedError(Exception):
    """Raised when `plan_or_raise` can't afford any model on the ladder."""


class Budget:
    """Tracks one agent run's spend against a dollar ceiling.

    Not a provider client — it decides what's affordable and records what
    was actually spent. Callers own the actual model/tool call.
    """

    def __init__(self, amount: float, *, ladder: tuple[str, ...] = DOWNGRADE_LADDER):
        self.ledger = Ledger(amount)
        self.ladder = ladder

    @property
    def remaining(self) -> float:
        return self.ledger.remaining

    def plan(
        self,
        *,
        model_id: str,
        input_tokens: int,
        max_tokens: int,
    ) -> PlannedCall:
        """Decide which model (if any) to call, and reserve its estimated cost.

        The reservation is held until `record` settles it or `abandon`
        releases it — never call the model for a plan you haven't reserved.
        """
        decision = decide(
            requested_model=model_id,
            remaining_budget=self.remaining,
            input_tokens=input_tokens,
            max_tokens=max_tokens,
            ladder=self.ladder,
        )
        if decision.action is Action.DENY:
            return PlannedCall(decision, None)
        reservation = self.ledger.reserve_or_raise(decision.model_id, decision.estimated_cost)
        return PlannedCall(decision, reservation)

    def plan_or_raise(
        self,
        *,
        model_id: str,
        input_tokens: int,
        max_tokens: int,
    ) -> PlannedCall:
        planned = self.plan(model_id=model_id, input_tokens=input_tokens, max_tokens=max_tokens)
        if planned.decision.action is Action.DENY:
            raise BudgetExhaustedError(planned.decision.reason)
        return planned

    def record(self, planned: PlannedCall, *, actual_cost: float) -> None:
        """Settle a planned call's reservation with its real cost from `usage`."""
        if planned.reservation is None:
            raise ValueError("cannot record a denied plan — nothing was reserved")
        self.ledger.settle(planned.reservation.id, actual_cost)

    def abandon(self, planned: PlannedCall) -> None:
        """Release a reservation for a call that never ran (error, skipped, etc.)."""
        if planned.reservation is None:
            return
        self.ledger.release(planned.reservation.id)

    def __enter__(self) -> Budget:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None
