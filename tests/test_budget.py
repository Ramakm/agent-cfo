import pytest

from agent_cfo.budget import Budget, BudgetExhaustedError, PlannedCall
from agent_cfo.policy import Action, Decision


def test_plan_reserves_against_the_ledger():
    budget = Budget(2.00)
    planned = budget.plan(model_id="gpt-4o", input_tokens=1000, max_tokens=1000)
    assert planned.decision.action is Action.ALLOW
    assert planned.reservation is not None
    assert budget.remaining < 2.00


def test_record_settles_with_actual_usage():
    budget = Budget(2.00)
    planned = budget.plan(model_id="gemini-2.0-flash", input_tokens=1000, max_tokens=1000)
    remaining_after_reserve = budget.remaining
    budget.record(planned, actual_cost=0.0001)
    # settling with a smaller real cost than the reservation frees budget back up
    assert budget.remaining >= remaining_after_reserve


def test_abandon_releases_reservation_on_call_failure():
    """A tool/model call that errors out must not permanently burn its
    reservation — the budget has to stay usable for a retry."""
    budget = Budget(0.10)
    planned = budget.plan(model_id="gpt-4o", input_tokens=1000, max_tokens=100)
    reserved_amount = planned.reservation.amount
    remaining_while_reserved = budget.remaining
    budget.abandon(planned)
    assert budget.remaining == pytest.approx(remaining_while_reserved + reserved_amount)
    assert budget.remaining == pytest.approx(0.10)


def test_plan_downgrades_to_free_floor_when_budget_is_zero():
    budget = Budget(0.0)
    planned = budget.plan(model_id="gpt-4o", input_tokens=100, max_tokens=100)
    assert planned.decision.action == Action.DOWNGRADE
    assert planned.decision.model_id == "llama-3"
    assert planned.reservation is not None
    assert budget.remaining == pytest.approx(0.0)


def test_plan_or_raise_raises_budget_exhausted_error():
    budget = Budget(1.00, ladder=("gpt-4o",))  # no free fallback on this ladder
    with pytest.raises(BudgetExhaustedError):
        budget.plan_or_raise(model_id="gpt-4o", input_tokens=10_000_000, max_tokens=10_000_000)


def test_plan_or_raise_returns_normally_when_affordable():
    budget = Budget(2.00)
    planned = budget.plan_or_raise(model_id="llama-3", input_tokens=10, max_tokens=10)
    assert planned.decision.action is Action.ALLOW


def test_context_manager_usable():
    with Budget(1.00) as budget:
        planned = budget.plan(model_id="llama-3", input_tokens=10, max_tokens=10)
        budget.record(planned, actual_cost=0.0)
    assert budget.remaining == pytest.approx(1.00)


def test_record_without_reservation_raises():
    budget = Budget(1.00)
    denied = PlannedCall(Decision(Action.DENY, None, None, "no budget"), None)
    with pytest.raises(ValueError):
        budget.record(denied, actual_cost=0.0)


def test_abandon_on_denied_plan_is_a_no_op():
    budget = Budget(1.00)
    denied = PlannedCall(Decision(Action.DENY, None, None, "no budget"), None)
    budget.abandon(denied)  # must not raise
    assert budget.remaining == pytest.approx(1.00)


def test_full_reserve_settle_cycle_across_multiple_calls():
    """End-to-end: several planned calls against one budget, some settled
    under estimate, one abandoned — remaining must reconcile exactly."""
    budget = Budget(1.00)

    call1 = budget.plan(model_id="gemini-2.0-flash", input_tokens=10_000, max_tokens=1_000)
    budget.record(call1, actual_cost=call1.decision.estimated_cost / 2)

    call2 = budget.plan(model_id="llama-3", input_tokens=10_000, max_tokens=1_000)
    budget.record(call2, actual_cost=0.0)

    call3 = budget.plan(model_id="gpt-4o", input_tokens=10_000, max_tokens=1_000)
    budget.abandon(call3)  # simulate the call failing before it ran

    expected_spent = call1.decision.estimated_cost / 2
    assert budget.remaining == pytest.approx(1.00 - expected_spent)
