import pytest

from agent_cfo.ledger import InsufficientBudgetError, Ledger


def test_reserve_reduces_remaining():
    ledger = Ledger(2.00)
    r = ledger.reserve("gpt-4o", 0.50)
    assert r is not None
    assert ledger.remaining == pytest.approx(1.50)


def test_reserve_beyond_budget_returns_none():
    ledger = Ledger(1.00)
    assert ledger.reserve("gpt-4o", 1.50) is None
    assert ledger.remaining == pytest.approx(1.00)


def test_reserve_or_raise_raises_on_overdraft():
    ledger = Ledger(1.00)
    with pytest.raises(InsufficientBudgetError):
        ledger.reserve_or_raise("gpt-4o", 2.00)


def test_reserve_rejects_negative_amount():
    ledger = Ledger(1.00)
    with pytest.raises(ValueError):
        ledger.reserve("gpt-4o", -0.10)


def test_settle_moves_reservation_to_settled_total():
    ledger = Ledger(2.00)
    r = ledger.reserve("gpt-4o", 0.50)
    ledger.settle(r.id, 0.30)  # actual came in under estimate
    assert ledger.settled_total == pytest.approx(0.30)
    assert ledger.outstanding == pytest.approx(0.0)
    assert ledger.remaining == pytest.approx(1.70)


def test_settle_can_exceed_reservation():
    ledger = Ledger(2.00)
    r = ledger.reserve("gpt-4o", 0.50)
    ledger.settle(r.id, 0.75)  # actual came in over estimate
    assert ledger.remaining == pytest.approx(1.25)


def test_release_returns_reservation_to_pool_unspent():
    ledger = Ledger(1.00)
    r = ledger.reserve("gpt-4o", 0.40)
    ledger.release(r.id)
    assert ledger.remaining == pytest.approx(1.00)
    assert ledger.outstanding == pytest.approx(0.0)


def test_release_on_call_failure_frees_budget_for_retry():
    """The scenario CLAUDE.md's testing section calls out by name: a
    reservation must be released, not silently spent, when the call never
    happens — so a retry at a cheaper tier can still fit."""
    ledger = Ledger(1.00)
    r = ledger.reserve("gpt-4o", 0.90)
    assert ledger.reserve("gemini-2.0-flash", 0.20) is None  # not enough left
    ledger.release(r.id)
    r2 = ledger.reserve("gemini-2.0-flash", 0.20)  # now it fits
    assert r2 is not None


def test_cannot_settle_a_released_reservation():
    ledger = Ledger(1.00)
    r = ledger.reserve("gpt-4o", 0.40)
    ledger.release(r.id)
    with pytest.raises(ValueError):
        ledger.settle(r.id, 0.10)


def test_cannot_double_settle():
    ledger = Ledger(1.00)
    r = ledger.reserve("gpt-4o", 0.40)
    ledger.settle(r.id, 0.40)
    with pytest.raises(ValueError):
        ledger.settle(r.id, 0.10)


def test_cannot_release_a_settled_reservation():
    ledger = Ledger(1.00)
    r = ledger.reserve("gpt-4o", 0.40)
    ledger.settle(r.id, 0.40)
    with pytest.raises(ValueError):
        ledger.release(r.id)


def test_unknown_reservation_id_raises():
    ledger = Ledger(1.00)
    with pytest.raises(KeyError):
        ledger.settle(999, 0.10)
    with pytest.raises(KeyError):
        ledger.release(999)


def test_concurrent_reservations_share_one_budget():
    """Multiple outstanding reservations against one budget must all count
    against `remaining` simultaneously — not just the most recent one."""
    ledger = Ledger(1.00)
    r1 = ledger.reserve("gpt-4o", 0.40)
    r2 = ledger.reserve("gemini-2.0-flash", 0.40)
    assert r1 is not None
    assert r2 is not None
    assert ledger.remaining == pytest.approx(0.20)
    assert ledger.reserve("llama-3", 0.30) is None  # doesn't fit anymore
    ledger.settle(r1.id, 0.40)
    ledger.settle(r2.id, 0.40)
    assert ledger.remaining == pytest.approx(0.20)


def test_history_is_append_only_and_ordered():
    ledger = Ledger(1.00)
    r = ledger.reserve("gpt-4o", 0.40)
    ledger.settle(r.id, 0.35)
    kinds = [e.kind for e in ledger.history()]
    assert kinds == ["reserve", "settle"]


def test_history_returns_a_copy_not_the_live_log():
    ledger = Ledger(1.00)
    ledger.reserve("gpt-4o", 0.10)
    snapshot = ledger.history()
    ledger.reserve("gpt-4o", 0.10)
    assert len(snapshot) == 1  # mutating internal state didn't leak back


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        Ledger(-1.00)


def test_zero_budget_is_valid_and_allows_free_reservations():
    ledger = Ledger(0.0)
    r = ledger.reserve("llama-3", 0.0)
    assert r is not None
    assert ledger.remaining == pytest.approx(0.0)
