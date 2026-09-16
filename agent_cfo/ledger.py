"""Append-only ledger of reservations and settlements.

Reserve-then-settle is the enforcement mechanism: a call may not run until
its estimated cost is reserved against the budget, and the reservation is
replaced with the real cost as soon as `usage` comes back. See
CLAUDE.md § The core loop.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum


class ReservationStatus(Enum):
    OPEN = "open"
    SETTLED = "settled"
    RELEASED = "released"


@dataclass
class Reservation:
    id: int
    model_id: str
    amount: float
    status: ReservationStatus = ReservationStatus.OPEN
    settled_amount: float | None = None


@dataclass(frozen=True)
class LedgerEvent:
    """One append-only entry — reserve, settle, or release."""

    kind: str
    reservation_id: int
    model_id: str
    amount: float
    timestamp: float = field(default_factory=time.monotonic)


class InsufficientBudgetError(Exception):
    """Raised by `Ledger.reserve_or_raise` when the budget can't cover the call."""


class Ledger:
    """Tracks a single budget's reservations, settlements, and remaining balance.

    Not thread-safe by design — one ledger per agent run. Wrap externally
    with a lock if a run fans out across concurrent tool calls.
    """

    def __init__(self, budget: float):
        if budget < 0:
            raise ValueError("budget must be >= 0")
        self.budget = budget
        self._reservations: dict[int, Reservation] = {}
        self._events: list[LedgerEvent] = []
        self._ids = itertools.count(1)
        self._settled_total = 0.0

    @property
    def outstanding(self) -> float:
        """Sum of open reservations not yet settled or released."""
        return sum(
            r.amount for r in self._reservations.values() if r.status is ReservationStatus.OPEN
        )

    @property
    def settled_total(self) -> float:
        return self._settled_total

    @property
    def remaining(self) -> float:
        return self.budget - self._settled_total - self.outstanding

    def reserve(self, model_id: str, amount: float) -> Reservation | None:
        """Reserve `amount` against the budget. Returns None if it won't fit."""
        if amount < 0:
            raise ValueError("reservation amount must be >= 0")
        if amount > self.remaining:
            return None
        r = Reservation(id=next(self._ids), model_id=model_id, amount=amount)
        self._reservations[r.id] = r
        self._events.append(LedgerEvent("reserve", r.id, model_id, amount))
        return r

    def reserve_or_raise(self, model_id: str, amount: float) -> Reservation:
        r = self.reserve(model_id, amount)
        if r is None:
            raise InsufficientBudgetError(
                f"cannot reserve ${amount:.4f} for {model_id!r}: "
                f"only ${self.remaining:.4f} remaining"
            )
        return r

    def settle(self, reservation_id: int, actual_amount: float) -> None:
        """Replace a reservation's estimate with the real settled cost.

        `actual_amount` may be more or less than the reservation — this is
        where the ledger reconciles the estimate against `response.usage`.
        """
        r = self._reservations.get(reservation_id)
        if r is None:
            raise KeyError(f"no reservation with id {reservation_id}")
        if r.status is not ReservationStatus.OPEN:
            raise ValueError(
                f"reservation {reservation_id} is already {r.status.value}, cannot settle"
            )
        r.status = ReservationStatus.SETTLED
        r.settled_amount = actual_amount
        self._settled_total += actual_amount
        self._events.append(LedgerEvent("settle", reservation_id, r.model_id, actual_amount))

    def release(self, reservation_id: int) -> None:
        """Release a reservation without spending it (call failed / was skipped)."""
        r = self._reservations.get(reservation_id)
        if r is None:
            raise KeyError(f"no reservation with id {reservation_id}")
        if r.status is not ReservationStatus.OPEN:
            raise ValueError(
                f"reservation {reservation_id} is already {r.status.value}, cannot release"
            )
        r.status = ReservationStatus.RELEASED
        self._events.append(LedgerEvent("release", reservation_id, r.model_id, 0.0))

    def history(self) -> list[LedgerEvent]:
        """A copy of the append-only event log, in order."""
        return list(self._events)
