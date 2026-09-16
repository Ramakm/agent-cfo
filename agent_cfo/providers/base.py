"""Provider adapter protocol.

Every provider — hosted API or local runtime — implements this shape so
the budget/policy layer can call any of them uniformly. Adapters translate
their SDK's response into a `CallResult` carrying real usage numbers;
`Ledger.settle` needs those, not an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Usage:
    """Normalized token usage, regardless of provider-specific field names."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class CallResult:
    model_id: str
    content: str
    usage: Usage
    cost: float


class Provider(Protocol):
    """Adapter contract: given a model + messages, make the call and return
    normalized usage so the ledger can settle against real numbers."""

    def call(self, model_id: str, messages: list[dict], *, max_tokens: int) -> CallResult: ...
