"""Fallback provider used when a tier has no usable API key.

Lets the chat demo run end-to-end with zero external credentials: it still
produces real token counts (via the same char-based estimator `estimate.py`
uses) and a real dollar cost from `pricing.py`, so the ledger settles against
honest numbers even though no network call happened. Never used silently in
place of a real provider that IS configured — `webapp/server.py` only reaches
for this when the tier's key is missing or the real call raised.
"""

from __future__ import annotations

import random

from agent_cfo.estimate import estimate_input_tokens
from agent_cfo.pricing import get_pricing
from agent_cfo.providers.base import CallResult, Usage

_OPENERS = [
    "Here's my take:",
    "Looking at this,",
    "Quick answer:",
    "Sure —",
    "Happy to help:",
]


class SimulatedProvider:
    """Deterministic-ish stand-in for a real model call. Cost 0 to the wallet
    isn't the point here — it fabricates plausible usage so the ledger still
    has real numbers to settle, at the tier's real per-token price."""

    def call(self, model_id: str, messages: list[dict], *, max_tokens: int) -> CallResult:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        input_tokens = estimate_input_tokens("\n".join(m["content"] for m in messages))
        opener = random.choice(_OPENERS)
        reply = (
            f"{opener} ({model_id}, simulated — no API key configured for this tier) "
            f"you said: “{last_user[:280]}”. In a live deployment this reply "
            f"would come straight from {model_id}."
        )
        output_tokens = min(max_tokens, estimate_input_tokens(reply))
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        cost = get_pricing(model_id).cost(
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
        )
        return CallResult(model_id=model_id, content=reply, usage=usage, cost=cost)
