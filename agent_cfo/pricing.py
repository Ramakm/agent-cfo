"""Model pricing table — the single source of truth for $/1M tokens.

Update `LAST_VERIFIED` whenever prices are checked against the provider's
published rates. A stale price here silently corrupts every cost number
the system reports (see CLAUDE.md § Rules that matter here).
"""

from __future__ import annotations

from dataclasses import dataclass

LAST_VERIFIED = "2026-06-24"


@dataclass(frozen=True)
class ModelPricing:
    """$/1M-token rates for one model."""

    id: str
    provider: str
    tier: str  # "open-source" | "medium" | "high-end"
    context_window: int
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float = 0.0
    cache_write_per_million: float = 0.0

    def cost(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Dollar cost for one call's token usage."""
        return (
            input_tokens * self.input_per_million
            + output_tokens * self.output_per_million
            + cache_read_tokens * self.cache_read_per_million
            + cache_write_tokens * self.cache_write_per_million
        ) / 1_000_000


PRICING: dict[str, ModelPricing] = {
    "llama-3": ModelPricing(
        id="llama-3",
        provider="local",
        tier="open-source",
        context_window=8_000,
        input_per_million=0.0,
        output_per_million=0.0,
    ),
    "gemini-2.0-flash": ModelPricing(
        id="gemini-2.0-flash",
        provider="google",
        tier="medium",
        context_window=1_000_000,
        input_per_million=0.075,
        output_per_million=0.30,
    ),
    "gpt-4o": ModelPricing(
        id="gpt-4o",
        provider="openai",
        tier="high-end",
        context_window=128_000,
        input_per_million=5.00,
        output_per_million=15.00,
    ),
    # Kept for reference/fallback — not part of the active downgrade ladder.
    "claude-opus-5": ModelPricing(
        id="claude-opus-5",
        provider="anthropic",
        tier="high-end",
        context_window=1_000_000,
        input_per_million=5.00,
        output_per_million=25.00,
        cache_read_per_million=0.50,
        cache_write_per_million=6.25,
    ),
}

# Highest cost/capability first — the order the policy layer downgrades
# through when a requested model doesn't fit the remaining budget.
DOWNGRADE_LADDER: tuple[str, ...] = ("gpt-4o", "gemini-2.0-flash", "llama-3")


class UnknownModelError(KeyError):
    """Raised when a model ID isn't in the pricing table."""


def get_pricing(model_id: str) -> ModelPricing:
    try:
        return PRICING[model_id]
    except KeyError as exc:
        raise UnknownModelError(
            f"No pricing entry for {model_id!r}. Add it to agent_cfo/pricing.py "
            f"before using it — an unpriced call can't be budgeted."
        ) from exc


def estimate_cost(
    model_id: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Dollar cost for a call's real (or hypothetical) token usage."""
    return get_pricing(model_id).cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
