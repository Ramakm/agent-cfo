"""Pre-call cost estimation.

Estimates must over-estimate, never under (CLAUDE.md § Rules that matter
here) — `max_tokens` stands in for output because the model's actual
output length isn't knowable before the call runs.
"""

from __future__ import annotations

from agent_cfo.pricing import get_pricing

# Rough fallback when no tokenizer is available: ~4 characters per token
# for English text. Always prefer a real tokenizer / provider `count_tokens`
# call when one is available — this is a last resort, and it must stay on
# the generous side since underestimating breaks the one guarantee this
# project makes.
_CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_input_tokens(text: str) -> int:
    """Conservative fallback token count when no provider tokenizer is available.

    Rounds up — see module docstring on why estimates must never undercount.
    """
    if not text:
        return 0
    return -(-len(text) // _CHARS_PER_TOKEN_ESTIMATE)  # ceil division


def estimate_call_cost(
    model_id: str,
    *,
    input_tokens: int,
    max_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Worst-case dollar cost of a call, reserved before it runs.

    Output is priced at `max_tokens` — the honest ceiling — not a guess at
    what the model will actually produce.
    """
    pricing = get_pricing(model_id)
    return pricing.cost(
        input_tokens=input_tokens,
        output_tokens=max_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
