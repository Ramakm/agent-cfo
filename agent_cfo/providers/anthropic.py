"""Anthropic adapter — kept for reference/fallback (CLAUDE.md § Model pricing).

Not on the active downgrade ladder (`DOWNGRADE_LADDER` in `agent_cfo.pricing`
is now gpt-4o / gemini-2.0-flash / llama-3), but `claude-opus-5` stays in the
pricing table so this adapter and any run that names it directly still work.
The `anthropic` package is an optional dependency (`pip install
agent-cfo[anthropic]`); the import is lazy so importing `agent_cfo` never
requires it.
"""

from __future__ import annotations

from agent_cfo.pricing import get_pricing
from agent_cfo.providers.base import CallResult, Usage


class AnthropicProvider:
    def __init__(self, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicProvider requires the `anthropic` package: "
                "pip install agent-cfo[anthropic]"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)

    def call(self, model_id: str, messages: list[dict], *, max_tokens: int) -> CallResult:
        response = self._client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=messages,
        )
        usage_data = response.usage
        usage = Usage(
            input_tokens=usage_data.input_tokens,
            output_tokens=usage_data.output_tokens,
            cache_read_tokens=getattr(usage_data, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage_data, "cache_creation_input_tokens", 0) or 0,
        )
        cost = get_pricing(model_id).cost(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
        content = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return CallResult(model_id=model_id, content=content, usage=usage, cost=cost)
