"""OpenAI adapter — high-end tier (`gpt-4o`, CLAUDE.md § Model pricing).

The `openai` package is an optional dependency (`pip install agent-cfo[openai]`);
the import is lazy so importing `agent_cfo` never requires it.
"""

from __future__ import annotations

from agent_cfo.pricing import get_pricing
from agent_cfo.providers.base import CallResult, Usage


class OpenAIProvider:
    def __init__(self, api_key: str | None = None):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAIProvider requires the `openai` package: pip install agent-cfo[openai]"
            ) from exc
        self._client = openai.OpenAI(api_key=api_key)

    def call(self, model_id: str, messages: list[dict], *, max_tokens: int) -> CallResult:
        response = self._client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=messages,
        )
        usage_data = response.usage
        cached = getattr(getattr(usage_data, "prompt_tokens_details", None), "cached_tokens", 0)
        usage = Usage(
            input_tokens=usage_data.prompt_tokens,
            output_tokens=usage_data.completion_tokens,
            cache_read_tokens=cached or 0,
        )
        cost = get_pricing(model_id).cost(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
        )
        content = response.choices[0].message.content or ""
        return CallResult(model_id=model_id, content=content, usage=usage, cost=cost)
