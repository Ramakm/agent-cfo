"""OpenRouter adapter — routes to whichever backing model the slug names.

OpenRouter's API is OpenAI-compatible, so this reuses the `openai` SDK
pointed at OpenRouter's endpoint. `:free`-suffixed model slugs are $0 but
rate-limited (~20 req/min) and rotate over time — check
https://openrouter.ai/models?fmt=free for what's currently live before
relying on any specific one.
"""

from __future__ import annotations

from agent_cfo.pricing import get_pricing
from agent_cfo.providers.base import CallResult, Usage

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    def __init__(self, api_key: str | None = None):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenRouterProvider requires the `openai` package: pip install agent-cfo[openai]"
            ) from exc
        self._client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def call(self, model_id: str, messages: list[dict], *, max_tokens: int) -> CallResult:
        response = self._client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=messages,
        )
        usage_data = response.usage
        usage = Usage(
            input_tokens=usage_data.prompt_tokens,
            output_tokens=usage_data.completion_tokens,
        )
        cost = get_pricing(model_id).cost(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        content = response.choices[0].message.content or ""
        return CallResult(model_id=model_id, content=content, usage=usage, cost=cost)
