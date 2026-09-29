"""Groq adapter — cheap, open-weight tier via Groq's hosted inference.

Groq's API is OpenAI-compatible, so this reuses the `openai` SDK pointed at
Groq's endpoint rather than adding a new dependency. Note: as of 2026-08-26
Groq moved Llama 3 models to enterprise-only pricing with no published
per-token rate, so the CLAUDE.md `llama-3` ($0/local) ladder entry cannot be
served through this provider. `openai/gpt-oss-20b` is the current cheapest
self-serve open-weight model on Groq and is what this adapter targets —
priced, not free, and not Llama. Don't conflate it with the local tier.
"""

from __future__ import annotations

from agent_cfo.pricing import get_pricing
from agent_cfo.providers.base import CallResult, Usage

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider:
    def __init__(self, api_key: str | None = None):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "GroqProvider requires the `openai` package: pip install agent-cfo[openai]"
            ) from exc
        self._client = openai.OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    def call(self, model_id: str, messages: list[dict], *, max_tokens: int) -> CallResult:
        response = self._client.chat.completions.create(
            model=model_id,
            max_completion_tokens=max_tokens,
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
