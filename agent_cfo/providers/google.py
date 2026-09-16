"""Google adapter — medium tier (`gemini-2.0-flash`, CLAUDE.md § Model pricing).

The `google-generativeai` package is an optional dependency
(`pip install agent-cfo[google]`); the import is lazy so importing
`agent_cfo` never requires it.
"""

from __future__ import annotations

from agent_cfo.pricing import get_pricing
from agent_cfo.providers.base import CallResult, Usage


class GoogleProvider:
    def __init__(self, api_key: str | None = None):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "GoogleProvider requires the `google-generativeai` package: "
                "pip install agent-cfo[google]"
            ) from exc
        if api_key:
            genai.configure(api_key=api_key)
        self._genai = genai

    def call(self, model_id: str, messages: list[dict], *, max_tokens: int) -> CallResult:
        model = self._genai.GenerativeModel(model_id)
        # The budget layer only needs role/content pairs collapsed to text;
        # Google's SDK takes a flat prompt or its own richer content format.
        prompt = "\n\n".join(m["content"] for m in messages)
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens},
        )
        usage_meta = response.usage_metadata
        usage = Usage(
            input_tokens=usage_meta.prompt_token_count,
            output_tokens=usage_meta.candidates_token_count,
        )
        cost = get_pricing(model_id).cost(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        return CallResult(model_id=model_id, content=response.text, usage=usage, cost=cost)
