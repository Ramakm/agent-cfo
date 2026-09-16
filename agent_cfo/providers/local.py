"""Local adapter — open-source tier (`llama-3` via Ollama).

Cost is always $0 (CLAUDE.md § Model pricing), but a call still isn't
free — it costs wall time. Callers that care about latency should read
`last_elapsed_seconds` off the instance rather than assume local is instant.
"""

from __future__ import annotations

import time

from agent_cfo.providers.base import CallResult, Usage


class LocalProvider:
    """Talks to a local Ollama server. No API key, no billed usage."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.last_elapsed_seconds: float | None = None

    def call(self, model_id: str, messages: list[dict], *, max_tokens: int) -> CallResult:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "LocalProvider requires `httpx` to talk to Ollama: pip install agent-cfo[local]"
            ) from exc

        started = time.monotonic()
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model_id,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        self.last_elapsed_seconds = time.monotonic() - started

        data = response.json()
        usage = Usage(
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )
        return CallResult(
            model_id=model_id,
            content=data.get("message", {}).get("content", ""),
            usage=usage,
            cost=0.0,
        )
