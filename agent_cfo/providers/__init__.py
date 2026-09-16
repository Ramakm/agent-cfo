"""Provider adapters — one per model source, all conforming to `Provider`.

Each adapter's SDK is an optional dependency and is imported lazily inside
`call()`/`__init__`, so importing this package never requires any of
`openai`, `google-generativeai`, `anthropic`, or `httpx` to be installed.
"""

from agent_cfo.providers.base import CallResult, Provider, Usage

__all__ = ["CallResult", "Provider", "Usage"]
