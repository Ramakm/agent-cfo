"""Provider adapter tests.

No test here hits a real API (CLAUDE.md § Testing). Each provider's SDK is
faked out via `sys.modules` injection with a canned `usage` payload, and we
assert the adapter normalizes it into `Usage`/`CallResult` correctly.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest


@pytest.fixture
def fake_openai_module(monkeypatch):
    calls = {}

    @dataclass
    class FakeUsageDetails:
        cached_tokens: int

    @dataclass
    class FakeUsage:
        prompt_tokens: int
        completion_tokens: int
        prompt_tokens_details: FakeUsageDetails

    @dataclass
    class FakeMessage:
        content: str

    @dataclass
    class FakeChoice:
        message: FakeMessage

    @dataclass
    class FakeResponse:
        choices: list
        usage: FakeUsage

    class FakeCompletions:
        def create(self, *, model, max_tokens, messages):
            calls["model"] = model
            calls["max_tokens"] = max_tokens
            calls["messages"] = messages
            return FakeResponse(
                choices=[FakeChoice(message=FakeMessage(content="hello from gpt-4o"))],
                usage=FakeUsage(
                    prompt_tokens=1000,
                    completion_tokens=200,
                    prompt_tokens_details=FakeUsageDetails(cached_tokens=100),
                ),
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.chat = FakeChat()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return calls


def test_openai_provider_normalizes_usage(fake_openai_module):
    from agent_cfo.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="fake-key")
    result = provider.call("gpt-4o", [{"role": "user", "content": "hi"}], max_tokens=200)

    assert result.model_id == "gpt-4o"
    assert result.content == "hello from gpt-4o"
    assert result.usage.input_tokens == 1000
    assert result.usage.output_tokens == 200
    assert result.usage.cache_read_tokens == 100
    assert result.cost == pytest.approx((1000 * 5.00 + 200 * 15.00) / 1_000_000)
    assert fake_openai_module["model"] == "gpt-4o"


@pytest.fixture
def fake_google_module(monkeypatch):
    @dataclass
    class FakeUsageMetadata:
        prompt_token_count: int
        candidates_token_count: int

    @dataclass
    class FakeResponse:
        text: str
        usage_metadata: FakeUsageMetadata

    class FakeGenerativeModel:
        def __init__(self, model_id):
            self.model_id = model_id

        def generate_content(self, prompt, generation_config=None):
            return FakeResponse(
                text="hello from gemini",
                usage_metadata=FakeUsageMetadata(
                    prompt_token_count=2000, candidates_token_count=300
                ),
            )

    fake_module = types.ModuleType("google.generativeai")
    fake_module.GenerativeModel = FakeGenerativeModel
    fake_module.configure = lambda api_key=None: None

    fake_google_pkg = types.ModuleType("google")
    fake_google_pkg.generativeai = fake_module

    monkeypatch.setitem(sys.modules, "google", fake_google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_module)
    return fake_module


def test_google_provider_normalizes_usage(fake_google_module):
    from agent_cfo.providers.google import GoogleProvider

    provider = GoogleProvider(api_key="fake-key")
    result = provider.call(
        "gemini-2.0-flash", [{"role": "user", "content": "hi"}], max_tokens=300
    )

    assert result.model_id == "gemini-2.0-flash"
    assert result.content == "hello from gemini"
    assert result.usage.input_tokens == 2000
    assert result.usage.output_tokens == 300
    assert result.cost == pytest.approx((2000 * 0.075 + 300 * 0.30) / 1_000_000)


@pytest.fixture
def fake_anthropic_module(monkeypatch):
    @dataclass
    class FakeUsage:
        input_tokens: int
        output_tokens: int
        cache_read_input_tokens: int
        cache_creation_input_tokens: int

    @dataclass
    class FakeTextBlock:
        text: str
        type: str = "text"

    @dataclass
    class FakeMessage:
        content: list
        usage: FakeUsage

    class FakeMessages:
        def create(self, *, model, max_tokens, messages):
            return FakeMessage(
                content=[FakeTextBlock(text="hello from opus")],
                usage=FakeUsage(
                    input_tokens=500,
                    output_tokens=150,
                    cache_read_input_tokens=50,
                    cache_creation_input_tokens=0,
                ),
            )

    class FakeAnthropic:
        def __init__(self, api_key=None):
            self.messages = FakeMessages()

    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return fake_module


def test_anthropic_provider_normalizes_usage(fake_anthropic_module):
    from agent_cfo.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="fake-key")
    result = provider.call(
        "claude-opus-5", [{"role": "user", "content": "hi"}], max_tokens=150
    )

    assert result.content == "hello from opus"
    assert result.usage.input_tokens == 500
    assert result.usage.cache_read_tokens == 50
    assert result.cost == pytest.approx(
        (500 * 5.00 + 150 * 25.00 + 50 * 0.50) / 1_000_000
    )


@pytest.fixture
def fake_httpx_module(monkeypatch):
    @dataclass
    class FakeResponse:
        _payload: dict

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, json, timeout):
        return FakeResponse(
            _payload={
                "message": {"content": "hello from llama"},
                "prompt_eval_count": 42,
                "eval_count": 17,
            }
        )

    fake_module = types.ModuleType("httpx")
    fake_module.post = fake_post
    monkeypatch.setitem(sys.modules, "httpx", fake_module)
    return fake_module


def test_local_provider_costs_nothing_and_normalizes_usage(fake_httpx_module):
    from agent_cfo.providers.local import LocalProvider

    provider = LocalProvider()
    result = provider.call("llama-3", [{"role": "user", "content": "hi"}], max_tokens=17)

    assert result.model_id == "llama-3"
    assert result.content == "hello from llama"
    assert result.cost == 0.0
    assert result.usage.input_tokens == 42
    assert result.usage.output_tokens == 17
    assert provider.last_elapsed_seconds is not None


def test_openai_provider_raises_clean_error_without_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)  # simulate package not installed
    from agent_cfo.providers.openai import OpenAIProvider

    with pytest.raises(ImportError, match="agent-cfo\\[openai\\]"):
        OpenAIProvider()
