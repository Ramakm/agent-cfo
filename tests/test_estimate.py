from agent_cfo.estimate import estimate_call_cost, estimate_input_tokens
from agent_cfo.pricing import estimate_cost as actual_call_cost


def test_estimate_input_tokens_rounds_up():
    assert estimate_input_tokens("abcde") == 2  # 5 chars / 4 -> ceil -> 2
    assert estimate_input_tokens("abcd") == 1  # exact multiple, no over-round
    assert estimate_input_tokens("") == 0


def test_estimate_call_cost_uses_max_tokens_as_output_ceiling():
    cost_small_cap = estimate_call_cost("gpt-4o", input_tokens=100, max_tokens=100)
    cost_large_cap = estimate_call_cost("gpt-4o", input_tokens=100, max_tokens=1000)
    assert cost_large_cap > cost_small_cap


def test_estimate_never_undercounts_relative_to_actual_output():
    """The estimate must be >= any actual output cost up to max_tokens — it
    prices the ceiling, not a guess (CLAUDE.md: 'estimates must
    over-estimate, never under')."""
    max_tokens = 500
    estimate = estimate_call_cost("gpt-4o", input_tokens=200, max_tokens=max_tokens)
    for actual_output in (0, 100, 250, 500):
        actual = actual_call_cost("gpt-4o", input_tokens=200, output_tokens=actual_output)
        assert estimate >= actual


def test_estimate_call_cost_includes_cache_tokens():
    plain = estimate_call_cost("claude-opus-5", input_tokens=1000, max_tokens=100)
    with_cache_write = estimate_call_cost(
        "claude-opus-5", input_tokens=1000, max_tokens=100, cache_write_tokens=1000
    )
    assert with_cache_write > plain
