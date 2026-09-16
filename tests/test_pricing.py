import pytest

from agent_cfo.pricing import (
    DOWNGRADE_LADDER,
    PRICING,
    UnknownModelError,
    estimate_cost,
    get_pricing,
)


def test_all_ladder_models_are_priced():
    for model_id in DOWNGRADE_LADDER:
        assert model_id in PRICING


def test_open_source_model_is_free():
    pricing = get_pricing("llama-3")
    assert pricing.input_per_million == 0.0
    assert pricing.output_per_million == 0.0


def test_ladder_is_ordered_by_descending_output_cost():
    costs = [get_pricing(m).output_per_million for m in DOWNGRADE_LADDER]
    assert costs == sorted(costs, reverse=True)


def test_unknown_model_raises():
    with pytest.raises(UnknownModelError):
        get_pricing("not-a-real-model")


def test_estimate_cost_basic():
    cost = estimate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(5.00)


def test_estimate_cost_scales_with_output_tokens():
    small = estimate_cost("gpt-4o", input_tokens=0, output_tokens=1_000_000)
    large = estimate_cost("gpt-4o", input_tokens=0, output_tokens=2_000_000)
    assert large == pytest.approx(2 * small)


def test_estimate_cost_cache_read_cheaper_than_fresh_input():
    fresh = estimate_cost("claude-opus-5", input_tokens=1_000_000, output_tokens=0)
    cached = estimate_cost("claude-opus-5", cache_read_tokens=1_000_000, output_tokens=0)
    assert cached < fresh


def test_gemini_medium_tier_cheaper_than_gpt4o_high_end():
    """CLAUDE.md's tier design: medium must cost less than high-end, per token."""
    gemini = get_pricing("gemini-2.0-flash")
    gpt4o = get_pricing("gpt-4o")
    assert gemini.input_per_million < gpt4o.input_per_million
    assert gemini.output_per_million < gpt4o.output_per_million
