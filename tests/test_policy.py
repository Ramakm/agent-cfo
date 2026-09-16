from agent_cfo.policy import Action, decide
from agent_cfo.pricing import DOWNGRADE_LADDER


def test_allows_requested_model_when_affordable():
    decision = decide(
        requested_model="gpt-4o",
        remaining_budget=10.0,
        input_tokens=1000,
        max_tokens=1000,
    )
    assert decision.action is Action.ALLOW
    assert decision.model_id == "gpt-4o"


def test_downgrades_when_requested_model_too_expensive():
    decision = decide(
        requested_model="gpt-4o",
        remaining_budget=0.001,
        input_tokens=1000,
        max_tokens=1000,
    )
    assert decision.action is Action.DOWNGRADE
    assert decision.model_id in ("gemini-2.0-flash", "llama-3")


def test_bottoms_out_at_free_local_model():
    """However tight the budget (as long as non-negative), llama-3 at $0
    must still be reachable — it's the floor of the downgrade ladder that
    the README's core pitch ('use local model instead') depends on."""
    decision = decide(
        requested_model="gpt-4o",
        remaining_budget=0.0,
        input_tokens=1_000_000,
        max_tokens=1_000_000,
    )
    assert decision.action is Action.DOWNGRADE
    assert decision.model_id == "llama-3"
    assert decision.estimated_cost == 0.0


def test_denies_when_budget_is_negative():
    decision = decide(
        requested_model="gpt-4o",
        remaining_budget=-1.0,
        input_tokens=100,
        max_tokens=100,
    )
    assert decision.action is Action.DENY
    assert decision.model_id is None
    assert decision.estimated_cost is None


def test_top_of_ladder_allowed_with_ample_budget():
    decision = decide(
        requested_model=DOWNGRADE_LADDER[0],
        remaining_budget=1_000.0,
        input_tokens=10,
        max_tokens=10,
    )
    assert decision.action is Action.ALLOW


def test_requested_model_not_on_ladder_is_priced_standalone():
    decision = decide(
        requested_model="claude-opus-5",
        remaining_budget=10.0,
        input_tokens=100,
        max_tokens=100,
    )
    assert decision.action is Action.ALLOW
    assert decision.model_id == "claude-opus-5"


def test_requested_model_not_on_ladder_falls_back_when_unaffordable():
    decision = decide(
        requested_model="claude-opus-5",
        remaining_budget=0.0001,
        input_tokens=1_000_000,
        max_tokens=1_000_000,
    )
    assert decision.action is Action.DOWNGRADE
    assert decision.model_id == "llama-3"


def test_custom_ladder_without_free_floor_can_deny():
    decision = decide(
        requested_model="gpt-4o",
        remaining_budget=1.00,
        input_tokens=10_000_000,
        max_tokens=10_000_000,
        ladder=("gpt-4o",),
    )
    assert decision.action is Action.DENY
