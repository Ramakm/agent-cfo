# agent-cfo

Budget enforcement for autonomous agents.

Every agent run gets a dollar budget. `agent-cfo` tracks spend across model
calls in real time and forces the agent to degrade gracefully — cheaper
model, local model, or hard stop — before the budget runs out.

```
Agent
 ↓
Budget = $2
 ↓
Task
 ↓
Tool calls
 ↓
Model calls
 ↓
Cost tracking
 ↓
"Only $0.42 remaining"
```

Then it decides:

- Use a cheaper model instead.
- Use a local model instead.
- Stop execution.

This becomes increasingly relevant as agent workloads become long-running
and cost management becomes an operational problem.

See [CLAUDE.md](CLAUDE.md) for the full design — the reserve-then-settle
model, the pricing rules, and why estimates must never undercount.

## Install

```bash
uv sync --group dev          # core + dev deps (pytest, ruff)
uv sync --group dev --extra openai --extra google --extra anthropic --extra local
```

Each provider's SDK is an optional extra — `agent_cfo` itself has zero
required dependencies, so you only install what you actually call.

Copy `.env.example` to `.env` and fill in keys for the providers you use:

```bash
cp .env.example .env
```

## Quick start

```python
from agent_cfo import Budget

with Budget(2.00) as budget:
    planned = budget.plan(model_id="gpt-4o", input_tokens=50_000, max_tokens=20_000)

    if planned.decision.action.value == "deny":
        raise SystemExit("out of budget")

    # planned.decision.model_id is the model to actually call — it may be
    # cheaper than the one you asked for if the budget forced a downgrade.
    print(planned.decision.model_id, planned.decision.reason)

    # ... call the model, then settle the reservation with its real cost:
    budget.record(planned, actual_cost=0.03)

    print(f"${budget.remaining:.2f} remaining")
```

## Model tiers

| Tier | Model | Input $/1M | Output $/1M |
|---|---|---|---|
| Open-source | Llama 3 (local, via Ollama) | $0.00 | $0.00 |
| Medium | Gemini 2.0 Flash | $0.075 | $0.30 |
| High-end | GPT-4o | $5.00 | $15.00 |

The downgrade ladder walks `gpt-4o → gemini-2.0-flash → llama-3` as the
budget tightens. Full pricing table (including the reference-only
`claude-opus-5` entry) in `agent_cfo/pricing.py`.

## Project layout

```
agent_cfo/
  pricing.py      # model → $/1M, single source of truth
  ledger.py        # reserve / settle / release, remaining balance
  estimate.py      # pre-call cost estimate (always the max_tokens ceiling)
  policy.py         # decide(): allow / downgrade / deny
  budget.py         # public Budget context manager
  providers/         # adapters: openai, google, anthropic, local (Ollama)
tests/               # offline unit tests — no test hits a real API
```

## Testing

```bash
uv run pytest -q
uv run ruff check .
```
