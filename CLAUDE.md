# agent-cfo

Budget enforcement for autonomous agents. Every agent run gets a dollar budget; the
CFO tracks spend across model and tool calls in real time and forces the agent to
degrade gracefully — smaller model, local model, or hard stop — before the budget
is gone.

**Status: greenfield.** No implementation exists yet. This file describes the
intended design; update it as reality diverges.

## The core loop

```
Agent run  →  budget = $2.00
  ├─ before each priced call: reserve(estimated_cost)
  │    └─ if reserve fails → policy decision (downgrade / local / stop)
  ├─ after each call: settle(actual_cost from usage)
  └─ remaining = budget − settled − outstanding reservations
```

Reserve-then-settle (not settle-only) is the whole point. Estimating cost *before*
the call is what makes enforcement possible; settling afterward with real `usage`
numbers is what keeps the ledger honest. Never let a call happen that has not been
reserved.

## Stack

- **Python 3.11+**, `pyproject.toml`, `uv` for env/deps.
- `anthropic` SDK for API calls. Never hand-rolled `requests`/`httpx` against
  `/v1/messages`.
- `pytest` for tests, `ruff` for lint/format.

## Intended module layout

| Module | Responsibility |
|---|---|
| `agent_cfo/ledger.py` | Append-only record of reservations and settlements. Single source of truth for "what's left". |
| `agent_cfo/pricing.py` | Model → $/1M in/out. Cache-read and cache-write rates. Pure data + lookup, no I/O. |
| `agent_cfo/estimate.py` | Pre-call cost estimate. Input tokens via `count_tokens`; output via `max_tokens` as the worst case. |
| `agent_cfo/policy.py` | Given remaining budget and a proposed call, decide: allow / downgrade / route local / deny. |
| `agent_cfo/providers/` | Adapters. `anthropic.py` (API), `local.py` (Ollama/llama.cpp, cost 0 but not free — track wall time). |
| `agent_cfo/budget.py` | Public surface: the context manager / decorator an agent wraps its run in. |

## Rules that matter here

**Cost is always derived from `response.usage`, never estimated after the fact.**
Read `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, and
`cache_read_input_tokens` separately — cache reads are ~10× cheaper than fresh
input and a tracker that ignores them will overstate spend badly enough to trigger
false downgrades.

**Pricing lives in exactly one place** (`pricing.py`) and is dated. Prices change.
A stale price silently corrupts every number the system reports, so treat the table
as data with a `last_verified` stamp, not as constants scattered through the code.

**Estimates must over-estimate, never under.** `max_tokens` is the honest ceiling
for output. An estimator that guesses low lets the agent blow through the budget,
which is the one failure mode this project exists to prevent.

**Thinking tokens are billed as output tokens.** Adaptive thinking means output
cost is not predictable from the prompt — another reason to reserve against
`max_tokens`.

**Denial is a return value, not an exception in the hot path.** The agent needs to
*react* to "you can't afford this" (retry smaller, go local), so the policy layer
returns a decision object. Reserve for a genuine, unrecoverable overdraft.

## Model pricing (verified 2026-06-24, per 1M tokens)

| Tier | Model | ID | Context | Input | Output | Notes |
|---|---|---|---|---|---|---|
| Open-source | Llama 3 | `llama-3` | 8K | $0.00 | $0.00 | Local (Ollama), wall-time tracked |
| Medium | Gemini 2.0 Flash | `gemini-2.0-flash` | 1M | $0.075 | $0.30 | Google (via `google-generativeai` SDK) |
| High-end | GPT-4o | `gpt-4o` | 128K | $5.00 | $15.00 | OpenAI (via `openai` SDK) |

Use exact ID strings — no date suffixes. Downgrade ladder order (by cost-quality
tradeoff): `gpt-4o → gemini-2.0-flash → llama-3-local`. The budget enforcement 
layer picks which model to use based on available funds and task requirements.

**Provider adapters:** Each model requires its own SDK. Update `agent_cfo/providers/`
with `openai.py`, `google.py`, and keep `anthropic.py` for fallback context. Cost
comparison is within tier only; absolute costs vary by provider.

## Cheaper before smaller

The policy layer tries free levers before it degrades quality:

1. **Prompt caching** — cache the stable prefix (system prompt, tool definitions).
   Verify with `usage.cache_read_input_tokens`; a zero there across repeated calls
   means something volatile leaked into the prefix.
2. **`output_config.effort`** — `low`/`medium` cuts thinking spend within the same
   model. Cheaper than switching models and usually cheaper on *quality* too.
3. **Batch API** — 50% off for anything not latency-sensitive.
4. *Then* model downgrade, then local, then stop.

Note that caches are model-scoped: a downgrade forfeits cache reuse, so the
switch can cost more on the next call than it saves on this one. The policy layer
must account for this.

## Testing

Cost math is the product — test it directly and offline. No test may hit the API;
providers are fake, `usage` payloads are fixtures. Cover at minimum: reservation
released on failure, cache-read pricing, concurrent reservations against one
budget, and the downgrade ladder bottoming out into a stop.
