# Risk-aware downgrade (design proposal)

`agent_cfo/policy.py` currently walks `DOWNGRADE_LADDER` purely on cost: it
returns the cheapest model in the ladder that fits `remaining_budget`, with
no notion of what the call is for. That's the right default, but it's not
enough for domains — biomedical workflows being the motivating case — where
a wrong answer from an under-powered model is worse than an expensive one.

This doc captures the extension discussed for that case. Not yet implemented.

## Problem with cost-only downgrade

Budget-only `decide()` will happily downgrade a high-stakes call (e.g. a
clinical recommendation) to the free local tier the moment funds run low,
with no signal that this particular call needed the accuracy the top-tier
model provides. The ladder floor should sometimes be *risk*, not budget.

## Proposed extension

1. **Risk tier on the call.** Callers tag a request `risk="low"` or
   `risk="high"` (a crude two-level tag is enough to start — task type or
   "does this feed a clinical/scientific decision" is a fine heuristic).
   Add this as a field alongside `requested_model` in `decide()`'s
   signature.

2. **Hard floor for high-risk calls.** For `risk="high"`, `decide()` should
   never return `Action.DOWNGRADE` past a configured floor model (e.g. never
   below `gemini-2.0-flash`, never to `llama-3-local`) — even when the floor
   model doesn't fit `remaining_budget`. In that case it returns
   `Action.DENY` (or a new `Action.ESCALATE`) instead of silently serving a
   cheaper, less reliable answer. Cost protection must not silently become
   a reliability regression.

3. **Verification gate on accepted downgrades.** When a downgrade *is*
   allowed, run a cheap post-hoc check before the result is returned to the
   caller — e.g. confirm the facts carried forward via the handoff (see
   below) still appear in the downgraded model's answer. A failed check
   escalates back up the ladder rather than returning the degraded answer.

4. **Context handoff across downgrades.** Caches are model-scoped
   (CLAUDE.md § Cheaper before smaller), so a downgrade forfeits more than
   price — it forfeits the smaller model's chance to see what the larger
   model already established. Extract key facts/entities from the prior
   model's response into a structured object and prepend it to the next
   model's prompt on downgrade, instead of relying on free-text carryover.

5. **Log the decision, not just the cost.** Every `Decision` from a
   risk-aware `decide()` should record: risk tier, whether a downgrade was
   attempted, whether verification passed, and whether the call escalated.
   That log is the evidence base for reliability claims over time — more
   defensible than a one-off benchmark run.

## Why this order

Steps 1–2 (risk tier + hard floor) are the actual answer to "how do you
protect scientific validity when downgrading for cost" — they make
reliability a gate the system enforces, not a property you hope holds and
measure afterward. Steps 3–4 (verification + handoff) are what make a
downgrade that *is* allowed trustworthy rather than just cheaper. Step 5
is what turns real usage into an ongoing evaluation instead of a single
pilot.

## Open questions

- Risk classification: manual tag from the caller vs. inferred from task
  metadata? Start manual; infer later once there's usage data to train a
  heuristic on.
- What counts as "verification failed" for open-ended (non-multiple-choice)
  outputs, where there's no exact-match reference to check against.
- Whether `Action.DENY` is the right terminus for a blocked high-risk
  downgrade, or whether callers need a distinct `Action.ESCALATE` that
  retries at a higher tier automatically rather than surfacing a bare
  denial.
