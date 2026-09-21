"""FastAPI backend for the agent-cfo chat demo.

Thin wrapper around the real budget engine (`agent_cfo.Budget`) — every
request still goes through reserve -> call -> settle
(CLAUDE.md § The core loop). Nothing here estimates or prices anything
itself; that stays in `estimate.py` / `pricing.py` / `policy.py`.

Provider selection per tier: use the real SDK adapter if that tier's API key
is set and the call succeeds, otherwise fall back to `SimulatedProvider` so
the demo runs with zero configuration. `llama-3` tries a local Ollama server
first (also free) and simulates only if that's unreachable.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_cfo.budget import Budget
from agent_cfo.estimate import estimate_input_tokens
from agent_cfo.policy import Action
from agent_cfo.pricing import DOWNGRADE_LADDER, PRICING
from agent_cfo.webapp.simulated import SimulatedProvider

DEFAULT_MAX_TOKENS = 400

app = FastAPI(title="agent-cfo")

_simulated = SimulatedProvider()


class Session:
    def __init__(self, amount: float):
        self.id = uuid.uuid4().hex[:12]
        self.budget = Budget(amount)
        self.starting_amount = amount
        self.messages: list[dict] = []
        self.events: list[dict] = []


_sessions: dict[str, Session] = {}


def _get_session(session_id: str) -> Session:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return session


def _provider_for(model_id: str):
    """Best available provider for a tier: real SDK if keyed, else simulated."""
    if model_id == "llama-3":
        try:
            from agent_cfo.providers.local import LocalProvider

            return LocalProvider(base_url=os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434")
        except ImportError:
            return _simulated
    if model_id == "gemini-2.0-flash" and os.environ.get("GOOGLE_API_KEY"):
        try:
            from agent_cfo.providers.google import GoogleProvider

            return GoogleProvider(api_key=os.environ["GOOGLE_API_KEY"])
        except ImportError:
            return _simulated
    if model_id == "gpt-4o" and os.environ.get("OPENAI_API_KEY"):
        try:
            from agent_cfo.providers.openai import OpenAIProvider

            return OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
        except ImportError:
            return _simulated
    return _simulated


class CreateSessionRequest(BaseModel):
    budget: float = 2.00


class ChatRequest(BaseModel):
    content: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    requested_model: str = DOWNGRADE_LADDER[0]


def _ledger_snapshot(session: Session) -> dict:
    ledger = session.budget.ledger
    return {
        "budget": session.starting_amount,
        "settled_total": round(ledger.settled_total, 6),
        "outstanding": round(ledger.outstanding, 6),
        "remaining": round(ledger.remaining, 6),
        "events": session.events,
    }


@app.post("/api/sessions")
def create_session(req: CreateSessionRequest):
    if req.budget <= 0:
        raise HTTPException(status_code=400, detail="budget must be > 0")
    session = Session(req.budget)
    _sessions[session.id] = session
    return {
        "session_id": session.id,
        "ladder": list(DOWNGRADE_LADDER),
        "pricing": {
            model_id: {
                "tier": p.tier,
                "input_per_million": p.input_per_million,
                "output_per_million": p.output_per_million,
            }
            for model_id, p in PRICING.items()
            if model_id in DOWNGRADE_LADDER
        },
        **_ledger_snapshot(session),
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = _get_session(session_id)
    return {"messages": session.messages, **_ledger_snapshot(session)}


@app.post("/api/sessions/{session_id}/messages")
def post_message(session_id: str, req: ChatRequest):
    session = _get_session(session_id)
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")

    session.messages.append({"role": "user", "content": req.content})

    history = [{"role": m["role"], "content": m["content"]} for m in session.messages]
    input_tokens = estimate_input_tokens("\n".join(m["content"] for m in history))

    planned = session.budget.plan(
        model_id=req.requested_model,
        input_tokens=input_tokens,
        max_tokens=req.max_tokens,
    )
    decision = planned.decision

    if decision.action is Action.DENY:
        assistant_message = {
            "role": "assistant",
            "content": (
                "⛔ Budget exhausted — I can't afford any model on the downgrade "
                f"ladder with ${session.budget.remaining:.4f} remaining. Run stopped."
            ),
            "model": None,
            "action": "deny",
            "estimated_cost": None,
            "actual_cost": None,
        }
        session.messages.append(assistant_message)
        session.events.append({"kind": "deny", "reason": decision.reason})
        return {"message": assistant_message, **_ledger_snapshot(session)}

    provider = _provider_for(decision.model_id)
    try:
        result = provider.call(decision.model_id, history, max_tokens=req.max_tokens)
    except Exception:
        session.budget.abandon(planned)
        result = _simulated.call(decision.model_id, history, max_tokens=req.max_tokens)
        planned = session.budget.plan_or_raise(
            model_id=decision.model_id,
            input_tokens=input_tokens,
            max_tokens=req.max_tokens,
        )

    session.budget.record(planned, actual_cost=result.cost)

    assistant_message = {
        "role": "assistant",
        "content": result.content,
        "model": decision.model_id,
        "action": decision.action.value,
        "reason": decision.reason,
        "estimated_cost": round(decision.estimated_cost, 6),
        "actual_cost": round(result.cost, 6),
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
    }
    session.messages.append(assistant_message)
    session.events.append(
        {
            "kind": "settle",
            "model": decision.model_id,
            "action": decision.action.value,
            "estimated_cost": decision.estimated_cost,
            "actual_cost": result.cost,
        }
    )
    return {"message": assistant_message, **_ledger_snapshot(session)}


_static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
