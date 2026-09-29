"""Real head-to-head test: GPT-4o vs Gemini 2.0 Flash on PubMedQA.

Not a fixture-based unit test — this makes real, billed API calls and
belongs outside `tests/` (CLAUDE.md: "No test may hit the API"). Run
manually:

    uv run python bench/run_pubmedqa.py --n 25

Requires OPENAI_API_KEY and GOOGLE_API_KEY in a local `.env` (gitignored).
Downloads the real PubMedQA labeled set (pubmedqa/pubmedqa, 1000 expert-
annotated yes/no/maybe items) into .scratch/ on first run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_cfo.pricing import PRICING  # noqa: E402
from agent_cfo.providers.base import CallResult, Usage  # noqa: E402
from agent_cfo.providers.google import GoogleProvider  # noqa: E402
from agent_cfo.providers.groq import GroqProvider  # noqa: E402
from agent_cfo.providers.openai import OpenAIProvider  # noqa: E402
from agent_cfo.providers.openrouter import OpenRouterProvider  # noqa: E402

DATA_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
DATA_PATH = ROOT / ".scratch" / "pqal.json"

PROMPT_TEMPLATE = """You are answering a biomedical research question based only on the \
abstract context given. Respond with exactly one word: yes, no, or maybe.

Context: {context}

Question: {question}

Answer (one word: yes, no, or maybe):"""


def load_env(path: Path) -> None:
    import os

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value:
            os.environ[key.strip()] = value  # .env wins over a stale/invalid shell value
        else:
            os.environ.pop(key.strip(), None)  # blank in .env means "unset", not "inherit"


def load_dataset() -> dict:
    DATA_PATH.parent.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Downloading PubMedQA labeled set to {DATA_PATH}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return json.loads(DATA_PATH.read_text())


def sample_balanced(data: dict, n: int, seed: int = 42) -> list[dict]:
    by_label: dict[str, list[dict]] = {"yes": [], "no": [], "maybe": []}
    for pmid, item in data.items():
        label = item.get("final_decision")
        if label in by_label:
            by_label[label].append({**item, "pmid": pmid})
    rng = random.Random(seed)
    for label in by_label:
        rng.shuffle(by_label[label])
    per_label = max(1, n // 3)
    sample = (
        by_label["yes"][:per_label] + by_label["no"][:per_label] + by_label["maybe"][:per_label]
    )
    rng.shuffle(sample)
    return sample[:n]


def normalize_answer(raw: str) -> str:
    cleaned = raw.strip().lower().strip(".").split()
    for token in cleaned:
        if token in ("yes", "no", "maybe"):
            return token
    return cleaned[0] if cleaned else "unparsable"


def call_gemini_unpriced(provider: GoogleProvider, model_id: str, prompt: str, max_tokens: int):
    """Like GoogleProvider.call, but for a model not yet in pricing.py.

    Accuracy only, no cost figure — never fabricate a price. Used because
    Google retired gemini-2.0-flash; gemini-3.8-flash has no verified rate
    in agent_cfo/pricing.py yet (see docs note in this script's summary).
    """
    model = provider._genai.GenerativeModel(model_id)
    response = model.generate_content(
        prompt, generation_config={"max_output_tokens": max_tokens}
    )
    usage_meta = response.usage_metadata
    usage = Usage(
        input_tokens=usage_meta.prompt_token_count,
        output_tokens=usage_meta.candidates_token_count,
    )
    try:
        text = response.text
    except Exception:  # noqa: BLE001 - no text Part (e.g. truncated by MAX_TOKENS)
        text = ""
    return CallResult(model_id=model_id, content=text, usage=usage, cost=None)


def run_model(
    provider, model_id: str, items: list[dict], label: str, *, max_tokens: int, delay: float
) -> dict:
    priced = model_id in PRICING
    correct = 0
    total_cost = 0.0
    errors = 0
    per_item = []
    for i, item in enumerate(items, 1):
        context = " ".join(item["CONTEXTS"])
        prompt = PROMPT_TEMPLATE.format(context=context[:4000], question=item["QUESTION"])
        attempt = 0
        while True:
            attempt += 1
            try:
                if priced:
                    result = provider.call(
                        model_id,
                        [{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                    )
                else:
                    result = call_gemini_unpriced(provider, model_id, prompt, max_tokens=max_tokens)
                break
            except Exception as exc:  # noqa: BLE001 - surfacing any provider failure as a scored miss
                if "429" in str(exc) and attempt < 4:
                    wait = 60
                    print(f"  [{label}] item {i}/{len(items)} rate-limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"  [{label}] item {i}/{len(items)} ERROR: {exc}")
                errors += 1
                per_item.append({"pmid": item["pmid"], "error": str(exc)})
                result = None
                break
        if result is None:
            continue
        if delay:
            time.sleep(delay)
        predicted = normalize_answer(result.content)
        gold = item["final_decision"]
        is_correct = predicted == gold
        correct += is_correct
        if priced:
            total_cost += result.cost
        per_item.append(
            {
                "pmid": item["pmid"],
                "gold": gold,
                "predicted": predicted,
                "correct": is_correct,
                "cost": result.cost if priced else None,
            }
        )
        print(f"  [{label}] item {i}/{len(items)}: gold={gold} predicted={predicted}")
    scored = len(items) - errors
    return {
        "model": model_id,
        "n": len(items),
        "errors": errors,
        "correct": correct,
        "accuracy": correct / scored if scored else 0.0,
        "total_cost": total_cost if priced else None,
        "priced": priced,
        "per_item": per_item,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="items per model (balanced yes/no/maybe)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gemini-model", default="gemini-3.8-flash")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument(
        "--gemini-delay", type=float, default=13.0, help="seconds between Gemini calls (free tier: 5/min)"
    )
    parser.add_argument("--skip-gemini", action="store_true", help="skip Gemini (e.g. daily free-tier quota hit)")
    args = parser.parse_args()

    load_env(ROOT / ".env")
    import os

    openai_key = os.environ.get("OPENAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not google_key and not args.skip_gemini:
        raise SystemExit("Missing GOOGLE_API_KEY in .env")

    data = load_dataset()
    items = sample_balanced(data, args.n, seed=args.seed)
    print(f"Sampled {len(items)} PubMedQA items (balanced yes/no/maybe).\n")

    summary = {}

    if openai_key:
        print("Running GPT-4o...")
        summary["gpt-4o"] = run_model(
            OpenAIProvider(api_key=openai_key),
            "gpt-4o",
            items,
            "gpt-4o",
            max_tokens=args.max_tokens,
            delay=0,
        )
    else:
        print("Skipping GPT-4o: no OPENAI_API_KEY set.")

    if args.skip_gemini:
        print("Skipping Gemini (--skip-gemini, e.g. daily free-tier quota exhausted).")
    else:
        print(f"\nRunning {args.gemini_model}...")
        summary[args.gemini_model] = run_model(
            GoogleProvider(api_key=google_key),
            args.gemini_model,
            items,
            "gemini",
            max_tokens=args.max_tokens,
            delay=args.gemini_delay,
        )

    if groq_key:
        groq_model = "openai/gpt-oss-20b"
        print(f"\nRunning {groq_model} (Groq, open-weight tier)...")
        summary[groq_model] = run_model(
            GroqProvider(api_key=groq_key),
            groq_model,
            items,
            "groq",
            max_tokens=args.max_tokens,
            delay=0,
        )
    else:
        print("Skipping Groq open-weight tier: no GROQ_API_KEY set.")

    if openrouter_key:
        or_model = "nvidia/nemotron-3-super-120b-a12b:free"
        print(f"\nRunning {or_model} (OpenRouter, free 120B open-weight tier)...")
        summary[or_model] = run_model(
            OpenRouterProvider(api_key=openrouter_key),
            or_model,
            items,
            "openrouter",
            max_tokens=args.max_tokens,
            delay=3.0,  # free tier ~20 req/min
        )
    else:
        print("Skipping OpenRouter: no OPENROUTER_API_KEY set.")

    out_path = ROOT / ".scratch" / "pubmedqa_results.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    for name, r in summary.items():
        cost_str = f"${r['total_cost']:.4f} spent" if r["priced"] else "cost not tracked (unpriced model)"
        print(
            f"{name}: {r['correct']}/{r['n'] - r['errors']} correct "
            f"({r['accuracy']:.1%}), {cost_str}, {r['errors']} errors"
        )
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
