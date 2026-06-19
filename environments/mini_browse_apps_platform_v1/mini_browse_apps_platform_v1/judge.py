"""LLM judge for the browse-apps local-app tasks (structured-output verdict)."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.utils.client_utils import load_prime_config
from verifiers.v1.clients.config import BaseClientConfig

JUDGE_TEMPERATURE = 0

# Strict structured output: the judge must return exactly these fields, always valid JSON.
JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "correct_fields": {"type": "integer"},
                "total_fields": {"type": "integer"},
                "score": {"type": "number"},
                "verdict": {"type": "string", "enum": ["yes", "partial", "no"]},
                "explanation": {"type": "string"},
            },
            "required": [
                "correct_fields",
                "total_fields",
                "score",
                "verdict",
                "explanation",
            ],
        },
    },
}

JUDGE_PROMPT = """You evaluate a browser automation agent's submitted result for a deterministic local flight-search task.

Use the evaluation contract and gold answer as the source of truth. Score only
expected fields where score is true or absent. Ignore verifier metadata, ids,
internal keys, hidden seed fields, and non-scoreable diagnostics.

Treat equivalent formatting as correct when the same fact is clearly attached to
the same flight leg, date, row, provider, fare, comparison slot, or outcome.
Examples: "Nonstop" equals "0 stops"; prices match after removing currency
symbols and commas; date formats match when they refer to the same calendar date;
duration strings match when they have the same total minutes.

Extra fields are fine unless they contradict the gold answer. Critical fields
with critical=true are hard gates: if any scoreable critical field is missing or
wrong, the verdict must be "no".

Report `correct_fields` / `total_fields` over the scoreable expected fields, a
`score` from 0 to 1, a `verdict`, and a one-sentence `explanation`.
"""


class JudgeConfig(BaseConfig):
    """The judge model and the OpenAI-compatible endpoint it runs on (Prime auto-resolved)."""

    model: str = "openai/gpt-4.1-mini"
    """A model that supports strict structured output (`json_schema`)."""
    client: BaseClientConfig = Field(default_factory=BaseClientConfig)


async def judge_answer_key(
    *,
    task_instruction: str,
    submitted_result: Any,
    answer_key: dict[str, Any],
    output_schema: dict[str, Any],
    config: JudgeConfig,
) -> dict[str, Any]:
    context = {
        "task_instruction": task_instruction,
        "submitted_result": submitted_result,
        "evaluation_contract": answer_key.get("evaluator") or {},
        "gold_answer": answer_key.get("gold_answer") or answer_key,
        "output_schema": output_schema,
    }
    response = await judge_client(config.client).chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False, sort_keys=True),
            },
        ],
        temperature=JUDGE_TEMPERATURE,
        response_format=JUDGE_RESPONSE_FORMAT,
    )
    content = response.choices[0].message.content or "{}"
    return parse_json_object(content)


def judge_client(config: BaseClientConfig) -> AsyncOpenAI:
    # base_url + team header are resolved by BaseClientConfig; the key falls back to the Prime
    # CLI config for pinference (mirrors verifiers' resolve_client).
    api_key = os.environ.get(config.api_key_var)
    if not api_key and config.api_key_var == "PRIME_API_KEY":
        api_key = load_prime_config().get("api_key")
    return AsyncOpenAI(
        base_url=config.base_url,
        api_key=api_key or "EMPTY",
        default_headers=config.headers or None,
    )


def score_from_judge_payload(payload: dict[str, Any]) -> float:
    correct = payload.get("correct_fields")
    total = payload.get("total_fields")
    if isinstance(correct, int) and isinstance(total, int) and total > 0:
        return max(0.0, min(1.0, correct / total))
    score = payload.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return max(0.0, min(1.0, float(score)))
    verdict = str(payload.get("verdict") or "").lower()
    if verdict == "yes":
        return 1.0
    if verdict == "partial":
        return 0.5
    return 0.0


def parse_json_object(content: str) -> dict[str, Any]:
    # Strict structured output is always valid JSON; this stays tolerant (code fences, an
    # unterminated object) as a backstop for an overridden/non-conforming judge model.
    fenced = content.strip()
    if fenced.startswith("```"):
        fenced = fenced.split("```", 2)[1].removeprefix("json").strip()
    start = fenced.find("{")
    span = fenced[start:] if start >= 0 else ""
    for candidate in (content, fenced, span, _balance_json(span)):
        if not candidate.strip():
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {"score": 0.0, "explanation": content[:500], "verdict": "no"}


def _balance_json(text: str) -> str:
    """Close an unterminated JSON object/array: append the missing `}`/`]` for any brackets left
    open outside of strings, after dropping a dangling trailing comma."""
    stack: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    trimmed = text.rstrip()
    if trimmed.endswith(","):
        trimmed = trimmed[:-1]
    return trimmed + "".join(reversed(stack))
