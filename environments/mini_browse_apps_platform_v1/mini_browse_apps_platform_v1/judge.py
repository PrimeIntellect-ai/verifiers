"""LLM judge support for Mini Browse local-app tasks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

JUDGE_TEMPERATURE = 0

JUDGE_PROMPT = """You evaluate a browser automation agent's submitted result for a deterministic local flight-search task.

Use the evaluation contract and gold answer as the source of truth. Score only
expected_fields where score is true or absent. Ignore verifier metadata, ids,
internal keys, hidden seed fields, and non-scoreable diagnostics.

Treat equivalent formatting as correct when the same fact is clearly attached to
the same flight leg, date, row, provider, fare, comparison slot, or outcome.
Examples: "Nonstop" equals "0 stops"; prices match after removing currency
symbols and commas; date formats match when they refer to the same calendar date;
duration strings match when they have the same total minutes.

Extra fields are fine unless they contradict the gold answer. Critical fields
with critical=true are hard gates: if any scoreable critical field is missing or
wrong, the verdict must be "no".

Return exactly one JSON object, no prose and no code fence:
{
  "correct_fields": <integer>,
  "total_fields": <integer>,
  "score": <number from 0 to 1>,
  "verdict": "yes" | "partial" | "no",
  "explanation": "<one concise sentence>",
  "field_verdicts": [
    {
      "field_path": "<expected field path>",
      "verdict": "exact_match" | "semantic_match" | "wrong" | "missing",
      "reason": "<short reason>"
    }
  ]
}
"""


async def judge_answer_key(
    *,
    task_instruction: str,
    submitted_result: Any,
    answer_key: dict[str, Any],
    output_schema: dict[str, Any],
    model: str,
    base_url: str | None,
    api_key_env: str,
) -> dict[str, Any]:
    context = {
        "task_instruction": task_instruction,
        "submitted_result": submitted_result,
        "evaluation_contract": answer_key.get("evaluator") or {},
        "gold_answer": answer_key.get("gold_answer") or answer_key,
        "output_schema": output_schema,
    }
    response = await judge_client(base_url, api_key_env).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False, sort_keys=True),
            },
        ],
        temperature=JUDGE_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return parse_json_object(content)


def judge_client(base_url: str | None, api_key_env: str) -> AsyncOpenAI:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Missing judge API key env var {api_key_env}")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    default_headers = prime_default_headers(base_url)
    if default_headers:
        kwargs["default_headers"] = default_headers
    return AsyncOpenAI(**kwargs)


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
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            return {"score": 0.0, "explanation": content[:500], "verdict": "no"}
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        return {
            "score": 0.0,
            "explanation": "judge returned non-object",
            "verdict": "no",
        }
    return parsed


def prime_team_id() -> str | None:
    for name in ("PRIME_TEAM_ID", "PI_TEAM_ID", "X_PRIME_TEAM_ID"):
        value = os.environ.get(name)
        if value:
            return value
    config_path = Path.home() / ".prime" / "config.json"
    try:
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if isinstance(config, dict):
                value = config.get("team_id")
                if value:
                    return str(value)
    except (json.JSONDecodeError, OSError):
        return None
    return None


def prime_default_headers(base_url: str | None) -> dict[str, str]:
    if not base_url or "pinference" not in base_url.lower():
        return {}
    team_id = prime_team_id()
    return {"X-Prime-Team-ID": team_id} if team_id else {}
