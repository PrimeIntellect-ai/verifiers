import json
import re
from collections.abc import Sequence
from typing import Any

from datasets import Dataset
import verifiers as vf

SYSTEM_PROMPT = """You are solving a deterministic tool-use simulation task.

Return only JSON with this schema:
{
  "tool_calls": [
    {"name": "tool_name", "arguments": {"key": "value"}}
  ],
  "answer": "final short answer"
}

Do not execute tools. Infer the exact call sequence from the available tools,
conversation, and target objective."""

TASKS: list[dict[str, Any]] = [
    {
        "id": "weather_then_calendar",
        "available_tools": [
            {
                "name": "weather.lookup",
                "description": "Look up the forecast for a city and date.",
                "required_arguments": ["city", "date"],
            },
            {
                "name": "calendar.create_event",
                "description": "Create a calendar event.",
                "required_arguments": ["title", "date", "time"],
            },
        ],
        "conversation": (
            "User: If the forecast in Austin tomorrow is sunny, schedule "
            "tennis practice for 7 AM tomorrow.\n"
            "Context: Tomorrow is 2026-05-12. The embedded forecast table says "
            "Austin on 2026-05-12 is sunny."
        ),
        "target": "Check the forecast before creating the event.",
        "expected_tool_calls": [
            {
                "name": "weather.lookup",
                "arguments": {"city": "Austin", "date": "2026-05-12"},
            },
            {
                "name": "calendar.create_event",
                "arguments": {
                    "title": "tennis practice",
                    "date": "2026-05-12",
                    "time": "07:00",
                },
            },
        ],
        "answer": "Scheduled tennis practice for 7 AM tomorrow.",
    },
    {
        "id": "refund_lookup",
        "available_tools": [
            {
                "name": "orders.find",
                "description": "Find an order by email and item.",
                "required_arguments": ["email", "item"],
            },
            {
                "name": "refunds.create",
                "description": "Create a refund for an order.",
                "required_arguments": ["order_id", "reason"],
            },
        ],
        "conversation": (
            "User: Refund the blue backpack order for sam@example.com because "
            "the zipper broke.\nContext: Matching order id is ORD-8842."
        ),
        "target": "Look up the order before creating the refund.",
        "expected_tool_calls": [
            {
                "name": "orders.find",
                "arguments": {"email": "sam@example.com", "item": "blue backpack"},
            },
            {
                "name": "refunds.create",
                "arguments": {
                    "order_id": "ORD-8842",
                    "reason": "zipper broke",
                },
            },
        ],
        "answer": "Created a refund for order ORD-8842.",
    },
    {
        "id": "repo_issue_summary",
        "available_tools": [
            {
                "name": "github.search_issues",
                "description": "Search repository issues.",
                "required_arguments": ["repo", "query"],
            },
            {
                "name": "notion.create_page",
                "description": "Create a Notion page.",
                "required_arguments": ["database", "title", "summary"],
            },
        ],
        "conversation": (
            "User: Summarize open crash issues in acme/mobile into the "
            "Engineering Triage database.\nContext: The issue search should use "
            "query 'is:issue is:open crash'."
        ),
        "target": "Search GitHub, then create a triage page.",
        "expected_tool_calls": [
            {
                "name": "github.search_issues",
                "arguments": {
                    "repo": "acme/mobile",
                    "query": "is:issue is:open crash",
                },
            },
            {
                "name": "notion.create_page",
                "arguments": {
                    "database": "Engineering Triage",
                    "title": "Open crash issues in acme/mobile",
                    "summary": "Summarize open crash issues from acme/mobile.",
                },
            },
        ],
        "answer": "Created an Engineering Triage page for open crash issues.",
    },
    {
        "id": "currency_invoice",
        "available_tools": [
            {
                "name": "fx.convert",
                "description": "Convert money between currencies.",
                "required_arguments": ["amount", "from_currency", "to_currency"],
            },
            {
                "name": "invoice.draft",
                "description": "Draft an invoice.",
                "required_arguments": ["customer", "amount", "currency"],
            },
        ],
        "conversation": (
            "User: Draft an invoice for Contoso for 1200 EUR converted to USD.\n"
            "Context: The conversion result is 1296 USD."
        ),
        "target": "Convert the amount before drafting the invoice.",
        "expected_tool_calls": [
            {
                "name": "fx.convert",
                "arguments": {
                    "amount": "1200",
                    "from_currency": "EUR",
                    "to_currency": "USD",
                },
            },
            {
                "name": "invoice.draft",
                "arguments": {
                    "customer": "Contoso",
                    "amount": "1296",
                    "currency": "USD",
                },
            },
        ],
        "answer": "Drafted a USD invoice for Contoso for 1296.",
    },
]


def _normalize_value(value: Any) -> str:
    return str(value).strip().lower()


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion:
        last_message = completion[-1]
        if isinstance(last_message, dict):
            return str(last_message.get("content", ""))
    return str(completion)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from a raw or fenced completion."""
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise TypeError("completion must be a JSON object")
    return parsed


def format_question(task: dict[str, Any]) -> str:
    tools_json = json.dumps(task["available_tools"], indent=2, sort_keys=True)
    return (
        f"Available tools:\n{tools_json}\n\n"
        f"Conversation:\n{task['conversation']}\n\n"
        f"Target objective:\n{task['target']}"
    )


def build_dataset(num_examples: int = -1) -> Dataset:
    tasks = TASKS if num_examples < 0 else TASKS[:num_examples]
    rows = []
    for task in tasks:
        rows.append(
            {
                "question": format_question(task),
                "answer": json.dumps(
                    {
                        "tool_calls": task["expected_tool_calls"],
                        "answer": task["answer"],
                    },
                    sort_keys=True,
                ),
                "info": {
                    "task_id": task["id"],
                    "available_tools": task["available_tools"],
                    "target": task["target"],
                },
            }
        )
    return Dataset.from_list(rows)


def _coerce_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [call for call in value if isinstance(call, dict)]


def tool_sequence_reward(completion: Any, answer: str, **kwargs: Any) -> float:
    del kwargs
    try:
        actual = extract_json_object(_extract_completion_text(completion))
        expected = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    actual_calls = _coerce_tool_calls(actual.get("tool_calls"))
    expected_calls = _coerce_tool_calls(expected.get("tool_calls"))
    if not actual_calls or len(actual_calls) != len(expected_calls):
        return 0.0

    sequence_score = float(
        [call.get("name") for call in actual_calls]
        == [call.get("name") for call in expected_calls]
    )

    argument_matches = 0
    argument_total = 0
    for actual_call, expected_call in zip(actual_calls, expected_calls, strict=True):
        actual_arguments = actual_call.get("arguments", {})
        expected_arguments = expected_call.get("arguments", {})
        if not isinstance(expected_arguments, dict):
            continue
        argument_total += len(expected_arguments)
        if not isinstance(actual_arguments, dict):
            continue
        for key, expected_value in expected_arguments.items():
            if _normalize_value(actual_arguments.get(key, "")) == _normalize_value(
                expected_value
            ):
                argument_matches += 1

    argument_score = argument_matches / argument_total if argument_total else 0.0
    answer_score = float(
        _normalize_value(actual.get("answer", ""))
        == _normalize_value(expected["answer"])
    )
    return 0.4 * sequence_score + 0.4 * argument_score + 0.2 * answer_score


def load_environment(
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    system_prompt: str = SYSTEM_PROMPT,
) -> vf.Environment:
    train_dataset = build_dataset(num_train_examples)
    eval_dataset = build_dataset(num_eval_examples)
    rubric = vf.Rubric(funcs=[tool_sequence_reward], weights=[1.0])
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
    )
