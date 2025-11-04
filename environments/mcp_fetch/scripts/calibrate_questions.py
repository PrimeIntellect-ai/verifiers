#!/usr/bin/env python3
"""Calibrate question difficulty by driving the fetch tool via OpenAI models.

This script spins up the deterministic mini HTTP server, loads deterministic
tasks from `environments/mcp_fetch/tasks/qa.jsonl`, and evaluates them with one
or more OpenAI models using tool-calling. The assistant is required to call the
`fetch` function to gather data before responding with a final answer in the
form `ANSWER: <value>`.

For each model we record pass/fail outcomes (exact match against the task's
`expected` field) and emit a JSON report under `environments/mcp_fetch/reports/`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from types import SimpleNamespace

from openai import OpenAI

from environments.mcp_fetch.tools.fetch_mcp_server import fetch_url_async
from environments.mcp_fetch.utils.mini_httpd import serve
from environments.mcp_fetch.verifiers import get_judge_prompt
SCRIPT_ROOT = Path(__file__).resolve().parent
ENV_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = ENV_ROOT.parent
TASKS_PATH = ENV_ROOT / "tasks" / "qa.jsonl"
REPORTS_DIR = ENV_ROOT / "reports"
DEFAULT_PORT = 31415
DEFAULT_MAX_TURNS = 6

FETCH_TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": (
                "Fetch a URL from the deterministic mini site. "
                "Use this for every information request; hosts are restricted "
                "to http://127.0.0.1:31415 by default."
            ),
            "parameters": {
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["GET", "HEAD", "get", "head"],
                        "default": "GET",
                    },
                    "headers": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "params": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "timeout_s": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 60.0,
                        "default": 8.0,
                    },
                    "max_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1_000_000,
                        "default": 200_000,
                    },
                },
            },
        },
    }
]

def _model_requires_responses(model: str) -> bool:
    lowered = model.lower()
    return lowered.startswith("gpt-5") or lowered.startswith("o3")


def _supports_temperature(model: str) -> bool:
    lowered = model.lower()
    if lowered.startswith("gpt-5"):
        return False
    return True


def _convert_text_block(role: str, text: str) -> Dict[str, str]:
    block_type = "text"
    if role == "user":
        block_type = "input_text"
    elif role == "assistant":
        block_type = "output_text"
    return {"type": block_type, "text": text}


def _format_messages_for_responses(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            blocks = []
            for call in tool_calls:
                func = call.get("function", {})
                blocks.append(
                    {
                        "type": "tool_call",
                        "id": call.get("id") or f"call_{len(blocks)}",
                        "name": func.get("name", "tool"),
                        "arguments": func.get("arguments", "{}"),
                    }
                )
            formatted.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            formatted.append(
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": msg.get("tool_call_id") or "",
                            "output": str(msg.get("content", "")),
                            "is_error": False,
                        }
                    ],
                }
            )
            continue

        content = msg.get("content")
        if isinstance(content, list):
            text_parts: List[str] = []
            for chunk in content:
                if isinstance(chunk, dict) and "text" in chunk:
                    text_parts.append(str(chunk["text"]))
                elif isinstance(chunk, str):
                    text_parts.append(chunk)
            content = "\n".join(text_parts) if text_parts else ""
        if content is None:
            content = ""
        blocks = [_convert_text_block(role, str(content))]
        formatted.append({"role": role, "content": blocks})
    return formatted


def _responses_to_chat_choice(response) -> SimpleNamespace:
    """Convert Responses API output to a chat-completion-like object."""

    outputs = getattr(response, "output", None)
    if outputs is None and hasattr(response, "model_dump"):
        outputs = response.model_dump().get("output", [])

    tool_calls = []
    text_parts: List[str] = []
    finish_reason: Optional[str] = None

    for raw_item in outputs or []:
        item = raw_item.model_dump() if hasattr(raw_item, "model_dump") else raw_item
        item_type = item.get("type")
        if item_type == "message":
            for chunk in item.get("content", []):
                chunk_type = chunk.get("type")
                if chunk_type in {"output_text", "text"}:
                    text_parts.append(chunk.get("text", ""))
            finish_reason = (
                item.get("metadata", {}).get("finish_reason")
                or item.get("status")
                or finish_reason
                or "stop"
            )
        elif item_type == "tool_call":
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {})
            tool_calls.append(
                {
                    "id": item.get("id") or item.get("tool_call_id") or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": item.get("name") or item.get("function_name") or "tool",
                        "arguments": arguments,
                    },
                }
            )
            finish_reason = "tool_calls"

    message_obj = SimpleNamespace(
        role="assistant",
        content="\n".join(part for part in text_parts if part).strip() or None,
        tool_calls=[SimpleNamespace(**call) for call in tool_calls] if tool_calls else None,
    )
    choice_obj = SimpleNamespace(
        message=message_obj,
        finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
    )
    return SimpleNamespace(choices=[choice_obj])


def _create_completion(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
):
    if _model_requires_responses(model):
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": _format_messages_for_responses(messages),
        }
        if temperature and _supports_temperature(model):
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return _responses_to_chat_choice(client.responses.create(**kwargs))
    return client.chat.completions.create(
        model=model,
        temperature=temperature if _supports_temperature(model) else 0,
        messages=messages,
        tools=tools,
    )

SYSTEM_PROMPT = """\
You are a calibration assistant for the Fetch MCP environment.
Every task references the deterministic fixtures server at http://127.0.0.1:31415.

Rules:
1. Always call the `fetch` tool to read the requested URL before answering.
2. Use GET or HEAD exactly as requested; include headers/query params when needed.
3. After gathering the necessary data, reply with `ANSWER: <value>` and nothing else.
4. Do not guess without fetching; if something fails, briefly explain then provide `ANSWER: <value>`.
"""


@dataclass
class Task:
    raw: Dict[str, Any]

    @property
    def id(self) -> str:
        return self.raw.get("id", "")

    @property
    def question(self) -> str:
        return self.raw.get("question", "")

    @property
    def expected(self) -> str:
        return str(self.raw.get("expected", "")).strip()

    @property
    def verifier_type(self) -> str:
        verifier = self.raw.get("verifier") or {}
        return str(verifier.get("type", ""))

    @property
    def verifier(self) -> Dict[str, Any]:
        return self.raw.get("verifier") or {}

    @property
    def rubric_id(self) -> Optional[str]:
        return self.verifier.get("rubric")


def load_local_env(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file if present."""

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def load_tasks(
    *,
    include_judge: bool = False,
    ids: Optional[Iterable[str]] = None,
) -> List[Task]:
    selected_ids = {task_id.strip() for task_id in ids} if ids else None
    tasks: List[Task] = []
    with TASKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            task = Task(raw=raw)
            if not include_judge and task.verifier_type == "judge":
                continue
            if selected_ids and task.id not in selected_ids:
                continue
            tasks.append(task)
    return tasks


def start_fixture_server(port: int) -> threading.Thread:
    thread = threading.Thread(target=serve, kwargs={"port": port}, daemon=True)
    thread.start()
    time.sleep(0.5)
    return thread


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict) and "text" in chunk:
                parts.append(str(chunk["text"]))
        return "\n".join(parts)
    return str(content)


def _chat_message_to_dict(message: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"role": message.role}
    content = _message_content_to_text(message.content)
    if content:
        payload["content"] = content
    if message.tool_calls:
        payload["tool_calls"] = []
        for call in message.tool_calls:
            payload["tool_calls"].append(
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )
    return payload


def _invoke_fetch_tool(call: Any) -> Dict[str, Any]:
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return {"error": f"invalid arguments: {exc}"}

    url = args.get("url")
    if not url:
        return {"error": "url is required"}

    try:
        result = asyncio.run(
            fetch_url_async(
                url,
                method=args.get("method", "GET"),
                headers=args.get("headers"),
                params=args.get("params"),
                timeout_s=float(args.get("timeout_s", 8.0)),
                max_bytes=int(args.get("max_bytes", 200_000)),
            )
        )
    except Exception as exc:  # noqa: BLE001 - need to surface tool errors
        return {"error": str(exc)}
    return result


def extract_answer(output_text: str) -> str:
    if not output_text:
        return ""
    match = re.search(r"answer\s*:\s*(.+)", output_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return output_text.strip()


def normalize_answer(text: str) -> str:
    return text.strip().casefold()


def evaluate_judge_completion(
    client: OpenAI,
    *,
    submission: str,
    rubric_id: str,
    judge_model: str,
) -> Dict[str, Any]:
    prompt = get_judge_prompt(rubric_id)
    user_prompt = f"{prompt.strip()}\n\nSubmission:\n{submission.strip() or '(empty)'}"
    response = _create_completion(
        client,
        model=judge_model,
        temperature=0,
        messages=[{"role": "user", "content": user_prompt}],
    )
    content = _message_content_to_text(response.choices[0].message.content)
    decision = content.strip().lower()
    passed = decision.startswith("pass")
    return {"passed": passed, "judge_output": content}


def evaluate_task(
    client: OpenAI,
    model: str,
    task: Task,
    *,
    max_turns: int,
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": task.question})
    tool_calls = 0
    transcript: List[Dict[str, Any]] = []

    for _ in range(max_turns):
        completion = _create_completion(
            client,
            model=model,
            temperature=0,
            messages=messages,
            tools=FETCH_TOOL_SPEC,
        )
        choice = completion.choices[0]
        message = choice.message
        message_dict = _chat_message_to_dict(message)
        messages.append(message_dict)
        transcript.append(message_dict)

        if choice.finish_reason == "tool_calls":
            for call in message.tool_calls or []:
                tool_calls += 1
                payload = _invoke_fetch_tool(call)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(payload),
                }
                messages.append(tool_message)
                transcript.append(tool_message)
            continue

        final_text = _message_content_to_text(message.content)
        answer_value = extract_answer(final_text)
        return {
            "answer": answer_value,
            "raw_output": final_text,
            "tool_calls": tool_calls,
            "transcript": transcript,
        }

    return {
        "answer": "",
        "raw_output": "",
        "tool_calls": tool_calls,
        "transcript": transcript,
        "error": "max turns exceeded",
    }


def sanitize_filename(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Fetch question quality.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="OpenAI model to evaluate (can be passed multiple times).",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        help="Limit evaluation to specific task IDs (repeatable).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Limit the number of tasks (after filtering) to evaluate.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help="Maximum chat turns (assistant responses) per task.",
    )
    parser.add_argument(
        "--include-judge",
        action="store_true",
        help="Include JudgeRubric tasks (requires judge grading).",
    )
    parser.add_argument(
        "--judge-model",
        help="Model to use for JudgeRubric grading (defaults to the first --model).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for the fixtures server.",
    )
    args = parser.parse_args()

    load_local_env(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (load it in .env or environment).")

    client = OpenAI()
    judge_model = args.judge_model
    if not judge_model:
        for candidate in args.model:
            if not _model_requires_responses(candidate):
                judge_model = candidate
                break
        if not judge_model and args.model:
            judge_model = args.model[0]
    if args.include_judge and not judge_model:
        raise SystemExit("Judge model must be provided when including judge tasks.")
    start_fixture_server(args.port)

    tasks = load_tasks(ids=args.task_id, include_judge=args.include_judge)
    if args.max_questions is not None:
        tasks = tasks[: args.max_questions]

    REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    summaries: list[dict[str, Any]] = []

    for model in args.model:
        results = []
        correct = 0
        judge_model_for_run = judge_model or model
        for task in tasks:
            record = {
                "task_id": task.id,
                "question": task.question,
                "expected": task.expected,
            }
            try:
                outcome = evaluate_task(
                    client,
                    model,
                    task,
                    max_turns=args.max_turns,
                )
            except Exception as exc:  # noqa: BLE001
                record["error"] = str(exc)
                record["answer"] = ""
                record["raw_output"] = ""
                record["tool_calls"] = 0
            else:
                record.update(outcome)

            extra_detail = f", expected={task.expected!r}"
            if task.verifier_type == "judge":
                rubric_id = task.rubric_id
                if not rubric_id:
                    raise ValueError(f"Judge task {task.id} missing rubric id")
                judge_result = evaluate_judge_completion(
                    client,
                    submission=record.get("answer", ""),
                    rubric_id=rubric_id,
                    judge_model=judge_model_for_run,
                )
                record.update(judge_result)
                is_correct = judge_result["passed"] and not record.get("error")
                extra_detail = f", judge={judge_result['judge_output']!r}"
            else:
                actual = normalize_answer(record.get("answer", ""))
                expected = normalize_answer(task.expected)
                is_correct = actual == expected and not record.get("error")
            record["correct"] = is_correct
            if is_correct:
                correct += 1
            results.append(record)
            status = "PASS" if is_correct else "FAIL"
            print(f"[{model}] {task.id}: {status} (answer={record.get('answer')!r}{extra_detail})")

        accuracy = correct / len(tasks) if tasks else 0.0
        summary = {
            "model": model,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(tasks),
            "thresholds": {
                "small_model_max": "< 0.90",
                "strong_model_min": "> 0.10",
            },
            "results": results,
        }
        report_path = REPORTS_DIR / f"question_quality_{sanitize_filename(model)}.json"
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary_record = {
            "model": model,
            "correct": correct,
            "total": len(tasks),
            "accuracy": accuracy,
            "report_path": str(report_path),
        }
        summaries.append(summary_record)
        print(f"\nModel {model}: {correct}/{len(tasks)} correct ({accuracy:.1%}). Report: {report_path}\n")

    if summaries:
        print("\n=== Calibration Summary ===")
        for record in summaries:
            print(
                f"- {record['model']}: {record['correct']}/{record['total']} "
                f"({record['accuracy']:.1%}) -> {record['report_path']}"
            )
        print("==========================\n")


if __name__ == "__main__":
    main()
