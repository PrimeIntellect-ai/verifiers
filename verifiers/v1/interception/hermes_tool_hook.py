"""Hermes plugin that applies the rollout's two-phase tool policy."""

import json
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, build_opener

credentials_path = Path(os.environ.pop("VF_TOOL_INTERCEPTION_CONFIG"))
credentials_json = credentials_path.read_text()
credentials_path.unlink()
credentials = json.loads(credentials_json)
TOOL_URL = credentials["url"]
TOOL_SECRET = credentials["secret"]
if not isinstance(TOOL_URL, str) or not isinstance(TOOL_SECRET, str):
    raise TypeError("Tool interception configuration is unavailable")
OPENER = build_opener()
BLOCKED_TOOL_CALLS: set[str] = set()


def _text(content: Any) -> str:
    return (
        content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    )


def _intercept(
    phase: str, tool_call_id: str, tool_name: str, content: Any
) -> dict[str, Any]:
    body = json.dumps(
        {
            "phase": phase,
            "rewrite": {"content": "text"},
            "message": {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
                "name": tool_name,
            },
        }
    ).encode()
    request = Request(
        TOOL_URL,
        body,
        {
            "Authorization": f"Bearer {TOOL_SECRET}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with OPENER.open(request, timeout=30) as response:
        decision = json.load(response)
    if decision.get("action") not in {"allow", "rewrite", "stop"}:
        raise ValueError("tool interception returned an invalid action")
    if decision["action"] == "rewrite" and not decision.get("message"):
        raise ValueError("tool interception omitted the rewritten result")
    return decision


def _tool_execution(
    tool_name: str,
    args: dict[str, Any],
    next_call: Any,
    tool_call_id: str = "",
    **_: Any,
) -> Any:
    try:
        decision = _intercept("before", tool_call_id, tool_name, "")
    except Exception:  # noqa: BLE001 - native hooks must fail closed
        return json.dumps({"error": "Tool interception is unavailable."})
    if decision["action"] == "allow":
        return next_call(args)
    BLOCKED_TOOL_CALLS.add(tool_call_id)
    if decision["action"] == "rewrite":
        return _text(decision["message"]["content"])
    return json.dumps(
        {"error": decision.get("reason") or "Rollout terminated by interception."}
    )


def _transform_tool_result(
    tool_name: str,
    result: str,
    tool_call_id: str = "",
    status: str = "ok",
    **_: Any,
) -> str | None:
    if tool_call_id in BLOCKED_TOOL_CALLS:
        BLOCKED_TOOL_CALLS.discard(tool_call_id)
        return None
    phase = "after_failure" if status == "error" else "after"
    if phase == "after":
        try:
            parsed = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("exit_code") not in (None, 0):
            phase = "after_failure"
    try:
        decision = _intercept(phase, tool_call_id, tool_name, result)
    except Exception:  # noqa: BLE001 - native hooks must fail closed
        return json.dumps({"error": "Tool interception is unavailable."})
    if decision["action"] == "allow":
        return None
    if decision["action"] == "rewrite":
        return _text(decision["message"]["content"])
    return json.dumps(
        {"error": decision.get("reason") or "Rollout terminated by interception."}
    )


def register(ctx: Any) -> None:
    ctx.register_middleware("tool_execution", _tool_execution)
    ctx.register_hook("transform_tool_result", _transform_tool_result)
