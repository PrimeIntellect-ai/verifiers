"""Hermes plugin that bridges native tool hooks to the rollout's /tool policy.

Fail-closed: a hook that cannot reach interception (or is told to stop) exits the
agent process immediately, so a vetoed tool can never run and the harness cannot
advance on a result the policy has not seen."""

import json
import os
import sys
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
PRE_REPLACED_TOOL_CALLS: set[str] = set()


def _text(content: Any) -> str:
    return (
        content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    )


def _intercept(phase: str, tool_call_id: str, tool_name: str, content: Any) -> dict:
    try:
        body = json.dumps(
            {
                "phase": phase,
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
    except Exception as error:  # noqa: BLE001 - native hooks must fail closed
        print(f"Tool interception failed: {error}", file=sys.stderr, flush=True)
        os._exit(1)
    if decision["action"] == "stop":
        # The rollout already recorded the stop; ending the agent here guarantees
        # the vetoed call cannot execute and no further turn is taken.
        os._exit(1)
    return decision


def _tool_execution(
    tool_name: str,
    args: dict[str, Any],
    next_call: Any,
    tool_call_id: str = "",
    **_: Any,
) -> Any:
    decision = _intercept("before", tool_call_id, tool_name, "")
    if decision["action"] == "allow":
        return next_call(args)
    PRE_REPLACED_TOOL_CALLS.add(tool_call_id)
    return _text(decision["message"]["content"])


def _transform_tool_result(
    tool_name: str,
    result: str,
    tool_call_id: str = "",
    **_: Any,
) -> str | None:
    if tool_call_id in PRE_REPLACED_TOOL_CALLS:
        # The pre-execution hook already supplied this result; policy must not
        # run twice over it.
        PRE_REPLACED_TOOL_CALLS.discard(tool_call_id)
        return None
    decision = _intercept("after", tool_call_id, tool_name, result)
    if decision["action"] == "allow":
        return None
    return _text(decision["message"]["content"])


def register(ctx: Any) -> None:
    ctx.register_middleware("tool_execution", _tool_execution)
    ctx.register_hook("transform_tool_result", _transform_tool_result)
