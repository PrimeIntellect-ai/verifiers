# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2"]
# ///
"""Bridge Claude Code and Codex tool hooks to the rollout interception server."""

import json
import os
import sys
from typing import Any, Literal
from urllib.request import Request, urlopen

from pydantic import BaseModel, TypeAdapter


class NativeToolHook(BaseModel):
    hook_event_name: Literal["PreToolUse", "PostToolUse", "PostToolUseFailure"]
    tool_name: str
    tool_use_id: str
    tool_input: Any = None
    tool_response: Any = None
    error: str | None = None


class InterceptedToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: Any
    name: str | None = None


class ToolDecision(BaseModel):
    action: Literal["allow", "rewrite", "stop"]
    message: InterceptedToolMessage | None = None
    reason: str | None = None


def native_output(
    adapter: Literal["claude", "codex"],
    hook: NativeToolHook,
    decision: ToolDecision,
) -> dict | None:
    if decision.action == "allow":
        return None
    reason = decision.reason or "Rollout terminated by interception."
    if decision.action == "stop":
        if adapter == "codex" and hook.hook_event_name == "PreToolUse":
            return {"decision": "block", "reason": reason}
        return {"continue": False, "stopReason": reason}

    if decision.message is None:
        raise ValueError("rewrite decision omitted its tool message")
    content = decision.message.content
    text = (
        content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    )
    if adapter == "codex":
        return {"decision": "block", "reason": text}
    if hook.hook_event_name == "PreToolUse":
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": text,
            }
        }
    if hook.hook_event_name == "PostToolUseFailure":
        return {"decision": "block", "reason": text}

    original = hook.tool_response
    if isinstance(original, str):
        replacement = text
    elif (
        isinstance(original, dict)
        and {
            "stdout",
            "stderr",
            "interrupted",
            "isImage",
        }
        <= original.keys()
    ):
        replacement = {**original, "stdout": text, "stderr": ""}
    elif hook.tool_name.startswith("mcp__") and isinstance(original, dict):
        replacement = {
            **original,
            "content": [{"type": "text", "text": text}],
        }
    elif isinstance(original, dict) and isinstance(original.get("content"), str):
        replacement = {**original, "content": text}
    else:
        return {
            "continue": False,
            "stopReason": "Claude Code cannot safely replace this structured tool output.",
        }
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "updatedToolOutput": replacement,
        }
    }


def main() -> None:
    hook = NativeToolHook.model_validate_json(sys.stdin.read())
    adapter = TypeAdapter(Literal["claude", "codex"]).validate_python(
        os.environ["VF_TOOL_INTERCEPTION_ADAPTER"]
    )
    if hook.hook_event_name == "PreToolUse":
        content = json.dumps(hook.tool_input, ensure_ascii=False)
    elif hook.hook_event_name == "PostToolUse":
        content = (
            hook.tool_response
            if isinstance(hook.tool_response, str)
            else json.dumps(hook.tool_response, ensure_ascii=False)
        )
    elif hook.hook_event_name == "PostToolUseFailure":
        content = hook.error or "Tool execution failed."
    payload = {
        "phase": "before" if hook.hook_event_name == "PreToolUse" else "after",
        "can_rewrite": hook.hook_event_name == "PreToolUse"
        or (adapter == "claude" and hook.hook_event_name == "PostToolUse"),
        "message": {
            "role": "tool",
            "tool_call_id": hook.tool_use_id,
            "name": hook.tool_name,
            "content": content,
        },
    }
    try:
        request = Request(
            os.environ["VF_TOOL_INTERCEPTION_URL"],
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + os.environ["VF_TOOL_INTERCEPTION_SECRET"],
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=30) as response:
            decision = ToolDecision.model_validate_json(response.read())
    except Exception:  # noqa: BLE001 - native hooks must fail closed
        decision = ToolDecision(
            action="stop", reason="Tool interception is unavailable."
        )
    try:
        output = native_output(adapter, hook, decision)
    except Exception:  # noqa: BLE001 - malformed decisions must also fail closed
        output = native_output(
            adapter,
            hook,
            ToolDecision(action="stop", reason="Tool interception is unavailable."),
        )
    if output:
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
