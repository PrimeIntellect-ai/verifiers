# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["httpx", "mcp>=1.24,<2"]
# ///
"""Expose the rollout tool policy as a model-hidden Codex hook tool."""

import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# {tool_interception}

credentials_path = Path(os.environ.pop("VF_TOOL_INTERCEPTION_CONFIG"))
credentials = json.loads(credentials_path.read_bytes())
credentials_path.unlink()
interceptor = ToolInterceptionClient(  # noqa: F821
    credentials["url"], credentials["secret"]
)
server = FastMCP("vf-interceptor")


def content(value: Any) -> str | list[dict]:
    parts = value.get("content") if isinstance(value, dict) else None
    if not isinstance(parts, list):
        return (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
    result = []
    for part in parts:
        if part.get("type") == "text":
            result.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image":
            result.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{part.get('mimeType', 'image/png')};base64,"
                            f"{part.get('data', '')}"
                        )
                    },
                }
            )
        else:
            return json.dumps(value, ensure_ascii=False)
    return (
        "\n".join(part["text"] for part in result)
        if all(part["type"] == "text" for part in result)
        else result
    )


def output(phase: str, decision: dict) -> str:
    if decision["action"] == "allow":
        return "{}"
    replacement = decision.get("message", {}).get("content")
    reason = (
        replacement or decision.get("reason") or "Rollout terminated by interception."
    )
    if not isinstance(reason, str) or not reason:
        raise TypeError("Codex hooks require a non-empty text replacement")
    if phase == "before":
        return json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    return json.dumps({"decision": "block", "reason": reason})


@server.tool()
def before(tool_name: str, tool_use_id: str, tool_input: dict) -> str:
    body = {
        "phase": "before",
        "content": "nonempty_text",
        "detachedParent": "exec",
        "message": {
            "role": "tool",
            "tool_call_id": tool_use_id,
            "content": "",
            "name": tool_name,
        },
    }
    command = tool_input.get("command")
    if tool_name in {"Bash", "apply_patch"} and isinstance(command, str):
        body["rewrite_prefix"] = "Command blocked by PreToolUse hook: "
        body["rewrite_suffix"] = f". Command: {command}"
    else:
        body["rewrite_prefix"] = "Tool call blocked by PreToolUse hook: "
        body["rewrite_suffix"] = f". Tool: {tool_name}"
    decision = interceptor.request(body)
    return output("before", decision)


@server.tool()
def after(
    tool_name: str, tool_use_id: str, tool_input: dict, tool_response: Any
) -> str:
    decision = interceptor.request(
        {
            "phase": "after",
            "content": "nonempty_text",
            "detachedParent": "exec",
            "message": {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": content(tool_response),
                "name": tool_name,
            },
        }
    )
    return output("after", decision)


try:
    server.run(transport="stdio")
finally:
    interceptor.close()
