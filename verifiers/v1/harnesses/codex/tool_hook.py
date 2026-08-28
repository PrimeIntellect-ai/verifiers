# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["httpx", "mcp>=1.24,<2"]
# ///
"""Expose the rollout tool policy as a model-hidden Codex hook tool."""

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# {tool_interception}

credentials_path = Path(os.environ.pop("VF_TOOL_INTERCEPTION_CONFIG"))
credentials = json.loads(credentials_path.read_bytes())
credentials_path.unlink()
interceptor = ToolInterceptionClient(  # noqa: F821
    credentials["url"], credentials["secret"]
)
server = FastMCP("vf-interceptor")


def output(decision: dict) -> str:
    if decision["action"] == "allow":
        return "{}"
    replacement = decision.get("message", {}).get("content")
    reason = (
        replacement or decision.get("reason") or "Rollout terminated by interception."
    )
    if not isinstance(reason, str) or not reason:
        raise TypeError("Codex hooks require a non-empty text replacement")
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    )


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
    return output(decision)


try:
    server.run(transport="stdio")
finally:
    interceptor.close()
