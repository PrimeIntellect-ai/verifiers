# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Bridge Pool's native command hooks to rollout tool policy."""

import json
import os
import signal
import sys
from pathlib import Path
from urllib.request import Request, urlopen

HOOK_NAME = "verifiers-tool-interception"
POST_SUFFIX = (
    f'\n<hook name="{HOOK_NAME}" event="PostToolUse">'
    "modified this observation before it was shown to you</hook>"
)
MAX_RESULT_BYTES = 32 * 1024

try:
    hook_event = json.load(sys.stdin)
    if hook_event.get("hook_api_version") != "1.0":
        raise ValueError("Pool returned an unsupported hook API version")
    event_name = hook_event.get("hook_event_name")
    if event_name not in ("PreToolUse", "PostToolUse"):
        raise ValueError(f"Pool returned unsupported hook event {event_name!r}")
    tool_name = hook_event.get("tool_name")
    call_id = hook_event.get("tool_call_id")
    if not isinstance(tool_name, str) or not isinstance(call_id, str):
        raise TypeError("Pool returned an invalid tool identity")
    # Pool invokes an internal exit action after the model has already completed.
    # It has no issuing model tool call and therefore no policy boundary to cross.
    if tool_name == "exit":
        raise SystemExit(0)

    credentials = json.loads(Path(sys.argv[1]).read_text())
    url = credentials.get("url")
    secret = credentials.get("secret")
    if not isinstance(url, str) or not isinstance(secret, str):
        raise TypeError("Pool tool interception credentials are invalid")

    before = event_name == "PreToolUse"
    content = "" if before else hook_event.get("tool_output")
    if not isinstance(content, str):
        raise TypeError("Pool tool interception supports only text results")
    request_body = {
        "phase": "before" if before else "after",
        "content": "nonempty_text",
        "resultSuffix": "" if before else POST_SUFFIX,
        "message": {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
            "name": tool_name,
        },
    }
    request = Request(
        url,
        data=json.dumps(request_body).encode(),
        headers={
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        decision = json.load(response)
    action = decision.get("action") if isinstance(decision, dict) else None
    if action not in ("allow", "rewrite", "stop"):
        raise ValueError("tool interception returned an invalid decision")

    if action == "allow":
        raise SystemExit(0)
    if action == "rewrite":
        message = decision.get("message")
        replacement = message.get("content") if isinstance(message, dict) else None
        if not isinstance(replacement, str) or not replacement:
            raise TypeError(
                "Pool tool interception requires non-empty text replacements"
            )
        if message.get("name", tool_name) != tool_name:
            raise ValueError("Pool tool interception cannot replace the tool name")
    else:
        replacement = decision.get("reason") or "Rollout terminated by interception."
    if not isinstance(replacement, str):
        raise TypeError("Pool tool interception returned a non-text result")
    if not before:
        # RolloutSession applies the same normalization before adding Pool's
        # deterministic hook notice to the canonical delivered result.
        replacement = replacement.strip()
        if not replacement:
            raise ValueError("Pool tool interception requires non-empty text results")
    suffix = "" if before else POST_SUFFIX
    if len((replacement + suffix).encode()) > MAX_RESULT_BYTES:
        raise ValueError("Pool tool replacement exceeds its 32 KiB model-visible limit")

    if before:
        output = {
            "hook_specific_output": {
                "permission_decision": "deny",
                "permission_decision_reason": replacement,
            }
        }
    else:
        output = {"hook_specific_output": {"updated_tool_output": replacement}}
    print(json.dumps(output))
except SystemExit:
    raise
except Exception as error:
    print(f"Tool interception is unavailable: {error}", file=sys.stderr)
    # Pool command hooks fail open. Killing their direct parent makes a bridge failure
    # terminate the ACP agent instead of admitting an unchecked tool boundary.
    os.kill(os.getppid(), signal.SIGTERM)
    raise
