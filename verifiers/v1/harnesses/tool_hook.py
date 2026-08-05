# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "httpx>=0.27.0",
#   "pydantic>=2.12.3",
# ]
# ///
"""Send Claude and Codex tool hooks through Verifiers interception."""

import argparse
import json
import sys
from typing import Annotated, Literal, cast

import httpx
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    Field,
    JsonValue,
    SecretStr,
    TypeAdapter,
)


# The runtime uploads this script without the Verifiers package, so its public wire
# models live here as well as in verifiers.v1.interception.tool.
class ToolInterceptionRequest(BaseModel):
    tool_call_id: str
    name: str
    can_rewrite: bool
    phase: Literal["before", "after"] = "after"
    content: str | None = None


class ToolInterceptionDecision(BaseModel):
    action: Literal["allow", "rewrite", "terminate"]
    content: str | None = None
    reason: str | None = None


class ToolHookConfig(BaseModel):
    adapter: Literal["claude", "codex"]
    url: AnyHttpUrl
    secret: SecretStr


class PreToolUseEvent(BaseModel):
    hook_event_name: Literal["PreToolUse"]
    tool_use_id: str
    tool_name: str
    tool_input: JsonValue


class PostToolUseEvent(BaseModel):
    hook_event_name: Literal["PostToolUse"]
    tool_use_id: str
    tool_name: str
    tool_response: JsonValue


class PostToolUseFailureEvent(BaseModel):
    hook_event_name: Literal["PostToolUseFailure"]
    tool_use_id: str
    tool_name: str
    error: str


ToolHookEvent = Annotated[
    PreToolUseEvent | PostToolUseEvent | PostToolUseFailureEvent,
    Field(discriminator="hook_event_name"),
]
TOOL_HOOK_EVENT_ADAPTER = TypeAdapter(ToolHookEvent)


class StopHookOutput(BaseModel):
    continue_: Literal[False] = Field(False, serialization_alias="continue")
    stop_reason: str = Field(serialization_alias="stopReason")


class PreToolUseDecision(BaseModel):
    hook_event_name: Literal["PreToolUse"] = Field(
        "PreToolUse", serialization_alias="hookEventName"
    )
    permission_decision: Literal["deny"] = Field(
        "deny", serialization_alias="permissionDecision"
    )
    permission_decision_reason: str = Field(
        serialization_alias="permissionDecisionReason"
    )


class PreToolUseOutput(BaseModel):
    hook_specific_output: PreToolUseDecision = Field(
        serialization_alias="hookSpecificOutput"
    )


class PostToolUseRewrite(BaseModel):
    hook_event_name: Literal["PostToolUse"] = Field(
        "PostToolUse", serialization_alias="hookEventName"
    )
    updated_tool_output: str = Field(serialization_alias="updatedToolOutput")


class PostToolUseRewriteOutput(BaseModel):
    hook_specific_output: PostToolUseRewrite = Field(
        serialization_alias="hookSpecificOutput"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=("claude", "codex"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--secret", required=True)
    config = ToolHookConfig.model_validate(vars(parser.parse_args()))
    event = TOOL_HOOK_EVENT_ADAPTER.validate_json(sys.stdin.buffer.read())
    before = isinstance(event, PreToolUseEvent)
    native = (
        None
        if before
        else event.tool_response
        if isinstance(event, PostToolUseEvent)
        else event.error
    )
    request = ToolInterceptionRequest(
        tool_call_id=event.tool_use_id,
        name=event.tool_name,
        phase="before" if before else "after",
        content=native
        if native is None or isinstance(native, str)
        else json.dumps(native),
        # Both agents can deny before execution. Only Claude can replace plain post-tool output.
        can_rewrite=before
        or (
            config.adapter == "claude"
            and isinstance(event, PostToolUseEvent)
            and isinstance(native, str)
        ),
    )
    response = httpx.post(
        str(config.url),
        content=request.model_dump_json(),
        headers={
            "Authorization": f"Bearer {config.secret.get_secret_value()}",
            "Content-Type": "application/json",
        },
        timeout=None,
    )
    response.raise_for_status()
    decision = ToolInterceptionDecision.model_validate_json(response.content)

    if decision.action == "allow":
        print("{}")
        return
    elif decision.action == "rewrite" and before:
        output: BaseModel = PreToolUseOutput(
            hook_specific_output=PreToolUseDecision(
                permission_decision_reason=cast(str, decision.content)
            )
        )
    elif decision.action == "rewrite" and request.can_rewrite:
        output = PostToolUseRewriteOutput(
            hook_specific_output=PostToolUseRewrite(
                updated_tool_output=cast(str, decision.content)
            )
        )
    else:
        output = StopHookOutput(
            stop_reason=cast(
                str,
                decision.reason if decision.action == "terminate" else decision.content,
            )
        )
    print(output.model_dump_json(by_alias=True, exclude_none=True))


# Hook runners execute this file; the guard keeps its models safe to import.
if __name__ == "__main__":
    main()
