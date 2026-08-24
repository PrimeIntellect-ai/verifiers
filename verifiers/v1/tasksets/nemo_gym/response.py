"""Convert Verifiers traces to NeMo Gym Responses objects."""

import json
from collections.abc import Collection
from typing import Any

from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, ToolMessage


def trace_to_nemo_response(
    trace: Trace,
    responses_create_params: dict[str, Any],
    tool_names: Collection[str],
) -> dict[str, Any]:
    """Convert the one completed V1 branch into a Gym Responses object."""

    branches = trace.branches
    if len(branches) != 1:
        raise ValueError(
            f"NeMo Gym scoring requires exactly one trace branch, got {len(branches)}"
        )

    output: list[dict[str, Any]] = []
    call_types: dict[str, str] = {}
    started = False

    for node in branches[0].nodes:
        message = node.message
        if isinstance(message, AssistantMessage) and node.sampled:
            started = True
            call_types.update((call.id, call.type) for call in message.tool_calls or [])
            if message.provider_state and all(
                item.get("type")
                in {"reasoning", "message", "function_call", "custom_tool_call"}
                for item in message.provider_state
            ):
                output.extend(message.provider_state)
                continue
            if message.reasoning_content:
                output.append(
                    {
                        "id": f"rs_{trace.id}_{len(output)}",
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": message.reasoning_content}
                        ],
                    }
                )
            if message.content:
                output.append(
                    {
                        "id": f"msg_{trace.id}_{len(output)}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": message.content,
                                "annotations": [],
                            }
                        ],
                    }
                )
            output.extend(
                {
                    "type": "custom_tool_call"
                    if call.type == "custom"
                    else "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    ("input" if call.type == "custom" else "arguments"): call.arguments,
                }
                for call in message.tool_calls or []
            )
        elif started and isinstance(message, ToolMessage):
            content = (
                message.content
                if isinstance(message.content, str)
                else json.dumps(
                    [part.model_dump(mode="json") for part in message.content]
                )
            )
            output.append(
                {
                    "type": "custom_tool_call_output"
                    if call_types.get(message.tool_call_id) == "custom"
                    else "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": content,
                }
            )

    known_names = sorted(tool_names, key=len, reverse=True)
    for index, item in enumerate(output):
        name = item.get("name")
        if item.get("type") != "function_call" or not isinstance(name, str):
            continue
        if name in tool_names:
            continue
        if name.startswith("mcp__"):
            for tool_name in known_names:
                if name.endswith(f"__{tool_name}"):
                    output[index] = item | {"name": tool_name}
                    break

    return {
        "id": f"resp_{trace.id}",
        "created_at": branches[0].nodes[-1].timestamp,
        "model": str(responses_create_params.get("model") or "verifiers"),
        "object": "response",
        "output": output,
        "parallel_tool_calls": responses_create_params.get("parallel_tool_calls", True),
        "tool_choice": responses_create_params.get("tool_choice", "auto"),
        "tools": responses_create_params.get("tools") or [],
        "status": "completed",
    }
