import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

import verifiers as vf
from verifiers.v1.runtime import Runtime
from verifiers.v1.state import State

DEEP_AGENTS_SCAFFOLDING_PROMPT = """\
You have access to deep-agent scaffolding tools (`write_todos`, `write_file`, \
`read_file`, `ls`, `edit_file`, `task`). Use them when they help: sketch a plan \
with `write_todos`, jot intermediate notes or findings in a file, and call \
`task` to spawn a focused sub-agent for a sub-problem. They are entirely \
optional."""


class AgentMessage(Protocol):
    role: str
    content: object


def deep_agent_system_prompt(state: State) -> str:
    system_prompt_messages = state.get("system_prompt")
    if not isinstance(system_prompt_messages, list):
        return ""
    return "\n\n".join(
        str(message.content or "")
        for message in vf.get_messages(system_prompt_messages)
    )


def runtime_tool_coroutine(
    runtime_tool: Callable[..., Awaitable[object]],
) -> Callable[..., Awaitable[str]]:
    async def call(**kwargs: Any) -> str:
        return str(await runtime_tool(**kwargs))

    return call


def langchain_tools_from_state(state: State, runtime: Runtime) -> list[Any]:
    """Convert the rollout's exposed runtime tools into LangChain tools.

    Each tool definition's name, description, and JSON-schema parameters become a
    LangChain ``StructuredTool`` whose coroutine dispatches back into the
    matching Verifiers runtime callable.
    """
    from langchain_core.tools import StructuredTool  # ty: ignore[unresolved-import]

    tool_defs = runtime.tool_defs(state) or []
    callables = state.get_tools()
    tools: list[Any] = []
    for tool_def in tool_defs:
        runtime_tool = callables.get(tool_def.name)
        if runtime_tool is None:
            continue
        parameters = dict(tool_def.parameters) or {"type": "object", "properties": {}}
        tools.append(
            StructuredTool.from_function(
                coroutine=runtime_tool_coroutine(runtime_tool),
                name=tool_def.name,
                description=tool_def.description or f"Call the {tool_def.name} tool.",
                args_schema=parameters,
            )
        )
    return tools


def serialize_agent_completion(
    messages: Sequence[AgentMessage | vf.JsonData],
) -> list[vf.JsonData]:
    role_aliases = {
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
        "system": "system",
    }
    call_names: dict[str, str] = {}
    serialized: list[vf.JsonData] = []
    for message in messages:
        if isinstance(message, dict):
            payload = dict(message)
        else:
            model_dump = getattr(message, "model_dump", None)
            payload = (
                model_dump(mode="json", exclude_none=True)
                if callable(model_dump)
                else {
                    "role": getattr(message, "role", None)
                    or getattr(message, "type", "assistant"),
                    "content": getattr(message, "content", str(message)),
                    "name": getattr(message, "name", None),
                    "tool_call_id": getattr(message, "tool_call_id", None),
                    "tool_calls": getattr(message, "tool_calls", None),
                }
            )
        raw_role = payload.get("role") or payload.get("type") or "assistant"
        role = role_aliases.get(str(raw_role), str(raw_role))
        item: vf.JsonData = {
            "role": role,
            "content": payload.get("content", ""),
        }
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized_tool_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_payload = dict(tool_call)
                name = tool_call_payload.get("name")
                tool_id = tool_call_payload.get("id") or tool_call_payload.get(
                    "tool_call_id"
                )
                if isinstance(tool_id, str) and isinstance(name, str):
                    call_names[tool_id] = name
                arguments = tool_call_payload.get("arguments")
                if not isinstance(arguments, str):
                    args = tool_call_payload.get("args", {})
                    try:
                        arguments = json.dumps(args if args is not None else {})
                    except (TypeError, ValueError):
                        arguments = str(args)
                    tool_call_payload["arguments"] = arguments
                normalized_tool_calls.append(tool_call_payload)
            item["tool_calls"] = normalized_tool_calls
        name = payload.get("name")
        if isinstance(name, str):
            item["name"] = name
        tool_call_id = payload.get("tool_call_id")
        if isinstance(tool_call_id, str):
            item["tool_call_id"] = tool_call_id
            if item["role"] == "tool" and "name" not in item:
                name = call_names.get(tool_call_id)
                if name is not None:
                    item["name"] = name
        serialized.append(item)
    if serialized and serialized[0].get("role") == "user":
        return serialized[1:]
    return serialized
