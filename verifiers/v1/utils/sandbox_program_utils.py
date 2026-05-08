from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from typing import Any, cast

from verifiers.utils.interception_utils import serialize_tool_defs

from ..runtime import Runtime, serializable
from ..state import State
from ..task import Task
from .sandbox_utils import (
    VF_STATE_INPUT_PATH_KEY,
    python_runtime_command,
    python_runtime_setup_command,
    run_sandbox_command,
)

TASK_PATH = "/tmp/vf_task.json"
STATE_INPUT_PATH = "/tmp/vf_state_in.json"
STATE_OUTPUT_PATH = "/tmp/vf_state_out.json"
RUNNER_CONFIG_PATH = "/tmp/vf_runner_config.json"
TOOL_DEFS_PATH = "/tmp/vf_tool_defs.json"
TOOL_DEFS_BY_PROTOCOL_PATH = "/tmp/vf_tool_defs_by_protocol.json"
RUNNER_PATH = "/tmp/vf_program_runner.py"
STATE_ARTIFACT = "__vf_state"
PYTHON_PROGRAM_PACKAGES = ("openai", "anthropic", "requests")


def python_program_sandbox(sandbox_config: Mapping[str, object]) -> dict[str, object]:
    config = dict(sandbox_config)
    raw_packages = config.get("packages") or []
    if isinstance(raw_packages, str):
        packages = shlex.split(raw_packages)
    elif isinstance(raw_packages, list):
        packages = [str(package) for package in raw_packages]
    else:
        raise TypeError("sandbox.packages must be a list or string.")
    for package in PYTHON_PROGRAM_PACKAGES:
        if not any(is_python_package(existing, package) for existing in packages):
            packages.append(package)
    config["packages"] = packages
    return config


def is_python_package(requirement: str, package: str) -> bool:
    return (
        requirement == package
        or requirement.startswith(f"{package}[")
        or requirement.startswith(f"{package}=")
        or requirement.startswith(f"{package}<")
        or requirement.startswith(f"{package}>")
        or requirement.startswith(f"{package}~")
        or requirement.startswith(f"{package}!")
    )


async def run_sandbox_python_program(
    program: Mapping[str, object],
    sandbox_config: Mapping[str, object],
    task: Task,
    state: State,
    runtime: Runtime,
    mode: str,
    fn_ref: str | None,
    max_turns: int,
) -> State:
    runner_program = sandbox_runner_program(
        program=program,
        task=task,
        state=state,
        mode=mode,
        fn_ref=fn_ref,
        max_turns=max_turns,
        tool_defs=runtime.tool_defs(state),
    )
    command_record = state.get("command")
    await run_sandbox_command(runner_program, sandbox_config, task, state, runtime)
    output = state.get("artifacts", {}).pop(STATE_ARTIFACT, None)
    if not isinstance(output, Mapping):
        raise RuntimeError("Sandbox Python program did not return state.")
    patch = dict(cast(Mapping[str, Any], output))
    apply_internal_state_patch(state, patch, mode=mode)
    patch_artifacts = patch.pop("artifacts", None)
    if isinstance(patch_artifacts, Mapping):
        state.setdefault("artifacts", {})
        state["artifacts"].update(dict(patch_artifacts))
    state.update(patch)
    if command_record is not None:
        state["command"] = command_record
    return state


def apply_internal_state_patch(
    state: State, patch: dict[str, Any], *, mode: str
) -> None:
    for key in State.INTERNAL_KEYS:
        if key not in patch:
            continue
        value = patch.pop(key)
        if value == state.get(key):
            continue
        if mode != "base" or key == "is_completed":
            raise RuntimeError(
                f"Sandbox Python program cannot set framework-managed state key {key!r}."
            )
        if key == "stop_condition":
            state._set_stop_condition(cast(str | None, value), overwrite=True)
        elif key == "is_truncated":
            state._set_truncated(bool(value), overwrite=True)
        elif key == "error":
            state._set_error(value)
        else:
            raise RuntimeError(
                f"Sandbox Python program cannot set framework-managed state key {key!r}."
            )


def sandbox_runner_program(
    program: Mapping[str, object],
    task: Task,
    state: State,
    mode: str,
    fn_ref: str | None,
    max_turns: int,
    tool_defs: object,
) -> dict[str, object]:
    files = dict(cast(Mapping[str, object], program.get("files") or {}))
    files[TASK_PATH] = json.dumps(task)
    files[TOOL_DEFS_PATH] = json.dumps(
        serializable(serialize_tool_defs(tool_defs or [], "openai_chat_completions"))
    )
    files[TOOL_DEFS_BY_PROTOCOL_PATH] = json.dumps(
        {
            protocol: serializable(serialize_tool_defs(tool_defs or [], protocol))
            for protocol in (
                "vf",
                "openai_chat_completions",
                "openai_responses",
                "anthropic_messages",
            )
        }
    )
    files[RUNNER_PATH] = runner_source()
    files[RUNNER_CONFIG_PATH] = json.dumps({"max_turns": max_turns})
    artifacts = dict(cast(Mapping[str, object], program.get("artifacts") or {}))
    artifacts[STATE_ARTIFACT] = {"path": STATE_OUTPUT_PATH, "format": "json"}
    command = python_runtime_command(
        RUNNER_PATH, *([mode] if fn_ref is None else [mode, fn_ref])
    )
    setup = program.get("setup") or []
    if isinstance(setup, str):
        setup = [setup]
    if not isinstance(setup, list):
        setup = [setup]
    return {
        **dict(program),
        "files": files,
        "command": command,
        "env": dict(cast(Mapping[str, object], program.get("env") or {})),
        "setup": [python_runtime_setup_command(), *setup],
        "artifacts": artifacts,
        VF_STATE_INPUT_PATH_KEY: STATE_INPUT_PATH,
    }


def runner_source() -> str:
    return r"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys

import requests
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

TASK_PATH = "/tmp/vf_task.json"
STATE_INPUT_PATH = "/tmp/vf_state_in.json"
STATE_OUTPUT_PATH = "/tmp/vf_state_out.json"
RUNNER_CONFIG_PATH = "/tmp/vf_runner_config.json"
TOOL_DEFS_PATH = "/tmp/vf_tool_defs.json"
TOOL_DEFS_BY_PROTOCOL_PATH = "/tmp/vf_tool_defs_by_protocol.json"


class Client:
    def __init__(self, state):
        self.openai = AsyncOpenAI(
            api_key=endpoint_api_key(),
            base_url=os.environ.get("OPENAI_BASE_URL")
            or state["endpoint_base_url"],
        )
        self.anthropic = AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
            or endpoint_api_key(),
            base_url=os.environ.get("ANTHROPIC_BASE_URL")
            or state["endpoint_root_url"],
        )
        self.chat = self.openai.chat
        self.responses = self.openai.responses
        self.messages = self.anthropic.messages

    async def close(self):
        await self.openai.close()
        await self.anthropic.close()


def endpoint_api_key():
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "intercepted"


def endpoint_headers():
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "python-requests/2.32.3",
        "Authorization": f"Bearer {endpoint_api_key()}",
    }


def vf_url(state, path):
    return f"{state['endpoint_root_url'].rstrip('/')}/vf/{path}"


def endpoint_timeout():
    return 300.0


def post_json(url, payload, headers=None):
    response = requests.post(
        url,
        json=payload,
        headers=headers or endpoint_headers(),
        timeout=endpoint_timeout(),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(response.text) from exc
    if not response.content:
        return {}
    return response.json()


async def vf_post(state, path, payload):
    return await asyncio.to_thread(
        post_json, vf_url(state, path), payload, endpoint_headers()
    )


async def call_tool(state, name, arguments):
    payload = await vf_post(state, f"tools/{name}", {"arguments": arguments})
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload.get("result")


async def call_user(state, transcript):
    payload = await vf_post(state, "user", {"transcript": transcript})
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload.get("messages") or []


def set_stop_condition(state, value, *, overwrite=False):
    if overwrite or state.get("stop_condition") is None:
        state["stop_condition"] = value


async def check_stop(state):
    payload = await vf_post(state, "stop", {})
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    if payload.get("done"):
        if payload.get("stop_condition"):
            set_stop_condition(state, payload["stop_condition"])
        return True
    return False


async def maybe_call(fn, **objects):
    sig = inspect.signature(fn)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        result = fn(**objects)
    else:
        result = fn(**{key: value for key, value in objects.items() if key in sig.parameters})
    if inspect.isawaitable(result):
        return await result
    return result


def import_ref(ref):
    module_name, _, attr_path = ref.partition(":")
    obj = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def to_plain(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "model_dump_json"):
        return json.loads(value.model_dump_json(exclude_none=True))
    return json.loads(json.dumps(value))


def message_from_response(response):
    choice = response.choices[0]
    message = choice.message
    data = {"role": getattr(message, "role", "assistant")}
    content = getattr(message, "content", None)
    if content is not None:
        data["content"] = content
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = [
            {
                "id": call.id,
                "type": getattr(call, "type", "function"),
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
    return data


def tool_call_name(tool_call):
    return tool_call["function"]["name"]


def tool_call_arguments(tool_call):
    raw = tool_call["function"].get("arguments") or "{}"
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def tool_error_content(error):
    return str(error)


def client_type(state):
    return state.get("runtime", {}).get("client_type") or "openai_chat_completions"


def sampling_args(state):
    raw = state.get("runtime", {}).get("sampling_args") or {}
    if not isinstance(raw, dict):
        raise RuntimeError("state.runtime.sampling_args must be a mapping.")
    return dict(raw)


def model_name(state):
    model = state.get("runtime", {}).get("model")
    if not model:
        raise RuntimeError("sandbox base program requires state.runtime.model.")
    return model


def load_tool_defs(protocol):
    defs = json.loads(open(TOOL_DEFS_BY_PROTOCOL_PATH).read())
    return defs.get(protocol) or []


class ToolProxy:
    def __init__(self, state, name, description=None):
        self.state = state
        self.name = name
        self.__name__ = name
        self.__doc__ = description or ""

    async def __call__(self, **arguments):
        return await call_tool(self.state, self.name, arguments)


def load_tools(state):
    return {
        tool["name"]: ToolProxy(state, tool["name"], tool.get("description"))
        for tool in load_tool_defs("vf")
    }


def response_input(messages):
    items = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": str(message.get("content") or ""),
                }
            )
            continue
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            if message.get("content"):
                items.append({"role": "assistant", "content": message["content"]})
            for tool_call in tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"].get("arguments") or "{}",
                    }
                )
            continue
        items.append({"role": role or "user", "content": message.get("content") or ""})
    return items


def anthropic_payload_messages(messages):
    payload_messages = []
    system = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if content:
                system.append(str(content))
            continue
        if role == "tool":
            payload_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": str(content or ""),
                        }
                    ],
                }
            )
            continue
        if role == "assistant":
            blocks = []
            if content:
                blocks.append({"type": "text", "text": str(content)})
            for tool_call in message.get("tool_calls") or []:
                arguments = tool_call["function"].get("arguments") or "{}"
                try:
                    tool_input = json.loads(arguments)
                except json.JSONDecodeError:
                    tool_input = {"arguments": arguments}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": tool_input,
                    }
                )
            payload_messages.append({"role": "assistant", "content": blocks or ""})
            continue
        payload_messages.append({"role": "user", "content": str(content or "")})
    return "\n".join(system), payload_messages


def message_from_responses_response(response):
    message = {"role": "assistant"}
    text_parts = []
    tool_calls = []
    for item in response.get("output") or []:
        if item.get("type") == "message":
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"}:
                    text_parts.append(str(content.get("text") or ""))
        elif item.get("type") == "function_call":
            call_id = item.get("call_id") or item.get("id")
            if call_id:
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": item.get("arguments") or "{}",
                        },
                    }
                )
    if text_parts:
        message["content"] = "\n".join(text_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def message_from_anthropic_response(response):
    message = {"role": "assistant"}
    text_parts = []
    tool_calls = []
    for block in response.get("content") or []:
        if block.get("type") == "text":
            text_parts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
    if text_parts:
        message["content"] = "\n".join(text_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


async def create_model_message(state, messages, client):
    protocol = client_type(state)
    sampling = sampling_args(state)
    model = model_name(state)
    if protocol == "openai_chat_completions":
        payload = {"model": model, "messages": messages, **sampling}
        tool_defs = load_tool_defs(protocol)
        if tool_defs:
            payload["tools"] = tool_defs
        response = await client.chat.completions.create(**payload)
        return message_from_response(response)
    if protocol == "openai_responses":
        payload = {"model": model, "input": response_input(messages), **sampling}
        tool_defs = load_tool_defs(protocol)
        if tool_defs:
            payload["tools"] = tool_defs
        response = await client.responses.create(**payload)
        return message_from_responses_response(to_plain(response))
    if protocol == "anthropic_messages":
        system, provider_messages = anthropic_payload_messages(messages)
        if "max_tokens" in sampling:
            max_tokens = int(sampling.pop("max_tokens"))
        else:
            max_tokens = int(sampling.pop("max_completion_tokens", 4096))
        payload = {
            "model": model,
            "messages": provider_messages,
            "max_tokens": max_tokens,
            **sampling,
        }
        if system:
            payload["system"] = system
        tool_defs = load_tool_defs(protocol)
        if tool_defs:
            payload["tools"] = tool_defs
        response = await client.messages.create(**payload)
        return message_from_anthropic_response(to_plain(response))
    raise RuntimeError(f"Unsupported sandbox base client type: {protocol}")


async def run_base(task, state, client):
    prompt_messages = [*(state.get("system_prompt") or []), *(state.get("prompt") or [])]
    messages = list(prompt_messages)
    config = json.loads(open(RUNNER_CONFIG_PATH).read())
    max_turns = int(config["max_turns"])
    turn = 0
    while max_turns <= 0 or turn < max_turns:
        if await check_stop(state):
            break
        message = await create_model_message(state, messages, client)
        turn += 1
        messages.append(message)
        tool_calls = list(message.get("tool_calls") or [])
        if not tool_calls:
            user_messages = await call_user(state, messages)
            if user_messages:
                messages.extend(user_messages)
                continue
            set_stop_condition(state, "no_tools")
            break
        for tool_call in tool_calls:
            try:
                result = await call_tool(
                    state, tool_call_name(tool_call), tool_call_arguments(tool_call)
                )
                content = str(result)
            except Exception as exc:
                content = tool_error_content(exc)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": content,
                }
            )
            if await check_stop(state):
                completed = True
                break
        else:
            completed = False
        if completed:
            break
    state["completion"] = messages[len(prompt_messages):]
    set_stop_condition(state, "max_turns_reached")
    return state


async def main():
    mode = sys.argv[1]
    task = json.loads(open(TASK_PATH).read())
    state = json.loads(open(STATE_INPUT_PATH).read())
    original_state = json.loads(json.dumps(state))
    client = Client(state)
    try:
        if mode == "base":
            result = await run_base(task, state, client)
        elif mode == "fn":
            result = await maybe_call(
                import_ref(sys.argv[2]),
                task=task,
                state=state,
                client=client,
                tools=load_tools(state),
                tool_defs=load_tool_defs("vf"),
            )
        else:
            raise ValueError(f"Unknown sandbox program mode: {mode}")
    finally:
        await client.close()
    if result is not None:
        if not isinstance(result, dict):
            raise TypeError("Sandbox Python program must return None or a mapping.")
        state.update(result)
    patch = {
        key: value
        for key, value in state.items()
        if key not in original_state or original_state[key] != value
    }
    with open(STATE_OUTPUT_PATH, "w") as f:
        json.dump(patch, f)


asyncio.run(main())
"""
