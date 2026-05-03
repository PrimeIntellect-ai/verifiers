from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from ..runtime import Runtime, serializable
from ..state import State
from ..task import Task
from .sandbox_utils import run_sandbox_command

TASK_PATH = "/tmp/vf_task.json"
STATE_INPUT_PATH = "/tmp/vf_state_in.json"
STATE_OUTPUT_PATH = "/tmp/vf_state_out.json"
TOOL_DEFS_PATH = "/tmp/vf_tool_defs.json"
RUNNER_PATH = "/tmp/vf_program_runner.py"
STATE_ARTIFACT = "__vf_state"


async def run_sandbox_python_program(
    program: Mapping[str, object],
    sandbox_config: Mapping[str, object],
    task: Task,
    state: State,
    runtime: Runtime,
    mode: str,
    entrypoint: str | None,
    max_turns: int,
) -> State:
    runner_program = sandbox_runner_program(
        program=program,
        task=task,
        state=state,
        mode=mode,
        entrypoint=entrypoint,
        max_turns=max_turns,
        tool_defs=runtime.tool_defs(state),
    )
    command_record = state.get("command")
    await run_sandbox_command(runner_program, sandbox_config, task, state, runtime)
    output = state.get("artifacts", {}).pop(STATE_ARTIFACT, None)
    if not isinstance(output, Mapping):
        raise RuntimeError("Sandbox Python program did not return state.")
    patch = dict(cast(Mapping[str, Any], output))
    patch_artifacts = patch.pop("artifacts", None)
    if isinstance(patch_artifacts, Mapping):
        state.setdefault("artifacts", {})
        state["artifacts"].update(dict(patch_artifacts))
    state.update(patch)
    if command_record is not None:
        state["command"] = command_record
    return state


def sandbox_runner_program(
    program: Mapping[str, object],
    task: Task,
    state: State,
    mode: str,
    entrypoint: str | None,
    max_turns: int,
    tool_defs: object,
) -> dict[str, object]:
    files = dict(cast(Mapping[str, object], program.get("files") or {}))
    files[TASK_PATH] = json.dumps(task)
    files[STATE_INPUT_PATH] = json.dumps(state)
    files[TOOL_DEFS_PATH] = json.dumps(serializable(tool_defs or []))
    files[RUNNER_PATH] = runner_source()
    artifacts = dict(cast(Mapping[str, object], program.get("artifacts") or {}))
    artifacts[STATE_ARTIFACT] = {"path": STATE_OUTPUT_PATH, "format": "json"}
    command = ["python", RUNNER_PATH, mode]
    if entrypoint is not None:
        command.append(entrypoint)
    env = dict(cast(Mapping[str, object], program.get("env") or {}))
    env["VF_MAX_TURNS"] = str(max_turns)
    return {
        **dict(program),
        "files": files,
        "command": command,
        "env": env,
        "artifacts": artifacts,
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
import urllib.error
import urllib.request
from types import SimpleNamespace

TASK_PATH = "/tmp/vf_task.json"
STATE_INPUT_PATH = "/tmp/vf_state_in.json"
STATE_OUTPUT_PATH = "/tmp/vf_state_out.json"
TOOL_DEFS_PATH = "/tmp/vf_tool_defs.json"


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [namespace(item) for item in value]
    return value


class ChatCompletions:
    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def create(self, **payload):
        return namespace(await asyncio.to_thread(self._create, payload))

    def _create(self, payload):
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"model request failed: {detail}") from exc


class Chat:
    def __init__(self, base_url, api_key):
        self.completions = ChatCompletions(base_url, api_key)


class Client:
    def __init__(self, base_url, api_key):
        self.chat = Chat(base_url, api_key)


async def call_tool(state, name, arguments):
    data = json.dumps({"arguments": arguments}).encode()
    request = urllib.request.Request(
        f"{state['endpoint_root_url'].rstrip('/')}/vf/tools/{name}",
        data=data,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {os.environ.get('OPENAI_API_KEY') or 'intercepted'}",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"tool request failed: {detail}") from exc
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload.get("result")


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


async def run_base(task, state, client):
    if state.get("runtime", {}).get("client_type") not in (None, "openai_chat_completions"):
        raise RuntimeError("sandbox base program currently supports OpenAI Chat Completions.")
    messages = list(state.get("prompt") or [])
    tool_defs = json.loads(open(TOOL_DEFS_PATH).read())
    max_turns = int(os.environ.get("VF_MAX_TURNS") or "10")
    model = state.get("runtime", {}).get("model") or state.get("model")
    if not model:
        raise RuntimeError("sandbox base program requires state.runtime.model.")
    for _ in range(max_turns):
        payload = {"model": model, "messages": messages}
        if tool_defs:
            payload["tools"] = tool_defs
        response = await client.chat.completions.create(**payload)
        message = message_from_response(response)
        messages.append(message)
        tool_calls = list(message.get("tool_calls") or [])
        if not tool_calls:
            break
        for tool_call in tool_calls:
            result = await call_tool(
                state, tool_call_name(tool_call), tool_call_arguments(tool_call)
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(result),
                }
            )
    state["completion"] = messages[len(state.get("prompt") or []):]
    state["stop_condition"] = state.get("stop_condition") or "no_tools"
    return state


async def main():
    mode = sys.argv[1]
    task = json.loads(open(TASK_PATH).read())
    state = json.loads(open(STATE_INPUT_PATH).read())
    original_state = json.loads(json.dumps(state))
    client = Client(
        state["endpoint_base_url"],
        os.environ.get("OPENAI_API_KEY") or "intercepted",
    )
    if mode == "base":
        result = await run_base(task, state, client)
    elif mode == "entrypoint":
        result = await maybe_call(import_ref(sys.argv[2]), task=task, state=state, client=client)
    else:
        raise ValueError(f"Unknown sandbox program mode: {mode}")
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
