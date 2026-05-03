from __future__ import annotations

import asyncio
import os
import shlex
from collections.abc import Mapping
from typing import Any, cast

from verifiers.errors import InfraError
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.utils.interception_utils import serialize_tool_defs

from ..runtime import Runtime, _read_path, serializable
from ..state import State
from ..task import Task
from .mcp_proxy_utils import local_proxy_program, validate_tool_protocol


async def run_local_command(
    program: Mapping[str, object], task: Task, state: State, runtime: Runtime
) -> State:
    cleanup_path = None
    if runtime_tool_protocol(runtime) == "mcp":
        program, cleanup_path = local_proxy_program(program)
    argv = await command_argv(program, task, state, runtime)
    try:
        env = await command_env(program, task, state, runtime, include_base=True)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)
    state["command"] = {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }
    state["completion"] = [
        {"role": "assistant", "content": state["command"]["stdout"].strip()}
    ]
    if proc.returncode:
        raise InfraError(
            f"Command exited with {proc.returncode}: {state['command']['stderr']}"
        )
    state["stop_condition"] = state.get("stop_condition") or "command_completed"
    return state


async def command_argv(
    program: Mapping[str, object], task: Task, state: State, runtime: Runtime
) -> list[str]:
    command = program.get("command")
    if isinstance(command, str):
        argv = shlex.split(command)
    elif isinstance(command, list):
        argv = [
            str(await resolve_program_value(part, task, state, runtime))
            for part in command
        ]
    else:
        raise TypeError("program.command must be a string or list.")
    args = program.get("args", [])
    if not isinstance(args, list):
        raise TypeError("program.args must be a list.")
    for arg in args:
        argv.append(str(await resolve_program_value(arg, task, state, runtime)))
    if not argv:
        raise ValueError("program.command cannot be empty.")
    return argv


async def command_env(
    program: Mapping[str, object],
    task: Task,
    state: State,
    runtime: Runtime,
    include_base: bool,
) -> dict[str, str]:
    env = dict(os.environ) if include_base else {}
    endpoint_base_url = state.get("endpoint_base_url")
    if isinstance(endpoint_base_url, str):
        api_key = endpoint_api_key(runtime)
        endpoint_root_url = state.get("endpoint_root_url")
        env.setdefault("OPENAI_BASE_URL", endpoint_base_url)
        env.setdefault("OPENAI_API_KEY", api_key)
        env.setdefault("VF_ENDPOINT_API_KEY", api_key)
        if isinstance(endpoint_root_url, str):
            env.setdefault("ANTHROPIC_BASE_URL", endpoint_root_url)
            env.setdefault("ANTHROPIC_API_KEY", api_key)
            env.setdefault("VF_TOOL_BASE_URL", f"{endpoint_root_url}/vf/tools")
            env.setdefault("VF_TOOL_API_KEY", api_key)
        env.setdefault("VF_ENDPOINT_BASE_URL", endpoint_base_url)
        client_type = str(
            state.get("runtime", {}).get("client_type") or "openai_chat_completions"
        )
        tool_defs = runtime.tool_defs(state) or []
        env.setdefault("VF_CLIENT_TYPE", client_type)
        env.setdefault("VF_TOOL_PROTOCOL", runtime_tool_protocol(runtime))
        env.setdefault(
            "VF_TOOLS_JSON",
            json_dumps(serialize_tool_defs(tool_defs, client_type)),
        )
        env.setdefault(
            "VF_TOOL_DEFS_JSON",
            json_dumps(serialize_tool_defs(tool_defs, "vf")),
        )
    raw_env = program.get("env", {})
    if not isinstance(raw_env, Mapping):
        raise TypeError("program.env must be a mapping.")
    for key, value in raw_env.items():
        if not isinstance(key, str):
            raise TypeError("program.env keys must be strings.")
        env[key] = str(await resolve_program_value(value, task, state, runtime))
    return env


async def resolve_program_value(
    value: object, task: Task, state: State, runtime: Runtime
) -> object:
    _ = runtime
    if callable(value):
        return await maybe_call_with_named_args(value, task=task, state=state)
    if isinstance(value, str):
        root, separator, tail = value.partition(".")
        if separator and root == "task":
            return _read_path(task, tail)
        if separator and root == "state":
            return _read_path(state, tail)
        if separator and root == "runtime":
            return _read_path(state.get("runtime", {}), tail)
    if isinstance(value, Mapping):
        if len(value) != 1:
            raise ValueError("Program value mappings must have exactly one root.")
        root, path = next(iter(value.items()))
        if root == "task":
            return _read_path(task, str(path))
        if root == "state":
            return _read_path(state, str(path))
        if root == "runtime":
            return _read_path(state.get("runtime", {}), str(path))
        raise ValueError(f"Unknown program value root {root!r}.")
    return value


def float_config(config: Mapping[str, object], key: str, default: float) -> float:
    value = config.get(key)
    return default if value is None else float(cast(Any, value))


def int_config(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key)
    return default if value is None else int(cast(Any, value))


def endpoint_api_key(runtime: Runtime) -> str:
    harness = getattr(runtime, "harness", None)
    endpoint = getattr(harness, "endpoint", None)
    secret = getattr(endpoint, "secret", None)
    return str(secret or "intercepted")


def runtime_tool_protocol(runtime: Runtime) -> str:
    return validate_tool_protocol(
        getattr(getattr(runtime, "harness", None), "tool_protocol", "callable")
    )


def json_dumps(value: object) -> str:
    import json

    return json.dumps(serializable(value))
