from __future__ import annotations

import asyncio
import os
import shlex
from collections.abc import Mapping
from typing import Any, Callable, cast

from verifiers.errors import InfraError
from verifiers.utils.async_utils import maybe_call_with_named_args

from ..config import resolve_config_object
from ..runtime import (
    Runtime,
    _read_path,
    binding_key_parts,
    binding_source_root,
    function_name,
    validate_binding_source_root,
    validate_bound_arg,
)
from ..state import State
from ..task import Task
from .mcp_proxy_utils import ProgramToolType, validate_program_tool_types


async def run_local_command(
    program: Mapping[str, object], task: Task, state: State, runtime: Runtime
) -> State:
    if "mcp" in program_tool_types(program):
        raise ValueError("program.tools='mcp' requires sandbox command placement.")
    validate_program_bindings(program)
    argv = await command_argv(program, task, state, runtime)
    env = await command_env(program, task, state, runtime, include_base=True)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()
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
    state._set_stop_condition("command_completed")
    return state


async def command_argv(
    program: Mapping[str, object], task: Task, state: State, runtime: Runtime
) -> list[str]:
    command = program.get("command")
    if isinstance(command, str):
        argv = shlex.split(command)
    elif isinstance(command, list):
        argv = [
            str(await resolve_program_value(part, task, state, runtime, program))
            for part in command
        ]
    else:
        raise TypeError("program.command must be a string or list.")
    args = program.get("args", [])
    if not isinstance(args, list):
        raise TypeError("program.args must be a list.")
    for arg in args:
        argv.append(
            str(await resolve_program_value(arg, task, state, runtime, program))
        )
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
        if isinstance(endpoint_root_url, str):
            env.setdefault("ANTHROPIC_BASE_URL", endpoint_root_url)
            env.setdefault("ANTHROPIC_API_KEY", api_key)
    raw_env = program.get("env", {})
    if not isinstance(raw_env, Mapping):
        raise TypeError("program.env must be a mapping.")
    for key, value in raw_env.items():
        if not isinstance(key, str):
            raise TypeError("program.env keys must be strings.")
        env[key] = str(
            await resolve_program_value(value, task, state, runtime, program)
        )
    return env


async def resolve_program_value(
    value: object,
    task: Task,
    state: State,
    runtime: Runtime,
    program: Mapping[str, object] | None = None,
) -> object:
    fn = program_value_callable(value)
    if fn is not None:
        kwargs = await program_binding_kwargs(fn, program, task, state, runtime)
        return await maybe_call_with_named_args(fn, task=task, state=state, **kwargs)
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


def program_value_callable(value: object) -> Callable[..., object] | None:
    if callable(value):
        return cast(Callable[..., object], value)
    if isinstance(value, Mapping) and "fn" in value:
        spec = cast(Mapping[str, object], value)
        unknown = set(spec) - {"fn"}
        if unknown:
            raise ValueError(
                f"Program callable value has unknown keys: {sorted(unknown)}."
            )
        fn = resolve_config_object(spec["fn"])
        if not callable(fn):
            raise TypeError("Program callable value requires callable fn.")
        return cast(Callable[..., object], fn)
    return None


async def program_binding_kwargs(
    fn: Callable[..., object],
    program: Mapping[str, object] | None,
    task: Task,
    state: State,
    runtime: Runtime,
) -> dict[str, object]:
    if program is None:
        return {}
    raw_bindings = program.get("bindings") or {}
    if not isinstance(raw_bindings, Mapping):
        raise TypeError("program.bindings must be a mapping.")
    if not raw_bindings:
        return {}
    name = function_name(fn)
    kwargs: dict[str, object] = {}
    for binding_key, source in raw_bindings.items():
        target_name, arg_name = binding_key_parts(binding_key)
        if target_name != name:
            continue
        validate_bound_arg(fn, arg_name, f"Program binding {binding_key!r}")
        source_root = binding_source_root(source)
        validate_binding_source_root(source_root, f"Program binding {binding_key!r}")
        if source_root == "objects":
            raise ValueError("program.bindings cannot use objects.* sources.")
        if arg_name in kwargs:
            raise ValueError(f"Program binding arg {arg_name!r} is defined twice.")
        kwargs[arg_name] = await runtime.resolve_binding(source, task, state)
    return kwargs


def validate_program_bindings(program: Mapping[str, object]) -> None:
    raw_bindings = program.get("bindings") or {}
    if not isinstance(raw_bindings, Mapping):
        raise TypeError("program.bindings must be a mapping.")
    if not raw_bindings:
        return
    targets = program_binding_targets(program)
    for binding_key, source in raw_bindings.items():
        target_name, arg_name = binding_key_parts(binding_key)
        fn = targets.get(target_name)
        if fn is None:
            if target_name in program_setup_callable_names(program):
                raise ValueError(
                    "program.setup callables cannot use program.bindings; move "
                    "bound runtime setup under program.tools.<interface>."
                )
            raise ValueError(
                f"Program binding {binding_key!r} does not match a callable "
                "owned by the same program."
            )
        validate_bound_arg(fn, arg_name, f"Program binding {binding_key!r}")
        source_root = binding_source_root(source)
        validate_binding_source_root(source_root, f"Program binding {binding_key!r}")
        if source_root == "objects":
            raise ValueError("program.bindings cannot use objects.* sources.")


def program_binding_targets(
    program: Mapping[str, object],
) -> dict[str, Callable[..., object]]:
    targets: dict[str, Callable[..., object]] = {}

    def add(value: object) -> None:
        fn = program_value_callable(value)
        if fn is None:
            return
        name = function_name(fn)
        existing = targets.get(name)
        if existing is not None and existing is not fn:
            raise ValueError(f"Program binding target {name!r} is defined twice.")
        targets[name] = fn

    def add_items(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                add(item)
        elif value is not None and not isinstance(value, str):
            add(value)

    command = program.get("command")
    if isinstance(command, list):
        for item in command:
            add(item)
    add_items(program.get("args"))
    for _, item, _ in program_tools_setup(program):
        add(item)
    for key in ("files", "dirs", "env"):
        value = program.get(key)
        if isinstance(value, Mapping):
            for item in value.values():
                add(item)
    return targets


def program_setup_callable_names(program: Mapping[str, object]) -> set[str]:
    names: set[str] = set()
    setup = program.get("setup")
    items = setup if isinstance(setup, list) else [setup]
    for item in items:
        fn = program_value_callable(item)
        if fn is not None:
            names.add(function_name(fn))
    return names


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


def program_tool_types(program: Mapping[str, object]) -> tuple[ProgramToolType, ...]:
    return validate_program_tool_types(program.get("tools"))


def program_tools_setup(
    program: Mapping[str, object],
) -> list[tuple[ProgramToolType, object, int]]:
    tools = program.get("tools")
    if tools is None or isinstance(tools, str):
        return []
    if isinstance(tools, list):
        result: list[tuple[ProgramToolType, object, int]] = []
        for item in tools:
            result.extend(program_tools_setup({"tools": item}))
        return result
    if not isinstance(tools, Mapping):
        validate_program_tool_types(tools)
        return []
    tool_types = validate_program_tool_types(tools)
    tools_map = cast(Mapping[str, object], tools)
    priority = cast(int, tools_map.get("priority", -100))
    result = []
    for tool_type in tool_types:
        value = tools_map[tool_type]
        if value is None or value is True:
            continue
        if value is False:
            raise ValueError("program.tools setup should be removed instead of false.")
        items = value if isinstance(value, list) else [value]
        for item in items:
            result.append((tool_type, item, priority))
    return result
