from __future__ import annotations

import asyncio
import glob
import json
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from verifiers.clients import Client
from verifiers.types import Messages, Response, Tool
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.utils.response_utils import parse_response_message, parse_response_tokens
from verifiers.utils.tool_utils import convert_func_to_tool_def

from .lifecycle import collect_handlers, run_handlers
from .scoring import SignalRecord, build_signals, collect_signals
from .scoring import score_group as score_group_signals
from .scoring import score_rollout as score_rollout_signals
from .state import State
from .task import Task
from .toolset import Toolset, flatten_toolsets, iter_toolsets, tool_name

_global_runtime: ContextVar[object | None] = ContextVar(
    "verifiers_v1_global_runtime", default=None
)


class Runtime:
    def __init__(self, taskset: object | None = None, harness: object | None = None):
        self.taskset = taskset
        self.harness = harness
        self.toolsets = self._collect_toolsets()
        self.tool_owners = self._build_tool_owners()
        self.tools = self._build_tools(apply_visibility=False)
        self.exposed_tools = self._build_tools(apply_visibility=True)
        self.bindings = self._build_bindings()
        self.object_specs = self._build_object_specs()
        self.objects: dict[str, object] = {}
        self.model_clients: dict[str, Client] = {}
        self.tool_sandboxes: dict[tuple[int, str], object] = {}
        self.tool_sandbox_lock = asyncio.Lock()
        signals = self._build_signals()
        self.rollout_signals = [
            signal for signal in signals if signal["stage"] == "rollout"
        ]
        self.group_signals = [
            signal for signal in signals if signal["stage"] == "group"
        ]
        self.rollout_cleanup = collect_handlers(
            (taskset, harness, *self.toolsets),
            "cleanup",
            self._extra_cleanup(),
            stage="rollout",
        )
        self.group_cleanup = collect_handlers(
            (taskset, harness, *self.toolsets),
            "cleanup",
            self._extra_cleanup(),
            stage="group",
        )
        self.teardown_handlers = collect_handlers(
            (taskset, harness, *self.toolsets), "teardown", self._extra_teardown()
        )

    def prepare_state(self, task: Task, state: State) -> None:
        state.setdefault("task", dict(task))
        state.setdefault("runtime", {})
        state["runtime"].setdefault("tools", sorted(self.exposed_tools))
        state["tools"] = sorted(self.exposed_tools)

    def bind_model_client(self, state: State, client: Client | None) -> None:
        if client is None:
            return
        key = str(state["runtime"].get("client_key") or "default")
        self.model_clients[key] = client
        state["runtime"]["client_key"] = key

    def model_client(self, state: State) -> Client:
        key = str(state.get("runtime", {}).get("client_key") or "default")
        client = self.model_clients.get(key)
        if client is None:
            raise RuntimeError("Harness has no model client for intercepted requests.")
        return client

    def model(self, state: State) -> str:
        model = state.get("runtime", {}).get("model") or state.get("model")
        if not isinstance(model, str) or not model:
            raise RuntimeError("Harness has no model for intercepted requests.")
        return model

    def sampling_args(self, state: State) -> dict[str, Any]:
        sampling = state.get("runtime", {}).get("sampling_args") or {}
        if not isinstance(sampling, Mapping):
            raise TypeError("state.runtime.sampling_args must be a mapping.")
        return dict(cast(Mapping[str, Any], sampling))

    def tool_defs(self) -> list[Tool] | None:
        defs: list[Tool] = []
        for name, tool in self.exposed_tools.items():
            if callable(tool):
                defs.append(self._tool_def(name, tool))
        return defs or None

    def _tool_def(self, name: str, tool: object) -> Tool:
        tool_def = convert_func_to_tool_def(tool)
        hidden_args = {"task", "state"}
        owner = self.tool_owners.get(name)
        if owner is not None and owner.sandbox is not None:
            hidden_args.add("sandbox")
        for binding_key in self.bindings:
            tool_name_prefix, separator, arg_name = binding_key.partition(".")
            if separator == "." and tool_name_prefix == name:
                hidden_args.add(arg_name)
        parameters = dict(tool_def.parameters)
        properties = dict(
            cast(Mapping[str, object], parameters.get("properties") or {})
        )
        for arg_name in hidden_args:
            properties.pop(arg_name, None)
        parameters["properties"] = properties
        required = parameters.get("required")
        if isinstance(required, list):
            parameters["required"] = [arg for arg in required if arg not in hidden_args]
        return Tool(
            name=tool_def.name,
            description=tool_def.description,
            parameters=parameters,
            strict=tool_def.strict,
        )

    async def call_tool(
        self, name: str, task: Task, state: State, **kwargs: object
    ) -> object:
        return await self._call_tool(name, task, state, True, **kwargs)

    def tool_calls(
        self, task: Task, state: State
    ) -> dict[str, Callable[..., Awaitable[object]]]:
        calls: dict[str, Callable[..., Awaitable[object]]] = {}
        for name in self.exposed_tools:
            calls[name] = self._tool_call(name, task, state, exposed=True)
        return calls

    def _tool_call(
        self, name: str, task: Task, state: State, exposed: bool
    ) -> Callable[..., Awaitable[object]]:
        async def call(**kwargs: object) -> object:
            return await self._call_tool(name, task, state, exposed, **kwargs)

        return call

    async def _call_tool(
        self,
        name: str,
        task: Task,
        state: State,
        exposed: bool,
        **kwargs: object,
    ) -> object:
        tools = self.exposed_tools if exposed else self.tools
        if name not in tools:
            kind = "exposed tool" if exposed else "tool"
            raise KeyError(f"Unknown {kind} {name!r}.")
        tool_kwargs = dict(kwargs)
        owner = self.tool_owners.get(name)
        if owner is not None and owner.sandbox is not None:
            tool_kwargs.setdefault(
                "sandbox", await self.resolve_tool_sandbox(owner, task, state)
            )
        for binding_key, source in self.bindings.items():
            tool_name_prefix, separator, arg_name = binding_key.partition(".")
            if separator != ".":
                raise ValueError(f"Tool binding {binding_key!r} must be 'tool.arg'.")
            if tool_name_prefix != name:
                continue
            if arg_name in tool_kwargs:
                raise ValueError(f"Tool arg {name}.{arg_name} is already set.")
            tool_kwargs[arg_name] = await self.resolve_binding(source, task, state)
        return await maybe_call_with_named_args(
            cast(Callable[..., object], tools[name]),
            task=task,
            state=state,
            **tool_kwargs,
        )

    async def submit_model_request(
        self,
        prompt: Messages,
        task: Task,
        state: State,
        tool_defs: list[Tool] | None = None,
        extras: dict[str, object] | None = None,
    ) -> Response:
        client = self.model_client(state)
        response = await client.get_response(
            prompt=prompt,
            model=self.model(state),
            tools=tool_defs,
            sampling_args=self.sampling_args(state),
            state=state,
        )
        completion = await parse_response_message(response)
        tokens = await parse_response_tokens(response)
        is_truncated = response.message.is_truncated or (
            tokens is not None and bool(tokens.get("is_truncated"))
        )
        state["trajectory"].append(
            {
                "prompt": serializable(prompt),
                "completion": serializable(completion),
                "response": serializable(response),
                "tokens": serializable(tokens),
                "reward": None,
                "advantage": None,
                "is_truncated": bool(is_truncated),
                "trajectory_id": str(state["trajectory_id"]),
                "extras": extras or {},
            }
        )
        return response

    async def score_rollout(self, task: Task, state: State) -> State:
        await score_rollout_signals(self.rollout_signals, task, state)
        return state

    async def score_group(self, tasks: list[Task], states: list[State]) -> list[State]:
        await score_group_signals(
            self.group_signals,
            cast(list[Mapping[str, Any]], tasks),
            cast(list[dict[str, Any]], states),
        )
        return states

    async def cleanup_rollout(self, task: Task, state: State) -> None:
        await run_handlers(
            self.rollout_cleanup, task=task, state=state, runtime=self, **self.objects
        )
        await self.release_tool_sandboxes(state, scope="rollout")

    async def cleanup_group(self, tasks: list[Task], states: list[State]) -> None:
        await run_handlers(
            self.group_cleanup, tasks=tasks, states=states, runtime=self, **self.objects
        )
        for state in states:
            await self.release_tool_sandboxes(state, scope="group")

    async def collect_artifacts(self, task: Task, state: State) -> None:
        program = getattr(self.harness, "program", None)
        if not isinstance(program, Mapping):
            return
        artifacts = program.get("artifacts")
        if artifacts is None:
            return
        if not isinstance(artifacts, Mapping):
            raise TypeError("program.artifacts must be a mapping.")
        state.setdefault("artifacts", {})
        for name, spec in artifacts.items():
            if not isinstance(name, str):
                raise TypeError("program.artifacts keys must be strings.")
            if name in state["artifacts"]:
                continue
            state["artifacts"][name] = await self._collect_artifact(spec, task, state)

    async def teardown(self) -> None:
        await run_handlers(self.teardown_handlers, runtime=self, **self.objects)
        for handle in list(self.tool_sandboxes.values()):
            await maybe_call_with_named_args(getattr(handle, "delete"))
        self.tool_sandboxes.clear()

    async def resolve_binding(self, source: object, task: Task, state: State) -> object:
        if isinstance(source, str):
            return await self._resolve_path(source, task, state)
        if callable(source):
            return await maybe_call_with_named_args(
                source, task=task, state=state, runtime=self
            )
        return source

    def _build_signals(self) -> list[SignalRecord]:
        taskset_signals = self._owner_signals(self.taskset)
        harness_signals = self._owner_signals(self.harness)
        return collect_signals(taskset_signals, harness_signals)

    def _collect_toolsets(self) -> list[Toolset]:
        owners = (self.taskset, self.harness)
        groups: list[Toolset] = []
        for owner in owners:
            if owner is None:
                continue
            groups.extend(iter_toolsets(getattr(owner, "toolsets", ())))
        return groups

    def _build_tool_owners(self) -> dict[str, Toolset]:
        owners: dict[str, Toolset] = {}

        def visit(toolset: Toolset) -> None:
            for item in toolset.tools:
                if isinstance(item, Toolset):
                    visit(item)
                    continue
                name = tool_name(item)
                if name in owners:
                    raise ValueError(f"Tool {name!r} is defined twice.")
                owners[name] = toolset

        for toolset in self.toolsets:
            visit(toolset)
        return owners

    def _build_tools(self, apply_visibility: bool) -> dict[str, object]:
        tools: dict[str, object] = {}
        for owner in (self.taskset, self.harness):
            if owner is None:
                continue
            for tool in flatten_toolsets(
                getattr(owner, "toolsets", ()), apply_visibility=apply_visibility
            ):
                name = tool_name(tool)
                if name in tools:
                    raise ValueError(f"Tool {name!r} is defined twice.")
                tools[name] = tool
        return tools

    def _build_bindings(self) -> dict[str, object]:
        bindings: dict[str, object] = {}
        for toolset in self.toolsets:
            for key, value in toolset.bindings.items():
                if key in bindings:
                    raise ValueError(f"Tool binding {key!r} is defined twice.")
                bindings[key] = value
        return bindings

    def _build_object_specs(self) -> dict[str, object]:
        objects: dict[str, object] = {}
        for toolset in self.toolsets:
            for key, value in toolset.objects.items():
                if key in objects:
                    raise ValueError(f"Runtime object {key!r} is defined twice.")
                objects[key] = value
        return objects

    def _owner_signals(self, owner: object | None) -> list[SignalRecord]:
        if owner is None:
            return []
        config = getattr(owner, "config", None)
        return build_signals(
            owner=owner,
            scoring=getattr(config, "scoring", {}),
            metrics=getattr(owner, "metrics", ()),
            rewards=getattr(owner, "rewards", ()),
        )

    def _extra_cleanup(self) -> list[Callable[..., object]]:
        handlers: list[Callable[..., object]] = []
        for owner in (self.taskset, self.harness, *self.toolsets):
            if owner is None:
                continue
            for handler in getattr(owner, "__dict__", {}).get("cleanup", ()):
                if not callable(handler):
                    raise TypeError("cleanup entries must be callable.")
                handlers.append(cast(Callable[..., object], handler))
        return handlers

    def _extra_teardown(self) -> list[Callable[..., object]]:
        handlers: list[Callable[..., object]] = []
        for owner in (self.taskset, self.harness, *self.toolsets):
            if owner is None:
                continue
            for handler in getattr(owner, "__dict__", {}).get("teardown", ()):
                if not callable(handler):
                    raise TypeError("teardown entries must be callable.")
                handlers.append(cast(Callable[..., object], handler))
        return handlers

    async def _collect_artifact(self, spec: object, task: Task, state: State) -> object:
        if callable(spec):
            return await maybe_call_with_named_args(spec, task=task, state=state)
        if not isinstance(spec, Mapping):
            raise TypeError("Artifact specs must be callables or mappings.")
        spec_map = cast(Mapping[str, object], spec)
        path = spec_map.get("path")
        if not isinstance(path, str):
            raise TypeError("Artifact mapping specs require a string path.")
        matches = sorted(glob.glob(path.format(**state)))
        if not matches:
            raise FileNotFoundError(f"Artifact path matched no files: {path!r}")
        artifact_format = spec_map.get("format", "text")
        with open(matches[0], encoding="utf-8") as f:
            if artifact_format == "json":
                data: object = json.load(f)
            elif artifact_format == "text":
                data = f.read()
            else:
                raise ValueError(f"Unsupported artifact format: {artifact_format!r}")
        key = spec_map.get("key")
        if key is not None:
            if not isinstance(key, str):
                raise TypeError("Artifact key must be a string.")
            data = cast(Mapping[str, Any], data)[key]
        return data

    async def _resolve_path(self, path: str, task: Task, state: State) -> object:
        root, separator, tail = path.partition(".")
        if root == "task":
            value: object = task
        elif root == "state":
            value = state
        elif root == "runtime":
            value = state.get("runtime", {})
        elif root == "objects":
            if not separator:
                return self.objects
            name, _, rest = tail.partition(".")
            value = await self._resolve_object(name, task, state)
            tail = rest
        elif root == "tools":
            if not separator:
                return self.tools
            name, _, rest = tail.partition(".")

            value = self._tool_call(name, task, state, exposed=False)
            tail = rest
        else:
            raise ValueError(f"Unknown binding root {root!r}.")
        if separator and root not in {"objects", "tools"}:
            return _read_path(value, tail)
        if tail:
            return _read_path(value, tail)
        return value

    async def _resolve_object(self, name: str, task: Task, state: State) -> object:
        if name in self.objects:
            return self.objects[name]
        if name not in self.object_specs:
            raise KeyError(f"Unknown runtime object {name!r}.")
        spec = self.object_specs[name]
        if callable(spec):
            obj = await maybe_call_with_named_args(spec, task=task, state=state)
        else:
            obj = spec
        self.objects[name] = obj
        return obj

    async def resolve_tool_sandbox(
        self, toolset: Toolset, task: Task, state: State
    ) -> object:
        from .sandbox_utils import (
            SandboxHandle,
            create_tool_sandbox_lease,
            sandbox_scope,
            tool_sandbox_key,
        )

        _ = task
        sandbox = toolset.sandbox
        if not isinstance(sandbox, Mapping):
            raise TypeError("Toolset sandbox must be a mapping.")
        scope = sandbox_scope(sandbox)
        state_key = 0 if scope == "global" else id(state)
        key = (state_key, tool_sandbox_key(toolset))
        async with self.tool_sandbox_lock:
            lease = self.tool_sandboxes.get(key)
            if lease is None:
                lease = await create_tool_sandbox_lease(toolset)
                self.tool_sandboxes[key] = lease
        return SandboxHandle(cast(Any, lease), state)

    async def release_tool_sandboxes(self, state: State, scope: str) -> None:
        for key, handle in list(self.tool_sandboxes.items()):
            state_id, _ = key
            if state_id != id(state):
                continue
            handle_scope = getattr(handle, "scope", None)
            if handle_scope != scope:
                continue
            await maybe_call_with_named_args(getattr(handle, "delete"))
            del self.tool_sandboxes[key]

    async def ensure_global_tool_sandboxes(self) -> None:
        from .sandbox_utils import (
            create_tool_sandbox_lease,
            sandbox_scope,
            tool_sandbox_key,
        )

        async with self.tool_sandbox_lock:
            for toolset in self.toolsets:
                sandbox = toolset.sandbox
                if not isinstance(sandbox, Mapping):
                    continue
                if sandbox_scope(sandbox) != "global":
                    continue
                key = (0, tool_sandbox_key(toolset))
                if key not in self.tool_sandboxes:
                    self.tool_sandboxes[key] = await create_tool_sandbox_lease(toolset)

    def bind_global_tool_sandboxes(self, state: State) -> None:
        from .sandbox_utils import attach_tool_sandbox_ref

        for key, lease in self.tool_sandboxes.items():
            state_id, _ = key
            if state_id != 0:
                continue
            attach_tool_sandbox_ref(state, cast(Any, lease))


def _read_path(value: object, path: str) -> object:
    if not path:
        return value
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = cast(Mapping[str, object], current)[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def serializable(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, tuple):
        return [serializable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): serializable(item) for key, item in value.items()}
    return value


@contextmanager
def runtime_context(runtime: Runtime):
    token = _global_runtime.set(runtime)
    try:
        yield
    finally:
        _global_runtime.reset(token)


def current_runtime() -> Runtime:
    raw_runtime = _global_runtime.get()
    if raw_runtime is None:
        raise RuntimeError("v1 runtime helpers must be called inside Harness.run().")
    return cast(Runtime, raw_runtime)
