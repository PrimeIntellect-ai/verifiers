from __future__ import annotations

import asyncio
import glob
import inspect
import json
import uuid
import weakref
from contextlib import AsyncExitStack
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, cast, get_args

from verifiers.clients import Client, resolve_client
from verifiers.types import Messages, Response, Tool
from verifiers.types import ClientConfig, ClientType
from verifiers.utils.client_utils import resolve_client_config
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.response_utils import parse_response_message, parse_response_tokens
from verifiers.utils.tool_utils import convert_func_to_tool_def

from .config import resolve_config_object
from .utils.lifecycle_utils import (
    collect_handlers,
    run_handlers,
    unique_handlers,
    validate_handler_args,
)
from .utils.scoring_utils import SignalRecord, build_signals, collect_signals
from .utils.scoring_utils import score_group as score_group_signals
from .utils.scoring_utils import score_rollout as score_rollout_signals
from .state import State
from .task import Task
from .toolset import (
    MCPTool,
    Toolset,
    flatten_toolsets,
    iter_toolsets,
    normalize_toolset_result,
    tool_name,
)
from .user import User

_RUNTIME_REGISTRY: weakref.WeakValueDictionary[str, Runtime] = (
    weakref.WeakValueDictionary()
)


class Runtime:
    def __init__(self, taskset: object | None = None, harness: object | None = None):
        self.runtime_id = uuid.uuid4().hex
        _RUNTIME_REGISTRY[self.runtime_id] = self
        self.taskset = taskset
        self.harness = harness
        self.toolsets = self._collect_toolsets()
        self.named_toolsets = self._collect_named_toolsets()
        self.rollout_toolsets: dict[str, list[Toolset]] = {}
        self.objects: dict[tuple[int, str, str], object] = {}
        self.user_objects: dict[tuple[int, str, str], object] = {}
        self.model_clients: dict[str, Client] = {}
        self.owned_model_clients: set[str] = set()
        self.sandbox_leases: dict[tuple[str, str], object] = {}
        self.sandbox_lock = asyncio.Lock()
        self.mcp_exit_stacks: dict[str, AsyncExitStack] = {}
        self.mcp_tools: dict[str, dict[str, object]] = {}
        self.exposed_mcp_tools: dict[str, dict[str, object]] = {}
        self.stop_conditions = collect_handlers(
            self._handler_owners(),
            "stop",
            self._extra_handlers("stop", builtins=[state_done]),
        )
        validate_handler_args(
            self.stop_conditions, {"task", "state"}, "stop", "rollout"
        )
        self.rollout_render = collect_handlers(
            self._handler_owners(),
            "render",
            self._extra_handlers("render"),
            stage="rollout",
        )
        self.group_render = collect_handlers(
            self._handler_owners(),
            "render",
            self._extra_handlers("render"),
            stage="group",
        )
        validate_handler_args(
            self.rollout_render, {"task", "state"}, "render", "rollout"
        )
        validate_handler_args(self.group_render, {"tasks", "states"}, "render", "group")
        signals = self._build_signals()
        self.rollout_signals = [
            signal for signal in signals if signal["stage"] == "rollout"
        ]
        self.group_signals = [
            signal for signal in signals if signal["stage"] == "group"
        ]
        self.rollout_cleanup = collect_handlers(
            self._handler_owners(),
            "cleanup",
            self._extra_handlers("cleanup"),
            stage="rollout",
        )
        self.group_cleanup = collect_handlers(
            self._handler_owners(),
            "cleanup",
            self._extra_handlers("cleanup"),
            stage="group",
        )
        validate_handler_args(
            self.rollout_cleanup, {"task", "state"}, "cleanup", "rollout"
        )
        validate_handler_args(
            self.group_cleanup, {"tasks", "states"}, "cleanup", "group"
        )
        self.teardown_handlers = collect_handlers(
            (self.taskset, self.harness, *self.toolsets),
            "teardown",
            self._extra_handlers(
                "teardown", owners=(self.taskset, self.harness, *self.toolsets)
            ),
        )

    @property
    def has_group_signals(self) -> bool:
        return bool(self.group_signals)

    @property
    def has_group_rewards(self) -> bool:
        return any(signal["kind"] == "reward" for signal in self.group_signals)

    @property
    def has_group_advantages(self) -> bool:
        return any(signal["kind"] == "advantage" for signal in self.group_signals)

    def prepare_state(self, task: Task, state: State) -> None:
        state.setdefault("task", dict(task))
        state.setdefault("runtime", {})
        state["runtime"]["runtime_id"] = self.runtime_id
        state["runtime"].setdefault(
            "tool_protocol",
            getattr(getattr(self, "harness", None), "tool_protocol", "callable"),
        )
        state["tools"] = sorted(self.all_exposed_tools(state))

    def bind_model_client(
        self, state: State, client: Client | ClientConfig | None
    ) -> None:
        if client is None:
            return
        owns_client = False
        if isinstance(client, ClientConfig):
            resolved_config = resolve_client_config(client)
            client = resolve_client(resolved_config)
            client_type: ClientType = resolved_config.client_type
            owns_client = True
        else:
            config = getattr(client, "_config", None)
            if isinstance(config, ClientConfig):
                client_type = config.client_type
            else:
                client_type = "openai_chat_completions"
        key = str(
            state["runtime"].get("client_key")
            or state.get("trajectory_id")
            or f"client_{uuid.uuid4().hex}"
        )
        self.model_clients[key] = client
        if owns_client:
            self.owned_model_clients.add(key)
        state["runtime"]["client_key"] = key
        state["runtime"]["client_type"] = client_type

    def inherit_model_controls(
        self, parent_state: State, child_runtime: Runtime, child_state: State
    ) -> None:
        parent = parent_state.get("runtime", {})
        child_harness = getattr(child_runtime, "harness", None)
        child_has_client = getattr(child_harness, "client", None) is not None
        child_has_model = getattr(child_harness, "model", None) is not None
        child_has_sampling_args = bool(getattr(child_harness, "sampling_args", None))
        child_state.setdefault("runtime", {})
        for key in ("model", "sampling_args", "client_type", "score_rollout"):
            if key == "model" and child_has_model:
                continue
            if key == "sampling_args" and child_has_sampling_args:
                continue
            if key == "client_type" and child_has_client:
                continue
            if key in parent and key not in child_state["runtime"]:
                child_state["runtime"][key] = parent[key]
        if "group_key" in parent and "group_key" not in child_state["runtime"]:
            child_state["runtime"]["group_key"] = parent["group_key"]
        client_key = parent.get("client_key")
        if (
            not child_has_client
            and "client_key" not in child_state["runtime"]
            and isinstance(client_key, str)
            and client_key in self.model_clients
        ):
            child_runtime.model_clients[client_key] = self.model_clients[client_key]
            child_state["runtime"]["client_key"] = client_key

    async def run_harness(
        self,
        harness: object,
        task: Task | Mapping[str, Any],
        parent_state: State,
        state: State | None = None,
    ) -> State:
        from .harness import Harness

        if not isinstance(harness, Harness):
            raise TypeError("run_harness expects a verifiers.v1.Harness.")
        task = task if isinstance(task, Task) else Task(task).freeze()
        child_state = State.for_task(task) if state is None else state
        self.inherit_model_controls(parent_state, harness.runtime, child_state)
        child_state = await harness.run(task, child_state)
        self.record_child_rollout(parent_state, task, child_state)
        return child_state

    def record_child_rollout(
        self, parent_state: State, task: Task, child_state: State
    ) -> None:
        child_record = State(cast(Mapping[str, Any], serializable(child_state)))
        child_record.strip_runtime_handles()
        parent_state.setdefault("child_rollouts", [])
        parent_state["child_rollouts"].append(
            {
                "task": serializable(task),
                "state": serializable(child_record),
            }
        )

    def model_client(self, state: State) -> Client:
        key = str(state.get("runtime", {}).get("client_key") or "default")
        client = self.model_clients.get(key)
        if client is None:
            raise RuntimeError("Harness has no model client for intercepted requests.")
        return client

    def client_type(self, state: State) -> ClientType:
        raw_client_type = state.get("runtime", {}).get("client_type")
        if raw_client_type is None:
            return "openai_chat_completions"
        if raw_client_type not in get_args(ClientType):
            raise ValueError(f"Unsupported client type: {raw_client_type!r}")
        return cast(ClientType, raw_client_type)

    def model(self, state: State) -> str:
        model = state.get("runtime", {}).get("model")
        if not isinstance(model, str) or not model:
            raise RuntimeError("Harness has no model for intercepted requests.")
        return model

    def sampling_args(self, state: State) -> dict[str, Any]:
        sampling = state.get("runtime", {}).get("sampling_args") or {}
        if not isinstance(sampling, Mapping):
            raise TypeError("state.runtime.sampling_args must be a mapping.")
        return dict(cast(Mapping[str, Any], sampling))

    def tool_defs(self, state: State) -> list[Tool] | None:
        defs: list[Tool] = []
        for name, tool in self.all_exposed_tools(state).items():
            if callable(tool):
                defs.append(self._tool_def(name, tool, state))
        return defs or None

    async def user_messages(
        self, task: Task, state: State, transcript: Sequence[object] | None = None
    ) -> list[dict[str, object]]:
        user = self._resolve_user()
        if user is None:
            return []
        kwargs: dict[str, object] = {}
        fn = user.fn
        if user.sandbox is not None:
            kwargs["sandbox"] = await self.resolve_user_sandbox(user, task, state)
        for name, source in user.bindings.items():
            kwargs[name] = await self.resolve_user_binding(
                user, source, task, state, transcript
            )
        raw_messages = await maybe_call_with_named_args(
            fn, task=task, state=state, **kwargs
        )
        if raw_messages is None:
            return []
        messages = normalize_messages(raw_messages, field_name="user")
        return [message.model_dump(exclude_none=True) for message in messages]

    def _resolve_user(self) -> User | None:
        users = [
            user
            for user in (
                getattr(self.taskset, "user", None),
                getattr(self.harness, "user", None),
            )
            if user is not None
        ]
        if len(users) > 1:
            raise ValueError("Taskset and harness cannot both define user.")
        return cast(User | None, users[0] if users else None)

    def _tool_def(self, name: str, tool: object, state: State) -> Tool:
        mcp_tool_def = getattr(tool, "tool_def", None)
        if isinstance(mcp_tool_def, Tool):
            return mcp_tool_def
        tool_def = convert_func_to_tool_def(tool)
        hidden_args = {"runtime", "task", "state"}
        owner = self.tool_owner(name, state)
        if owner is not None and owner.sandbox is not None:
            hidden_args.add("sandbox")
        for binding_key in self.bindings_for_state(state):
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
        self, tool_name: str, task: Task, state: State, **kwargs: object
    ) -> object:
        return await self._call_tool(tool_name, task, state, True, **kwargs)

    async def is_completed(self, task: Task, state: State) -> bool:
        conditions = unique_handlers(
            [*self.stop_conditions, *self._rollout_handlers("stop", state)]
        )
        for condition in conditions:
            completed = await maybe_call_with_named_args(
                condition, task=task, state=state
            )
            if completed:
                state["is_completed"] = True
                state["is_truncated"] = state.get("is_truncated", False) or any(
                    step.get("is_truncated", False)
                    for step in state.get("trajectory", [])
                    if isinstance(step, Mapping)
                )
                state["stop_condition"] = state.get("stop_condition") or getattr(
                    condition, "__name__", type(condition).__name__
                )
                return True
        return False

    def tool_calls(
        self, task: Task, state: State
    ) -> dict[str, Callable[..., Awaitable[object]]]:
        calls: dict[str, Callable[..., Awaitable[object]]] = {}
        for name in self.all_exposed_tools(state):
            calls[name] = self._tool_call(name, task, state, exposed=True)
        return calls

    def _tool_call(
        self, tool_name: str, task: Task, state: State, exposed: bool
    ) -> Callable[..., Awaitable[object]]:
        async def call(**kwargs: object) -> object:
            return await self._call_tool(tool_name, task, state, exposed, **kwargs)

        return call

    async def _call_tool(
        self,
        tool_name: str,
        task: Task,
        state: State,
        exposed: bool,
        **kwargs: object,
    ) -> object:
        tools = self.all_exposed_tools(state) if exposed else self.all_tools(state)
        if tool_name not in tools:
            kind = "exposed tool" if exposed else "tool"
            raise KeyError(f"Unknown {kind} {tool_name!r}.")
        tool_kwargs = dict(kwargs)
        owner = self.tool_owner(tool_name, state)
        for hidden_arg in ("runtime", "task", "state"):
            if hidden_arg in tool_kwargs:
                raise ValueError(f"Tool arg {tool_name}.{hidden_arg} is reserved.")
        if owner is not None and owner.sandbox is not None:
            if "sandbox" in tool_kwargs:
                raise ValueError(f"Tool arg {tool_name}.sandbox is reserved.")
            tool_kwargs["sandbox"] = await self.resolve_tool_sandbox(owner, task, state)
        for binding_key, source in self.bindings_for_state(state).items():
            tool_name_prefix, separator, arg_name = binding_key.partition(".")
            if separator != ".":
                raise ValueError(f"Tool binding {binding_key!r} must be 'tool.arg'.")
            if tool_name_prefix != tool_name:
                continue
            if arg_name in tool_kwargs:
                raise ValueError(f"Tool arg {tool_name}.{arg_name} is already set.")
            tool_kwargs[arg_name] = await self.resolve_binding(source, task, state)
        return await maybe_call_with_named_args(
            cast(Callable[..., object], tools[tool_name]),
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
        step = {
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
        keep_step = getattr(self.harness, "keep_trajectory_step", None)
        if keep_step is not None:
            headers = {}
            if extras is not None and isinstance(extras.get("headers"), Mapping):
                headers = dict(cast(Mapping[str, object], extras["headers"]))
            keep = await maybe_call_with_named_args(
                keep_step, step=step, state=state, headers=headers
            )
            if not keep:
                return response
        state["trajectory"].append(step)
        return response

    async def render_rollout(self, task: Task, state: State) -> State:
        handlers = unique_handlers(
            [
                *self.rollout_render,
                *self._rollout_handlers("render", state, stage="rollout"),
            ]
        )
        validate_handler_args(handlers, {"task", "state"}, "render", "rollout")
        await run_handlers(handlers, task=task, state=state)
        return state

    async def render_group(self, tasks: list[Task], states: list[State]) -> list[State]:
        handlers = unique_handlers(
            [
                *self.group_render,
                *self._group_handlers("render", states, stage="group"),
            ]
        )
        validate_handler_args(handlers, {"tasks", "states"}, "render", "group")
        await run_handlers(handlers, tasks=tasks, states=states)
        return states

    async def score_rollout(self, task: Task, state: State) -> State:
        await score_rollout_signals(self.rollout_signals, task, state)
        return state

    async def score_group(self, tasks: list[Task], states: list[State]) -> list[State]:
        await self.render_group(tasks, states)
        await score_group_signals(
            self.group_signals,
            cast(list[Mapping[str, Any]], tasks),
            cast(list[dict[str, Any]], states),
        )
        return states

    async def cleanup_rollout(self, task: Task, state: State) -> None:
        handlers = unique_handlers(
            [
                *self.rollout_cleanup,
                *self._rollout_handlers("cleanup", state, stage="rollout"),
            ]
        )
        await run_handlers(handlers, task=task, state=state)
        await self.release_objects("rollout", state)
        await self.release_user_objects("rollout", state)
        await self.release_sandboxes(scope="rollout", state=state)
        await self.close_mcp_tools(state)
        await self.release_model_client(state)

    async def cleanup_group(self, tasks: list[Task], states: list[State]) -> None:
        handlers = unique_handlers(
            [
                *self.group_cleanup,
                *self._group_handlers("cleanup", states, stage="group"),
            ]
        )
        await run_handlers(handlers, tasks=tasks, states=states)
        for state in states:
            await self.release_objects("group", state)
            await self.release_user_objects("group", state)
            await self.release_sandboxes(scope="group", state=state)
            await self.close_mcp_tools(state, scope="group")
            self.rollout_toolsets.pop(self.scope_key("rollout", state), None)

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
        await run_handlers(self.teardown_handlers)
        await self.release_objects("global")
        await self.release_user_objects("global")
        for handle in list(self.sandbox_leases.values()):
            await maybe_call_with_named_args(getattr(handle, "delete"))
        self.sandbox_leases.clear()
        await self.close_all_mcp_tools()
        await self.release_all_model_clients()
        _RUNTIME_REGISTRY.pop(self.runtime_id, None)

    async def resolve_binding(self, source: object, task: Task, state: State) -> object:
        if isinstance(source, str):
            return await self._resolve_path(source, task, state)
        if callable(source):
            return await maybe_call_with_named_args(source, task=task, state=state)
        return source

    async def resolve_user_binding(
        self,
        user: User,
        source: object,
        task: Task,
        state: State,
        transcript: Sequence[object] | None = None,
    ) -> object:
        if isinstance(source, str):
            root, separator, tail = source.partition(".")
            if root == "objects" and separator:
                name, _, rest = tail.partition(".")
                if name in user.objects:
                    value = await self.resolve_user_object(user, name, task, state)
                else:
                    value = await self.resolve_object(name, task, state)
                if rest:
                    return _read_path(value, rest)
                return value
        if callable(source):
            return await maybe_call_with_named_args(
                source, task=task, state=state, transcript=transcript
            )
        return await self.resolve_binding(source, task, state)

    async def resolve_user_object(
        self, user: User, name: str, task: Task, state: State
    ) -> object:
        if name not in user.objects:
            raise KeyError(f"Unknown user object {name!r}.")
        key = (id(user), self.scope_key(user.scope, state), name)
        if key in self.user_objects:
            return self.user_objects[key]
        spec = user.objects[name]
        if callable(spec):
            obj = await maybe_call_with_named_args(spec, task=task, state=state)
        else:
            obj = spec
        self.user_objects[key] = obj
        return obj

    async def release_user_objects(
        self, scope: str, state: State | None = None
    ) -> None:
        scope_key = self.scope_key(scope, state) if scope != "global" else "global"
        for key, obj in list(self.user_objects.items()):
            _, object_scope_key, _ = key
            if object_scope_key != scope_key:
                continue
            await close_object(obj)
            del self.user_objects[key]

    async def ensure_rollout_toolsets(self, task: Task, state: State) -> None:
        key = self.scope_key("rollout", state)
        if key in self.rollout_toolsets:
            return
        self.rollout_toolsets[key] = await self._task_toolset_additions(task, state)

    def _task_toolsets_config(self, task: Mapping[str, Any]) -> Mapping[str, object]:
        raw_toolsets = task.get("toolsets")
        if raw_toolsets is None:
            return {}
        if not isinstance(raw_toolsets, Mapping):
            raise TypeError("task.toolsets must be a mapping.")
        return cast(Mapping[str, object], raw_toolsets)

    async def _task_toolset_additions(self, task: Task, state: State) -> list[Toolset]:
        toolsets: list[Toolset] = []
        config = self._task_toolsets_config(task)
        for name, spec in config.items():
            if name in {"show", "hide"}:
                continue
            if name in self.named_toolsets:
                raise ValueError(f"Task toolset {name!r} is already defined.")
            toolsets.append(await self._runtime_named_toolset(name, spec, task, state))
        return toolsets

    async def _runtime_named_toolset(
        self, name: str, spec: object, task: Task, state: State
    ) -> Toolset:
        spec = resolve_config_object(spec)
        if isinstance(spec, Toolset):
            return spec
        if isinstance(spec, Mapping):
            mapping = cast(Mapping[str, object], spec)
            if "fn" in mapping:
                fn = resolve_config_object(mapping.get("fn"))
                if not callable(fn):
                    raise TypeError(f"Task toolset {name!r} requires callable fn.")
                kwargs = {key: value for key, value in mapping.items() if key != "fn"}
                result = await maybe_call_with_named_args(
                    cast(Callable[..., object], fn),
                    task=task,
                    state=state,
                    **kwargs,
                )
                toolsets = normalize_toolset_result(result)
                if len(toolsets) != 1:
                    raise ValueError(
                        f"Task toolset {name!r} fn must return exactly one Toolset."
                    )
                return toolsets[0]
            return Toolset(config=mapping)
        if callable(spec):
            result = await maybe_call_with_named_args(
                cast(Callable[..., object], spec), task=task, state=state
            )
            toolsets = normalize_toolset_result(result)
            if len(toolsets) != 1:
                raise ValueError(
                    f"Task toolset {name!r} fn must return exactly one Toolset."
                )
            return toolsets[0]
        return normalize_toolset_result(spec)[0]

    def _rollout_toolsets(self, state: State) -> list[Toolset]:
        return self.rollout_toolsets.get(self.scope_key("rollout", state), [])

    def active_toolsets(self, state: State) -> list[Toolset]:
        return [*self._static_toolsets_for_state(state), *self._rollout_toolsets(state)]

    def _static_toolsets_for_state(self, state: State) -> list[Toolset]:
        task = cast(Mapping[str, Any], state.get("task") or {})
        selected = self._selected_toolset_names(task)
        ids_to_names = {
            id(toolset): name for name, toolset in self.named_toolsets.items()
        }
        active: list[Toolset] = []
        for toolset in self.toolsets:
            name = ids_to_names.get(id(toolset))
            if name is not None and name not in selected:
                continue
            active.append(toolset)
        return active

    def _selected_toolset_names(self, task: Mapping[str, Any]) -> set[str]:
        names = set(self.named_toolsets)
        config = self._task_toolsets_config(task)
        show = config.get("show")
        hide = config.get("hide")
        if show is not None and hide is not None:
            raise ValueError("task.toolsets accepts show or hide, not both.")
        if show is not None:
            selected = set(string_list(show, "task.toolsets.show"))
            unknown = sorted(selected - names)
            if unknown:
                raise KeyError(f"Unknown shown toolsets: {unknown}.")
            return selected
        if hide is not None:
            hidden = set(string_list(hide, "task.toolsets.hide"))
            unknown = sorted(hidden - names)
            if unknown:
                raise KeyError(f"Unknown hidden toolsets: {unknown}.")
            return names - hidden
        return names

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

    def _collect_named_toolsets(self) -> dict[str, Toolset]:
        named: dict[str, Toolset] = {}
        for owner in (self.taskset, self.harness):
            if owner is None:
                continue
            owner_named = getattr(owner, "named_toolsets", {})
            if not isinstance(owner_named, Mapping):
                raise TypeError("named_toolsets must be a mapping.")
            for name, toolset in owner_named.items():
                if not isinstance(name, str):
                    raise TypeError("Toolset names must be strings.")
                if name in named:
                    raise ValueError(f"Toolset {name!r} is defined twice.")
                if not isinstance(toolset, Toolset):
                    raise TypeError("named_toolsets values must be Toolsets.")
                named[name] = toolset
        return named

    def _tools_for_toolsets(
        self, toolsets: Iterable[object], apply_visibility: bool
    ) -> dict[str, object]:
        tools: dict[str, object] = {}
        for tool in flatten_toolsets(toolsets, apply_visibility=apply_visibility):
            if isinstance(tool, MCPTool):
                continue
            name = tool_name(tool)
            if name in tools:
                raise ValueError(f"Tool {name!r} is defined twice.")
            tools[name] = tool
        return tools

    def _tool_owners_for(self, toolsets: Sequence[Toolset]) -> dict[str, Toolset]:
        owners: dict[str, Toolset] = {}

        def visit(toolset: Toolset) -> None:
            for item in toolset.tools:
                if isinstance(item, Toolset):
                    visit(item)
                    continue
                if isinstance(item, MCPTool):
                    continue
                name = tool_name(item)
                if name in owners:
                    raise ValueError(f"Tool {name!r} is defined twice.")
                owners[name] = toolset

        for toolset in toolsets:
            visit(toolset)
        return owners

    def tool_owner(self, name: str, state: State) -> Toolset | None:
        return self._tool_owners_for(self.active_toolsets(state)).get(name)

    def bindings_for_state(self, state: State) -> dict[str, object]:
        bindings: dict[str, object] = {}
        for toolset in iter_toolsets(self.active_toolsets(state)):
            for key, value in toolset.bindings.items():
                if key in bindings:
                    raise ValueError(f"Tool binding {key!r} is defined twice.")
                bindings[key] = value
        return bindings

    def object_specs_for_state(self, state: State) -> dict[str, tuple[Toolset, object]]:
        objects: dict[str, tuple[Toolset, object]] = {}
        for toolset in iter_toolsets(self.active_toolsets(state)):
            for key, value in toolset.objects.items():
                if key in objects:
                    raise ValueError(f"Runtime object {key!r} is defined twice.")
                objects[key] = (toolset, value)
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
            advantages=getattr(owner, "advantages", ()),
        )

    def _handler_owners(self) -> tuple[object | None, ...]:
        return (self.taskset, self.harness)

    def _extra_handlers(
        self,
        attr: str,
        builtins: Sequence[Callable[..., object]] = (),
        owners: Sequence[object | None] | None = None,
    ) -> list[Callable[..., object]]:
        handlers: list[Callable[..., object]] = list(builtins)
        for owner in owners or self._handler_owners():
            if owner is None:
                continue
            for handler in getattr(owner, "__dict__", {}).get(attr, ()):
                if not callable(handler):
                    raise TypeError(f"{attr} entries must be callable.")
                handlers.append(cast(Callable[..., object], handler))
        return handlers

    def _rollout_handlers(
        self,
        attr: str,
        state: State,
        stage: str | None = None,
    ) -> list[Callable[..., object]]:
        handlers: list[Callable[..., object]] = []
        for toolset in iter_toolsets(self.active_toolsets(state)):
            for handler in getattr(toolset, attr, ()):
                if not callable(handler):
                    raise TypeError(f"{attr} entries must be callable.")
                if (
                    stage is not None
                    and getattr(handler, f"{attr}_stage", "rollout") != stage
                ):
                    continue
                handlers.append(cast(Callable[..., object], handler))
            for _, method in inspect.getmembers(toolset, predicate=callable):
                if not getattr(method, attr, False):
                    continue
                if (
                    stage is not None
                    and getattr(method, f"{attr}_stage", "rollout") != stage
                ):
                    continue
                handlers.append(method)
        return sorted(
            unique_handlers(handlers),
            key=lambda handler: (
                -int(getattr(handler, f"{attr}_priority", 0)),
                str(getattr(handler, "__name__", "")),
            ),
        )

    def _group_handlers(
        self,
        attr: str,
        states: Sequence[State],
        stage: str | None = None,
    ) -> list[Callable[..., object]]:
        handlers: list[Callable[..., object]] = []
        for state in states:
            handlers.extend(self._rollout_handlers(attr, state, stage=stage))
        return sorted(
            unique_handlers(handlers),
            key=lambda handler: (
                -int(getattr(handler, f"{attr}_priority", 0)),
                str(getattr(handler, "__name__", "")),
            ),
        )

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
                return self.all_tools(state)
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
        object_specs = self.object_specs_for_state(state)
        if name not in object_specs:
            raise KeyError(f"Unknown runtime object {name!r}.")
        toolset, spec = object_specs[name]
        scope = toolset_object_scope(toolset)
        key = (id(toolset), self.scope_key(scope, state), name)
        if key in self.objects:
            return self.objects[key]
        if callable(spec):
            obj = await maybe_call_with_named_args(spec, task=task, state=state)
        else:
            obj = spec
        self.objects[key] = obj
        return obj

    async def resolve_object(self, name: str, task: Task, state: State) -> object:
        return await self._resolve_object(name, task, state)

    def scope_key(self, scope: str, state: State | None = None) -> str:
        if scope == "global":
            return "global"
        if state is None:
            raise ValueError(f"{scope} object cleanup requires state.")
        runtime = state.get("runtime", {})
        if scope == "group":
            return str(runtime.get("group_key") or state.get("trajectory_id"))
        if scope == "rollout":
            return str(state.get("trajectory_id"))
        raise ValueError("Object scope must be 'rollout', 'group', or 'global'.")

    async def release_objects(self, scope: str, state: State | None = None) -> None:
        scope_key = self.scope_key(scope, state) if scope != "global" else "global"
        for key, obj in list(self.objects.items()):
            _, object_scope_key, _ = key
            if object_scope_key != scope_key:
                continue
            await close_object(obj)
            del self.objects[key]

    async def release_model_client(self, state: State) -> None:
        key = state.get("runtime", {}).get("client_key")
        if not isinstance(key, str):
            return
        client = self.model_clients.pop(key, None)
        if key not in self.owned_model_clients:
            return
        self.owned_model_clients.remove(key)
        if client is not None:
            await close_object(client)

    async def release_all_model_clients(self) -> None:
        for key, client in list(self.model_clients.items()):
            del self.model_clients[key]
            if key in self.owned_model_clients:
                self.owned_model_clients.remove(key)
                await close_object(client)

    async def resolve_tool_sandbox(
        self, toolset: Toolset, task: Task, state: State
    ) -> object:
        from .utils.sandbox_utils import (
            SandboxHandle,
            create_tool_sandbox_lease,
            sandbox_scope,
            tool_sandbox_key,
        )

        _ = task
        sandbox = toolset.sandbox
        if isinstance(sandbox, str):
            if sandbox != "program":
                raise ValueError("Toolset sandbox string must be 'program'.")
            sandbox_record = state.get("runtime", {}).get("sandbox")
            if not isinstance(sandbox_record, Mapping):
                raise RuntimeError(
                    "Toolset sandbox='program' requires an active program sandbox."
                )
            lease_key = sandbox_record.get("lease_key")
            if (
                not isinstance(lease_key, list)
                or len(lease_key) != 2
                or not all(isinstance(item, str) for item in lease_key)
            ):
                raise RuntimeError(
                    "Program sandbox state is missing a runtime lease key."
                )
            lease = self.sandbox_leases.get((lease_key[0], lease_key[1]))
            if lease is None:
                raise RuntimeError("Program sandbox lease is no longer active.")
            return SandboxHandle(cast(Any, lease), state)
        if not isinstance(sandbox, Mapping):
            raise TypeError("Toolset sandbox must be a mapping.")
        scope = sandbox_scope(sandbox)
        key = (self.scope_key(scope, state), tool_sandbox_key(toolset))
        async with self.sandbox_lock:
            lease = self.sandbox_leases.get(key)
            if lease is None:
                lease = await create_tool_sandbox_lease(toolset)
                self.sandbox_leases[key] = lease
        return SandboxHandle(cast(Any, lease), state)

    async def resolve_program_sandbox(
        self, sandbox_config: Mapping[str, object], task: Task, state: State
    ) -> Any:
        from .utils.sandbox_utils import (
            create_sandbox_lease,
            program_sandbox_key,
            sandbox_scope,
        )

        _ = task
        scope = sandbox_scope(sandbox_config)
        key = (self.scope_key(scope, state), program_sandbox_key(sandbox_config))
        async with self.sandbox_lock:
            lease = self.sandbox_leases.get(key)
            if lease is None:
                lease = await create_sandbox_lease(sandbox_config, key[1])
                self.sandbox_leases[key] = lease
        return lease

    async def resolve_user_sandbox(
        self, user: User, task: Task, state: State
    ) -> object:
        from .utils.sandbox_utils import (
            SandboxHandle,
            create_scoped_sandbox_lease,
            sandbox_owner_key,
            sandbox_scope,
        )

        _ = task
        sandbox = user.sandbox
        if not isinstance(sandbox, Mapping):
            raise TypeError("User sandbox must be a mapping.")
        scope = sandbox_scope(sandbox)
        key = (self.scope_key(scope, state), sandbox_owner_key(user))
        async with self.sandbox_lock:
            lease = self.sandbox_leases.get(key)
            if lease is None:
                lease = await create_scoped_sandbox_lease(user, key[1])
                self.sandbox_leases[key] = lease
        return SandboxHandle(cast(Any, lease), state)

    async def release_sandboxes(self, scope: str, state: State) -> None:
        scope_key = self.scope_key(scope, state)
        for key, handle in list(self.sandbox_leases.items()):
            lease_scope_key, _ = key
            if lease_scope_key != scope_key:
                continue
            handle_scope = getattr(handle, "scope", None)
            if handle_scope != scope:
                continue
            await maybe_call_with_named_args(getattr(handle, "delete"))
            del self.sandbox_leases[key]

    async def ensure_global_sandboxes(self, state: State | None = None) -> None:
        from .utils.sandbox_utils import (
            create_scoped_sandbox_lease,
            create_tool_sandbox_lease,
            sandbox_owner_key,
            sandbox_scope,
            tool_sandbox_key,
        )

        async with self.sandbox_lock:
            for owner in self.sandbox_owners(state):
                sandbox = getattr(owner, "sandbox", None)
                if not isinstance(sandbox, Mapping):
                    continue
                if sandbox_scope(sandbox) != "global":
                    continue
                if isinstance(owner, Toolset):
                    sandbox_key = tool_sandbox_key(owner)
                else:
                    sandbox_key = sandbox_owner_key(owner)
                key = ("global", sandbox_key)
                if key in self.sandbox_leases:
                    continue
                if isinstance(owner, Toolset):
                    self.sandbox_leases[key] = await create_tool_sandbox_lease(owner)
                else:
                    self.sandbox_leases[key] = await create_scoped_sandbox_lease(
                        owner, sandbox_key
                    )

    def bind_global_sandboxes(self, state: State) -> None:
        from .utils.sandbox_utils import attach_sandbox_ref

        for key, lease in self.sandbox_leases.items():
            scope_key, _ = key
            if scope_key != "global":
                continue
            attach_sandbox_ref(state, cast(Any, lease))

    def sandbox_owners(self, state: State | None = None) -> list[object]:
        owners: list[object] = [*self.toolsets]
        if state is not None:
            owners.extend(self._rollout_toolsets(state))
        user = self._resolve_user()
        if user is not None:
            owners.append(user)
        return owners

    async def ensure_mcp_tools(self, state: State) -> None:
        from .utils.mcp_utils import connect_mcp_tool

        for key in self.mcp_scope_keys(state):
            if key in self.mcp_exit_stacks:
                continue
            exit_stack = AsyncExitStack()
            tools: dict[str, object] = {}
            exposed_tools: dict[str, object] = {}
            try:
                for toolset in self.active_toolsets(state):
                    await self._register_mcp_tools(
                        toolset,
                        [toolset],
                        connect_mcp_tool,
                        exit_stack,
                        tools,
                        exposed_tools,
                        state,
                        key,
                    )
            except BaseException:
                await exit_stack.aclose()
                raise
            self.mcp_exit_stacks[key] = exit_stack
            self.mcp_tools[key] = tools
            self.exposed_mcp_tools[key] = exposed_tools

    async def _register_mcp_tools(
        self,
        toolset: Toolset,
        parents: list[Toolset],
        connect_mcp_tool: Callable[
            [MCPTool, AsyncExitStack[bool | None]], Awaitable[Sequence[object]]
        ],
        exit_stack: AsyncExitStack,
        tools: dict[str, object],
        exposed_tools: dict[str, object],
        state: State,
        target_key: str,
    ) -> None:
        for item in toolset.tools:
            if isinstance(item, Toolset):
                await self._register_mcp_tools(
                    item,
                    [*parents, item],
                    connect_mcp_tool,
                    exit_stack,
                    tools,
                    exposed_tools,
                    state,
                    target_key,
                )
                continue
            if not isinstance(item, MCPTool):
                continue
            if self.mcp_scope_key(toolset, state) != target_key:
                continue
            handles = await connect_mcp_tool(item, exit_stack)
            for handle in handles:
                name = tool_name(handle)
                if (
                    name
                    in self._tools_for_toolsets(
                        self.active_toolsets(state), apply_visibility=False
                    )
                    or name in tools
                ):
                    raise ValueError(f"Tool {name!r} is defined twice.")
                tools[name] = handle
                if all(tool_visible(parent, name) for parent in parents):
                    exposed_tools[name] = handle

    async def close_mcp_tools(self, state: State, scope: str = "rollout") -> None:
        for key in self.mcp_scope_keys(state, scope=scope):
            exit_stack = self.mcp_exit_stacks.pop(key, None)
            self.mcp_tools.pop(key, None)
            self.exposed_mcp_tools.pop(key, None)
            if exit_stack is not None:
                await exit_stack.aclose()

    async def close_all_mcp_tools(self) -> None:
        for key, exit_stack in list(self.mcp_exit_stacks.items()):
            self.mcp_tools.pop(key, None)
            self.exposed_mcp_tools.pop(key, None)
            del self.mcp_exit_stacks[key]
            await exit_stack.aclose()

    def all_tools(self, state: State) -> dict[str, object]:
        tools = self._tools_for_toolsets(
            self.active_toolsets(state), apply_visibility=False
        )
        for name, tool in self.mcp_tools_for_state(state, exposed=False).items():
            if name in tools:
                raise ValueError(f"Tool {name!r} is defined twice.")
            tools[name] = tool
        return tools

    def unfiltered_exposed_tools(self, state: State) -> dict[str, object]:
        tools = self._tools_for_toolsets(
            self.active_toolsets(state), apply_visibility=True
        )
        for name, tool in self.mcp_tools_for_state(state, exposed=True).items():
            if name in tools:
                raise ValueError(f"Tool {name!r} is defined twice.")
            tools[name] = tool
        return tools

    def all_exposed_tools(self, state: State) -> dict[str, object]:
        tools = self.unfiltered_exposed_tools(state)
        selected = state.get("runtime", {}).get("tools")
        if selected is None:
            return tools
        if isinstance(selected, Mapping):
            unknown_keys = set(selected) - {"show", "hide"}
            if unknown_keys:
                raise ValueError(
                    f"state.runtime.tools has unknown keys: {sorted(unknown_keys)}."
                )
            if selected.get("show") is not None and selected.get("hide") is not None:
                raise ValueError("state.runtime.tools accepts show or hide, not both.")
            if selected.get("show") is not None:
                selected_names = string_list(
                    selected["show"], "state.runtime.tools.show"
                )
                unknown = sorted(set(selected_names) - set(tools))
                if unknown:
                    raise KeyError(f"Unknown requested tools: {unknown}.")
                return {name: tools[name] for name in selected_names}
            elif selected.get("hide") is not None:
                hidden_names = set(
                    string_list(selected["hide"], "state.runtime.tools.hide")
                )
                unknown = sorted(hidden_names - set(tools))
                if unknown:
                    raise KeyError(f"Unknown hidden tools: {unknown}.")
                return {
                    name: tool
                    for name, tool in tools.items()
                    if name not in hidden_names
                }
            else:
                return tools
        raise TypeError("state.runtime.tools must be a mapping with show or hide.")

    def mcp_tools_for_state(self, state: State, exposed: bool) -> dict[str, object]:
        source = self.exposed_mcp_tools if exposed else self.mcp_tools
        tools: dict[str, object] = {}
        for key in self.mcp_scope_keys(state):
            for name, tool in source.get(key, {}).items():
                if name in tools:
                    raise ValueError(f"Tool {name!r} is defined twice.")
                tools[name] = tool
        return tools

    def mcp_scope_keys(self, state: State, scope: str | None = None) -> list[str]:
        keys: list[str] = []

        def visit(toolset: Toolset) -> None:
            for item in toolset.tools:
                if isinstance(item, Toolset):
                    visit(item)
                    continue
                if not isinstance(item, MCPTool):
                    continue
                item_scope = toolset_object_scope(toolset)
                if scope is not None and item_scope != scope:
                    continue
                key = self.mcp_scope_key(toolset, state)
                if key not in keys:
                    keys.append(key)

        for toolset in self.active_toolsets(state):
            visit(toolset)
        return keys

    def mcp_scope_key(self, toolset: Toolset, state: State) -> str:
        scope = toolset_object_scope(toolset)
        return f"{scope}:{self.scope_key(scope, state)}:{id(toolset)}"


def tool_visible(toolset: Toolset, name: str) -> bool:
    if toolset.show is not None and name not in toolset.show:
        return False
    if toolset.hide is not None and name in toolset.hide:
        return False
    return True


def toolset_object_scope(toolset: Toolset) -> str:
    if toolset.scope is not None:
        return toolset.scope
    return "rollout" if toolset.write else "global"


async def state_done(task: Task, state: State) -> bool:
    _ = task
    return bool(state.get("done") or state.get("is_completed"))


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


def string_list(value: object, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence) or isinstance(value, bytes):
        raise TypeError(f"{field} must be a string or list of strings.")
    result = [str(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate names.")
    return result


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


async def close_object(obj: object) -> None:
    for name in ("aclose", "close", "delete", "teardown"):
        fn = getattr(obj, name, None)
        if callable(fn):
            await maybe_call_with_named_args(fn)
            return


def load_runtime(runtime_id: str) -> Runtime:
    runtime = _RUNTIME_REGISTRY.get(runtime_id)
    if runtime is None:
        raise RuntimeError(f"No live v1 runtime registered for id {runtime_id!r}.")
    return runtime


def load_runtime_from_state(state: Mapping[str, object]) -> Runtime:
    runtime_state = state.get("runtime")
    if not isinstance(runtime_state, Mapping):
        raise RuntimeError("State has no runtime metadata.")
    runtime_state = cast(Mapping[str, object], runtime_state)
    runtime_id = runtime_state.get("runtime_id")
    if not isinstance(runtime_id, str) or not runtime_id:
        raise RuntimeError("State has no live runtime id.")
    return load_runtime(runtime_id)
