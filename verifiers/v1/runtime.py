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
    sort_handlers,
    unique_handlers,
    validate_handler_args,
)
from .utils.scoring_utils import SignalRecord, build_signals, collect_signals
from .utils.scoring_utils import score_group as score_group_signals
from .utils.scoring_utils import score_rollout as score_rollout_signals
from .utils.trajectory_utils import render_completion
from .utils.artifact_utils import artifact_format, artifact_key, artifact_optional
from .utils.artifact_utils import artifact_path
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


class BorrowedTool:
    def __init__(self, runtime: Runtime, handle_id: str, name: str):
        self.runtime = runtime
        self.handle_id = handle_id
        self.name = name
        self.__name__ = name

    @property
    def tool_def(self) -> Tool:
        return self.runtime.borrowed_tool_def(self.handle_id, self.name)

    async def __call__(self, **kwargs: object) -> object:
        return await self.runtime.call_borrowed_tool(
            self.handle_id, self.name, **kwargs
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
        self.trajectories: dict[str, list[Mapping[str, object]]] = {}
        self.tool_handles: dict[str, tuple[Task, State, tuple[str, ...]]] = {}
        self.stop_conditions = collect_handlers(
            self._handler_owners(),
            "stop",
            self._extra_handlers("stop", builtins=[state_done]),
        )
        validate_handler_args(
            self.stop_conditions, {"task", "state"}, "stop", "rollout"
        )
        self.rollout_update = collect_handlers(
            self._handler_owners(),
            "update",
            self._extra_handlers("update", builtins=[render_completion]),
            stage="rollout",
        )
        self.group_update = collect_handlers(
            self._handler_owners(),
            "update",
            self._extra_handlers("update"),
            stage="group",
        )
        validate_handler_args(
            self.rollout_update, {"task", "state"}, "update", "rollout"
        )
        validate_handler_args(self.group_update, {"tasks", "states"}, "update", "group")
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
    def has_group_stage(self) -> bool:
        return bool(self.group_update or self.group_signals or self.group_cleanup)

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
        state["tools"] = sorted(self.all_exposed_tools(state))
        self.register_trajectory(state)

    def register_tool_handle(self, state: State, names: Sequence[str]) -> str:
        task = Task(cast(Mapping[str, Any], state["task"])).freeze()
        available = self.all_exposed_tools(state)
        unknown = sorted(set(names) - set(available))
        if unknown:
            raise KeyError(f"Unknown borrowed tools: {unknown}.")
        handle_id = uuid.uuid4().hex
        self.tool_handles[handle_id] = (task, state, tuple(names))
        return handle_id

    def _borrowed_tool_handle(
        self, handle_id: str
    ) -> tuple[Task, State, tuple[str, ...]]:
        handle = self.tool_handles.get(handle_id)
        if handle is None:
            raise RuntimeError(f"No live tool handle registered for {handle_id!r}.")
        return handle

    def borrowed_tool_def(self, handle_id: str, name: str) -> Tool:
        _, source_state, names = self._borrowed_tool_handle(handle_id)
        if name not in names:
            raise KeyError(f"Tool handle does not expose {name!r}.")
        source_tool = self.all_exposed_tools(source_state)[name]
        return self._tool_def(name, source_tool, source_state)

    async def call_borrowed_tool(
        self, handle_id: str, name: str, **kwargs: object
    ) -> object:
        source_task, source_state, names = self._borrowed_tool_handle(handle_id)
        if name not in names:
            raise KeyError(f"Tool handle does not expose {name!r}.")
        return await self._call_tool(
            name, source_task, source_state, exposed=True, **kwargs
        )

    def release_tool_handles(self, state: State) -> None:
        for handle_id, (_, source_state, _) in list(self.tool_handles.items()):
            if source_state is state:
                del self.tool_handles[handle_id]

    def register_trajectory(self, state: State) -> None:
        trajectory = state.get("trajectory")
        if trajectory is None:
            return
        if not isinstance(trajectory, list):
            raise TypeError("state.trajectory must be a list.")
        self.trajectories[str(state["trajectory_id"])] = cast(
            list[Mapping[str, object]], trajectory
        )

    def resolved_handles(self, state: State) -> Mapping[str, object]:
        runtime = state.get("runtime", {})
        if not isinstance(runtime, Mapping):
            return {}
        resolved = runtime.get("resolved") or {}
        if not isinstance(resolved, Mapping):
            raise TypeError("state.runtime.resolved must be a mapping.")
        return cast(Mapping[str, object], resolved)

    def resolved_handle(self, state: State, name: str) -> Mapping[str, object] | None:
        handle = self.resolved_handles(state).get(name)
        if handle is None:
            return None
        if not isinstance(handle, Mapping):
            raise TypeError(f"state.runtime.resolved.{name} must be a mapping.")
        return cast(Mapping[str, object], handle)

    def handle_runtime(self, handle: Mapping[str, object], name: str) -> Runtime:
        runtime_id = handle.get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id:
            raise TypeError(
                f"state.runtime.resolved.{name}.runtime_id must be a string."
            )
        return load_runtime(runtime_id)

    def resolve_trajectory(self, state: State) -> None:
        handle = self.resolved_handle(state, "trajectory")
        if handle is None:
            state.setdefault("trajectory", [])
            self.register_trajectory(state)
            return
        if handle.get("mode", "append") != "append":
            raise ValueError("state.runtime.resolved.trajectory.mode must be 'append'.")
        trajectory_id = handle.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise TypeError(
                "state.runtime.resolved.trajectory.trajectory_id must be a string."
            )
        runtime = self.handle_runtime(handle, "trajectory")
        trajectory = runtime.trajectories.get(trajectory_id)
        if trajectory is None:
            raise RuntimeError(f"No live trajectory registered for {trajectory_id!r}.")
        state["trajectory"] = trajectory

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

    def model_client(self, state: State) -> Client:
        handle = self.resolved_handle(state, "model")
        runtime = self
        if handle is None:
            key = str(state.get("runtime", {}).get("client_key") or "default")
        else:
            runtime = self.handle_runtime(handle, "model")
            key = handle.get("client_key")
            if not isinstance(key, str) or not key:
                raise TypeError(
                    "state.runtime.resolved.model.client_key must be a string."
                )
        client = runtime.model_clients.get(key)
        if client is None:
            raise RuntimeError("Harness has no model client for intercepted requests.")
        return client

    def client_type(self, state: State) -> ClientType:
        raw_client_type = state.get("runtime", {}).get("client_type")
        if raw_client_type is None:
            handle = self.resolved_handle(state, "model")
            if handle is not None:
                raw_client_type = handle.get("client_type")
        if raw_client_type is None:
            return "openai_chat_completions"
        if raw_client_type not in get_args(ClientType):
            raise ValueError(f"Unsupported client type: {raw_client_type!r}")
        return cast(ClientType, raw_client_type)

    def model(self, state: State) -> str:
        model = state.get("runtime", {}).get("model")
        if model is None:
            handle = self.resolved_handle(state, "model")
            if handle is not None:
                model = handle.get("model")
        if not isinstance(model, str) or not model:
            raise RuntimeError("Harness has no model for intercepted requests.")
        return model

    def sampling_args(self, state: State) -> dict[str, Any]:
        sampling = state.get("runtime", {}).get("sampling_args") or {}
        if not sampling:
            handle = self.resolved_handle(state, "model")
            if handle is not None:
                sampling = handle.get("sampling_args") or {}
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
            validate_bound_arg(user.fn, name, f"User binding {name!r}")
            validate_binding_source_root(
                binding_source_root(source), f"User binding {name!r}"
            )
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
        hidden_args = self.hidden_tool_args(name, state)
        schema_tool = tool
        filtered_signature = self._tool_signature(name, tool, state)
        if hidden_args and filtered_signature is not None:
            schema_tool = schema_callable(tool, filtered_signature)
        tool_def = convert_func_to_tool_def(schema_tool)
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

    def hidden_tool_args(self, name: str, state: State) -> set[str]:
        hidden_args = {"runtime", "task", "state"}
        owner = self.tool_owner(name, state)
        if owner is not None and owner.sandbox is not None:
            hidden_args.add("sandbox")
        if owner is not None:
            for binding_key in owner.bindings:
                tool_name_prefix, arg_name = binding_key_parts(binding_key)
                if tool_name_prefix == name:
                    validate_bound_arg(
                        self.all_tools(state)[name],
                        arg_name,
                        f"Tool binding {binding_key!r}",
                    )
                    hidden_args.add(arg_name)
        return hidden_args

    async def call_tool(
        self, tool_name: str, task: Task, state: State, **kwargs: object
    ) -> object:
        return await self._call_tool(tool_name, task, state, True, **kwargs)

    async def is_completed(self, task: Task, state: State) -> bool:
        conditions = unique_handlers(
            [*self.stop_conditions, *self._rollout_handlers("stop", state)]
        )
        for condition in conditions:
            extra_kwargs = await self.binding_kwargs(condition, task, state)
            completed = await maybe_call_with_named_args(
                condition, task=task, state=state, **extra_kwargs
            )
            if completed:
                state._set_completed(True)
                state._set_truncated(
                    any(
                        step.get("is_truncated", False)
                        for step in state.get("trajectory", [])
                        if isinstance(step, Mapping)
                    )
                )
                state._set_stop_condition(
                    getattr(condition, "__name__", type(condition).__name__)
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

        tools = self.all_exposed_tools(state) if exposed else self.all_tools(state)
        tool = tools[tool_name]
        tool_def = self._tool_def(tool_name, tool, state)
        call.__name__ = tool_def.name
        call.__doc__ = tool_def.description
        signature = self._tool_signature(tool_name, tool, state)
        if signature is not None:
            setattr(call, "__signature__", signature)
        return call

    def _tool_signature(
        self, tool_name: str, tool: object, state: State
    ) -> inspect.Signature | None:
        if not callable(tool):
            return None
        try:
            signature = inspect.signature(tool)
        except (TypeError, ValueError):
            return None
        hidden_args = self.hidden_tool_args(tool_name, state)
        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.name not in hidden_args
        ]
        return signature.replace(parameters=parameters)

    async def _call_tool_callable(
        self,
        tool: Callable[..., object],
        tool_name: str,
        task: Task,
        state: State,
        visible_kwargs: Mapping[str, object],
        hidden_kwargs: Mapping[str, object],
    ) -> object:
        call_kwargs = dict(visible_kwargs)
        try:
            signature = inspect.signature(tool)
        except (TypeError, ValueError):
            if hidden_kwargs:
                raise TypeError(
                    f"Tool {tool_name!r} uses hidden args, but its signature "
                    "cannot be inspected."
                )
            result = tool(**call_kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        parameters = signature.parameters
        hidden_values: dict[str, object] = {
            "task": task,
            "state": state,
            **hidden_kwargs,
        }
        for arg_name, value in hidden_values.items():
            if arg_name in parameters:
                call_kwargs[arg_name] = value
            elif arg_name in hidden_kwargs:
                raise TypeError(
                    f"Tool {tool_name!r} has hidden arg {arg_name!r}, but does "
                    "not declare it in its signature."
                )
        result = tool(**call_kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

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
        visible_kwargs = dict(kwargs)
        hidden_kwargs: dict[str, object] = {}
        owner = self.tool_owner(tool_name, state)
        for hidden_arg in ("runtime", "task", "state"):
            if hidden_arg in visible_kwargs:
                raise ValueError(f"Tool arg {tool_name}.{hidden_arg} is reserved.")
        if owner is not None and owner.sandbox is not None:
            if "sandbox" in visible_kwargs:
                raise ValueError(f"Tool arg {tool_name}.sandbox is reserved.")
            hidden_kwargs["sandbox"] = await self.resolve_tool_sandbox(
                owner, task, state
            )
        for binding_key, source in (
            owner.bindings if owner is not None else {}
        ).items():
            tool_name_prefix, arg_name = binding_key_parts(binding_key)
            if tool_name_prefix != tool_name:
                continue
            if arg_name in visible_kwargs or arg_name in hidden_kwargs:
                raise ValueError(f"Tool arg {tool_name}.{arg_name} is already set.")
            hidden_kwargs[arg_name] = await self.resolve_tool_binding(
                owner, source, task, state
            )
        return await self._call_tool_callable(
            cast(Callable[..., object], tools[tool_name]),
            tool_name,
            task=task,
            state=state,
            visible_kwargs=visible_kwargs,
            hidden_kwargs=hidden_kwargs,
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

    async def update_rollout(self, task: Task, state: State) -> State:
        handlers = sort_handlers(
            unique_handlers(
                [
                    *self.rollout_update,
                    *self._rollout_handlers("update", state, stage="rollout"),
                ]
            ),
            "update",
        )
        validate_handler_args(handlers, {"task", "state"}, "update", "rollout")
        await self.run_rollout_handlers(handlers, task=task, state=state)
        return state

    async def update_group(self, tasks: list[Task], states: list[State]) -> list[State]:
        handlers = sort_handlers(
            unique_handlers(
                [
                    *self.group_update,
                    *self._group_handlers("update", states, stage="group"),
                ]
            ),
            "update",
        )
        validate_handler_args(handlers, {"tasks", "states"}, "update", "group")
        await run_handlers(handlers, tasks=tasks, states=states)
        return states

    async def score_rollout(self, task: Task, state: State) -> State:
        await score_rollout_signals(
            self.rollout_signals,
            task,
            state,
            resolve_kwargs=self.binding_kwargs,
        )
        return state

    async def score_group(self, tasks: list[Task], states: list[State]) -> list[State]:
        await self.update_group(tasks, states)
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
        await self.run_rollout_handlers(handlers, task=task, state=state)
        await self.release_objects("rollout", state)
        await self.release_user_objects("rollout", state)
        await self.release_sandboxes(scope="rollout", state=state)
        await self.close_mcp_tools(state)
        await self.release_model_client(state)
        self.release_tool_handles(state)

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
        self.tool_handles.clear()
        await self.close_all_mcp_tools()
        await self.release_all_model_clients()
        _RUNTIME_REGISTRY.pop(self.runtime_id, None)

    async def run_rollout_handlers(
        self,
        handlers: Iterable[Callable[..., object]],
        task: Task,
        state: State,
    ) -> None:
        for handler in handlers:
            extra_kwargs = await self.binding_kwargs(handler, task, state)
            await maybe_call_with_named_args(
                handler, task=task, state=state, **extra_kwargs
            )

    async def binding_kwargs(
        self,
        fn: Callable[..., object],
        task: Mapping[str, Any],
        state: dict[str, Any],
    ) -> dict[str, object]:
        name = function_name(fn)
        kwargs: dict[str, object] = {}
        for binding_key, source in self.bindings_for_callable(
            fn, cast(State, state)
        ).items():
            prefix, arg_name = binding_key_parts(binding_key)
            if prefix != name:
                continue
            validate_bound_arg(fn, arg_name, f"Binding {binding_key!r}")
            kwargs[arg_name] = await self.resolve_binding(
                source, cast(Task, task), cast(State, state)
            )
        return kwargs

    async def resolve_binding(self, source: object, task: Task, state: State) -> object:
        if isinstance(source, str):
            if binding_source_root(source) == "objects":
                raise ValueError(
                    "objects.* bindings are private to the owning Toolset/User "
                    "callable."
                )
            return await self._resolve_path(source, task, state)
        if callable(source):
            return await maybe_call_with_named_args(source, task=task, state=state)
        return source

    async def resolve_tool_binding(
        self, toolset: Toolset | None, source: object, task: Task, state: State
    ) -> object:
        if isinstance(source, str):
            root, separator, tail = source.partition(".")
            if root == "objects":
                if toolset is None:
                    raise ValueError("objects.* tool bindings require a Toolset owner.")
                if not separator:
                    raise ValueError("objects binding sources must name an object.")
                name, _, rest = tail.partition(".")
                value = await self._resolve_toolset_object(toolset, name, task, state)
                if rest:
                    return _read_path(value, rest)
                return value
        return await self.resolve_binding(source, task, state)

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
                    raise KeyError(f"Unknown user object {name!r}.")
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
        _ = task, state
        if name not in user.objects:
            raise KeyError(f"Unknown user object {name!r}.")
        key = (id(user), self.scope_key(user.scope, state), name)
        if key in self.user_objects:
            return self.user_objects[key]
        spec = user.objects[name]
        obj = await resolve_object_factory(spec, f"User object {name!r}")
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

    def validate_bindings(self, state: State) -> None:
        for toolset in iter_toolsets(self.active_toolsets(state)):
            self._validate_toolset_bindings(toolset)
        user = self._resolve_user()
        if user is not None:
            for name, source in user.bindings.items():
                validate_bound_arg(user.fn, name, f"User binding {name!r}")
                source_root = binding_source_root(source)
                validate_binding_source_root(source_root, f"User binding {name!r}")
                if source_root == "objects":
                    object_name = binding_object_name(source)
                    if object_name not in user.objects:
                        raise KeyError(
                            f"User binding {name!r} references unknown User object "
                            f"{object_name!r}."
                        )

    def _validate_toolset_bindings(self, toolset: Toolset) -> None:
        targets = self._toolset_binding_targets(toolset)
        for binding_key, source in toolset.bindings.items():
            target_name, arg_name = binding_key_parts(binding_key)
            target = targets.get(target_name)
            if target is None:
                raise ValueError(
                    f"Binding {binding_key!r} does not match a callable owned by "
                    "the same Toolset."
                )
            target_kind, fn = target
            validate_bound_arg(fn, arg_name, f"Binding {binding_key!r}")
            source_root = binding_source_root(source)
            validate_binding_source_root(source_root, f"Binding {binding_key!r}")
            if source_root == "objects" and target_kind != "tool":
                raise ValueError(
                    f"Binding {binding_key!r} uses objects.*, which is only valid "
                    "for callable tools owned by the same Toolset."
                )
            if source_root == "objects":
                object_name = binding_object_name(source)
                if object_name not in toolset.objects:
                    raise KeyError(
                        f"Binding {binding_key!r} references unknown Toolset object "
                        f"{object_name!r}."
                    )

    def _toolset_binding_targets(
        self, toolset: Toolset
    ) -> dict[str, tuple[str, Callable[..., object]]]:
        targets: dict[str, tuple[str, Callable[..., object]]] = {}

        def add_target(name: str, kind: str, fn: Callable[..., object]) -> None:
            if name in targets:
                raise ValueError(f"Toolset binding target {name!r} is defined twice.")
            targets[name] = (kind, fn)

        for item in toolset.tools:
            if isinstance(item, Toolset | MCPTool):
                continue
            if callable(item):
                add_target(tool_name(item), "tool", cast(Callable[..., object], item))
        for attr in ("stops", "updates", "cleanups"):
            for fn in getattr(toolset, attr):
                if callable(fn):
                    add_target(
                        function_name(fn),
                        attr[:-1],
                        cast(Callable[..., object], fn),
                    )
        for _, method in inspect.getmembers(toolset, predicate=callable):
            if any(
                getattr(method, attr, False) for attr in ("stop", "update", "cleanup")
            ):
                add_target(
                    function_name(method),
                    "handler",
                    cast(Callable[..., object], method),
                )
        return targets

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

    def bindings_for_callable(
        self, fn: Callable[..., object], state: State
    ) -> dict[str, object]:
        bindings: dict[str, object] = {}
        for toolset in iter_toolsets(self.active_toolsets(state)):
            targets = self._toolset_binding_targets(toolset)
            target = targets.get(function_name(fn))
            if target is None or not same_callable(target[1], fn):
                continue
            for key, value in toolset.bindings.items():
                target_name, _ = binding_key_parts(key)
                if target_name != function_name(fn):
                    continue
                if key in bindings:
                    raise ValueError(f"Binding {key!r} is defined twice.")
                bindings[key] = value
        return bindings

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
        collection_attr = handler_collection_attr(attr)
        for owner in owners or self._handler_owners():
            if owner is None:
                continue
            for handler in getattr(owner, "__dict__", {}).get(collection_attr, ()):
                if not callable(handler):
                    raise TypeError(f"{collection_attr} entries must be callable.")
                handlers.append(cast(Callable[..., object], handler))
        return handlers

    def _rollout_handlers(
        self,
        attr: str,
        state: State,
        stage: str | None = None,
    ) -> list[Callable[..., object]]:
        handlers: list[Callable[..., object]] = []
        collection_attr = handler_collection_attr(attr)
        for toolset in iter_toolsets(self.active_toolsets(state)):
            for handler in getattr(toolset, collection_attr, ()):
                if not callable(handler):
                    raise TypeError(f"{collection_attr} entries must be callable.")
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
        return sort_handlers(unique_handlers(handlers), attr)

    def _group_handlers(
        self,
        attr: str,
        states: Sequence[State],
        stage: str | None = None,
    ) -> list[Callable[..., object]]:
        handlers: list[Callable[..., object]] = []
        for state in states:
            handlers.extend(self._rollout_handlers(attr, state, stage=stage))
        return sort_handlers(unique_handlers(handlers), attr)

    async def _collect_artifact(self, spec: object, task: Task, state: State) -> object:
        if callable(spec):
            return await maybe_call_with_named_args(spec, task=task, state=state)
        if not isinstance(spec, Mapping):
            raise TypeError("Artifact specs must be callables or mappings.")
        spec_map = cast(Mapping[str, object], spec)
        path = artifact_path(spec_map)
        optional = artifact_optional(spec_map)
        matches = sorted(glob.glob(path.format(**state)))
        if not matches:
            if optional:
                return None
            raise FileNotFoundError(f"Artifact path matched no files: {path!r}")
        format_name = artifact_format(spec_map)
        with open(matches[0], encoding="utf-8") as f:
            if format_name == "json":
                data: object = json.load(f)
            elif format_name == "text":
                data = f.read()
            else:
                raise ValueError(f"Unsupported artifact format: {format_name!r}")
        key = artifact_key(spec_map)
        if key is not None:
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
            raise ValueError(
                "objects.* bindings are private to the owning Toolset/User callable."
            )
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

    async def _resolve_toolset_object(
        self, toolset: Toolset, name: str, task: Task, state: State
    ) -> object:
        _ = task
        if name not in toolset.objects:
            raise KeyError(f"Unknown Toolset object {name!r}.")
        spec = toolset.objects[name]
        scope = toolset_object_scope(toolset)
        key = (id(toolset), self.scope_key(scope, state), name)
        if key in self.objects:
            return self.objects[key]
        obj = await resolve_object_factory(spec, f"Toolset object {name!r}")
        self.objects[key] = obj
        return obj

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
        if self.resolved_handle(state, "model") is not None:
            return
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
            lease = self._active_program_sandbox_lease(state)
            if lease is None:
                raise RuntimeError(
                    "Toolset sandbox='program' requires an active program sandbox."
                )
            return SandboxHandle(cast(Any, lease), state)
        if not isinstance(sandbox, Mapping):
            raise TypeError("Toolset sandbox must be a mapping.")
        prefer = sandbox.get("prefer")
        if prefer is not None:
            if prefer != "program":
                raise ValueError("Toolset sandbox.prefer must be 'program'.")
            lease = self._active_program_sandbox_lease(state)
            if lease is not None:
                return SandboxHandle(cast(Any, lease), state)
        scope = sandbox_scope(sandbox)
        key = (self.scope_key(scope, state), tool_sandbox_key(toolset))
        async with self.sandbox_lock:
            lease = self.sandbox_leases.get(key)
            if lease is None:
                lease = await create_tool_sandbox_lease(toolset)
                self.sandbox_leases[key] = lease
        return SandboxHandle(cast(Any, lease), state)

    def _active_program_sandbox_lease(self, state: State) -> object | None:
        sandbox_handle = self.resolved_handle(state, "sandbox")
        if sandbox_handle is not None:
            return self._sandbox_lease_from_handle(sandbox_handle, "sandbox")
        sandbox_record = state.get("runtime", {}).get("sandbox")
        if not isinstance(sandbox_record, Mapping):
            return None
        lease_key = sandbox_record.get("lease_key")
        if (
            not isinstance(lease_key, list)
            or len(lease_key) != 2
            or not all(isinstance(item, str) for item in lease_key)
        ):
            raise RuntimeError("Program sandbox state is missing a runtime lease key.")
        resolved_lease_key = (str(lease_key[0]), str(lease_key[1]))
        lease = self.sandbox_leases.get(resolved_lease_key)
        if lease is None:
            raise RuntimeError("Program sandbox lease is no longer active.")
        setattr(lease, "scope_key", resolved_lease_key[0])
        return lease

    async def resolve_program_sandbox(
        self, sandbox_config: Mapping[str, object], task: Task, state: State
    ) -> Any:
        from .utils.sandbox_utils import (
            create_sandbox_lease,
            program_sandbox_key,
            sandbox_scope,
        )

        _ = task
        sandbox_handle = self.resolved_handle(state, "sandbox")
        if sandbox_handle is not None:
            return self._sandbox_lease_from_handle(sandbox_handle, "sandbox")
        scope = sandbox_scope(sandbox_config)
        key = (self.scope_key(scope, state), program_sandbox_key(sandbox_config))
        async with self.sandbox_lock:
            lease = self.sandbox_leases.get(key)
            if lease is None:
                lease = await create_sandbox_lease(sandbox_config, key[1])
                self.sandbox_leases[key] = lease
            setattr(lease, "scope_key", key[0])
        return lease

    def _sandbox_lease_from_handle(
        self, handle: Mapping[str, object], name: str
    ) -> object:
        runtime = self.handle_runtime(handle, name)
        lease_key = handle.get("lease_key")
        if (
            not isinstance(lease_key, list)
            or len(lease_key) != 2
            or not all(isinstance(item, str) for item in lease_key)
        ):
            raise RuntimeError(
                f"state.runtime.resolved.{name} is missing a runtime lease key."
            )
        resolved_lease_key = (str(lease_key[0]), str(lease_key[1]))
        lease = runtime.sandbox_leases.get(resolved_lease_key)
        if lease is None:
            raise RuntimeError(f"Resolved {name} sandbox lease is no longer active.")
        setattr(lease, "scope_key", resolved_lease_key[0])
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
        for name, tool in self.borrowed_tools_for_state(state).items():
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
        for name, tool in self.borrowed_tools_for_state(state).items():
            if name in tools:
                raise ValueError(f"Tool {name!r} is defined twice.")
            tools[name] = tool
        return tools

    def borrowed_tools_for_state(self, state: State) -> dict[str, object]:
        handle = self.resolved_handle(state, "tools")
        if handle is None:
            return {}
        handle_id = handle.get("handle_id")
        if not isinstance(handle_id, str) or not handle_id:
            raise TypeError("state.runtime.resolved.tools.handle_id must be a string.")
        names = string_list(handle.get("names"), "state.runtime.resolved.tools.names")
        source_runtime = self.handle_runtime(handle, "tools")
        return {name: BorrowedTool(source_runtime, handle_id, name) for name in names}

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
    return bool(state.get("done"))


def handler_collection_attr(attr: str) -> str:
    return {
        "stop": "stops",
        "update": "updates",
        "cleanup": "cleanups",
        "teardown": "teardowns",
    }.get(attr, attr)


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


def schema_callable(
    tool: object, signature: inspect.Signature
) -> Callable[..., object]:
    def call_for_schema(**kwargs: object) -> None:
        _ = kwargs
        return None

    call_for_schema.__name__ = tool_name(tool)
    call_for_schema.__doc__ = getattr(tool, "__doc__", None)
    setattr(call_for_schema, "__signature__", signature)
    return call_for_schema


def function_name(fn: Callable[..., object]) -> str:
    name = getattr(fn, "__name__", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Callable bindings require a stable __name__.")
    return name


def binding_key_parts(key: object) -> tuple[str, str]:
    if not isinstance(key, str):
        raise TypeError("Binding keys must be strings.")
    target, separator, arg_name = key.partition(".")
    if separator != "." or not target or not arg_name or "." in arg_name:
        raise ValueError(f"Binding key {key!r} must be 'callable.arg'.")
    return target, arg_name


def binding_source_root(source: object) -> str | None:
    if not isinstance(source, str):
        return None
    root, _, _ = source.partition(".")
    return root


def validate_binding_source_root(root: str | None, context: str) -> None:
    if root is None:
        return
    if root not in {"task", "state", "runtime", "objects", "tools"}:
        raise ValueError(
            f"{context} source root must be task, state, runtime, objects, or tools."
        )


def binding_object_name(source: object) -> str:
    if not isinstance(source, str):
        raise TypeError("Object binding source must be a string.")
    root, separator, tail = source.partition(".")
    if root != "objects" or not separator:
        raise ValueError("Object binding source must be 'objects.name'.")
    name, _, _ = tail.partition(".")
    if not name:
        raise ValueError("Object binding source must be 'objects.name'.")
    return name


def validate_bound_arg(
    fn: Callable[..., object] | object, arg_name: str, context: str
) -> None:
    if arg_name in {"task", "state", "runtime"}:
        raise ValueError(f"{context} cannot bind reserved arg {arg_name!r}.")
    if not callable(fn):
        raise TypeError(f"{context} target is not callable.")
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} target signature cannot be inspected.") from exc
    if arg_name not in signature.parameters:
        name = (
            getattr(fn, "__name__", None)
            or getattr(fn, "name", None)
            or type(fn).__name__
        )
        raise TypeError(
            f"{context} targets {name!r}, but {name!r} does not declare "
            f"arg {arg_name!r}."
        )


def same_callable(left: Callable[..., object], right: Callable[..., object]) -> bool:
    if left is right:
        return True
    left_self = getattr(left, "__self__", None)
    right_self = getattr(right, "__self__", None)
    left_func = getattr(left, "__func__", None)
    right_func = getattr(right, "__func__", None)
    return left_self is right_self and left_func is not None and left_func is right_func


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


async def resolve_object_factory(spec: object, context: str) -> object:
    if not callable(spec):
        return spec
    try:
        signature = inspect.signature(spec)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} factory signature cannot be inspected.") from exc
    if signature.parameters:
        raise TypeError(f"{context} factory must accept no arguments.")
    value = cast(Callable[[], object], spec)()
    if inspect.isawaitable(value):
        return await cast(Awaitable[object], value)
    return value


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
