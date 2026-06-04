import asyncio
import glob
import hashlib
import inspect
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from importlib.abc import Traversable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    TypeAlias,
    cast,
    get_args,
    runtime_checkable,
)

from verifiers.clients import Client, resolve_client
from verifiers.types import Messages, Response, Tool
from verifiers.types import ClientConfig, ClientType, SamplingArgs
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.utils.client_utils import resolve_client_config
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.response_utils import parse_response_message, parse_response_tokens
from verifiers.utils.tool_utils import convert_func_to_tool_def

from .utils.binding_utils import (
    BindingSource,
    GROUP_FRAMEWORK_ARGS,
    ROLLOUT_FRAMEWORK_ARGS,
    binding_key_parts,
    binding_object_name,
    binding_source_root,
    function_name,
    owner_object_name,
    read_path,
    same_callable,
    validate_binding_source,
    validate_bound_arg,
    validate_callable_source,
)
from .utils.config_callable_utils import CallableKind
from .utils.config_utils import resolve_config_object
from .utils.lifecycle_utils import collect_handlers, handler_is_marked, handler_stage
from .utils.lifecycle_utils import run_handlers, sort_handlers
from .utils.lifecycle_utils import state_done, unique_handlers, validate_handler_args
from .utils.object_utils import close_object, resolve_object_factory
from .utils.runtime_registry import load_runtime, register_runtime, unregister_runtime
from .utils.runtime_owner_utils import RuntimeOwnerMixin
from .utils.scoring_utils import SignalRecord, build_signals, collect_signals
from .utils.scoring_utils import group_framework_kwargs, rollout_framework_kwargs
from .utils.scoring_utils import score_group as score_group_signals
from .utils.scoring_utils import score_rollout as score_rollout_signals
from .utils.serialization_utils import serializable
from .artifact import ArtifactConfig
from .sandbox import SandboxConfig
from .runtime_handles import (
    ModelRuntimeHandleConfig,
    ResolvedRuntimeHandlesConfig,
    RuntimeHandleConfig,
    SandboxRuntimeHandleConfig,
    SandboxRuntimeStateConfig,
)
from .utils.tool_utils import schema_callable, tool_schema, tool_visible
from .utils.tool_utils import toolset_object_scope
from .utils.usage_utils import record_response_usage
from .state import State
from .task import Task
from .toolset import (
    MCPTool,
    ToolEntry,
    Toolset,
    VisibilityConfig,
    iter_toolsets,
    tool_name,
)
from .user import User, state_messages
from .types import (
    ConfigData,
    Handler,
    JsonData,
    PromptMessage,
    RuntimeCallable,
    RuntimeCallableResult,
    RuntimeData,
    RuntimeObject,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .harness import Harness
    from .taskset import Taskset
    from .utils.mcp_utils import MCPToolHandle
    from .utils.sandbox_utils import SandboxClient, SandboxLease

BindingOwner = Toolset | RuntimeOwnerMixin | None
BindingEntry = tuple[str, BindingSource, BindingOwner]
ArtifactOwner = RuntimeOwnerMixin | Toolset | User | None
TrajectoryVisibility = Literal["append", "hidden"]
RuntimeObjectOwner = Literal["toolset", "user", "taskset", "harness"]
RuntimeObjectKey = tuple[int, str, str]
RuntimeObjectStore = dict[RuntimeObjectKey, RuntimeObject]
RUNTIME_OBJECT_OWNERS: tuple[RuntimeObjectOwner, ...] = (
    "toolset",
    "user",
    "taskset",
    "harness",
)


@runtime_checkable
class ToolDefinitionProvider(Protocol):
    @property
    def tool_def(self) -> Tool: ...

    def __call__(self, **kwargs: object) -> RuntimeCallableResult: ...


RuntimeTool: TypeAlias = RuntimeCallable | Tool | ToolDefinitionProvider
RuntimeTools: TypeAlias = dict[str, RuntimeTool]


def lifecycle_handlers(
    owner: RuntimeOwnerMixin | Toolset, kind: CallableKind
) -> Iterable[Handler]:
    if kind == "stop":
        return owner.stops
    if kind == "setup":
        return owner.setups
    if kind == "update":
        return owner.updates
    if kind == "cleanup":
        return owner.cleanups
    if kind == "teardown":
        return owner.teardowns
    if isinstance(owner, Toolset):
        return ()
    if kind == "metric":
        return owner.metrics
    if kind == "reward":
        return owner.rewards
    if kind == "advantage":
        return owner.advantages
    raise ValueError(f"Unknown lifecycle kind: {kind!r}.")


@dataclass(frozen=True)
class ModelRequestContext:
    source: Literal["direct", "endpoint"] = "direct"
    endpoint_request_id: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    trajectory_visibility: TrajectoryVisibility = "append"

    def extras(self) -> ConfigData:
        data: ConfigData = {}
        if self.source == "endpoint":
            data["endpoint"] = True
        if self.endpoint_request_id is not None:
            data["endpoint_request_id"] = self.endpoint_request_id
        if self.headers:
            data["headers"] = self.headers
        if self.trajectory_visibility != "append":
            data["trajectory_visibility"] = self.trajectory_visibility
        return data


@dataclass(frozen=True)
class RuntimeArtifact:
    name: str
    config: ArtifactConfig
    owner: ArtifactOwner = None


class BorrowedTool:
    def __init__(self, runtime: "Runtime", handle_id: str, name: str):
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


class AsyncRateLimiter:
    def __init__(self, rate_per_second: float | None):
        self.interval = 0.0 if rate_per_second is None else 1.0 / rate_per_second
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self.lock:
            now = time.monotonic()
            delay = self.next_at - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = time.monotonic()
            self.next_at = max(now, self.next_at) + self.interval


class Runtime:
    def __init__(
        self, taskset: "Taskset | None" = None, harness: "Harness | None" = None
    ):
        self.runtime_id = uuid.uuid4().hex
        register_runtime(self.runtime_id, self)
        self.taskset = taskset
        self.harness = harness
        owners = (self.taskset, self.harness)
        self.toolsets = []
        for owner in owners:
            if owner is not None:
                self.toolsets.extend(iter_toolsets(owner.toolsets))
        self.named_toolsets = self._collect_named_toolsets()
        self.scoped_tools: dict[tuple[int, str, str], list[ToolEntry]] = {}
        self.runtime_objects: dict[RuntimeObjectOwner, RuntimeObjectStore] = {
            "toolset": {},
            "user": {},
            "taskset": {},
            "harness": {},
        }
        self.model_clients: dict[str, Client] = {}
        self.owned_model_clients: set[str] = set()
        self._sandbox_client = None
        self.sandbox_leases: dict[tuple[str, str], SandboxLease] = {}
        self.sandbox_creation_tasks: dict[
            tuple[str, str], asyncio.Task[SandboxLease]
        ] = {}
        self.upload_archive_tasks: dict[tuple[str, str, str], asyncio.Task[Path]] = {}
        self.sandbox_lock = asyncio.Lock()
        sandbox_config = (
            self.harness.sandbox
            if self.harness is not None and self.harness.sandbox is not None
            else SandboxConfig()
        )
        create_concurrency = sandbox_config.create_concurrency
        create_rate = sandbox_config.create_rate_per_second
        delete_concurrency = sandbox_config.delete_concurrency
        delete_rate = sandbox_config.delete_rate_per_second
        self.sandbox_create_semaphore = asyncio.Semaphore(create_concurrency)
        self.sandbox_create_rate_limiter = AsyncRateLimiter(create_rate)
        self.sandbox_delete_semaphore = asyncio.Semaphore(delete_concurrency)
        self.sandbox_delete_rate_limiter = AsyncRateLimiter(delete_rate)
        self.mcp_exit_stacks: dict[str, AsyncExitStack] = {}
        self.mcp_tools: dict[str, RuntimeTools] = {}
        self.mcp_tool_parents: dict[str, dict[str, tuple[Toolset, ...]]] = {}
        self.trajectories: dict[str, list[JsonData]] = {}
        self.tool_handles: dict[str, tuple[Task, State, tuple[str, ...]]] = {}
        self.stop_conditions = collect_handlers(
            self._handler_owners(),
            "stop",
            self._extra_handlers("stop", builtins=[state_done]),
        )
        validate_handler_args(
            self.stop_conditions, {"task", "state"}, "stop", "rollout"
        )
        self.rollout_setup = collect_handlers(
            self._handler_owners(),
            "setup",
            self._extra_handlers("setup"),
        )
        validate_handler_args(self.rollout_setup, {"task", "state"}, "setup", "rollout")
        self.rollout_update = collect_handlers(
            self._handler_owners(),
            "update",
            self._extra_handlers("update"),
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
        signals = collect_signals(
            self._owner_signals(self.taskset),
            self._owner_signals(self.harness),
        )
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
        state["task"] = task
        state.runtime_state()["runtime_id"] = self.runtime_id
        self.resolve_trajectory(state)
        self.refresh_tools(state, validate=False)

    def refresh_tools(self, state: State, *, validate: bool = True) -> None:
        state["tools"] = sorted(self.all_exposed_tools(state, validate=validate))

    def task_for_state(self, state: State) -> Task:
        return cast(Task, state["task"])

    def register_tool_handle(self, state: State, names: Sequence[str]) -> str:
        task = self.task_for_state(state)
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
        return self.tool_def(name, source_tool, source_state)

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

    def add_tool(self, toolset: str, tool: ToolEntry, state: State) -> None:
        if toolset not in self.named_toolsets:
            raise KeyError(f"Unknown toolset {toolset!r}.")
        if isinstance(tool, Toolset):
            raise TypeError("State.add_tool accepts a tool, not a Toolset.")
        toolset_value = self.named_toolsets[toolset]
        scope = toolset_object_scope(toolset_value)
        key = (id(toolset_value), scope, self.scope_key(scope, state))
        tools = self.scoped_tools.setdefault(key, [])
        if not isinstance(tool, MCPTool):
            name = tool_name(tool)
            existing = self.tools_for_toolsets(
                [toolset_value], apply_visibility=False, state=state
            )
            if name in existing:
                raise ValueError(f"Tool {name!r} is defined twice.")
        tools.append(tool)

    def release_scoped_tools(self, scope: str, state: State) -> None:
        scope_key = self.scope_key(scope, state)
        for key in list(self.scoped_tools):
            _, tool_scope, tool_scope_key = key
            if tool_scope == scope and tool_scope_key == scope_key:
                del self.scoped_tools[key]

    def scoped_tool_entries(
        self, toolset: Toolset, state: State | None
    ) -> list[ToolEntry]:
        if state is None:
            return []
        scope = toolset_object_scope(toolset)
        key = (id(toolset), scope, self.scope_key(scope, state))
        return list(self.scoped_tools.get(key, ()))

    def register_trajectory(self, state: State) -> None:
        trajectory = state.get("trajectory")
        if trajectory is None:
            return
        if not isinstance(trajectory, list):
            raise TypeError("state.trajectory must be a list.")
        self.trajectories[str(state["trajectory_id"])] = cast(
            list[JsonData], trajectory
        )

    def resolved_handles(self, state: State) -> ResolvedRuntimeHandlesConfig:
        runtime = state.runtime_state()
        resolved = runtime.get("resolved") or {}
        return ResolvedRuntimeHandlesConfig.model_validate(resolved)

    def resolved_runtime(self, handle: RuntimeHandleConfig) -> "Runtime":
        return load_runtime(handle.runtime_id)

    def model_handle(self, state: State) -> ModelRuntimeHandleConfig | None:
        return self.resolved_handles(state).model

    def endpoint_handle(self, state: State) -> RuntimeHandleConfig | None:
        return self.resolved_handles(state).endpoint

    def sandbox_handle(self, state: State) -> SandboxRuntimeHandleConfig | None:
        return self.resolved_handles(state).sandbox

    def resolve_trajectory(self, state: State) -> None:
        handle = self.resolved_handles(state).trajectory
        if handle is None:
            state.setdefault("trajectory", [])
            self.register_trajectory(state)
            return
        if handle.mode != "append":
            raise ValueError("state.runtime.resolved.trajectory.mode must be 'append'.")
        runtime = self.resolved_runtime(handle)
        trajectory = runtime.trajectories.get(handle.trajectory_id)
        if trajectory is None:
            raise RuntimeError(
                f"No live trajectory registered for {handle.trajectory_id!r}."
            )
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
        elif isinstance(client, Client):
            config = client.config
            if isinstance(config, ClientConfig):
                client_type = config.client_type
            else:
                client_type = "openai_chat_completions"
        else:
            client_type = "openai_chat_completions"
        key = str(
            state.runtime_state().get("client_key")
            or state.get("trajectory_id")
            or f"client_{uuid.uuid4().hex}"
        )
        self.model_clients[key] = client
        if owns_client:
            self.owned_model_clients.add(key)
        runtime = state.runtime_state()
        runtime["client_key"] = key
        runtime["client_type"] = client_type

    def model_client(self, state: State) -> Client:
        handle = self.model_handle(state)
        runtime = self
        if handle is None:
            key = str(state.runtime_state().get("client_key") or "default")
        else:
            runtime = self.resolved_runtime(handle)
            key = handle.client_key
        client = runtime.model_clients.get(key)
        if client is None:
            raise RuntimeError("Harness has no model client for intercepted requests.")
        return client

    def client_type(self, state: State) -> ClientType:
        raw_client_type = state.runtime_state().get("client_type")
        if raw_client_type is None:
            handle = self.model_handle(state)
            if handle is not None:
                raw_client_type = handle.client_type
        if raw_client_type is None:
            return "openai_chat_completions"
        if raw_client_type not in get_args(ClientType):
            raise ValueError(f"Unsupported client type: {raw_client_type!r}")
        return cast(ClientType, raw_client_type)

    def model(self, state: State) -> str:
        model = state.runtime_state().get("model")
        if model is None:
            handle = self.model_handle(state)
            if handle is not None:
                model = handle.model
        if not isinstance(model, str) or not model:
            raise RuntimeError("Harness has no model for intercepted requests.")
        return model

    def sampling_args(self, state: State) -> SamplingArgs:
        sampling = state.runtime_state().get("sampling_args") or {}
        if not sampling:
            handle = self.model_handle(state)
            if handle is not None:
                sampling = handle.sampling_args or {}
        if not isinstance(sampling, dict):
            raise TypeError("state.runtime.sampling_args must be a mapping.")
        return cast(SamplingArgs, dict(cast(ConfigData, sampling)))

    def tool_defs(self, state: State) -> list[Tool] | None:
        defs: list[Tool] = []
        for name, tool in self.all_exposed_tools(state).items():
            if (
                isinstance(tool, Tool)
                or isinstance(tool, ToolDefinitionProvider)
                or callable(tool)
            ):
                defs.append(self.tool_def(name, tool, state))
        return defs or None

    async def user_messages(
        self,
        task: Task,
        state: State,
        transcript: Sequence[PromptMessage] | None = None,
    ) -> list[JsonData]:
        user = self.active_user()
        if user is None:
            return []
        kwargs: RuntimeData = {}
        if user.sandbox is not None:
            kwargs["sandbox"] = await self.resolve_user_sandbox(user, task, state)
        for name, source in user.bindings.items():
            validate_bound_arg(user.get_response, name, f"User binding {name!r}")
            validate_binding_source(source, f"User binding {name!r}")
            kwargs[name] = await self.resolve_user_binding(
                user, source, task, state, transcript
            )
        raw_messages = await maybe_call_with_named_args(
            user.get_response,
            task=task,
            state=state,
            messages=state_messages(state, transcript),
            **kwargs,
        )
        if raw_messages is None:
            return []
        messages = normalize_messages(raw_messages, field_name="user")
        return [message.model_dump(exclude_none=True) for message in messages]

    def active_user(self) -> User | None:
        users = []
        if self.taskset is not None and self.taskset.user is not None:
            users.append(self.taskset.user)
        if self.harness is not None and self.harness.user is not None:
            users.append(self.harness.user)
        if len(users) > 1:
            raise ValueError("Taskset and harness cannot both define user.")
        if not users:
            return None
        return users[0]

    def tool_def(self, name: str, tool: object, state: State) -> Tool:
        hidden_args = self.hidden_tool_args(name, state)
        if isinstance(tool, Tool):
            return tool_schema(tool, hidden_args)
        if isinstance(tool, ToolDefinitionProvider):
            return tool_schema(tool.tool_def, hidden_args)
        schema_tool = tool
        filtered_signature = self._tool_signature(name, tool, state)
        if hidden_args and filtered_signature is not None:
            schema_tool = schema_callable(tool, filtered_signature)
        return tool_schema(convert_func_to_tool_def(schema_tool), hidden_args)

    def hidden_tool_args(self, name: str, state: State) -> set[str]:
        hidden_args = {"runtime", "task", "state"}
        owner = self.tool_owner(name, state)
        if owner is not None and owner.sandbox is not None:
            hidden_args.add("sandbox")
        if owner is not None:
            for binding_key in owner.bindings:
                tool_name_prefix, arg_name = binding_key_parts(binding_key)
                if tool_name_prefix == name:
                    tool = self.all_tools(state)[name]
                    target = owner.handler if isinstance(tool, Tool) else tool
                    if target is None:
                        raise TypeError(
                            f"Schema-backed tool {name!r} requires a Toolset handler."
                        )
                    validate_bound_arg(
                        target,
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
            framework_kwargs = rollout_framework_kwargs(task, state)
            extra_kwargs = await self.binding_kwargs(
                condition, task, state, set(framework_kwargs)
            )
            completed = await maybe_call_with_named_args(
                condition, **extra_kwargs, **framework_kwargs
            )
            if completed:
                state._set_completed(True)
                state._set_truncated(
                    any(
                        step.get("is_truncated", False)
                        for step in state.get("trajectory", [])
                        if isinstance(step, dict)
                    )
                )
                state._set_stop_condition(function_name(condition))
                return True
        return False

    def tool_calls(self, task: Task, state: State) -> dict[str, RuntimeCallable]:
        return {
            name: self._tool_call(name, task, state, exposed=True)
            for name in self.all_exposed_tools(state)
        }

    def _tool_call(
        self, tool_name: str, task: Task, state: State, exposed: bool
    ) -> RuntimeCallable:
        async def call(**kwargs: object) -> object:
            return await self._call_tool(tool_name, task, state, exposed, **kwargs)

        tools = self.all_exposed_tools(state) if exposed else self.all_tools(state)
        tool = tools[tool_name]
        tool_def = self.tool_def(tool_name, tool, state)
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
        tool: RuntimeCallable,
        tool_name: str,
        task: Task,
        state: State,
        visible_kwargs: RuntimeData,
        hidden_kwargs: RuntimeData,
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
        hidden_values: RuntimeData = {
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

    async def _call_schema_tool(
        self,
        tool: Tool,
        handler: RuntimeCallable,
        tool_name: str,
        task: Task,
        state: State,
        visible_kwargs: RuntimeData,
        hidden_kwargs: RuntimeData,
    ) -> object:
        call_kwargs: RuntimeData = {
            "tool": tool,
            "tool_name": tool_name,
            "arguments": dict(visible_kwargs),
            "task": task,
            "state": state,
            "runtime": self,
            **hidden_kwargs,
        }
        try:
            signature = inspect.signature(handler)
        except (TypeError, ValueError):
            if hidden_kwargs:
                raise TypeError(
                    f"Toolset handler for {tool_name!r} uses hidden args, but its "
                    "signature cannot be inspected."
                )
            result = handler(tool=tool, arguments=dict(visible_kwargs))
            if inspect.isawaitable(result):
                return await result
            return result
        if not any(
            parameter.kind == parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            for arg_name in hidden_kwargs:
                if arg_name not in signature.parameters:
                    raise TypeError(
                        f"Toolset handler for {tool_name!r} has hidden arg "
                        f"{arg_name!r}, but does not declare it in its signature."
                    )
        return await maybe_call_with_named_args(handler, **call_kwargs)

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
        hidden_kwargs: RuntimeData = {}
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
        tool = tools[tool_name]
        if isinstance(tool, Tool):
            if owner is None or owner.handler is None:
                raise TypeError(
                    f"Schema-backed tool {tool_name!r} requires a Toolset handler."
                )
            return await self._call_schema_tool(
                tool,
                owner.handler,
                tool_name,
                task=task,
                state=state,
                visible_kwargs=visible_kwargs,
                hidden_kwargs=hidden_kwargs,
            )
        if not callable(tool):
            raise TypeError(f"Tool {tool_name!r} must be callable or schema-backed.")
        return await self._call_tool_callable(
            cast(RuntimeCallable, tool),
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
        context: ModelRequestContext | None = None,
    ) -> Response:
        context = context or ModelRequestContext()
        client = self.model_client(state)
        request_start = time.time()
        response = await client.get_response(
            prompt=prompt,
            model=self.model(state),
            tools=tool_defs,
            sampling_args=self.sampling_args(state),
            state=state,
        )
        request_end = time.time()
        state.record_model_timing(request_start, request_end)
        record_response_usage(state, response)
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
            "extras": context.extras(),
        }
        if context.trajectory_visibility == "append":
            state["trajectory"].append(step)
        elif context.trajectory_visibility != "hidden":
            raise AssertionError(
                f"Unknown trajectory visibility: {context.trajectory_visibility!r}"
            )
        return response

    async def setup_rollout(
        self,
        task: Task,
        state: State,
        setup_handlers: Iterable[Handler] = (),
        **kwargs: object,
    ) -> State:
        handlers = sort_handlers(
            unique_handlers(
                [
                    *setup_handlers,
                    *self.rollout_setup,
                    *self._rollout_handlers("setup", state),
                ]
            ),
            "setup",
        )
        validate_handler_args(handlers, {"task", "state"}, "setup", "rollout")
        await self.run_rollout_handlers(handlers, task=task, state=state, **kwargs)
        await self.ensure_mcp_tools(state)
        self.validate_bindings(state)
        self.refresh_tools(state)
        return state

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
        await self.run_group_handlers(handlers, tasks=tasks, states=states)
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
            tasks,
            states,
            resolve_kwargs=self.group_binding_kwargs,
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
        await self.release_runtime_objects("rollout", state)
        await self.release_sandboxes(scope="rollout", state=state)
        await self.close_mcp_tools(state)
        self.release_scoped_tools("rollout", state)
        await self.release_model_client(state)
        self.release_tool_handles(state)

    async def cleanup_group(self, tasks: list[Task], states: list[State]) -> None:
        handlers = unique_handlers(
            [
                *self.group_cleanup,
                *self._group_handlers("cleanup", states, stage="group"),
            ]
        )
        await self.run_group_handlers(handlers, tasks=tasks, states=states)
        for state in states:
            await self.release_runtime_objects("group", state)
            await self.release_sandboxes(scope="group", state=state)
            await self.close_mcp_tools(state, scope="group")
            self.release_scoped_tools("group", state)
            await self.release_model_client(state, group=True)

    async def collect_artifacts(self, task: Task, state: State) -> None:
        artifacts = self.runtime_artifacts(task, state)
        if not artifacts:
            return
        state_artifacts = state.setdefault("artifacts", {})
        if not isinstance(state_artifacts, dict):
            raise TypeError("state.artifacts must be a mapping.")
        for name in artifacts:
            if name in state_artifacts:
                raise ValueError(f"Artifact {name!r} is already present on state.")
        values = await asyncio.gather(
            *(
                self.collect_runtime_artifact(artifact, task, state)
                for artifact in artifacts.values()
            )
        )
        state_artifacts.update(dict(zip(artifacts, values, strict=True)))

    def runtime_artifacts(self, task: Task, state: State) -> dict[str, RuntimeArtifact]:
        artifacts: dict[str, RuntimeArtifact] = {}

        sources: list[tuple[ArtifactOwner, dict[str, ArtifactConfig]]] = []
        if self.taskset is not None:
            sources.append((self.taskset, self.taskset.artifacts))
        if self.harness is not None:
            sources.append((self.harness, self.harness.artifacts))
            sources.append(
                (
                    None,
                    self.harness.program_config.artifacts.artifacts(
                        "harness.program.artifacts"
                    ),
                )
            )
        for toolset in iter_toolsets(self.active_toolsets(state)):
            sources.append((toolset, toolset.artifacts))
        user = self.active_user()
        if user is not None:
            sources.append((user, user.artifacts))

        for owner, source in sources:
            for name, artifact in source.items():
                if name in artifacts:
                    raise ValueError(f"Artifact {name!r} is defined twice.")
                artifacts[name] = RuntimeArtifact(
                    name=name, config=artifact, owner=owner
                )
        for name, artifact in (
            task.artifacts_config().artifacts("task.artifacts").items()
        ):
            if name in artifacts:
                raise ValueError(f"Artifact {name!r} is defined twice.")
            artifacts[name] = RuntimeArtifact(name=name, config=artifact)
        return artifacts

    async def teardown(self) -> None:
        await run_handlers(self.teardown_handlers)
        await self.release_runtime_objects()
        failed_sandbox_deletions = []
        try:
            if self.sandbox_creation_tasks:
                await self.clear_sandbox_creation_tasks(
                    list(self.sandbox_creation_tasks.items())
                )
            for key, handle in list(self.sandbox_leases.items()):
                try:
                    await self.close_sandbox_lease(handle)
                except Exception as exc:
                    logger.warning(
                        "Failed to delete sandbox %s during teardown: %s",
                        handle.id,
                        exc,
                        exc_info=True,
                    )
                    failed_sandbox_deletions.append(handle)
                else:
                    async with self.sandbox_lock:
                        if self.sandbox_leases.get(key) is handle:
                            del self.sandbox_leases[key]
        finally:
            if not any(
                handle.client is self._sandbox_client
                for handle in failed_sandbox_deletions
            ):
                await self.teardown_sandbox_client()
            await self.cleanup_upload_archives()
        self.tool_handles.clear()
        self.scoped_tools.clear()
        await self.close_all_mcp_tools()
        await self.release_all_model_clients()
        if not failed_sandbox_deletions:
            unregister_runtime(self.runtime_id)

    def sandbox_client(self) -> "SandboxClient":
        if self._sandbox_client is None:
            from verifiers.utils.threaded_sandbox_client import (
                ThreadedAsyncSandboxClient,
            )

            from .utils.sandbox_utils import SandboxClient

            self._sandbox_client = cast(
                SandboxClient,
                ThreadedAsyncSandboxClient(),
            )
        return self._sandbox_client

    async def teardown_sandbox_client(self) -> None:
        if self._sandbox_client is None:
            return
        from .utils.sandbox_utils import close_sandbox_client

        await close_sandbox_client(self._sandbox_client)
        self._sandbox_client = None

    async def cached_upload_archive(
        self, local_source: Path | Traversable, remote_path: str
    ) -> Path:
        from .utils.sandbox_utils import UPLOAD_IGNORE_PARTS, build_dir_archive

        if isinstance(local_source, Path):
            root = local_source.resolve()
            digest = hashlib.sha256()
            paths = [root]
            if root.is_dir():
                paths = []
                directories = [root]
                while directories:
                    for path in sorted(directories.pop().iterdir()):
                        if path.name in UPLOAD_IGNORE_PARTS:
                            continue
                        paths.append(path)
                        if path.is_dir():
                            directories.append(path)
            for path in paths:
                relative = (
                    path.relative_to(root).as_posix() if path != root else path.name
                )
                stat = path.stat()
                kind = "d" if path.is_dir() else "f"
                digest.update(
                    f"{kind}:{relative}:{stat.st_mtime_ns}:{stat.st_size}\0".encode()
                )
            key = (remote_path, str(root), digest.hexdigest())
        else:
            key = (remote_path, str(local_source), "resource")
        task = self.upload_archive_tasks.get(key)
        if task is None:
            task = asyncio.create_task(
                asyncio.to_thread(build_dir_archive, local_source, remote_path)
            )
            self.upload_archive_tasks[key] = task
        try:
            return await asyncio.shield(task)
        except Exception:
            if self.upload_archive_tasks.get(key) is task:
                del self.upload_archive_tasks[key]
            raise

    async def cleanup_upload_archives(self) -> None:
        tasks = list(self.upload_archive_tasks.values())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.upload_archive_tasks.clear()
        for result in results:
            if not isinstance(result, Path):
                continue
            try:
                result.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning(
                    "Failed to delete cached upload archive %s: %s",
                    result,
                    exc,
                    exc_info=True,
                )

    async def close_sandbox_lease(self, lease: "SandboxLease") -> None:
        async with self.sandbox_delete_semaphore:
            await self.sandbox_delete_rate_limiter.wait()
            await close_object(lease)

    async def clear_sandbox_creation_tasks(
        self,
        creations: Sequence[tuple[tuple[str, str], asyncio.Task["SandboxLease"]]],
        *,
        state: State | None = None,
        scope: str | None = None,
    ) -> None:
        from .utils.sandbox_utils import SandboxLease as SandboxLeaseClass

        claimed_creations = []
        async with self.sandbox_lock:
            for key, task in creations:
                if self.sandbox_creation_tasks.get(key) is task:
                    del self.sandbox_creation_tasks[key]
                    claimed_creations.append((key, task))

        for _, task in claimed_creations:
            if not task.done():
                task.cancel()
        results = await asyncio.gather(
            *(task for _, task in claimed_creations), return_exceptions=True
        )

        for (key, _), result in zip(claimed_creations, results, strict=True):
            if not isinstance(result, SandboxLeaseClass):
                continue
            result.scope_key = key[0]
            try:
                await self.close_sandbox_lease(result)
            except Exception as exc:
                async with self.sandbox_lock:
                    if not result.deleted:
                        self.sandbox_leases[key] = result
                logger.warning(
                    "Failed to delete sandbox %s from cancelled creation: %s",
                    result.id,
                    exc,
                    exc_info=True,
                )
                if state is not None and scope is not None:
                    cleanup_errors = cast(
                        list[ConfigData], state.setdefault("cleanup_errors", [])
                    )
                    cleanup_errors.append(
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "scope": scope,
                        }
                    )

    async def resolve_sandbox_lease(
        self, key: tuple[str, str], factory: Callable[[], Awaitable["SandboxLease"]]
    ) -> "SandboxLease":
        async with self.sandbox_lock:
            lease = self.sandbox_leases.get(key)
            if lease is not None:
                if lease.deleted:
                    raise RuntimeError("Sandbox lease is being deleted.")
                return lease
            task = self.sandbox_creation_tasks.get(key)
            if task is None:

                async def create_sandbox_lease() -> "SandboxLease":
                    async with self.sandbox_create_semaphore:
                        await self.sandbox_create_rate_limiter.wait()
                        return await factory()

                task = asyncio.create_task(create_sandbox_lease())
                self.sandbox_creation_tasks[key] = task

        try:
            lease = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                async with self.sandbox_lock:
                    if self.sandbox_creation_tasks.get(key) is task:
                        del self.sandbox_creation_tasks[key]
            raise
        except BaseException:
            async with self.sandbox_lock:
                if self.sandbox_creation_tasks.get(key) is task:
                    del self.sandbox_creation_tasks[key]
            raise
        async with self.sandbox_lock:
            existing = self.sandbox_leases.get(key)
            if existing is not None:
                if existing.deleted:
                    raise RuntimeError("Sandbox lease is being deleted.")
                return existing
            if self.sandbox_creation_tasks.get(key) is not task:
                raise RuntimeError(
                    "Sandbox creation was cancelled before the lease was resolved."
                )
            del self.sandbox_creation_tasks[key]
            if lease.deleted:
                raise RuntimeError(
                    "Sandbox lease was deleted before it could be resolved."
                )
            lease.scope_key = key[0]
            self.sandbox_leases[key] = lease
            return lease

    async def run_rollout_handlers(
        self,
        handlers: Iterable[Handler],
        task: Task,
        state: State,
        **kwargs: object,
    ) -> None:
        for handler in handlers:
            framework_kwargs = rollout_framework_kwargs(task, state)
            protected_args = set(framework_kwargs) | set(kwargs)
            extra_kwargs = await self.binding_kwargs(
                handler, task, state, protected_args
            )
            await maybe_call_with_named_args(
                handler, **extra_kwargs, **kwargs, **framework_kwargs
            )

    async def run_group_handlers(
        self,
        handlers: Iterable[Handler],
        tasks: list[Task],
        states: list[State],
        **kwargs: object,
    ) -> None:
        for handler in handlers:
            framework_kwargs = group_framework_kwargs(tasks, states)
            protected_args = set(framework_kwargs) | set(kwargs)
            extra_kwargs = await self.group_binding_kwargs(
                handler,
                tasks,
                states,
                protected_args,
            )
            await maybe_call_with_named_args(
                handler, **extra_kwargs, **kwargs, **framework_kwargs
            )

    async def binding_kwargs(
        self,
        fn: Handler,
        task: Task,
        state: State,
        protected_args: set[str] | None = None,
    ) -> RuntimeData:
        name = function_name(fn)
        kwargs: RuntimeData = {}
        protected = protected_args or set()
        for binding_key, source, owner in self._binding_entries_for_callable(fn, state):
            prefix, arg_name = binding_key_parts(binding_key)
            if prefix != name:
                continue
            if arg_name in protected:
                continue
            validate_bound_arg(fn, arg_name, f"Binding {binding_key!r}", protected)
            if arg_name in kwargs:
                raise ValueError(f"Binding arg {arg_name!r} is defined twice.")
            kwargs[arg_name] = await self.resolve_owner_binding(
                owner, source, task, state
            )
        return kwargs

    async def group_binding_kwargs(
        self,
        fn: Handler,
        tasks: list[Task],
        states: list[State],
        protected_args: set[str] | None = None,
    ) -> RuntimeData:
        if not states:
            return {}
        state = states[0]
        name = function_name(fn)
        kwargs: RuntimeData = {}
        protected = protected_args or set()
        for binding_key, source, owner in self._binding_entries_for_callable(fn, state):
            prefix, arg_name = binding_key_parts(binding_key)
            if prefix != name:
                continue
            if arg_name in protected:
                continue
            validate_bound_arg(fn, arg_name, f"Binding {binding_key!r}", protected)
            if arg_name in kwargs:
                raise ValueError(f"Binding arg {arg_name!r} is defined twice.")
            kwargs[arg_name] = await self.resolve_group_binding(
                owner,
                source,
                tasks,
                states,
                state,
            )
        return kwargs

    async def resolve_owner_binding(
        self, owner: BindingOwner, source: BindingSource, task: Task, state: State
    ) -> object:
        if isinstance(owner, Toolset):
            return await self.resolve_tool_binding(owner, source, task, state)
        if self.taskset is not None and owner is self.taskset:
            return await self.resolve_attached_owner_binding(
                self.taskset, source, task, state
            )
        if self.harness is not None and owner is self.harness:
            return await self.resolve_attached_owner_binding(
                self.harness, source, task, state
            )
        return await self.resolve_binding(source, task, state)

    async def resolve_owner_object(
        self,
        owner: RuntimeOwnerMixin | Toolset | User,
        name: str,
        task: Task,
        state: State,
    ) -> object:
        if owner is self.taskset:
            return await self.resolve_taskset_object(name, task, state)
        if owner is self.harness:
            return await self.resolve_harness_object(name, task, state)
        if isinstance(owner, Toolset):
            return await self.resolve_toolset_object(owner, name, task, state)
        if isinstance(owner, User):
            return await self.resolve_user_object(owner, name, task, state)
        raise RuntimeError("Runtime object owner is not attached to this runtime.")

    async def resolve_attached_owner_binding(
        self,
        owner: RuntimeOwnerMixin,
        source: BindingSource,
        task: Task,
        state: State,
    ) -> object:
        if isinstance(source, str):
            root, separator, tail = source.partition(".")
            if root == "objects":
                if not separator:
                    raise ValueError("objects binding sources must name an object.")
                name, _, rest = tail.partition(".")
                value = await self.resolve_owner_object(owner, name, task, state)
                return read_path(value, rest) if rest else value
        return await self.resolve_binding(source, task, state)

    async def resolve_binding(
        self, source: BindingSource, task: Task, state: State
    ) -> object:
        if isinstance(source, str):
            if binding_source_root(source) == "objects":
                raise ValueError(
                    "objects.* bindings are private to the owning Taskset, Harness, "
                    "Toolset, or User callable. Use taskset.objects.* or "
                    "harness.objects.* for explicit cross-owner object bindings."
                )
            root, separator, tail = source.partition(".")
            if root in {"taskset", "harness"}:
                if not separator:
                    raise ValueError(
                        f"{root} binding sources must use {root}.objects.name."
                    )
                return await self.resolve_runtime_owner_path(root, tail, task, state)
            return self.resolve_path(source, task, state)
        if isinstance(source, dict) and "fn" in source:
            spec = cast(ConfigData, source)
            validate_callable_source(spec, "Callable binding source")
            fn = resolve_config_object(spec["fn"])
            if not callable(fn):
                raise TypeError("Callable binding source requires callable fn.")
            return await maybe_call_with_named_args(fn, task=task, state=state)
        if callable(source):
            return await maybe_call_with_named_args(source, task=task, state=state)
        raise TypeError("Binding sources must be framework paths or callables.")

    async def resolve_group_binding(
        self,
        owner: BindingOwner,
        source: BindingSource,
        tasks: list[Task],
        states: list[State],
        state: State,
    ) -> object:
        if isinstance(source, str):
            root, separator, tail = source.partition(".")
            if root == "tasks":
                return read_path(tasks, tail) if separator else tasks
            if root == "states":
                return read_path(states, tail) if separator else states
            if root in {"task", "state", "tools"}:
                raise ValueError("Group handler bindings must use tasks or states.")
            if root == "runtime":
                runtime = state.runtime_state()
                return read_path(runtime, tail) if separator else runtime
            if root == "objects":
                if not separator:
                    raise ValueError("objects binding sources must name an object.")
                name, _, rest = tail.partition(".")
                if owner is self.taskset:
                    value = await self.resolve_taskset_object(name, tasks[0], state)
                elif owner is self.harness:
                    value = await self.resolve_harness_object(name, tasks[0], state)
                elif isinstance(owner, Toolset):
                    if toolset_object_scope(owner) == "rollout":
                        raise ValueError(
                            "objects.* group bindings require a group or global Toolset scope."
                        )
                    value = await self.resolve_toolset_object(
                        owner, name, tasks[0], state
                    )
                else:
                    raise ValueError(
                        "objects.* group bindings require an object owner."
                    )
                return read_path(value, rest) if rest else value
            if root in {"taskset", "harness"}:
                if not separator:
                    raise ValueError(
                        f"{root} binding sources must use {root}.objects.name."
                    )
                return await self.resolve_runtime_owner_path(
                    root, tail, tasks[0], state
                )
        if isinstance(source, dict) and "fn" in source:
            spec = cast(ConfigData, source)
            validate_callable_source(spec, "Callable binding source")
            fn = resolve_config_object(spec["fn"])
            if not callable(fn):
                raise TypeError("Callable binding source requires callable fn.")
            return await maybe_call_with_named_args(fn, tasks=tasks, states=states)
        if callable(source):
            return await maybe_call_with_named_args(source, tasks=tasks, states=states)
        raise TypeError("Binding sources must be framework paths or callables.")

    async def resolve_tool_binding(
        self, toolset: Toolset | None, source: BindingSource, task: Task, state: State
    ) -> object:
        if isinstance(source, str):
            root, separator, tail = source.partition(".")
            if root == "objects":
                if toolset is None:
                    raise ValueError("objects.* tool bindings require a Toolset owner.")
                if not separator:
                    raise ValueError("objects binding sources must name an object.")
                name, _, rest = tail.partition(".")
                value = await self.resolve_toolset_object(toolset, name, task, state)
                if rest:
                    return read_path(value, rest)
                return value
        return await self.resolve_binding(source, task, state)

    async def resolve_user_binding(
        self,
        user: User,
        source: BindingSource,
        task: Task,
        state: State,
        transcript: Sequence[PromptMessage] | None = None,
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
                    return read_path(value, rest)
                return value
        if callable(source):
            return await maybe_call_with_named_args(
                source, task=task, state=state, transcript=transcript
            )
        return await self.resolve_binding(source, task, state)

    async def resolve_user_object(
        self, user: User, name: str, task: Task, state: State
    ) -> object:
        if user is not self.active_user():
            raise RuntimeError("User object owner is not attached to this runtime.")
        if name not in user.objects:
            raise KeyError(f"Unknown user object {name!r}.")
        key = (id(user), self.scope_key(user.scope, state), name)
        store = self.runtime_objects["user"]
        if key in store:
            return store[key]
        spec = user.objects[name]
        obj = await resolve_object_factory(spec, f"User object {name!r}")
        store[key] = obj
        return obj

    async def resolve_taskset_object(
        self, name: str, task: Task, state: State
    ) -> object:
        taskset = self.taskset
        if taskset is None:
            raise RuntimeError("Taskset objects require a Taskset.")
        return await self.resolve_attached_owner_object(
            "Taskset", taskset, "taskset", name, task, state
        )

    async def resolve_harness_object(
        self, name: str, task: Task, state: State
    ) -> object:
        harness = self.harness
        if harness is None:
            raise RuntimeError("Harness objects require a Harness.")
        return await self.resolve_attached_owner_object(
            "Harness", harness, "harness", name, task, state
        )

    async def resolve_attached_owner_object(
        self,
        label: Literal["Taskset", "Harness"],
        owner: RuntimeOwnerMixin,
        store_name: Literal["taskset", "harness"],
        name: str,
        task: Task,
        state: State,
    ) -> object:
        objects = owner.objects
        if name not in objects:
            raise KeyError(f"Unknown {label} object {name!r}.")
        object_bindings: dict[str, BindingSource] = {}
        for binding_key, source in owner.bindings.items():
            target_name, arg_name = binding_key_parts(binding_key)
            if target_name == name:
                object_bindings[arg_name] = source
        # Bound object factories can depend on task/state paths, so they are scoped
        # to the rollout. Unbound factories are process-global runtime objects.
        scope_key = self.scope_key("rollout", state) if object_bindings else "global"
        key = (id(owner), scope_key, name)
        store = self.runtime_objects[store_name]
        if key in store:
            return store[key]
        kwargs: RuntimeData = {}
        for arg_name, source in object_bindings.items():
            kwargs[arg_name] = await self.resolve_owner_binding(
                owner, source, task, state
            )
        obj = await resolve_object_factory(
            objects[name], f"{label} object {name!r}", kwargs
        )
        store[key] = obj
        return obj

    async def release_runtime_objects(
        self,
        scope: str | None = None,
        state: State | None = None,
        owner: RuntimeObjectOwner | None = None,
    ) -> None:
        scope_key = self.scope_key(scope, state) if scope is not None else None
        owners = RUNTIME_OBJECT_OWNERS if owner is None else (owner,)
        for owner_name in owners:
            store = self.runtime_objects[owner_name]
            for key, obj in list(store.items()):
                _, object_scope_key, _ = key
                if scope_key is not None and object_scope_key != scope_key:
                    continue
                await close_object(obj)
                del store[key]

    def validate_bindings(
        self, state: State, *, allow_unresolved_tool_bindings: bool = False
    ) -> None:
        for owner in (self.taskset, self.harness):
            self._validate_owner_bindings(owner)
        for toolset in iter_toolsets(self.active_toolsets(state)):
            self._validate_toolset_bindings(
                toolset,
                state,
                allow_unresolved=allow_unresolved_tool_bindings,
            )
        user = self.active_user()
        if user is not None:
            for name, source in user.bindings.items():
                validate_bound_arg(user.get_response, name, f"User binding {name!r}")
                source_root = binding_source_root(source)
                validate_binding_source(source, f"User binding {name!r}")
                if source_root == "objects":
                    object_name = binding_object_name(source)
                    if object_name not in user.objects:
                        raise KeyError(
                            f"User binding {name!r} references unknown User object "
                            f"{object_name!r}."
                        )
                self.validate_runtime_owner_object_source(
                    source, f"User binding {name!r}"
                )

    def _validate_owner_bindings(self, owner: RuntimeOwnerMixin | None) -> None:
        if owner is None:
            return
        targets = self._owner_binding_targets(owner)
        allow_objects = owner in (self.taskset, self.harness)
        for binding_key, source in owner.bindings.items():
            target_name, arg_name = binding_key_parts(binding_key)
            target = targets.get(target_name)
            if target is None:
                raise ValueError(
                    f"Binding {binding_key!r} does not match a Taskset/Harness "
                    "callable or object factory."
                )
            target_kind, fn = target
            if target_kind == "object":
                protected_args = frozenset()
            else:
                stage = handler_stage(fn, cast(CallableKind, target_kind))
                protected_args = (
                    GROUP_FRAMEWORK_ARGS if stage == "group" else ROLLOUT_FRAMEWORK_ARGS
                )
            if arg_name in protected_args:
                continue
            validate_bound_arg(
                fn,
                arg_name,
                f"Binding {binding_key!r}",
                protected_args,
                allow_reserved=target_kind == "object",
            )
            validate_binding_source(
                source, f"Binding {binding_key!r}", allow_objects=allow_objects
            )
            self.validate_runtime_owner_object_source(
                source, f"Binding {binding_key!r}"
            )
            source_root = binding_source_root(source)
            if source_root == "objects":
                object_name = binding_object_name(source)
                if object_name not in owner.objects:
                    raise KeyError(
                        f"Binding {binding_key!r} references unknown object "
                        f"{object_name!r}."
                    )

    def _validate_toolset_bindings(
        self, toolset: Toolset, state: State, *, allow_unresolved: bool
    ) -> None:
        targets = self._toolset_binding_targets(toolset, state)
        for binding_key, source in toolset.bindings.items():
            target_name, arg_name = binding_key_parts(binding_key)
            target = targets.get(target_name)
            if target is None:
                if allow_unresolved and toolset_object_scope(toolset) == "rollout":
                    validate_binding_source(source, f"Binding {binding_key!r}")
                    continue
                raise ValueError(
                    f"Binding {binding_key!r} does not match a callable or object "
                    "factory owned by the same Toolset."
                )
            target_kind, fn = target
            validate_bound_arg(
                fn,
                arg_name,
                f"Binding {binding_key!r}",
                allow_reserved=target_kind == "object",
            )
            source_root = binding_source_root(source)
            validate_binding_source(source, f"Binding {binding_key!r}")
            self.validate_runtime_owner_object_source(
                source, f"Binding {binding_key!r}"
            )
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

    def _owner_binding_targets(
        self, owner: RuntimeOwnerMixin
    ) -> dict[str, tuple[str, Handler]]:
        targets: dict[str, tuple[str, Handler]] = {}

        def add_target(kind: str, fn: Handler, name: str | None = None) -> None:
            name = name or function_name(fn)
            existing = targets.get(name)
            if existing is not None and not same_callable(existing[1], fn):
                raise ValueError(
                    f"Taskset/Harness binding target {name!r} is defined twice."
                )
            targets[name] = (kind, fn)

        for kind in (
            "stop",
            "setup",
            "update",
            "metric",
            "reward",
            "advantage",
            "cleanup",
        ):
            for fn in lifecycle_handlers(owner, kind):
                if callable(fn):
                    add_target(kind, cast(Handler, fn))
        for _, method in inspect.getmembers(owner, predicate=callable):
            for kind in (
                "stop",
                "setup",
                "update",
                "metric",
                "reward",
                "advantage",
                "cleanup",
            ):
                if handler_is_marked(method, kind):
                    add_target(kind, cast(Handler, method))
        for name, spec in owner.objects.items():
            if not isinstance(name, str):
                raise TypeError("Object names must be strings.")
            if callable(spec):
                add_target("object", cast(Handler, spec), name)
        return targets

    def _binding_entries_for_callable(
        self, fn: Handler, state: State
    ) -> list[BindingEntry]:
        target_name = function_name(fn)
        entries: list[BindingEntry] = []
        for owner in (self.taskset, self.harness):
            if owner is None:
                continue
            target = self._owner_binding_targets(owner).get(target_name)
            if target is None or not same_callable(target[1], fn):
                continue
            self._extend_binding_entries(entries, owner.bindings, target_name, owner)
        for toolset in iter_toolsets(self.active_toolsets(state)):
            target = self._toolset_binding_targets(toolset, state).get(target_name)
            if target is None or not same_callable(target[1], fn):
                continue
            self._extend_binding_entries(
                entries, toolset.bindings, target_name, toolset
            )
        return entries

    def validate_runtime_owner_object_source(
        self, source: object, context: str
    ) -> None:
        if not isinstance(source, str):
            return
        root = binding_source_root(source)
        if root == "taskset":
            if self.taskset is None:
                raise RuntimeError(
                    f"{context} references taskset, but no Taskset exists."
                )
            object_name = owner_object_name(source)
            if object_name not in self.taskset.objects:
                raise KeyError(
                    f"{context} references unknown Taskset object {object_name!r}."
                )
        if root == "harness":
            if self.harness is None:
                raise RuntimeError(
                    f"{context} references harness, but no Harness exists."
                )
            object_name = owner_object_name(source)
            if object_name not in self.harness.objects:
                raise KeyError(
                    f"{context} references unknown Harness object {object_name!r}."
                )

    def _extend_binding_entries(
        self,
        entries: list[BindingEntry],
        bindings: dict[str, BindingSource],
        target_name: str,
        owner: BindingOwner = None,
    ) -> None:
        existing = {key for key, _, _ in entries}
        for binding_key, source in bindings.items():
            prefix, _ = binding_key_parts(binding_key)
            if prefix != target_name:
                continue
            if binding_key in existing:
                raise ValueError(f"Binding {binding_key!r} is defined twice.")
            existing.add(binding_key)
            entries.append((binding_key, source, owner))

    def _toolset_binding_targets(
        self, toolset: Toolset, state: State | None = None
    ) -> dict[str, tuple[str, Handler]]:
        targets: dict[str, tuple[str, Handler]] = {}

        def add_target(name: str, kind: str, fn: Handler) -> None:
            if name in targets:
                raise ValueError(f"Toolset binding target {name!r} is defined twice.")
            targets[name] = (kind, fn)

        for item in self._toolset_entries(toolset, state):
            if isinstance(item, Toolset | MCPTool):
                continue
            if isinstance(item, Tool):
                if toolset.handler is None:
                    raise TypeError(
                        f"Schema-backed tool {item.name!r} requires a Toolset handler."
                    )
                add_target(item.name, "tool", toolset.handler)
                continue
            if callable(item):
                add_target(tool_name(item), "tool", cast(Handler, item))
        for name, spec in toolset.objects.items():
            if callable(spec):
                add_target(name, "object", cast(Handler, spec))
        for kind in ("stop", "setup", "update", "cleanup"):
            for fn in lifecycle_handlers(toolset, kind):
                if callable(fn):
                    add_target(
                        function_name(fn),
                        kind,
                        cast(Handler, fn),
                    )
        for _, method in inspect.getmembers(toolset, predicate=callable):
            if any(
                handler_is_marked(method, attr)
                for attr in ("stop", "setup", "update", "cleanup")
            ):
                handler = cast(Handler, method)
                add_target(
                    function_name(handler),
                    "handler",
                    handler,
                )
        return targets

    def active_toolsets(self, state: State) -> list[Toolset]:
        task = self.task_for_state(state)
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

    def _selected_toolset_names(self, task: Task) -> set[str]:
        names = set(self.named_toolsets)
        config = task.toolsets_config()
        show = config.show
        hide = config.hide
        if show is not None:
            selected = set(show)
            unknown = sorted(selected - names)
            if unknown:
                raise KeyError(f"Unknown shown toolsets: {unknown}.")
            return selected
        if hide is not None:
            hidden = set(hide)
            unknown = sorted(hidden - names)
            if unknown:
                raise KeyError(f"Unknown hidden toolsets: {unknown}.")
            return names - hidden
        return names

    def _collect_named_toolsets(self) -> dict[str, Toolset]:
        named: dict[str, Toolset] = {}
        for owner in (self.taskset, self.harness):
            if owner is None:
                continue
            for name, toolset in owner.named_toolsets.items():
                if not isinstance(name, str):
                    raise TypeError("Toolset names must be strings.")
                if name in named:
                    raise ValueError(f"Toolset {name!r} is defined twice.")
                if not isinstance(toolset, Toolset):
                    raise TypeError("named_toolsets values must be Toolsets.")
                named[name] = toolset
        return named

    def tools_for_toolsets(
        self,
        toolsets: Iterable[ToolEntry],
        apply_visibility: bool,
        state: State | None = None,
        tool_filters: dict[str, VisibilityConfig] | None = None,
    ) -> RuntimeTools:
        tools: RuntimeTools = {}

        def visit(item: ToolEntry, parents: list[Toolset]) -> None:
            if isinstance(item, Toolset):
                for child in self._toolset_entries(item, state):
                    visit(child, [*parents, item])
                return
            if isinstance(item, MCPTool):
                return
            name = tool_name(item)
            if apply_visibility and not all(
                self._tool_visible_for_task(toolset, name, tool_filters)
                for toolset in parents
            ):
                return
            if name in tools:
                raise ValueError(f"Tool {name!r} is defined twice.")
            tools[name] = cast(RuntimeTool, item)

        for toolset in toolsets:
            visit(toolset, [])
        return tools

    def _toolset_entries(
        self, toolset: Toolset, state: State | None
    ) -> list[ToolEntry]:
        return [*toolset.tools, *self.scoped_tool_entries(toolset, state)]

    def _tool_visible_for_task(
        self,
        toolset: Toolset,
        name: str,
        tool_filters: dict[str, VisibilityConfig] | None,
    ) -> bool:
        if not tool_visible(toolset, name):
            return False
        toolset_name = self._toolset_name(toolset)
        if toolset_name is None or tool_filters is None:
            return True
        selected = tool_filters.get(toolset_name)
        if selected is None:
            return True
        show = selected.show
        hide = selected.hide
        if show is not None and name not in show:
            return False
        if hide is not None and name in hide:
            return False
        return True

    def _toolset_name(self, toolset: Toolset) -> str | None:
        for name, named_toolset in self.named_toolsets.items():
            if named_toolset is toolset:
                return name
        return None

    def _tool_owners_for(
        self, toolsets: Sequence[Toolset], state: State | None = None
    ) -> dict[str, Toolset]:
        owners: dict[str, Toolset] = {}

        def visit(toolset: Toolset) -> None:
            for item in self._toolset_entries(toolset, state):
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
        return self._tool_owners_for(self.active_toolsets(state), state).get(name)

    def _owner_signals(self, owner: RuntimeOwnerMixin | None) -> list[SignalRecord]:
        if owner is None:
            return []
        return build_signals(
            owner=owner,
            scoring=owner.config.scoring,
            metrics=owner.metrics,
            rewards=owner.rewards,
            advantages=owner.advantages,
        )

    def _handler_owners(self) -> tuple["Taskset | Harness | None", ...]:
        return (self.taskset, self.harness)

    def _extra_handlers(
        self,
        attr: CallableKind,
        builtins: Sequence[Handler] = (),
        owners: Sequence["Taskset | Harness | Toolset | None"] | None = None,
    ) -> list[Handler]:
        handlers: list[Handler] = list(builtins)
        for owner in owners or self._handler_owners():
            if owner is None:
                continue
            for handler in lifecycle_handlers(owner, attr):
                if not callable(handler):
                    raise TypeError(f"{attr} entries must be callable.")
                handlers.append(cast(Handler, handler))
        return handlers

    def _rollout_handlers(
        self,
        attr: CallableKind,
        state: State,
        stage: str | None = None,
    ) -> list[Handler]:
        handlers: list[Handler] = []
        for toolset in iter_toolsets(self.active_toolsets(state)):
            for handler in lifecycle_handlers(toolset, attr):
                if not callable(handler):
                    raise TypeError(f"{attr} entries must be callable.")
                if stage is not None and handler_stage(handler, attr) != stage:
                    continue
                handlers.append(cast(Handler, handler))
            for _, method in inspect.getmembers(toolset, predicate=callable):
                if not handler_is_marked(method, attr):
                    continue
                if stage is not None and handler_stage(method, attr) != stage:
                    continue
                handlers.append(cast(Handler, method))
        return sort_handlers(unique_handlers(handlers), attr)

    def _group_handlers(
        self,
        attr: CallableKind,
        states: Sequence[State],
        stage: str | None = None,
    ) -> list[Handler]:
        handlers: list[Handler] = []
        for state in states:
            handlers.extend(self._rollout_handlers(attr, state, stage=stage))
        return sort_handlers(unique_handlers(handlers), attr)

    async def collect_artifact(
        self,
        spec: ArtifactConfig,
        task: Task,
        state: State,
        *,
        sandbox_lease: "SandboxLease | None",
    ) -> object:
        path = spec.path.format(**{**dict(task), **state})
        if sandbox_lease is not None:
            from .utils.sandbox_utils import read_sandbox_artifact

            try:
                content = await read_sandbox_artifact(
                    sandbox_lease.client, sandbox_lease.id, path
                )
            except FileNotFoundError:
                if spec.optional:
                    return None
                raise
            return spec.parse(content)

        matches = sorted(glob.glob(path))
        if not matches:
            if spec.optional:
                return None
            raise FileNotFoundError(f"Artifact path matched no files: {path!r}")
        with open(matches[0], encoding="utf-8") as f:
            return spec.parse(f.read())

    async def collect_runtime_artifact(
        self, artifact: RuntimeArtifact, task: Task, state: State
    ) -> object:
        sandbox_lease = self.active_artifact_sandbox_lease(artifact.owner, state)
        owner_requires_sandbox = (
            isinstance(artifact.owner, Toolset | User)
            and artifact.owner.sandbox is not None
        )
        if sandbox_lease is None and owner_requires_sandbox:
            if artifact.config.optional:
                return None
            raise RuntimeError(
                f"Artifact {artifact.name!r} requires an active owner sandbox."
            )
        return await self.collect_artifact(
            artifact.config,
            task,
            state,
            sandbox_lease=sandbox_lease,
        )

    def active_artifact_sandbox_lease(
        self, owner: ArtifactOwner, state: State
    ) -> "SandboxLease | None":
        if not isinstance(owner, Toolset | User):
            return self.active_program_sandbox_lease(state)
        from .utils.sandbox_utils import sandbox_owner_key, tool_sandbox_key

        sandbox = owner.sandbox
        if sandbox is None:
            return self.active_program_sandbox_lease(state)
        if sandbox == "program":
            return self.active_program_sandbox_lease(state)
        if not isinstance(sandbox, SandboxConfig):
            raise TypeError("Owner sandbox must be SandboxConfig or 'program'.")
        if sandbox.prefer == "program":
            lease = self.active_program_sandbox_lease(state)
            if lease is not None:
                return lease
        scope = sandbox.scope
        sandbox_key = (
            tool_sandbox_key(owner)
            if isinstance(owner, Toolset)
            else sandbox_owner_key(owner)
        )
        return self.sandbox_leases.get((self.scope_key(scope, state), sandbox_key))

    def resolve_path(self, path: str, task: Task, state: State) -> object:
        root, separator, tail = path.partition(".")
        if root == "task":
            value: object = task
        elif root == "state":
            value = state
        elif root == "runtime":
            value = state.runtime_state()
        elif root == "objects":
            raise ValueError(
                "objects.* bindings are private to the owning Taskset, Harness, "
                "Toolset, or User callable. Use taskset.objects.* or "
                "harness.objects.* for explicit cross-owner object bindings."
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
            return read_path(value, tail)
        if tail:
            return read_path(value, tail)
        return value

    async def resolve_runtime_owner_path(
        self, owner_name: str, path: str, task: Task, state: State
    ) -> object:
        root, separator, tail = path.partition(".")
        if root != "objects" or not separator:
            raise ValueError(
                f"{owner_name} binding sources must use {owner_name}.objects.name."
            )
        name, _, rest = tail.partition(".")
        if not name:
            raise ValueError(
                f"{owner_name} binding sources must use {owner_name}.objects.name."
            )
        if owner_name == "taskset":
            value = await self.resolve_taskset_object(name, task, state)
        elif owner_name == "harness":
            value = await self.resolve_harness_object(name, task, state)
        else:
            raise ValueError("Runtime owner must be 'taskset' or 'harness'.")
        return read_path(value, rest) if rest else value

    async def resolve_toolset_object(
        self, toolset: Toolset, name: str, task: Task, state: State
    ) -> object:
        if not any(
            toolset is active_toolset
            for active_toolset in iter_toolsets(self.active_toolsets(state))
        ):
            raise RuntimeError("Toolset object owner is not active in this runtime.")
        if name not in toolset.objects:
            raise KeyError(f"Unknown Toolset object {name!r}.")
        spec = toolset.objects[name]
        scope = toolset_object_scope(toolset)
        key = (id(toolset), self.scope_key(scope, state), name)
        store = self.runtime_objects["toolset"]
        if key in store:
            return store[key]
        kwargs: RuntimeData = {}
        for binding_key, source in toolset.bindings.items():
            target_name, arg_name = binding_key_parts(binding_key)
            if target_name == name:
                kwargs[arg_name] = await self.resolve_tool_binding(
                    toolset, source, task, state
                )
        obj = await resolve_object_factory(spec, f"Toolset object {name!r}", kwargs)
        store[key] = obj
        return obj

    def scope_key(self, scope: str, state: State | None = None) -> str:
        if scope == "global":
            return "global"
        if state is None:
            raise ValueError(f"{scope} object cleanup requires state.")
        if scope == "group":
            return str(
                state.runtime_state().get("group_key") or state.get("trajectory_id")
            )
        if scope == "rollout":
            return str(state.get("trajectory_id"))
        raise ValueError("Object scope must be 'rollout', 'group', or 'global'.")

    async def release_model_client(self, state: State, *, group: bool = False) -> None:
        if self.model_handle(state) is not None:
            return
        if not group and "group_key" in state.runtime_state():
            return
        key = state.runtime_state().get("client_key")
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
            create_scoped_sandbox_lease,
            tool_sandbox_key,
        )

        sandbox = toolset.sandbox
        if sandbox is None:
            raise TypeError("Toolset sandbox must be configured.")
        if isinstance(sandbox, str):
            if sandbox != "program":
                raise ValueError("Toolset sandbox string must be 'program'.")
            lease = self.active_program_sandbox_lease(state)
            if lease is None:
                raise RuntimeError(
                    "Toolset sandbox='program' requires an active program sandbox."
                )
            return SandboxHandle(lease, state)
        if not isinstance(sandbox, SandboxConfig):
            raise TypeError("Toolset sandbox must be SandboxConfig or 'program'.")
        if sandbox.prefer is not None:
            lease = self.active_program_sandbox_lease(state)
            if lease is not None:
                return SandboxHandle(lease, state)
        scope = sandbox.scope
        key = (self.scope_key(scope, state), tool_sandbox_key(toolset))
        lease = await self.resolve_sandbox_lease(
            key,
            lambda: create_scoped_sandbox_lease(
                toolset,
                key[1],
                client=self.sandbox_client(),
            ),
        )
        return SandboxHandle(lease, state)

    def active_program_sandbox_lease(self, state: State) -> "SandboxLease | None":
        sandbox_handle = self.sandbox_handle(state)
        if sandbox_handle is not None:
            return self.sandbox_lease_from_handle(sandbox_handle, "sandbox")
        sandbox_state = state.runtime_state().get("sandbox")
        if sandbox_state is None:
            return None
        sandbox_record = SandboxRuntimeStateConfig.model_validate(sandbox_state)
        resolved_lease_key = sandbox_record.lease_key
        lease = self.sandbox_leases.get(resolved_lease_key)
        if lease is None:
            raise RuntimeError("Program sandbox lease is no longer active.")
        lease.scope_key = resolved_lease_key[0]
        return lease

    async def resolve_program_sandbox(
        self, sandbox_config: SandboxConfig, task: Task, state: State
    ) -> "SandboxLease":
        from .utils.sandbox_utils import (
            create_sandbox_lease,
            program_sandbox_key,
        )

        sandbox_handle = self.sandbox_handle(state)
        if sandbox_handle is not None:
            return self.sandbox_lease_from_handle(sandbox_handle, "sandbox")
        scope = sandbox_config.scope
        key = (self.scope_key(scope, state), program_sandbox_key(sandbox_config))
        lease = await self.resolve_sandbox_lease(
            key,
            lambda: create_sandbox_lease(
                sandbox_config,
                key[1],
                client=self.sandbox_client(),
            ),
        )
        return lease

    def sandbox_lease_from_handle(
        self, handle: SandboxRuntimeHandleConfig, name: str
    ) -> "SandboxLease":
        runtime = self.resolved_runtime(handle)
        resolved_lease_key = handle.lease_key
        lease = runtime.sandbox_leases.get(resolved_lease_key)
        if lease is None:
            raise RuntimeError(f"Resolved {name} sandbox lease is no longer active.")
        lease.scope_key = resolved_lease_key[0]
        return lease

    async def resolve_user_sandbox(
        self, user: User, task: Task, state: State
    ) -> object:
        from .utils.sandbox_utils import (
            SandboxHandle,
            create_scoped_sandbox_lease,
            sandbox_owner_key,
        )

        sandbox = user.sandbox
        if sandbox is None:
            raise TypeError("User sandbox must be configured.")
        scope = sandbox.scope
        key = (self.scope_key(scope, state), sandbox_owner_key(user))
        lease = await self.resolve_sandbox_lease(
            key,
            lambda: create_scoped_sandbox_lease(
                user,
                key[1],
                client=self.sandbox_client(),
            ),
        )
        return SandboxHandle(lease, state)

    async def release_sandboxes(self, scope: str, state: State) -> None:
        scope_key = self.scope_key(scope, state)
        async with self.sandbox_lock:
            pending_creations = [
                (key, task)
                for key, task in self.sandbox_creation_tasks.items()
                if key[0] == scope_key
            ]
            scoped_leases = [
                (key, handle)
                for key, handle in self.sandbox_leases.items()
                if key[0] == scope_key and handle.scope == scope
            ]
        if pending_creations:
            await self.clear_sandbox_creation_tasks(
                pending_creations, state=state, scope=scope
            )
        deletion_failures = 0
        for key, handle in scoped_leases:
            try:
                await self.close_sandbox_lease(handle)
            except Exception as exc:
                deletion_failures += 1
                logger.warning(
                    "Failed to delete %s sandbox %s for scope key %s: %s",
                    scope,
                    handle.id,
                    key[0],
                    exc,
                    exc_info=True,
                )
                cleanup_errors = cast(
                    list[ConfigData], state.setdefault("cleanup_errors", [])
                )
                cleanup_errors.append(
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "scope": scope,
                    }
                )
            else:
                async with self.sandbox_lock:
                    if self.sandbox_leases.get(key) is handle:
                        del self.sandbox_leases[key]
        if deletion_failures:
            logger.error(
                "%s/%s %s sandbox deletions failed during cleanup",
                deletion_failures,
                len(scoped_leases),
                scope,
            )

    async def ensure_global_sandboxes(self, state: State | None = None) -> None:
        from .utils.sandbox_utils import (
            create_scoped_sandbox_lease,
            sandbox_owner_key,
            tool_sandbox_key,
        )

        toolsets = self.active_toolsets(state) if state is not None else self.toolsets
        owners: list[Toolset | User] = [*iter_toolsets(toolsets)]
        user = self.active_user()
        if user is not None:
            owners.append(user)
        for owner in owners:
            sandbox = owner.sandbox
            if sandbox is None or sandbox == "program":
                continue
            if not isinstance(sandbox, SandboxConfig):
                raise TypeError("Owner sandbox must be SandboxConfig or 'program'.")
            if sandbox.scope != "global":
                continue
            sandbox_key = (
                tool_sandbox_key(owner)
                if isinstance(owner, Toolset)
                else sandbox_owner_key(owner)
            )
            await self.resolve_sandbox_lease(
                ("global", sandbox_key),
                lambda owner=owner, sandbox_key=sandbox_key: (
                    create_scoped_sandbox_lease(
                        owner,
                        sandbox_key,
                        client=self.sandbox_client(),
                    )
                ),
            )

    def bind_global_sandboxes(self, state: State) -> None:
        from .utils.sandbox_utils import attach_sandbox_ref

        for key, lease in self.sandbox_leases.items():
            scope_key, _ = key
            if scope_key != "global":
                continue
            attach_sandbox_ref(state, lease)

    async def ensure_mcp_tools(self, state: State) -> None:
        from .utils.mcp_utils import connect_mcp_tool

        for key in self.mcp_scope_keys(state):
            if key in self.mcp_exit_stacks:
                continue
            exit_stack = AsyncExitStack()
            tools: RuntimeTools = {}
            tool_parents: dict[str, tuple[Toolset, ...]] = {}
            try:
                for toolset in self.active_toolsets(state):
                    await self._register_mcp_tools(
                        toolset,
                        [toolset],
                        connect_mcp_tool,
                        exit_stack,
                        tools,
                        tool_parents,
                        state,
                        key,
                    )
            except BaseException:
                await exit_stack.aclose()
                raise
            self.mcp_exit_stacks[key] = exit_stack
            self.mcp_tools[key] = tools
            self.mcp_tool_parents[key] = tool_parents

    async def _register_mcp_tools(
        self,
        toolset: Toolset,
        parents: list[Toolset],
        connect_mcp_tool: Callable[
            [MCPTool, AsyncExitStack[bool | None]],
            Awaitable[Sequence["MCPToolHandle"]],
        ],
        exit_stack: AsyncExitStack,
        tools: RuntimeTools,
        tool_parents: dict[str, tuple[Toolset, ...]],
        state: State,
        target_key: str,
    ) -> None:
        for item in self._toolset_entries(toolset, state):
            if isinstance(item, Toolset):
                await self._register_mcp_tools(
                    item,
                    [*parents, item],
                    connect_mcp_tool,
                    exit_stack,
                    tools,
                    tool_parents,
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
                    in self.tools_for_toolsets(
                        self.active_toolsets(state),
                        apply_visibility=False,
                        state=state,
                    )
                    or name in tools
                ):
                    raise ValueError(f"Tool {name!r} is defined twice.")
                tools[name] = handle
                tool_parents[name] = tuple(parents)

    async def close_mcp_tools(self, state: State, scope: str = "rollout") -> None:
        for key in self.mcp_scope_keys(state, scope=scope):
            exit_stack = self.mcp_exit_stacks.pop(key, None)
            self.mcp_tools.pop(key, None)
            self.mcp_tool_parents.pop(key, None)
            if exit_stack is not None:
                await exit_stack.aclose()

    async def close_all_mcp_tools(self) -> None:
        for key, exit_stack in list(self.mcp_exit_stacks.items()):
            self.mcp_tools.pop(key, None)
            self.mcp_tool_parents.pop(key, None)
            del self.mcp_exit_stacks[key]
            await exit_stack.aclose()

    def all_tools(self, state: State) -> RuntimeTools:
        tools = self.tools_for_toolsets(
            self.active_toolsets(state), apply_visibility=False, state=state
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

    def all_exposed_tools(self, state: State, *, validate: bool = True) -> RuntimeTools:
        active_toolsets = self.active_toolsets(state)
        tool_filters = self._task_tools_config(
            state, active_toolsets, validate=validate
        )
        tools = self.tools_for_toolsets(
            active_toolsets,
            apply_visibility=True,
            state=state,
            tool_filters=tool_filters,
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

    def borrowed_tools_for_state(self, state: State) -> RuntimeTools:
        handle = self.resolved_handles(state).tools
        if handle is None:
            return {}
        source_runtime = self.resolved_runtime(handle)
        return {
            name: cast(
                RuntimeTool, BorrowedTool(source_runtime, handle.handle_id, name)
            )
            for name in handle.names
        }

    def _task_tools_config(
        self,
        state: State,
        active_toolsets: Sequence[Toolset],
        *,
        validate: bool,
    ) -> dict[str, VisibilityConfig]:
        task = self.task_for_state(state)
        task_tools = task.tools_config()
        active_ids = {id(toolset) for toolset in iter_toolsets(active_toolsets)}
        active_named_toolsets = {
            name: toolset
            for name, toolset in self.named_toolsets.items()
            if id(toolset) in active_ids
        }
        filters: dict[str, VisibilityConfig] = {}
        for name, filter_config in task_tools.items():
            if name not in active_named_toolsets:
                raise KeyError(f"Unknown toolset tools filter: {name!r}.")
            show = filter_config.show
            hide = filter_config.hide
            if validate:
                available = set(
                    self._tool_names_for_toolset(active_named_toolsets[name], state)
                )
                selected = set(show or hide or [])
                unknown = sorted(selected - available)
                if unknown:
                    raise KeyError(f"Unknown tools for toolset {name!r}: {unknown}.")
            filters[name] = filter_config
        return filters

    def _tool_names_for_toolset(self, toolset: Toolset, state: State) -> list[str]:
        names: list[str] = []

        def visit_toolset(item: Toolset) -> None:
            names.extend(self.mcp_tools.get(self.mcp_scope_key(item, state), ()))
            for child in self._toolset_entries(item, state):
                visit_entry(child)

        def visit_entry(item: ToolEntry) -> None:
            if isinstance(item, Toolset):
                visit_toolset(item)
                return
            if isinstance(item, MCPTool):
                return
            names.append(tool_name(item))

        visit_toolset(toolset)
        return names

    def mcp_tools_for_state(self, state: State, exposed: bool) -> RuntimeTools:
        tool_filters = (
            self._task_tools_config(state, self.active_toolsets(state), validate=False)
            if exposed
            else {}
        )
        tools: RuntimeTools = {}
        for key in self.mcp_scope_keys(state):
            for name, tool in self.mcp_tools.get(key, {}).items():
                parents = self.mcp_tool_parents.get(key, {}).get(name, ())
                if exposed and not all(
                    self._tool_visible_for_task(parent, name, tool_filters)
                    for parent in parents
                ):
                    continue
                if name in tools:
                    raise ValueError(f"Tool {name!r} is defined twice.")
                tools[name] = tool
        return tools

    def mcp_scope_keys(self, state: State, scope: str | None = None) -> list[str]:
        keys: list[str] = []

        def visit(toolset: Toolset) -> None:
            for item in self._toolset_entries(toolset, state):
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
