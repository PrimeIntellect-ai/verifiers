from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from pydantic import TypeAdapter
from typing import TYPE_CHECKING, AsyncIterator, Generic, TypeVar, cast, final


from verifiers.clients import resolve_client
from verifiers.errors import Error, OverlongPromptError, ToolError
from verifiers.types import Messages, SamplingArgs, ToolMessage, UserMessage
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.utils.response_utils import parse_response_message

from .config import Config, ConfigSource
from .decorators import discover_decorated
from .interception import EndpointProtocol
from .protocols import default_protocols
from .mcp import BoundUpdate, ServerResult, MCPToolRegistry, ServerResponse
from .runtime import (
    RuntimeConfig,
    RuntimeConfigValue,
    RuntimeProvider,
    Runtime,
    SubprocessRuntimeConfig,
    make_runtime_provider,
    resolve_runtime_config,
)
from .state import Extras, State, TimeSpan, Turn, TurnTokens, TurnUsage
from .task import Task
from .types import Handler, JsonData, JsonValue, ModelClient, ModelConfig, Context
from .utils.config_utils import (
    coerce_config,
    config_ref_context,
    config_type_from_class,
    registered_config_type,
    register_config_type,
)
from .utils.json_utils import json_args, json_data
from .utils.prompt_utils import (
    SystemPrompt,
    SystemPromptStrategy,
    normalize_system_prompt,
    resolve_system_prompt,
)
from .utils.scoring_utils import SignalRecord, build_signals
from .utils.scoring_utils import score_rollout

if TYPE_CHECKING:
    from .taskset import Taskset

_MESSAGES_ADAPTER = TypeAdapter(Messages)


class HarnessConfig(Config):
    id: str | None = None
    system_prompt: SystemPrompt = None
    system_prompt_strategy: SystemPromptStrategy = "HT"
    max_turns: int = -1
    runtime: RuntimeConfig | None = None
    extras: Extras | None = None


ConfigT = TypeVar("ConfigT", bound=HarnessConfig)


class Harness(Generic[ConfigT]):
    config: ConfigT

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        config_type = config_type_from_class(
            cls,
            inherited=False,
            owner_base=Harness,
            config_base=HarnessConfig,
        )
        if config_type is not None:
            register_config_type(cls, config_type)

    @final
    def __init__(self, config: ConfigSource = None):
        config_type = registered_config_type(type(self), HarnessConfig)
        self.config = cast(ConfigT, coerce_config(config_type, config))
        with config_ref_context(self.config):
            resolved_id = self.config.id
            if resolved_id is not None and not isinstance(resolved_id, str):
                raise TypeError("harness id must be a string.")
            self.id = resolved_id or type(self).__name__
            self.system_prompt = normalize_system_prompt(
                self.load_system_prompt(self.config), field_name="harness.system_prompt"
            )
            self.system_prompt_strategy = self.config.system_prompt_strategy
            self.handlers = self.load_handlers()
            self.protocols = self.load_protocols()
            self.signals = build_signals(self)
            for signal in self.signals:
                if signal["kind"] != "metric":
                    raise ValueError("Harness signals must be metrics.")
        self.taskset: Taskset | None = None
        self.runtime_config = resolve_runtime_config(self.config.runtime)
        self.runtime_provider: RuntimeProvider | None = None
        self._env_toolsets: MCPToolRegistry | None = None
        self._env_user: MCPToolRegistry | None = None
        self._env_servers_lock = asyncio.Lock()
        self._env_scope_count = 0
        self.extras_schema: type[Extras] | None = Extras.schema_for(self.config.extras)
        self.extras_defaults: JsonData = Extras.defaults_for(self.config.extras)

    def load_system_prompt(self, config: ConfigT) -> SystemPrompt:
        return config.system_prompt

    def load_handlers(self) -> dict[str, list[Handler]]:
        handlers: dict[str, list[Handler]] = {
            "stop": [],
            "setup": [],
            "update": [],
            "cleanup": [],
            "teardown": [],
        }
        for kind in handlers:
            handlers[kind].extend(discover_decorated(self, kind))
        return handlers

    def load_protocols(self) -> list[EndpointProtocol]:
        return default_protocols()

    def load_model_client(self, config: ModelConfig) -> ModelClient:
        return ModelClient(config=config, client=resolve_client(config.client))

    async def close_model_client(self, model_client: ModelClient) -> None:
        await model_client.client.close()

    def bind(
        self,
        *,
        taskset: "Taskset | None" = None,
        runtime: RuntimeProvider | RuntimeConfigValue | None = None,
    ) -> None:
        self.taskset = taskset
        taskset_runtime = taskset.config.runtime if taskset is not None else None
        taskset_extras = None if taskset is None else taskset.config.extras
        self.extras_schema = Extras.realize_schema(
            Extras.schema_for(taskset_extras), Extras.schema_for(self.config.extras)
        )
        self.extras_defaults = Extras.merge_defaults(
            Extras.defaults_for(taskset_extras), Extras.defaults_for(self.config.extras)
        )
        if isinstance(runtime, RuntimeProvider):
            self.runtime_provider = runtime
            self.runtime_config = resolve_runtime_config(
                taskset_runtime, self.config.runtime
            )
            return
        self.runtime_config = resolve_runtime_config(
            taskset_runtime, self.config.runtime, runtime
        )
        self.runtime_provider = None

    def load_runtime_provider(self, config: RuntimeConfigValue) -> RuntimeProvider:
        return make_runtime_provider(config)

    def runtime_for(self, task: Task) -> RuntimeConfigValue:
        updates: JsonData = {}
        if task.image is not None:
            if isinstance(self.runtime_config, SubprocessRuntimeConfig):
                raise ValueError(
                    f"task {task.task_id!r} declares an image; use docker or "
                    "prime runtime."
                )
            if not isinstance(task.image, str) or not task.image:
                raise TypeError("task.image must be a non-empty string.")
            updates["image"] = task.image
        for field, value in task.resources.model_dump(exclude_none=True).items():
            spec = type(self.runtime_config).model_fields.get(field)
            if spec is None:
                raise ValueError(
                    f"task {task.task_id!r} declares resource {field!r}; runtime "
                    f"{self.runtime_config.type!r} does not support it."
                )
            if getattr(self.runtime_config, field) == spec.default:
                updates[field] = value
        if not updates:
            return self.runtime_config
        return self.runtime_config.model_copy(update=updates)

    def runtime_provider_for(self, task: Task) -> RuntimeProvider:
        if self.runtime_provider is not None:
            return self.runtime_provider
        return self.load_runtime_provider(self.runtime_for(task))

    @asynccontextmanager
    async def open_context(
        self,
        *,
        task: Task,
        state: State,
        model: ModelConfig,
        teacher: ModelConfig | None = None,
        runtime: Runtime | None = None,
        toolsets: MCPToolRegistry | None = None,
        user: MCPToolRegistry | None = None,
        parent: Context | None = None,
        score: bool = False,
    ) -> AsyncIterator[Context]:
        model_client = (
            parent.model_client
            if parent is not None and parent.model_client.config == model
            else self.load_model_client(model)
        )
        teacher_client = None
        if teacher is not None:
            teacher_client = (
                parent.teacher
                if parent is not None
                and parent.teacher is not None
                and parent.teacher.config == teacher
                else self.load_model_client(teacher)
            )
        try:
            yield Context(
                task=task,
                state=state,
                model_client=model_client,
                teacher=teacher_client,
                runtime=runtime,
                toolsets=toolsets,
                user=user,
                parent=parent,
                score=score,
            )
        finally:
            if model_client is not (
                parent.model_client if parent is not None else None
            ):
                await self.close_model_client(model_client)
            if teacher_client is not None and teacher_client is not (
                parent.teacher if parent is not None else None
            ):
                await self.close_model_client(teacher_client)

    async def run(
        self,
        task: Task | str,
        state: State | None = None,
        *,
        model: ModelConfig | str | None = None,
        teacher: ModelConfig | str | None = None,
        context: Context | None = None,
        score: bool = False,
    ) -> State:
        if isinstance(task, str):
            task = Task(prompt=task)
        if isinstance(model, str):
            model = ModelConfig(model=model)
        if isinstance(teacher, str):
            teacher = ModelConfig(model=teacher)
        if context is not None and score and context.has_active_scoring():
            raise RuntimeError("Nested scored harness runs are not supported.")
        using_context_state = state is None and context is not None
        if state is None:
            state = (
                context.state if context is not None else State(task_id=task.task_id)
            )
        if model is None:
            if context is None:
                raise TypeError("Harness.run requires model unless context is passed.")
            model = context.model_client.config
        if teacher is None and context is not None and context.teacher is not None:
            teacher = context.teacher.config
        if not using_context_state or state.task_id is None:
            state.task_id = task.task_id
        state.model = state.model or model
        state.teacher = state.teacher or teacher
        self.initialize_extras(state)

        if context is not None:
            async with self.open_context(
                task=task,
                state=state,
                model=model,
                teacher=teacher,
                runtime=context.runtime,
                toolsets=context.toolsets,
                user=context.user,
                parent=context,
                score=score,
            ) as child_context:
                if child_context.toolsets is not None:
                    child_context.toolsets.set_visibility(
                        toolsets=task.toolsets,
                        tools=task.tools,
                    )
                await self.run_lifecycle(child_context)
            return state

        from .lifecycle import EnvRun

        async with EnvRun(harness=self) as env_run:
            return await env_run.run_context(
                task,
                state,
                model=model,
                teacher=teacher,
                score=score,
            )

    async def run_lifecycle(self, context: Context) -> None:
        task = context.task
        state = context.state
        try:
            try:
                state.timing.setup.begin()
                await self.run_handlers("setup", "rollout", context)
                await self.resolve_toolsets(context)
                self.validate_extras(state)
                state.timing.setup.finish()
                state.timing.generation.begin()
                await self.run_with_context(context)
                state.timing.generation.finish()
                await self.run_handlers("update", "rollout", context)
                self.validate_extras(state)
                if context.score:
                    context.scoring = True
                    try:
                        await score_rollout(
                            self.owner_signals(),
                            task,
                            state,
                            runtime=context.runtime,
                            model_client=context.model_client,
                            teacher=context.teacher,
                            context=context,
                        )
                        self.validate_extras(state)
                    finally:
                        context.scoring = False
            except OverlongPromptError as exc:
                state.is_truncated = True
                state.capture_error(exc)
            except Error as exc:
                state.capture_error(exc)
            except BaseException as exc:
                state.capture_error(exc)
        finally:
            if not state.timing.generation.end:
                state.timing.generation.finish()
            if not state.timing.cleanup.end:
                state.timing.cleanup.begin()
                await self.run_handlers("cleanup", "rollout", context)
                self.validate_extras(state)
                state.timing.cleanup.finish()
            if "num_turns" not in state.metrics:
                state.metrics["num_turns"] = float(len(state.transcript))
            self.validate_extras(state)
            state.assert_serializable()

    def initialize_extras(self, state: State) -> None:
        for key, value in self.extras_defaults.items():
            state.extras.setdefault(key, deepcopy(value))

    def validate_extras(self, state: State) -> None:
        if self.extras_schema is None:
            return
        self.extras_schema.model_validate(state.extras)

    async def resolve_toolsets(self, context: Context) -> None:
        if context.toolsets is None:
            return
        task = context.task
        state = context.state
        await context.toolsets.resolve(
            context=self.binding_context(task, state),
            resolution_key=f"{state.id}:{task.task_id}",
            apply_updates=lambda updates: self.apply_bound_updates(state, updates),
        )
        self.validate_extras(state)
        state.assert_serializable()

    async def run_with_context(self, context: Context) -> None:
        task = context.task
        state = context.state
        toolsets = context.toolsets
        user = context.user
        messages = self.initial_messages(task)
        if not self.has_model_prompt(messages):
            bootstrap_messages = await self.call_user(user, task, state)
            if not bootstrap_messages and not state.is_completed:
                raise ValueError(
                    "Task prompt is empty and no user server produced an initial "
                    "message."
                )
            messages = [*messages, *bootstrap_messages]
            if state.is_completed:
                return
        max_turns = self.max_turns(task)
        turns = 0
        while max_turns <= 0 or turns < max_turns:
            if await self.is_completed(context):
                return
            sampling = self.sampling_args(task, context.sampling_args)
            start = time.time()
            response = await context.model_client.get_response(
                prompt=messages,
                model=context.model,
                sampling_args=sampling,
                tools=toolsets.tools() if toolsets is not None else None,
                state=state,
            )
            end = time.time()
            turn = Turn(
                prompt=messages,
                completion=await parse_response_message(response),
                tool_calls=list(response.message.tool_calls or []),
                response_id=response.id,
                model=response.model,
                created=response.created,
                finish_reason=response.message.finish_reason,
                usage=TurnUsage.from_usage(response.usage),
                tokens=TurnTokens.from_response(
                    response.message.tokens,
                    is_truncated=bool(response.message.is_truncated),
                ),
                is_truncated=bool(response.message.is_truncated),
                timing=TimeSpan(start=start, end=end),
            )
            state.transcript.append(turn)
            if turn.is_truncated:
                state.is_truncated = True
            messages = [*messages, *turn.completion]
            turns += 1
            if turn.tool_calls:
                if toolsets is None:
                    raise RuntimeError("Model requested tools but no tools are loaded.")
                tool_messages, server_messages = await self.call_tools(
                    task, state, toolsets, turn.tool_calls
                )
                turn.tool_results = tool_messages
                messages = [*messages, *tool_messages, *server_messages]
                continue
            user_messages = await self.call_user(user, task, state)
            if user_messages:
                messages = [*messages, *user_messages]
                continue
            state.stop("assistant_completed")
            return
        state.stop("max_turns")

    async def start_env_scope(self) -> None:
        async with self._env_servers_lock:
            if self._env_scope_count > 0:
                self._env_scope_count += 1
                return
            taskset = self.taskset
            started_toolsets = False
            try:
                if self._env_toolsets is None:
                    toolsets = (
                        {}
                        if taskset is None
                        else {
                            name: toolset
                            for name, toolset in taskset.toolsets.items()
                            if toolset.scope == "env"
                        }
                    )
                    env_toolsets = MCPToolRegistry(toolsets)
                    try:
                        await env_toolsets.__aenter__()
                    except BaseException:
                        await env_toolsets.__aexit__(None, None, None)
                        raise
                    self._env_toolsets = env_toolsets
                    started_toolsets = True
                if self._env_user is None:
                    user = None if taskset is None else taskset.user
                    user_toolsets = (
                        {} if user is None or user.scope != "env" else {"user": user}
                    )
                    env_user = MCPToolRegistry(user_toolsets)
                    try:
                        await env_user.__aenter__()
                    except BaseException:
                        await env_user.__aexit__(None, None, None)
                        raise
                    self._env_user = env_user
                self._env_scope_count = 1
            except BaseException:
                self._env_scope_count = 0
                if started_toolsets and self._env_toolsets is not None:
                    try:
                        await self._env_toolsets.__aexit__(None, None, None)
                    finally:
                        self._env_toolsets = None
                raise

    async def stop_env_scope(self, *, force: bool = False) -> None:
        async with self._env_servers_lock:
            if not force and self._env_scope_count > 1:
                self._env_scope_count -= 1
                return
            self._env_scope_count = 0
            try:
                if self._env_user is not None:
                    await self._env_user.__aexit__(None, None, None)
            finally:
                self._env_user = None
                try:
                    if self._env_toolsets is not None:
                        await self._env_toolsets.__aexit__(None, None, None)
                finally:
                    self._env_toolsets = None

    def rollout_toolsets(
        self, runtime: Runtime, user: MCPToolRegistry | None = None
    ) -> MCPToolRegistry:
        _ = runtime
        taskset = self.taskset
        parents = [
            parent for parent in (self._env_toolsets, user) if parent is not None
        ]
        toolsets = (
            {}
            if taskset is None
            else {
                name: toolset
                for name, toolset in taskset.toolsets.items()
                if toolset.scope == "rollout"
            }
        )
        return MCPToolRegistry(toolsets, runtime=runtime, parents=parents)

    def rollout_user(self, runtime: Runtime, task: Task) -> MCPToolRegistry:
        _ = runtime
        if task.user is False:
            return MCPToolRegistry({}, expose_tools=False)
        taskset = self.taskset
        parents = [self._env_user] if self._env_user is not None else []
        user = None if taskset is None else taskset.user
        user_toolsets = (
            {} if user is None or user.scope != "rollout" else {"user": user}
        )
        return MCPToolRegistry(user_toolsets, runtime=runtime, parents=parents)

    @property
    def stop_handlers(self) -> list[Handler]:
        return self.owner_handlers("stop")

    def owner_handlers(self, kind: str) -> list[Handler]:
        taskset_handlers: list[Handler] = []
        if self.taskset is not None:
            taskset_handlers = self.taskset.handlers[kind]
        return [*taskset_handlers, *self.handlers[kind]]

    def owner_signals(self) -> list[SignalRecord]:
        signals = list(getattr(self.taskset, "signals", [])) if self.taskset else []
        seen = {str(signal["name"]) for signal in signals}
        for signal in self.signals:
            if signal["name"] in seen:
                raise ValueError(f"Signal {signal['name']!r} is defined twice.")
            signals.append(signal)
        return sorted(signals, key=lambda signal: (-signal["priority"], signal["name"]))

    def initial_messages(self, task: Task) -> Messages:
        taskset_system_prompt = []
        if self.taskset is not None:
            taskset_system_prompt = getattr(self.taskset, "system_prompt", [])
        system_prompt = resolve_system_prompt(
            task=task,
            taskset_system_prompt=taskset_system_prompt,
            harness_system_prompt=self.system_prompt,
            strategy=self.system_prompt_strategy,
        )
        return [*_MESSAGES_ADAPTER.validate_python(system_prompt), *task.prompt]

    @staticmethod
    def has_model_prompt(messages: Messages) -> bool:
        return any(getattr(message, "role", None) != "system" for message in messages)

    def max_turns(self, task: Task) -> int:
        value = task.max_turns
        if value is None:
            return self.config.max_turns
        return value

    def sampling_args(self, task: Task, sampling_args: SamplingArgs) -> SamplingArgs:
        _ = task
        return dict(sampling_args)

    async def call_tools(
        self, task: Task, state: State, toolsets: MCPToolRegistry, tool_calls
    ) -> tuple[list[ToolMessage], Messages]:
        toolsets.set_context(self.binding_context(task, state))
        tool_results: list[ToolMessage] = []
        server_messages: Messages = []
        updates: list[BoundUpdate] = []
        for tool_call in tool_calls:
            try:
                arguments = json_args(tool_call.arguments or "{}")
                result = await toolsets.call(tool_call.name, arguments)
                updates.extend(result.updates)
                call_tool_results, call_messages = self.tool_response_messages(
                    tool_call.id, result.response
                )
            except Exception as exc:
                if isinstance(exc, ToolError):
                    content = f"Tool error: {exc}"
                else:
                    content = f"Tool error: {type(exc).__name__}: {exc}"
                call_tool_results = [
                    ToolMessage(tool_call_id=tool_call.id, content=content)
                ]
                call_messages = []
            tool_results.extend(call_tool_results)
            server_messages.extend(call_messages)
        try:
            self.apply_bound_updates(state, updates)
        except Exception as exc:
            message = f"Tool error: {type(exc).__name__}: {exc}"
            return (
                [
                    ToolMessage(tool_call_id=tool_call.id, content=message)
                    for tool_call in tool_calls
                ],
                [],
            )
        return tool_results, server_messages

    @staticmethod
    def tool_response_messages(
        tool_call_id: str, response: ServerResponse
    ) -> tuple[list[ToolMessage], Messages]:
        if response.content is not None:
            return [
                ToolMessage(tool_call_id=tool_call_id, content=response.content)
            ], list(response.messages)
        if not response.messages:
            return [ToolMessage(tool_call_id=tool_call_id, content="")], []
        tool_results = [
            message for message in response.messages if isinstance(message, ToolMessage)
        ]
        extra_messages: Messages = []
        for message in response.messages:
            if not isinstance(message, ToolMessage):
                extra_messages.append(message)
        if not tool_results:
            tool_results = [ToolMessage(tool_call_id=tool_call_id, content="")]
        return tool_results, extra_messages

    @staticmethod
    def binding_context(task: Task, state: State) -> JsonData:
        state_data = state.model_dump(
            mode="json",
            exclude_none=True,
            exclude_computed_fields=True,
        )
        state_data["prompt"] = State.serialized_messages(state.prompt)
        state_data["completion"] = State.serialized_messages(state.completion)
        state_data["messages"] = State.serialized_messages(state.messages)
        return json_data(
            {
                "task": task.model_dump(
                    mode="json", exclude_none=True, exclude_defaults=True
                ),
                "state": state_data,
                "extras": state.extras,
            },
            context="binding context",
        )

    def apply_tool_result(self, state: State, result: ServerResult) -> ServerResponse:
        self.apply_bound_updates(state, list(result.updates))
        return result.response

    def apply_bound_updates(self, state: State, updates: list[BoundUpdate]) -> None:
        assignments: list[tuple[str, JsonValue, str]] = []
        for update in updates:
            assignments.extend(self.bound_assignments(update))
        for index, (target, _, mode) in enumerate(assignments):
            for existing, _, existing_mode in assignments[:index]:
                if self.assignment_conflicts(existing, existing_mode, target, mode):
                    raise ValueError(
                        f"Conflicting bound state updates: {existing!r} and {target!r}."
                    )
        for target, value, mode in assignments:
            self.apply_assignment(state, target, value, mode)
        if assignments:
            state.assert_serializable()
            self.validate_extras(state)

    @staticmethod
    def bound_assignments(update: BoundUpdate) -> list[tuple[str, JsonValue, str]]:
        target = update.target
        if target.startswith("extras."):
            target = f"state.{target}"
        if not target.startswith("state."):
            raise ValueError(
                f"Bound return target {target!r} must start with state. or extras."
            )
        parts = target.split(".")
        if len(parts) < 2:
            raise ValueError(f"Bound return target {target!r} is incomplete.")
        if update.mode == "extend":
            if not isinstance(update.value, list):
                raise TypeError(f"Extend target {target!r} requires a list.")
            return [(target, deepcopy(update.value), "extend")]
        if update.mode == "set":
            return [(target, deepcopy(update.value), "set")]
        raise ValueError(f"Unknown bound return mode {update.mode!r}.")

    @staticmethod
    def apply_assignment(
        state: State, target: str, value: JsonValue, mode: str
    ) -> None:
        parts = target.split(".")
        field = parts[1]
        if field in {"extras", "metadata", "artifacts"}:
            if len(parts) < 3:
                raise ValueError(f"Bound return target {target!r} needs a key.")
            if field == "extras":
                container = state.extras
            elif field == "metadata":
                container = state.metadata
            else:
                container = state.artifacts
            if mode == "extend":
                Harness.extend_mapping_path(container, parts[2:], value)
            else:
                Harness.set_mapping_path(container, parts[2:], value)
            return
        if mode != "set":
            raise ValueError(f"Bound return target {target!r} only supports set.")
        if field == "transcript":
            if parts != ["state", "transcript", "last", "reward"]:
                raise ValueError(
                    "state.transcript only supports state.transcript.last.reward."
                )
            if not state.transcript:
                raise RuntimeError("state.transcript.last.reward requires a turn.")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError("state.transcript.last.reward requires a number.")
            state.transcript[-1].reward = float(value)
            return
        if field == "metrics":
            if len(parts) != 3:
                raise ValueError(f"Metric target {target!r} must name one metric.")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"Metric target {target!r} requires a number.")
            state.metrics[parts[2]] = float(value)
            return
        if field == "reward":
            if len(parts) != 2:
                raise ValueError("state.reward does not support nested targets.")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError("state.reward requires a number.")
            state.reward = float(value)
            return
        if field == "is_completed":
            if len(parts) != 2 or not isinstance(value, bool):
                raise TypeError("state.is_completed requires a boolean.")
            state.is_completed = value
            return
        if field == "is_truncated":
            if len(parts) != 2 or not isinstance(value, bool):
                raise TypeError("state.is_truncated requires a boolean.")
            state.is_truncated = value
            return
        if field == "stop_condition":
            if len(parts) != 2 or not isinstance(value, str):
                raise TypeError("state.stop_condition requires a string.")
            state.stop(value)
            return
        raise ValueError(f"Bound return target {target!r} is not writable.")

    @staticmethod
    def assignment_conflicts(
        left: str, left_mode: str, right: str, right_mode: str
    ) -> bool:
        if left_mode == right_mode == "extend" and left == right:
            return False
        return Harness.paths_conflict(left, right)

    @staticmethod
    def paths_conflict(left: str, right: str) -> bool:
        return (
            left == right
            or left.startswith(f"{right}.")
            or right.startswith(f"{left}.")
        )

    @staticmethod
    def set_mapping_path(
        container: JsonData, path: list[str], value: JsonValue
    ) -> None:
        target = Harness.nested_mapping(container, path[:-1])
        target[path[-1]] = deepcopy(value)

    @staticmethod
    def extend_mapping_path(
        container: JsonData, path: list[str], value: JsonValue
    ) -> None:
        if not isinstance(value, list):
            raise TypeError(f"Bound extend target {'.'.join(path)!r} requires a list.")
        target = Harness.nested_mapping(container, path[:-1])
        existing = target.get(path[-1])
        if existing is None:
            target[path[-1]] = deepcopy(value)
            return
        if not isinstance(existing, list):
            raise TypeError(f"Bound extend target {'.'.join(path)!r} is not a list.")
        existing.extend(deepcopy(value))

    @staticmethod
    def nested_mapping(container: JsonData, path: list[str]) -> JsonData:
        current = container
        for part in path:
            value = current.get(part)
            if value is None:
                child: JsonData = {}
                current[part] = child
                current = child
                continue
            if not isinstance(value, dict):
                raise TypeError(f"Bound return path {part!r} traverses a non-object.")
            current = value
        return current

    async def call_user(
        self, user: MCPToolRegistry | None, task: Task, state: State
    ) -> Messages:
        if task.user is False:
            return []
        if user is None or not user.has_hidden("respond"):
            if task.user is True:
                raise ValueError("Task requires a user server, but none is loaded.")
            return []
        user.set_context(self.binding_context(task, state))
        result = await user.call_hidden("respond", {})
        self.apply_bound_updates(state, list(result.updates))
        state.assert_serializable()
        messages = list(result.response.messages)
        if result.response.content is not None:
            messages.insert(0, UserMessage(content=result.response.content))
        return messages

    async def is_completed(self, context: Context) -> bool:
        task = context.task
        state = context.state
        if state.is_completed:
            return True
        for handler in self.stop_handlers:
            if await self.call_handler(
                handler,
                task,
                state,
                context=context,
                runtime=context.runtime,
                toolsets=context.toolsets,
                user=context.user,
                model=context.model_client,
                model_name=context.model,
                teacher=context.teacher,
                teacher_name=context.teacher.config.model
                if context.teacher is not None
                else None,
            ):
                state.stop(getattr(handler, "__name__", "stop"))
                return True
        return False

    async def run_handlers(
        self,
        kind: str,
        stage: str,
        context: Context,
    ) -> None:
        task = context.task
        state = context.state
        for handler in self.owner_handlers(kind):
            handler_stage = getattr(handler, f"{kind}_stage", "rollout")
            if handler_stage != stage:
                continue
            binding_context = self.binding_context(task, state)
            if context.toolsets is not None:
                context.toolsets.set_context(binding_context)
            if context.user is not None:
                context.user.set_context(binding_context)
            result = await self.call_handler(
                handler,
                task,
                state,
                context=context,
                runtime=context.runtime,
                toolsets=context.toolsets,
                user=context.user,
                model=context.model_client,
                model_name=context.model,
                teacher=context.teacher,
                teacher_name=context.teacher.config.model
                if context.teacher is not None
                else None,
            )
            if result is None:
                continue
            raise TypeError(
                f"{kind} handler {getattr(handler, '__name__', handler)!r} must mutate "
                f"state in place and return None, not {type(result).__name__}."
            )

    async def call_handler(
        self,
        handler: Handler,
        task: Task,
        state: State,
        runtime: Runtime | None = None,
        **extra: object,
    ) -> object:
        return await maybe_call_with_named_args(
            handler,
            task=task,
            state=state,
            extras=state.extras,
            transcript=state.transcript,
            completion=state.completion,
            metrics=state.metrics,
            reward=state.reward,
            prompt=state.prompt if state.transcript else task.prompt,
            example_id=task.row_id,
            harness=self,
            context=extra.pop("context", None),
            runtime=runtime,
            toolsets=extra.pop("toolsets", None),
            user=extra.pop("user", None),
            **extra,
        )

    async def teardown(self) -> None:
        for handler in self.handlers["teardown"]:
            result = handler()
            if inspect.isawaitable(result):
                await result

    async def close(self) -> None:
        try:
            await self.stop_env_scope(force=True)
        finally:
            await self.teardown()
