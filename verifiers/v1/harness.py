from __future__ import annotations

import inspect
import json
import time
from collections.abc import Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, AsyncIterator, Generic, TypeVar, cast, final


from verifiers.clients import resolve_client
from verifiers.errors import Error, OverlongPromptError, ToolError
from verifiers.types import Messages, SamplingArgs, ToolMessage
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.utils.message_utils import normalize_messages

from .config import Config, ConfigSource
from .decorators import discover_decorated
from .interception import EndpointProtocol
from .protocols import default_protocols
from .mcp import MCPToolRegistry
from .runtime import (
    LocalRuntimeConfig,
    RuntimeConfig,
    RuntimeConfigValue,
    RuntimeProvider,
    RuntimeSession,
    make_runtime_provider,
    resolve_runtime_config,
)
from .state import State
from .task import Task
from .toolset import Toolset
from .types import Handler, ModelClient, ModelConfig, RolloutContext
from .user import User
from .utils.config_utils import (
    coerce_config,
    config_ref_context,
    config_type_from_class,
    registered_config_type,
    register_config_type,
)
from .utils.prompt_utils import (
    SystemPrompt,
    SystemPromptStrategy,
    normalize_system_prompt,
    resolve_system_prompt,
)
from .utils.scoring_utils import build_signals
from .utils.scoring_utils import score_rollout

if TYPE_CHECKING:
    from .taskset import Taskset


class HarnessConfig(Config):
    id: str | None = None
    system_prompt: SystemPrompt = None
    system_prompt_strategy: SystemPromptStrategy = "HT"
    max_turns: int = -1
    runtime: RuntimeConfig | None = None


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
        self.taskset: Taskset | None = None
        self.runtime_config = resolve_runtime_config(self.config.runtime)
        self.runtime_provider: RuntimeProvider | None = None
        self._env_tools: MCPToolRegistry | None = None
        self._env_user: MCPToolRegistry | None = None

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
            handlers[kind].extend(cast(list[Handler], discover_decorated(self, kind)))
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
        image = task.image or getattr(self.runtime_config, "image", None)
        if image is None:
            return self.runtime_config
        if isinstance(self.runtime_config, LocalRuntimeConfig):
            raise ValueError(
                f"task {task.task_id!r} declares an image; use docker or prime runtime."
            )
        if not isinstance(image, str) or not image:
            raise TypeError("task.image must be a non-empty string.")
        return self.runtime_config.model_copy(update={"image": image})

    def runtime_provider_for(self, task: Task) -> RuntimeProvider:
        if self.runtime_provider is not None:
            return self.runtime_provider
        return self.load_runtime_provider(self.runtime_for(task))

    @asynccontextmanager
    async def model_context(
        self, model: ModelConfig, teacher: ModelConfig | None = None
    ) -> AsyncIterator[RolloutContext]:
        model_client = self.load_model_client(model)
        teacher_client = (
            self.load_model_client(teacher) if teacher is not None else None
        )
        try:
            yield RolloutContext(model_client=model_client, teacher=teacher_client)
        finally:
            await self.close_model_client(model_client)
            if teacher_client is not None:
                await self.close_model_client(teacher_client)

    async def run(
        self,
        task: Task,
        state: State,
        *,
        model: ModelConfig,
        teacher: ModelConfig | None = None,
    ) -> State:
        async with self.model_context(model, teacher) as ctx:
            state.task_id = task.task_id
            state.model = state.model or ctx.model_client.config
            state.teacher = state.teacher or (
                ctx.teacher.config if ctx.teacher is not None else None
            )
            async with self.runtime_provider_for(task).session() as runtime:
                await self.ensure_env_servers()
                tools: MCPToolRegistry | None = None
                user: MCPToolRegistry | None = None
                try:
                    async with AsyncExitStack() as stack:
                        tools = await stack.enter_async_context(
                            self.rollout_tools(runtime)
                        )
                        user = await stack.enter_async_context(
                            self.rollout_user(runtime)
                        )
                        try:
                            state.timing.setup.begin()
                            await self.run_handlers(
                                "setup",
                                "rollout",
                                task,
                                state,
                                ctx,
                                runtime,
                                tools=tools,
                                user=user,
                            )
                            state.timing.setup.finish()
                            state.timing.generation.begin()
                            await self._run(
                                task,
                                state,
                                ctx=ctx,
                                runtime=runtime,
                                tools=tools,
                                user=user,
                            )
                            state.timing.generation.finish()
                            await self.run_handlers(
                                "update",
                                "rollout",
                                task,
                                state,
                                ctx,
                                runtime,
                                tools=tools,
                                user=user,
                            )
                            await score_rollout(
                                self.owner_signals(),
                                task,
                                state,
                                runtime=runtime,
                                model_client=ctx.model_client,
                                teacher=ctx.teacher,
                            )
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
                            state.timing.cleanup.begin()
                            await self.run_handlers(
                                "cleanup",
                                "rollout",
                                task,
                                state,
                                ctx,
                                runtime,
                                tools=tools,
                                user=user,
                            )
                            state.timing.cleanup.finish()
                            if "num_turns" not in state.metrics:
                                state.metrics["num_turns"] = float(
                                    len(state.transcript)
                                )
                            state.finalize()
                except BaseException as exc:
                    state.capture_error(exc)
                    if not state.timing.generation.end:
                        state.timing.generation.finish()
                    if not state.timing.cleanup.end:
                        state.timing.cleanup.begin()
                        await self.run_handlers(
                            "cleanup", "rollout", task, state, ctx, runtime
                        )
                        state.timing.cleanup.finish()
                    if "num_turns" not in state.metrics:
                        state.metrics["num_turns"] = float(len(state.transcript))
                    state.finalize()
        return state

    async def _run(
        self,
        task: Task,
        state: State,
        *,
        ctx: RolloutContext,
        runtime: RuntimeSession | None = None,
        tools: MCPToolRegistry | None = None,
        user: MCPToolRegistry | None = None,
    ) -> None:
        _ = runtime
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
            if await self.is_completed(task, state, ctx=ctx):
                return
            sampling = self.sampling_args(task, ctx.sampling_args)
            start = time.time()
            response = await ctx.model_client.get_response(
                prompt=messages,
                model=ctx.model,
                sampling_args=sampling,
                tools=tools.tool_defs() if tools is not None else None,
                state=state,
            )
            end = time.time()
            turn = await state.add_response_turn(
                messages, response, start=start, end=end
            )
            messages = [*messages, *turn.completion]
            turns += 1
            if turn.tool_calls:
                if tools is None:
                    raise RuntimeError("Model requested tools but no tools are loaded.")
                tool_messages = await self.call_tools(
                    task, state, tools, turn.tool_calls
                )
                turn.tool_results = tool_messages
                messages = [*messages, *tool_messages]
                continue
            user_messages = await self.call_user(user, task, state)
            if user_messages:
                messages = [*messages, *user_messages]
                continue
            state.stop("assistant_completed")
            return
        state.stop("max_turns")

    async def ensure_env_servers(self) -> None:
        taskset = self.taskset
        if self._env_tools is None:
            toolsets = (
                []
                if taskset is None
                else [toolset for toolset in taskset.toolsets if toolset.scope == "env"]
            )
            self._env_tools = MCPToolRegistry(toolsets)
            await self._env_tools.__aenter__()
        if self._env_user is None:
            user = None if taskset is None else taskset.user
            user_toolsets = []
            if user is not None and user.scope == "env":
                user_toolsets = [
                    Toolset(name=user.name, server=user.server, scope=user.scope)
                ]
            self._env_user = MCPToolRegistry(user_toolsets, expose_tools=False)
            await self._env_user.__aenter__()

    def rollout_tools(self, runtime: RuntimeSession) -> MCPToolRegistry:
        _ = runtime
        taskset = self.taskset
        parents = [self._env_tools] if self._env_tools is not None else []
        toolsets = (
            []
            if taskset is None
            else [toolset for toolset in taskset.toolsets if toolset.scope == "rollout"]
        )
        return MCPToolRegistry(toolsets, parents=parents)

    def rollout_user(self, runtime: RuntimeSession) -> MCPToolRegistry:
        _ = runtime
        taskset = self.taskset
        parents = [self._env_user] if self._env_user is not None else []
        user = None if taskset is None else taskset.user
        user_toolsets = []
        if user is not None and user.scope == "rollout":
            user_toolsets = [
                Toolset(name=user.name, server=user.server, scope=user.scope)
            ]
        return MCPToolRegistry(user_toolsets, parents=parents, expose_tools=False)

    @property
    def stop_handlers(self) -> list[Handler]:
        return self.owner_handlers("stop")

    def owner_handlers(self, kind: str) -> list[Handler]:
        taskset_handlers: list[Handler] = []
        if self.taskset is not None:
            taskset_handlers = cast(list[Handler], self.taskset.handlers[kind])
        return [*taskset_handlers, *cast(list[Handler], self.handlers[kind])]

    def owner_signals(self) -> list[dict[str, object]]:
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
        prompt = normalize_messages(task.prompt, field_name="task.prompt")
        return [*normalize_messages(system_prompt, field_name="system_prompt"), *prompt]

    def has_model_prompt(self, messages: Messages) -> bool:
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
        self, task: Task, state: State, tools: MCPToolRegistry, tool_calls
    ) -> list[ToolMessage]:
        tool_messages: list[ToolMessage] = []
        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError("Tool arguments must decode to an object.")
                tools.set_context(self.context(task, state))
                result = await tools.call(tool_call.name, arguments)
                content = self.apply_tool_result(state, result)
            except Exception as exc:
                if isinstance(exc, ToolError):
                    content = f"Tool error: {exc}"
                else:
                    content = f"Tool error: {type(exc).__name__}: {exc}"
            tool_messages.append(
                ToolMessage(tool_call_id=tool_call.id, content=str(content))
            )
        return tool_messages

    def context(self, task: Task, state: State) -> dict[str, object]:
        return {
            "task": task.to_record(),
            "state": cast(
                dict[str, object],
                state.model_dump(mode="json", exclude_none=True),
            ),
            "transcript": [
                cast(dict[str, object], turn.model_dump(mode="json", exclude_none=True))
                for turn in state.transcript
            ],
        }

    def apply_tool_result(self, state: State, result: object) -> str:
        if not isinstance(result, Mapping):
            return result if isinstance(result, str) else json.dumps(result)
        result_map = cast(Mapping[str, object], result)
        if not any(
            key in result_map
            for key in (
                "content",
                "scratch",
                "metrics",
                "artifacts",
                "reward_delta",
                "stop_condition",
                "is_completed",
                "is_truncated",
            )
        ):
            return json.dumps(result_map)
        scratch = result_map.get("scratch")
        if isinstance(scratch, Mapping):
            state.scratch.update(deepcopy(dict(scratch)))
        metrics = result_map.get("metrics")
        if isinstance(metrics, Mapping):
            state.metrics.update(
                {
                    str(key): float(value)
                    for key, value in metrics.items()
                    if isinstance(value, int | float)
                }
            )
        artifacts = result_map.get("artifacts")
        if isinstance(artifacts, Mapping):
            state.artifacts.update(deepcopy(dict(artifacts)))
        reward_delta = result_map.get("reward_delta")
        if isinstance(reward_delta, int | float):
            state.reward += float(reward_delta)
        is_truncated = result_map.get("is_truncated")
        if isinstance(is_truncated, bool):
            state.is_truncated = is_truncated
        stop_condition = result_map.get("stop_condition")
        if isinstance(stop_condition, str):
            state.stop(stop_condition)
        else:
            is_completed = result_map.get("is_completed")
            if isinstance(is_completed, bool):
                state.is_completed = is_completed
        state.assert_serializable()
        content = result_map.get("content", "")
        return content if isinstance(content, str) else json.dumps(content)

    async def call_user(
        self, user: MCPToolRegistry | None, task: Task, state: State
    ) -> Messages:
        if user is None or not user.has_hidden("respond"):
            return []
        request = User.TurnRequest(
            task=task.to_record(),
            state=cast(dict[str, object], self.context(task, state)["state"]),
            transcript=cast(
                list[dict[str, object]], self.context(task, state)["transcript"]
            ),
        )
        result = await user.call_hidden("respond", request.model_dump(mode="json"))
        if isinstance(result, str):
            result = json.loads(result)
        if not isinstance(result, Mapping):
            raise TypeError("User MCP tool must return an object.")
        response = User.TurnResult.model_validate(result)
        state.scratch.update(deepcopy(response.scratch))
        state.metrics.update(response.metrics)
        state.artifacts.update(deepcopy(response.artifacts))
        state.reward += response.reward_delta
        if response.is_truncated is not None:
            state.is_truncated = response.is_truncated
        if response.stop_condition is not None:
            state.stop(response.stop_condition)
        elif response.is_completed is not None:
            state.is_completed = response.is_completed
        state.assert_serializable()
        return normalize_messages(response.messages, field_name="user.messages")

    async def is_completed(
        self,
        task: Task,
        state: State,
        *,
        ctx: RolloutContext | None = None,
    ) -> bool:
        if state.is_completed:
            return True
        for handler in self.stop_handlers:
            if await self.call_handler(
                handler,
                task,
                state,
                model_client=ctx.model_client if ctx is not None else None,
                client=ctx.client if ctx is not None else None,
                model=ctx.model if ctx is not None else None,
                teacher=ctx.teacher if ctx is not None else None,
                teacher_client=ctx.teacher.client
                if ctx is not None and ctx.teacher is not None
                else None,
                teacher_model=ctx.teacher.config.model
                if ctx is not None and ctx.teacher is not None
                else None,
            ):
                state.stop(getattr(handler, "__name__", "stop"))
                return True
        return False

    async def run_handlers(
        self,
        kind: str,
        stage: str,
        task: Task,
        state: State,
        ctx: RolloutContext,
        runtime: RuntimeSession | None = None,
        tools: MCPToolRegistry | None = None,
        user: MCPToolRegistry | None = None,
    ) -> None:
        for handler in self.owner_handlers(kind):
            handler_stage = getattr(handler, f"{kind}_stage", "rollout")
            if handler_stage != stage:
                continue
            result = await self.call_handler(
                handler,
                task,
                state,
                runtime=runtime,
                tools=tools,
                user=user,
                model_client=ctx.model_client,
                client=ctx.client,
                model=ctx.model,
                teacher=ctx.teacher,
                teacher_client=ctx.teacher.client if ctx.teacher is not None else None,
                teacher_model=ctx.teacher.config.model
                if ctx.teacher is not None
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
        runtime: RuntimeSession | None = None,
        **extra: object,
    ) -> object:
        return await maybe_call_with_named_args(
            handler,
            task=task,
            state=state,
            scratch=state.scratch,
            transcript=state.transcript,
            completion=state.completion,
            metrics=state.metrics,
            reward=state.reward,
            prompt=state.prompt or task.prompt,
            example_id=task.row_id,
            harness=self,
            runtime=runtime,
            tools=extra.pop("tools", None),
            user=extra.pop("user", None),
            **extra,
        )

    async def teardown(self) -> None:
        for handler in self.handlers["teardown"]:
            result = handler()
            if inspect.isawaitable(result):
                await result

    async def close(self) -> None:
        if self._env_user is not None:
            await self._env_user.__aexit__(None, None, None)
            self._env_user = None
        if self._env_tools is not None:
            await self._env_tools.__aexit__(None, None, None)
            self._env_tools = None
        await self.teardown()
