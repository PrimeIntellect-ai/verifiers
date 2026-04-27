from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast, final

from verifiers.decorators import discover_decorated, render, stop
from verifiers.errors import Error, OverlongPromptError, ToolCallError, ToolParseError
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    AssistantMessage,
    Message,
    MessageContent,
    Messages,
    Response,
    State,
    SystemMessage,
    ToolCall,
    Tool,
    ToolMessage,
    TrajectoryStep,
    UserMessage,
)
from verifiers.utils.async_utils import maybe_await, maybe_call_with_named_args
from verifiers.utils.error_utils import error_info
from verifiers.utils.message_utils import concat_messages, normalize_messages
from verifiers.utils.response_utils import parse_response_message, parse_response_tokens
from verifiers.utils.tool_utils import is_valid_tool_content_parts
from verifiers.utils.usage_utils import extract_usage_tokens

from .channels import (
    Channel,
    ChannelMap,
    ToolArgumentError,
    User,
)
from .configs import RunConfig
from .task import Task, task_to_input

if TYPE_CHECKING:
    from .resources import Resources

RubricSource = Rubric | Callable[[], Rubric] | None
ToolsSource = object | Callable[[], object] | None


@dataclass(frozen=True)
class ModelRequest:
    prompt: Messages
    tool_defs: list[Tool] | None = None
    extras: dict[str, object] | None = None
    context: dict[str, object] = field(default_factory=dict)


class Harness:
    """Base harness contract with shared rollout and tool-call primitives."""

    def __init__(
        self,
        channels: ChannelMap | None = None,
        run: RunConfig | dict[str, object] | None = None,
        rubric: RubricSource = None,
        system_prompt: str | None = None,
        tools: ToolsSource = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.resources: Resources | None = None
        self._channels = dict(channels or {})
        self._rubric_source = rubric
        self._rubric: Rubric | None = None
        self._rubric_loaded = False
        self.system_prompt = system_prompt
        self._tools_source = tools
        self._tools: object | None = None
        self._tools_loaded = False
        self.run_config = RunConfig.model_validate(run or {})
        self.max_turns = self.run_config.max_turns
        self.parallel_model_requests = self.run_config.parallel_model_requests
        self.error_formatter = self.run_config.error_formatter
        self.stop_errors = list(self.run_config.stop_errors)
        self.max_tool_calls_per_turn = self.run_config.max_tool_calls_per_turn
        self.tool_call_limit_message = self.run_config.tool_call_limit_message
        self._stop_conditions = discover_decorated(self, "stop")
        self._render_handlers = discover_decorated(self, "render")
        self._metric_handlers = discover_decorated(self, "metric")
        self._reward_handlers = discover_decorated(self, "reward")
        self._advantage_handlers = discover_decorated(self, "advantage")
        self._cleanup_handlers = discover_decorated(self, "cleanup")
        self._teardown_handlers = discover_decorated(self, "teardown")

    @property
    def rubric(self) -> Rubric | None:
        return self.get_rubric()

    @property
    def tools(self) -> object | None:
        return self.get_tools()

    def get_rubric(self) -> Rubric | None:
        if not self._rubric_loaded:
            source = self._rubric_source
            self._rubric = source() if callable(source) else source
            self._rubric_loaded = True
        return self._rubric

    def get_tools(self) -> object | None:
        if not self._tools_loaded:
            source = self._tools_source
            self._tools = source() if callable(source) else source
            self._tools_loaded = True
        return self._tools

    def channels(self, task: Task | None = None) -> ChannelMap:
        channels: dict[str, object] = dict(self._channels)
        if self.system_prompt:
            add_channel_shorthand(channels, "system_prompt", self.system_prompt)
        tools = self.get_tools()
        if tools is not None:
            add_channel_shorthand(channels, "tools", tools)
        rubrics = []
        rubric = self.get_rubric()
        if rubric is not None:
            rubrics.append(rubric)
        if rubrics:
            add_channel_shorthand(
                channels, "rubric", rubrics[0] if len(rubrics) == 1 else rubrics
            )
        if self._stop_conditions:
            add_lifecycle_channel(channels, "stop", self._stop_conditions)
        if self._render_handlers:
            add_lifecycle_channel(channels, "render", self._render_handlers)
        if self._metric_handlers:
            add_lifecycle_channel(channels, "metrics", self._metric_handlers)
        if self._reward_handlers:
            add_lifecycle_channel(channels, "rewards", self._reward_handlers)
        if self._advantage_handlers:
            add_lifecycle_channel(channels, "advantage", self._advantage_handlers)
        if self._cleanup_handlers:
            add_lifecycle_channel(channels, "cleanup", self._cleanup_handlers)
        if self._teardown_handlers:
            add_lifecycle_channel(channels, "teardown", self._teardown_handlers)
        return channels

    def channel_objects(self) -> dict[str, object]:
        return {}

    def channel_definitions(self) -> dict[str, Channel]:
        return {}

    @stop(priority=100)
    async def has_error(self, task: Task, state: State, resources: Resources) -> bool:
        return state.get("error") is not None

    @stop(priority=90)
    async def prompt_too_long(
        self, task: Task, state: State, resources: Resources
    ) -> bool:
        return bool(state.get("prompt_too_long"))

    @stop(priority=80)
    async def done(self, task: Task, state: State, resources: Resources) -> bool:
        return bool(state["done"])

    @stop
    async def max_turns_reached(
        self, task: Task, state: State, resources: Resources
    ) -> bool:
        requests = state.get("num_model_requests", len(state["trajectory"]))
        if self.max_turns <= 0 or requests < self.max_turns:
            return False
        if requests > len(state["trajectory"]):
            return True
        return state.get("final_env_response") is not None

    @stop(priority=-10)
    async def no_tools_or_user(
        self, task: Task, state: State, resources: Resources
    ) -> bool:
        if not state["trajectory"]:
            return False
        completion = state["trajectory"][-1]["completion"]
        if not completion or not isinstance(completion[-1], AssistantMessage):
            return False
        if completion[-1].tool_calls:
            return False
        if resources.get("user") is None:
            return True
        return bool(state.get("user_done"))

    @final
    async def is_completed(
        self, task: Task, state: State, resources: Resources
    ) -> bool:
        conditions = resources.current_handlers("stop")
        for condition in conditions:
            if await maybe_call_with_named_args(
                condition,
                task=task,
                state=state,
                harness=self,
                resources=resources,
            ):
                state["is_completed"] = True
                state["is_truncated"] = state.get("is_truncated", False) or any(
                    step.get("is_truncated", False)
                    for step in state.get("trajectory", [])
                )
                state["stop_condition"] = str(
                    getattr(condition, "__name__", condition.__class__.__name__)
                )
                return True
        return False

    async def setup_state(self, task: Task, resources: Resources) -> State:
        state = State(input=task_to_input(task))
        state["task"] = dict(task)
        state["prompt"] = normalize_messages(task.prompt, field_name="task.prompt")
        state["done"] = False
        state["is_completed"] = False
        state["is_truncated"] = False
        state["stop_condition"] = None
        state["trajectory"] = []
        state["num_model_requests"] = 0
        state["completion"] = None
        state["trajectory_id"] = uuid.uuid4().hex
        state["reward"] = None
        state["metrics"] = None
        state["error"] = None
        state["final_env_response"] = None
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": time.time(),
        }
        system_prompt = getattr(resources, "system_prompt", "")
        if system_prompt:
            state["prompt"] = [
                SystemMessage(content=system_prompt),
                *state["prompt"],
            ]
        return state

    async def get_prompt_messages(
        self, task: Task, state: State, resources: Resources
    ) -> Messages:
        if not state["trajectory"]:
            return state["prompt"]
        last = state["trajectory"][-1]
        messages = concat_messages([last["prompt"], last["completion"]])
        env_messages = await self.get_env_messages(task, state, resources)
        if env_messages:
            state["final_env_response"] = env_messages
        else:
            state.pop("final_env_response", None)
        return concat_messages([messages, env_messages])

    async def get_env_messages(
        self, task: Task, state: State, resources: Resources
    ) -> Messages:
        tool_calls = self.get_tool_calls(state)
        if tool_calls:
            if (
                self.max_tool_calls_per_turn is not None
                and len(tool_calls) > self.max_tool_calls_per_turn
            ):
                return [
                    UserMessage(
                        content=self.tool_call_limit_message.format(
                            limit=self.max_tool_calls_per_turn,
                            count=len(tool_calls),
                        )
                    )
                ]
            outputs = []
            for tool_call in tool_calls:
                outputs.append(await self.call_tool(tool_call, task, state, resources))
            return outputs
        if not state["trajectory"]:
            return []
        completion = state["trajectory"][-1]["completion"]
        if not completion or not isinstance(completion[-1], AssistantMessage):
            return []
        messages = await self.get_user_messages(task, state, resources)
        if not messages:
            state["user_done"] = True
            return []
        state.pop("user_done", None)
        return messages

    async def get_user_messages(
        self, task: Task, state: State, resources: Resources
    ) -> Messages:
        user = resources.get("user")
        if user is None:
            return []
        if not isinstance(user, User):
            raise TypeError("Resolved resource 'user' must implement User.")
        messages = await maybe_await(user.respond, task, state, resources)
        if messages is None:
            return []
        assert isinstance(messages, list), "user.respond must return vf.Messages."
        assert all(isinstance(message, Message) for message in messages), (
            "user.respond must return vf.Messages."
        )
        return cast(Messages, messages)

    def get_tool_calls(self, state: State) -> list[ToolCall]:
        if not state["trajectory"]:
            return []
        completion = state["trajectory"][-1]["completion"]
        if not completion:
            return []
        message = completion[-1]
        if not isinstance(message, AssistantMessage):
            return []
        return list(message.tool_calls or [])

    async def get_model_request(
        self, task: Task, state: State, resources: Resources
    ) -> ModelRequest | None:
        prompt = await self.get_prompt_messages(task, state, resources)
        assert isinstance(prompt, list), "get_prompt_messages must return vf.Messages."
        assert all(isinstance(message, Message) for message in prompt), (
            "get_prompt_messages must return vf.Messages."
        )
        if await self.is_completed(task, state, resources):
            return None
        state.pop("final_env_response", None)
        return ModelRequest(prompt=prompt)

    @final
    async def get_model_response(
        self,
        prompt: Messages,
        task: Task,
        state: State,
        resources: Resources,
        tool_defs: list[Tool] | None = None,
        context: dict[str, object] | None = None,
    ) -> Response:
        if tool_defs is None:
            tool_defs = resources.tools.defs()
        sampling = dict(resources.sampling_args)
        response = await resources.client.get_response(
            prompt=prompt,
            model=resources.model,
            tools=tool_defs,
            sampling_args=sampling,
            state=state,
        )
        input_tokens, output_tokens = extract_usage_tokens(response)
        if input_tokens or output_tokens:
            usage = dict(state.get("usage") or {})
            usage["input_tokens"] = float(usage.get("input_tokens", 0.0))
            usage["output_tokens"] = float(usage.get("output_tokens", 0.0))
            usage["input_tokens"] += float(input_tokens)
            usage["output_tokens"] += float(output_tokens)
            state["usage"] = usage
        return response

    async def add_model_response(
        self,
        prompt: Messages,
        response: Response,
        task: Task,
        state: State,
        resources: Resources,
        extras: dict[str, object] | None = None,
    ) -> None:
        completion = await parse_response_message(response)
        tokens = await parse_response_tokens(response)
        response_is_truncated = response.message.is_truncated or False
        is_truncated = response_is_truncated or (
            tokens is not None and bool(tokens.get("is_truncated"))
        )
        async with resources.trajectory_lock():
            state["trajectory"].append(
                TrajectoryStep(
                    prompt=prompt,
                    completion=completion,
                    response=response,
                    tokens=tokens,
                    reward=None,
                    advantage=None,
                    is_truncated=is_truncated,
                    trajectory_id=state["trajectory_id"],
                    extras=extras or {},
                )
            )

    async def submit_model_request(
        self,
        prompt: Messages,
        task: Task,
        state: State,
        resources: Resources,
        tool_defs: list[Tool] | None = None,
        extras: dict[str, object] | None = None,
        context: dict[str, object] | None = None,
    ) -> Response:
        response = await self.get_model_response(
            prompt,
            task,
            state,
            resources,
            tool_defs=tool_defs,
            context=context,
        )
        await self.add_model_response(
            prompt,
            response,
            task,
            state,
            resources,
            extras=extras,
        )
        return response

    def should_stop_for_tool_error(self, err: Exception) -> bool:
        return any(isinstance(err, err_type) for err_type in self.stop_errors)

    async def call_tool(
        self, tool_call: ToolCall, task: Task, state: State, resources: Resources
    ) -> ToolMessage:
        try:
            result = await resources.tools.call(
                tool_call.name,
                resources,
                tool_call.arguments,
                task=task,
                state=state,
            )
        except ToolArgumentError as e:
            if self.should_stop_for_tool_error(e):
                raise ToolParseError from e
            return ToolMessage(
                role="tool",
                tool_call_id=tool_call.id,
                content=self.error_formatter(e),
            )
        except Exception as e:
            if self.should_stop_for_tool_error(e):
                raise ToolCallError from e
            return ToolMessage(
                role="tool",
                tool_call_id=tool_call.id,
                content=self.error_formatter(e),
            )
        content = (
            cast(MessageContent, result)
            if is_valid_tool_content_parts(result)
            else str(result)
        )
        return ToolMessage(
            role="tool",
            tool_call_id=tool_call.id,
            content=content,
        )

    async def render_timing(
        self, task: Task, state: State, resources: Resources
    ) -> None:
        start = state["timing"]["start_time"]
        now = time.time()
        state["timing"]["generation_ms"] = (now - start) * 1000
        state["timing"]["total_ms"] = (now - start) * 1000

    async def render_completion(
        self, task: Task, state: State, resources: Resources
    ) -> None:
        if not state["trajectory"]:
            state["completion"] = []
            return
        first_prompt = state["prompt"]
        last = state["trajectory"][-1]
        full = concat_messages([last["prompt"], last["completion"]])
        if state.get("final_env_response"):
            final_env_response = state["final_env_response"]
            assert isinstance(final_env_response, list), (
                "final_env_response must be vf.Messages."
            )
            assert all(
                isinstance(message, Message) for message in final_env_response
            ), "final_env_response must be vf.Messages."
            full = concat_messages([full, cast(Messages, final_env_response)])
        state["completion"] = full[len(first_prompt) :]

    @render(priority=100)
    async def render_state(
        self, task: Task, state: State, resources: Resources
    ) -> None:
        await self.render_timing(task, state, resources)
        await self.render_completion(task, state, resources)

    def pending_model_requests(
        self, resources: Resources
    ) -> set[asyncio.Task[Response | None]]:
        pending = resources.runtime.setdefault("model_request_tasks", set())
        if not isinstance(pending, set):
            raise RuntimeError(
                "resources.runtime['model_request_tasks'] must be a set."
            )
        return cast(set[asyncio.Task[Response | None]], pending)

    def schedule_model_request(
        self,
        task: Task,
        request: ModelRequest,
        state: State,
        resources: Resources,
    ) -> asyncio.Task[Response | None]:
        state["num_model_requests"] = int(state.get("num_model_requests", 0)) + 1
        model_task = asyncio.create_task(
            self._run_model_request(task, request, state, resources)
        )
        self.pending_model_requests(resources).add(model_task)
        return model_task

    async def _run_model_request(
        self,
        task: Task,
        request: ModelRequest,
        state: State,
        resources: Resources,
    ) -> Response | None:
        try:
            return await self.submit_model_request(
                request.prompt,
                task,
                state,
                resources,
                tool_defs=request.tool_defs,
                extras=request.extras,
                context=request.context,
            )
        except OverlongPromptError:
            state["prompt_too_long"] = True
            state["is_truncated"] = True
        except Error as e:
            state["error"] = error_info(e)
        except Exception as e:
            state["error"] = error_info(e)
        finally:
            current_task = asyncio.current_task()
            if current_task is not None:
                self.pending_model_requests(resources).discard(
                    cast(asyncio.Task[Response | None], current_task)
                )
        return None

    async def wait_for_model_request(
        self,
        request_task: asyncio.Task[Response | None],
        state: State,
        resources: Resources,
    ) -> None:
        await request_task

    async def wait_for_pending_model_requests(
        self, state: State, resources: Resources
    ) -> None:
        while pending := set(self.pending_model_requests(resources)):
            await asyncio.gather(*pending, return_exceptions=True)

    async def cancel_pending_model_requests(
        self, state: State, resources: Resources
    ) -> None:
        pending = set(self.pending_model_requests(resources))
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    @final
    async def run(self, task: Task, resources: Resources) -> State:
        state = await self.setup_state(task, resources)
        cleanup_started = False
        try:
            while not await self.is_completed(task, state, resources):
                try:
                    request = await self.get_model_request(task, state, resources)
                    if request is None:
                        if not await self.is_completed(task, state, resources):
                            raise RuntimeError(
                                "get_model_request returned None without "
                                "a stop condition."
                            )
                        continue
                    model_task = self.schedule_model_request(
                        task, request, state, resources
                    )
                    if not self.parallel_model_requests:
                        await self.wait_for_model_request(model_task, state, resources)
                except OverlongPromptError:
                    state["prompt_too_long"] = True
                    state["is_truncated"] = True
                except Error as e:
                    state["error"] = error_info(e)
            await self.wait_for_pending_model_requests(state, resources)
            cleanup_started = True
            await resources.lifecycle.render_rollout(task, state, resources)
            if resources.score_rollout_enabled:
                await resources.scoring.rollout(task, state, resources)
            await resources.lifecycle.cleanup_rollout(task, state, resources)
            return state
        except asyncio.CancelledError:
            await self.cancel_pending_model_requests(state, resources)
            raise
        finally:
            if not cleanup_started:
                await resources.lifecycle.cleanup_rollout(task, state, resources)


def add_channel_shorthand(
    channels: dict[str, object], name: str, config: object
) -> None:
    if name in channels:
        if name == "rubric":
            channels[name] = [*as_config_list(channels[name]), config]
            return
        raise ValueError(
            f"Harness received {name!r} both in channels and as a shorthand arg."
        )
    channels[name] = config


def add_lifecycle_channel(
    channels: dict[str, object], name: str, handlers: list[object]
) -> None:
    existing = channels.get(name)
    if existing is None:
        channels[name] = list(handlers)
        return
    if name == "cleanup" and isinstance(existing, dict):
        cleanup = dict(existing)
        cleanup["harness"] = [*as_config_list(cleanup.get("harness")), *handlers]
        channels[name] = cleanup
        return
    channels[name] = [*as_config_list(existing), *handlers]


def as_config_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]
