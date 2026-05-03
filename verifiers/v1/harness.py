from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, cast

from verifiers.errors import Error, OverlongPromptError
from verifiers.utils.error_utils import error_info
from verifiers.utils.message_utils import normalize_messages
from verifiers.types import Messages, ToolMessage

from .config import (
    HarnessConfig,
    import_config_ref,
    merge_config_items,
    merge_config_value,
    resolve_config_object,
)
from .utils.endpoint_utils import (
    Endpoint,
    assistant_completion_from_messages,
    run_intercepted_program,
)
from .utils.json_utils import json_args
from .utils.program_utils import run_local_command
from .runtime import Runtime, runtime_context
from .utils.sandbox_utils import (
    release_group_sandboxes,
    release_rollout_sandboxes,
    run_sandbox_command,
)
from .utils.trajectory_utils import sync_trajectory
from .state import State
from .task import Task
from .toolset import normalize_toolsets
from .user import normalize_user
from .utils.tool_utils import load_tools_from_state


class Harness:
    config_type: ClassVar[type[HarnessConfig]] = HarnessConfig

    def __init__(
        self,
        program: Callable[..., object] | Mapping[str, object] | None = None,
        toolsets: list[object] | None = None,
        user: object | None = None,
        sandbox: Mapping[str, object] | None = None,
        metrics: list[Callable[..., object]] | None = None,
        rewards: list[Callable[..., object]] | None = None,
        cleanup: list[Callable[..., object]] | None = None,
        max_turns: int | None = None,
        config: HarnessConfig | Mapping[str, object] | None = None,
    ):
        self.config = type(self).config_type.model_validate(config or {})
        if max_turns is not None:
            self.config.max_turns = max_turns
        program_value = resolve_config_object(
            merge_config_value(program, self.config.program)
        )
        self.program = cast(
            Callable[..., object] | Mapping[str, object] | None, program_value
        )
        self.toolsets = normalize_toolsets(
            merge_config_items(toolsets or (), self.config.toolsets)
        )
        self.user = normalize_user(merge_config_value(user, self.config.user))
        self.sandbox = merge_config_value(sandbox, self.config.sandbox)
        self.metrics = cast(
            list[Callable[..., object]],
            merge_config_items(metrics or (), self.config.metrics),
        )
        self.rewards = cast(
            list[Callable[..., object]],
            merge_config_items(rewards or (), self.config.rewards),
        )
        self.cleanup = cast(
            list[Callable[..., object]],
            merge_config_items(cleanup or (), self.config.cleanup),
        )
        self.taskset: object | None = None
        self.runtime = self.resolve_runtime()
        self.endpoint = Endpoint(use_tunnel=self.sandbox is not None)
        self._program = self.compile_program(self.program)

    @classmethod
    def config_schema(cls) -> str:
        return cls.config_type.schema_text()

    def add_metric(self, fn: Callable[..., object]) -> None:
        self.metrics.append(fn)
        self.runtime = self.resolve_runtime()

    def add_reward(self, fn: Callable[..., object]) -> None:
        self.rewards.append(fn)
        self.runtime = self.resolve_runtime()

    def add_toolset(self, toolset: object) -> None:
        self.toolsets.extend(normalize_toolsets([toolset]))
        self.runtime = self.resolve_runtime()

    def add_cleanup(self, fn: Callable[..., object]) -> None:
        self.cleanup.append(fn)
        self.runtime = self.resolve_runtime()

    def attach_taskset(self, taskset: object) -> None:
        self.taskset = taskset
        self.runtime = self.resolve_runtime()

    def resolve_runtime(self) -> Runtime:
        return Runtime(taskset=self.taskset, harness=self)

    async def run(
        self, task: Task | Mapping[str, Any], state: State | None = None
    ) -> State:
        task = task if isinstance(task, Task) else Task(task).freeze()
        state = await self.init_state(task) if state is None else state
        timing_recorded = False
        completed = False
        with runtime_context(self.runtime):
            try:
                try:
                    state = await self.setup_state(task, state)
                    state = await self.run_program(task, state)
                    state["stop_condition"] = (
                        state.get("stop_condition") or "program_completed"
                    )
                    await self.runtime.collect_artifacts(task, state)
                except Error as e:
                    self.record_error(state, e)
                sync_trajectory(state)
                self.record_generation_timing(state)
                timing_recorded = True
                if state.get("runtime", {}).get("score_rollout", True):
                    await self.runtime.score_rollout(task, state)
                state["is_completed"] = True
                completed = True
            finally:
                if not timing_recorded:
                    self.record_generation_timing(state)
                await self.runtime.cleanup_rollout(task, state)
                await release_rollout_sandboxes([state])
                if not self.has_group_boundary(state):
                    await self.runtime.cleanup_group([task], [state])
                    await release_group_sandboxes([state])
                if completed:
                    state.assert_serializable()
        return state

    def record_error(self, state: State, error: Error) -> None:
        if isinstance(error, OverlongPromptError):
            state["prompt_too_long"] = True
            state["is_truncated"] = True
            state["stop_condition"] = "prompt_too_long"
            return
        state["error"] = error_info(error)
        state["stop_condition"] = "has_error"

    async def score_group(self, tasks: list[Task], states: list[State]) -> list[State]:
        return await self.runtime.score_group(tasks, states)

    async def cleanup_group(self, tasks: list[Task], states: list[State]) -> None:
        await self.runtime.cleanup_group(tasks, states)
        await release_group_sandboxes(states)

    def has_group_boundary(self, state: State) -> bool:
        runtime = state.get("runtime", {})
        return isinstance(runtime, Mapping) and "group_key" in runtime

    async def teardown(self) -> None:
        await self.runtime.teardown()
        await self.endpoint.teardown()

    async def init_state(self, task: Task) -> State:
        return State.for_task(task)

    async def setup_state(self, task: Task, state: State) -> State:
        state.setdefault("runtime", {})
        state["runtime"] = {**task.get("runtime", {}), **state["runtime"]}
        await self.runtime.ensure_mcp_tools(state)
        self.runtime.prepare_state(task, state)
        await self.runtime.ensure_global_sandboxes()
        self.runtime.bind_global_sandboxes(state)
        state.setdefault("trajectory", [])
        state.setdefault("artifacts", {})
        state.setdefault("metrics", {})
        state.setdefault("reward", 0.0)
        state.setdefault(
            "timing",
            {
                "generation_ms": 0.0,
                "scoring_ms": 0.0,
                "total_ms": 0.0,
                "start_time": time.time(),
            },
        )
        return state

    def record_generation_timing(self, state: State) -> None:
        timing = state["timing"]
        elapsed_ms = (time.time() - timing["start_time"]) * 1000
        timing["generation_ms"] = elapsed_ms
        timing["total_ms"] = elapsed_ms

    async def run_program(self, task: Task, state: State) -> State:
        result = await run_intercepted_program(
            self._program, self.endpoint, self.runtime, task, state
        )
        if result is None:
            return state
        if isinstance(result, State):
            return result
        if isinstance(result, Mapping):
            state.update(result)
            return state
        raise TypeError("Harness program must return None, State, or a mapping.")

    def compile_program(
        self, program: Callable[..., object] | Mapping[str, object] | None
    ) -> Callable[..., object]:
        if program is None:
            return self.base_program
        if callable(program):
            return program
        if not isinstance(program, Mapping):
            raise TypeError("program must be None, callable, or a mapping.")
        if "entrypoint" in program:
            entrypoint = program["entrypoint"]
            if not isinstance(entrypoint, str):
                raise TypeError("program.entrypoint must be a string ref.")
            fn = import_config_ref(entrypoint)
            if not callable(fn):
                raise TypeError("program.entrypoint did not resolve to a callable.")
            return fn
        if "command" in program:
            return self.command_program(cast(Mapping[str, object], program))
        raise ValueError("program mapping requires entrypoint or command.")

    async def base_program(self, task: Task, state: State) -> State:
        prompt = normalize_messages(state.get("prompt", []), field_name="state.prompt")
        prompt_messages = [message.model_dump(exclude_none=True) for message in prompt]
        messages = list(prompt)
        turn = 0
        while self.config.max_turns <= 0 or turn < self.config.max_turns:
            response = await self.runtime.submit_model_request(
                messages,
                task,
                state,
                tool_defs=self.runtime.tool_defs(state),
            )
            turn += 1
            messages.append(response.message)
            rendered_messages = [
                message.model_dump(exclude_none=True) for message in messages
            ]
            state["completion"] = assistant_completion_from_messages(
                prompt_messages, rendered_messages
            )
            tool_calls = list(response.message.tool_calls or [])
            if not tool_calls:
                transcript = [
                    message.model_dump(exclude_none=True) for message in messages
                ]
                user_messages = await self.runtime.user_messages(
                    task, state, transcript=transcript
                )
                if user_messages:
                    messages.extend(
                        normalize_messages(
                            cast(Messages, user_messages),
                            field_name="user_messages",
                        )
                    )
                    rendered_messages = [
                        message.model_dump(exclude_none=True) for message in messages
                    ]
                    state["completion"] = assistant_completion_from_messages(
                        prompt_messages, rendered_messages
                    )
                    continue
                state["stop_condition"] = state.get("stop_condition") or "no_tools"
                return state
            callable_tools = load_tools_from_state(state)
            for tool_call in tool_calls:
                name = tool_call.name
                result = await callable_tools[name](**json_args(tool_call.arguments))
                messages.append(
                    ToolMessage(tool_call_id=tool_call.id, content=str(result))
                )
                rendered_messages = [
                    message.model_dump(exclude_none=True) for message in messages
                ]
                state["completion"] = assistant_completion_from_messages(
                    prompt_messages, rendered_messages
                )
            if self.config.max_turns > 0 and turn >= self.config.max_turns:
                state["stop_condition"] = "max_turns_reached"
                return state
        return state

    def command_program(self, program: Mapping[str, object]) -> Callable[..., object]:
        async def run(task: Task, state: State) -> State:
            runtime = self.runtime
            if self.sandbox is not None:
                if not isinstance(self.sandbox, Mapping):
                    raise TypeError("sandbox must be a mapping.")
                sandbox_config = dict(self.sandbox)
                task_sandbox = task.get("sandbox")
                if isinstance(task_sandbox, Mapping):
                    sandbox_config.update(task_sandbox)
                return await run_sandbox_command(
                    program,
                    sandbox_config,
                    task,
                    state,
                    runtime,
                )
            return await run_local_command(program, task, state, runtime)

        return run
