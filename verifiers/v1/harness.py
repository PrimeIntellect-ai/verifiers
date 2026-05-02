from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, cast

from openai import AsyncOpenAI

from verifiers.errors import Error, OverlongPromptError
from verifiers.utils.error_utils import error_info
from verifiers.utils.message_utils import normalize_messages

from .config import HarnessConfig, merge_config_value
from .endpoint_utils import (
    Endpoint,
    append_openai_message,
    assistant_completion_from_messages,
    openai_tool_calls,
    run_intercepted_program,
)
from .program_utils import run_local_command
from .runtime import Runtime, runtime_context
from .scoring import import_ref
from .sandbox_utils import release_group_sandboxes, run_sandbox_command
from .state import State
from .task import Task
from .toolset import Toolset
from .utils.tool_utils import load_tools_from_state


class Harness:
    def __init__(
        self,
        program: Callable[..., object] | Mapping[str, object] | None = None,
        toolsets: list[Toolset] | None = None,
        sandbox: Mapping[str, object] | None = None,
        metrics: list[Callable[..., object]] | None = None,
        rewards: list[Callable[..., object]] | None = None,
        cleanup: list[Callable[..., object]] | None = None,
        config: HarnessConfig | Mapping[str, object] | None = None,
    ):
        self.config = HarnessConfig.model_validate(config or {})
        program_value = merge_config_value(program, self.config.program)
        self.program = cast(
            Callable[..., object] | Mapping[str, object] | None, program_value
        )
        self.toolsets = list(toolsets or [])
        self.sandbox = merge_config_value(sandbox, self.config.sandbox)
        self.metrics = list(metrics or [])
        self.rewards = list(rewards or [])
        self.cleanup = list(cleanup or [])
        self.taskset: object | None = None
        self.runtime = self.resolve_runtime()
        self.endpoint = Endpoint(use_tunnel=self.sandbox is not None)
        self._program = self.compile_program(self.program)

    def add_metric(self, fn: Callable[..., object]) -> None:
        self.metrics.append(fn)
        self.runtime = self.resolve_runtime()

    def add_reward(self, fn: Callable[..., object]) -> None:
        self.rewards.append(fn)
        self.runtime = self.resolve_runtime()

    def add_toolset(self, toolset: Toolset) -> None:
        self.toolsets.append(toolset)
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
        rendered = False
        completed = False
        with runtime_context(self.runtime):
            try:
                try:
                    state = await self.setup_state(task, state)
                    state = await self.run_program(task, state)
                    await self.runtime.collect_artifacts(task, state)
                except Error as e:
                    self.render_error(state, e)
                self.render_generation_timing(state)
                rendered = True
                await self.runtime.score_rollout(task, state)
                state["is_completed"] = True
                completed = True
            finally:
                if not rendered:
                    self.render_generation_timing(state)
                await self.runtime.cleanup_rollout(task, state)
                if completed:
                    state.assert_serializable()
        return state

    def render_error(self, state: State, error: Error) -> None:
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

    async def teardown(self) -> None:
        await self.runtime.teardown()
        await self.endpoint.teardown()

    async def init_state(self, task: Task) -> State:
        return State.for_task(task)

    async def setup_state(self, task: Task, state: State) -> State:
        state.setdefault("runtime", {})
        state["runtime"] = {**task.get("runtime", {}), **state["runtime"]}
        self.runtime.prepare_state(task, state)
        await self.runtime.ensure_global_tool_sandboxes()
        self.runtime.bind_global_tool_sandboxes(state)
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

    def render_generation_timing(self, state: State) -> None:
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
            return import_ref(entrypoint)
        if "command" in program:
            return self.command_program(cast(Mapping[str, object], program))
        raise ValueError("program mapping requires entrypoint or command.")

    async def base_program(
        self, task: Task, state: State, client: AsyncOpenAI
    ) -> State:
        prompt = normalize_messages(state.get("prompt", []), field_name="state.prompt")
        messages = [message.model_dump(exclude_none=True) for message in prompt]
        while True:
            response = await client.chat.completions.create(
                model=self.runtime.model(state),
                messages=messages,
                tools=self.runtime.tool_defs(),
                **self.runtime.sampling_args(state),
            )
            append_openai_message(messages, response)
            tool_calls = openai_tool_calls(response)
            if not tool_calls:
                state["completion"] = assistant_completion_from_messages(
                    [message.model_dump(exclude_none=True) for message in prompt],
                    messages,
                )
                return state
            callable_tools = load_tools_from_state(state)
            for tool_call in tool_calls:
                name = tool_call.function.name
                result = await callable_tools[name](
                    **json_args(tool_call.function.arguments)
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )
        return state

    def command_program(self, program: Mapping[str, object]) -> Callable[..., object]:
        async def run(task: Task, state: State) -> State:
            runtime = self.runtime
            if self.sandbox is not None:
                if not isinstance(self.sandbox, Mapping):
                    raise TypeError("sandbox must be a mapping.")
                return await run_sandbox_command(
                    program,
                    cast(Mapping[str, object], self.sandbox),
                    task,
                    state,
                    runtime,
                )
            return await run_local_command(program, task, state, runtime)

        return run


def json_args(value: str) -> dict[str, object]:
    import json

    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Tool call arguments must decode to a JSON object.")
    return cast(dict[str, object], parsed)
