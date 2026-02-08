from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import RolloutInput, SamplingArgs, State


class TaskProtocol(Protocol):
    task_id: str


class TasksetProtocol(Protocol):
    taskset_id: str

    def get_task(self, task_id: str) -> Any: ...


class RunContextProtocol(Protocol):
    model: str
    sampling_args: dict[str, Any]
    metadata: dict[str, Any]


class TraceEventProtocol(Protocol):
    event_type: str
    task_id: str
    payload: dict[str, Any]


TraceCallback = Callable[[TraceEventProtocol], Awaitable[None] | None]


class HarnessResultProtocol(Protocol):
    task_id: str
    output: Any
    messages: list[dict[str, Any]]
    metrics: dict[str, float]
    usage: dict[str, int | float]
    events: list[TraceEventProtocol]
    error: str | None


class HarnessProtocol(Protocol):
    async def run(
        self,
        task: Any,
        context: RunContextProtocol | None = None,
        on_event: TraceCallback | None = None,
    ) -> HarnessResultProtocol: ...

    def supports(self, taskset: TasksetProtocol) -> bool: ...

    def compile_rubric_spec(self, taskset: TasksetProtocol) -> dict[str, Any]: ...


@dataclass(slots=True)
class AgentRunContext:
    """Minimal run context for harness execution.

    Model/sampling live here (agent layer), not in Task.
    """

    model: str
    sampling_args: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(slots=True)
class AgentEnvConfig:
    """Minimal config sketch for AgentEnv."""

    taskset: TasksetProtocol
    harness: HarnessProtocol


class AgentEnv(vf.Environment):
    """Experimental environment that composes a taskset and harness.

    Contract sketch:
    - Main rubric is compiled from taskset + harness.
    - Users can add additional rubrics via add_rubric(...).
    - Rollout delegates execution to harness.run(task, context, on_event).
    """

    def __init__(
        self,
        taskset: TasksetProtocol,
        harness: HarnessProtocol,
        rubric: vf.Rubric | None = None,
        trace_callback: TraceCallback | None = None,
        **kwargs,
    ):
        self.taskset = taskset
        self.harness = harness
        self.trace_callback = trace_callback
        if not harness.supports(taskset):
            raise ValueError(
                f"Harness '{harness.__class__.__name__}' does not support taskset '{taskset.taskset_id}'"
            )

        compiled_rubric = rubric or self.compile_main_rubric(taskset, harness)
        super().__init__(rubric=compiled_rubric, **kwargs)

    def compile_main_rubric(
        self,
        taskset: TasksetProtocol,
        harness: HarnessProtocol,
    ) -> vf.Rubric:
        """Compile the base rubric from taskset+harness contracts.

        This is intentionally minimal; concrete grading behavior will evolve.
        """

        spec = harness.compile_rubric_spec(taskset)

        async def has_task_id(state: State) -> float:
            task = state.get("task")
            task_id = getattr(task, "task_id", None)
            return 1.0 if isinstance(task_id, str) and len(task_id) > 0 else 0.0

        async def harness_error_free(state: State) -> float:
            return 1.0 if state.get("harness_error") in (None, "") else 0.0

        base_rubric = vf.Rubric()
        base_rubric.add_metric(has_task_id)
        base_rubric.add_metric(harness_error_free)
        base_rubric.add_class_object("compiled_rubric_spec", spec)
        return base_rubric

    def resolve_task_id(self, state: State) -> str:
        task_obj = state.get("task")
        if task_obj is not None and hasattr(task_obj, "task_id"):
            return str(getattr(task_obj, "task_id"))

        info = state.get("info")
        if isinstance(info, dict) and "task_id" in info:
            return str(info["task_id"])

        raise ValueError(
            "AgentEnv requires task_id in state['task'].task_id or state['info']['task_id']"
        )

    def resolve_task(self, state: State) -> Any:
        task_id = self.resolve_task_id(state)
        return self.taskset.get_task(task_id)

    def make_run_context(
        self,
        model: str,
        sampling_args: SamplingArgs | None,
        state: State,
    ) -> AgentRunContext:
        normalized_sampling_args: dict[str, Any] = {}
        if sampling_args is not None:
            normalized_sampling_args = dict(sampling_args)
        return AgentRunContext(
            model=model,
            sampling_args=normalized_sampling_args,
            metadata={"trajectory_id": state.get("trajectory_id", "")},
        )

    def apply_harness_result(self, state: State, result: HarnessResultProtocol) -> None:
        state["harness_output"] = result.output
        state["harness_metrics"] = result.metrics
        state["harness_usage"] = result.usage
        state["harness_events"] = [
            {
                "event_type": event.event_type,
                "task_id": event.task_id,
                "payload": event.payload,
            }
            for event in result.events
        ]
        state["harness_error"] = result.error
        state["completion"] = result.messages
        if result.error:
            state["error"] = vf.GenerationError(result.error)

    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        task = self.resolve_task(state)
        stream_events: list[dict[str, Any]] = []

        async def on_event(event: TraceEventProtocol) -> None:
            event_record = {
                "event_type": event.event_type,
                "task_id": event.task_id,
                "payload": event.payload,
            }
            stream_events.append(event_record)
            if self.trace_callback is not None:
                maybe_awaitable = self.trace_callback(event)
                if maybe_awaitable is not None:
                    await maybe_awaitable

        run_context = self.make_run_context(model=model, sampling_args=sampling_args, state=state)
        result = await self.harness.run(task, context=run_context, on_event=on_event)
        self.apply_harness_result(state, result)
        if len(stream_events) > 0:
            state["harness_stream_events"] = stream_events
        return state
