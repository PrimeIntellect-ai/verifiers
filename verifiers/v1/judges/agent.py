"""A task-authored request for the environment's configured judge agent."""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace


@dataclass
class AgentJudgeRequest:
    task: Task
    model: str | None = None
    sampling: SamplingConfig | None = None
    isolated: bool = False
    """Run in a fresh native runtime; task.setup restores its captured inputs."""
    network: NetworkPolicyConfig | None = None
    """Effective execution policy for a fresh workspace; None keeps its config."""
    on_trace: Callable[[Trace], None] | None = None
    """Observe every native attempt, including attempts replaced by retries."""


@contextmanager
def collect_judge_traces(task: Task):
    """Collect native attempts during standalone regrading, without copying usage.

    The caller owns the executor's serving resources and can store these traces
    beside the rescored solver in an Episode. Use outside an active episode's
    scoring context, which already owns its child traces and their accounting.
    """
    traces: list[Trace] = []
    seen: set[int] = set()
    executor = task.judge_executor
    if executor is None:
        yield traces
        return

    async def execute(request, runtime):
        def watch(trace):
            # Native retries share an id, but each attempt is a distinct object.
            if id(trace) not in seen:
                seen.add(id(trace))
                traces.append(trace)
            if request.on_trace is not None:
                request.on_trace(trace)

        result = await executor(replace(request, on_trace=watch), runtime)
        if id(result) not in seen:
            seen.add(id(result))
            traces.append(result)
        return result

    task.judge_executor = execute
    try:
        yield traces
    finally:
        task.judge_executor = executor
