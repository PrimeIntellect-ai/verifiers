"""The environment: a taskset composed with an agent and a runtime.

It owns the per-rollout execution. It provisions the runtime (which the taskset
may refine per task, e.g. its image), starts the interception server, has the
agent run inside that runtime, then lets the taskset verify inside the SAME
runtime and score. The runtime is shared infrastructure, so a harbor-style task
with an in-runtime verifier runs under any agent.
"""

import time

from pydantic_config import BaseConfig

from verifiers.nano.agent import Agent, AgentConfig, DefaultAgentConfig
from verifiers.nano.clients import Client
from verifiers.nano.context import RolloutContext
from verifiers.nano.decorators import discover_decorated
from verifiers.nano.errors import RolloutError
from verifiers.nano.interception import InterceptionServer
from verifiers.nano.runtime import RuntimeConfig, SubprocessConfig, make_runtime
from verifiers.nano.task import Task
from verifiers.nano.taskset import Taskset, TasksetConfig
from verifiers.nano.transcript import Transcript
from verifiers.nano.types import SamplingConfig


class EnvConfig(BaseConfig):
    """The rollout's three peers, each with single field ownership: the taskset
    (data + scoring), the agent (which program drives it), and the runtime (where
    it runs). Subclass per env to narrow types / set defaults."""

    taskset: TasksetConfig = TasksetConfig()
    agent: AgentConfig = DefaultAgentConfig()
    runtime: RuntimeConfig = SubprocessConfig()


class Environment:
    def __init__(self, taskset: Taskset, agent: Agent, runtime: RuntimeConfig) -> None:
        self.taskset = taskset
        self.agent = agent
        self.runtime = runtime
        allowed = taskset.config.allowed_runtimes
        if allowed is not None and runtime.kind not in allowed:
            raise ValueError(
                f"taskset allows runtimes {allowed}, but the runtime is {runtime.kind!r}"
            )

    def tasks(self) -> list[Task]:
        return self.taskset.load_tasks()

    async def run_rollout(
        self, task: Task, client: Client, model: str, sampling_args: SamplingConfig
    ) -> Transcript:
        ctx = RolloutContext(
            client=client,
            model=model,
            sampling=sampling_args,
            user=self.taskset.user,
            toolset=self.taskset.toolset,
        )
        transcript = self.taskset.transcript_type(task=task)
        transcript.timing.generation.start = time.time()
        runtime = make_runtime(self.taskset.runtime_config(task, self.runtime))
        stops = discover_decorated(self.taskset, "stop")
        try:
            async with InterceptionServer(ctx, transcript, stops) as server:
                endpoint = await runtime.start(server.port)
                await self.agent.run(ctx, transcript, runtime, endpoint, server.secret)
            await self.taskset.verify(transcript, runtime)
            await self.taskset.score(transcript)
        except RolloutError as e:
            transcript.capture_error(e)
        finally:
            await runtime.stop()
            transcript.is_completed = True
            transcript.timing.generation.end = time.time()
            transcript.metrics["num_turns"] = float(len(transcript.trajectory))
        return transcript
