"""Taskset-level routing for ordinary and multi-step Harbor tasks."""

import asyncio

from verifiers.v1.agent import Agents
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.env import Env
from verifiers.v1.episode import Episode
from verifiers.v1.task import Task, TaskTimeout

from .taskset import (
    HarborTask,
    aggregate_step_rewards,
    min_reward_failure,
    run_healthcheck,
)


class HarborEnvConfig(EnvConfig):
    agent: AgentConfig = AgentConfig()
    resume_trajectory: bool = False
    """Continue one harness session across steps. False starts a fresh run per step."""


class HarborEnv(Env[HarborEnvConfig]):
    """Use the normal single-agent path unless a Harbor task declares ``[[steps]]``."""

    config: HarborEnvConfig

    async def run(self, task: Task, agents: Agents) -> None:
        if not isinstance(task, HarborTask) or not task.data.steps:
            await agents.agent.run(task)
            return

        async with agents.agent.provision(task) as runtime:
            if task.data.environment_healthcheck is not None:
                await run_healthcheck(runtime, task.data.environment_healthcheck)
            if self.config.resume_trajectory:
                await self._run_resumed(task, agents, runtime)
            else:
                with runtime.reuse():
                    await self._run_fresh(task, agents, runtime)

    async def _run_fresh(self, task: HarborTask, agents: Agents, runtime) -> None:
        for step in task.data.steps:
            trace = await agents.agent.run(task.for_step(step), runtime=runtime)
            if not trace.ok:
                break
            failure = min_reward_failure(
                trace.info.get("harbor_step_rewards"), step.min_reward
            )
            if failure is not None:
                trace.info["harbor_min_reward_failure"] = failure
                break

    async def _run_resumed(self, task: HarborTask, agents: Agents, runtime) -> None:
        first = task.data.steps[0]
        session_task = task.for_step(first, resume_session=True)
        # Per-step deadlines below enforce Harbor's independent budgets. Leaving the
        # rollout-level budget set to step one would make later resumed turns spend it.
        session_task.data = session_task.data.model_copy(
            update={
                "timeout": TaskTimeout(
                    setup=first.timeout.setup,
                    finalize=first.timeout.finalize,
                    scoring=first.timeout.scoring,
                )
            }
        )
        async with agents.agent.interaction(
            session_task, runtime=runtime
        ) as interaction:
            for index, step in enumerate(task.data.steps):
                step_task = task.for_step(step)
                if index:
                    await step_task.setup(interaction.trace, runtime)
                segment = await asyncio.wait_for(
                    interaction.turn(None if index == 0 else step.prompt),
                    step.timeout.agent,
                )
                if segment.terminated:
                    interaction.trace.info["harbor_stopped_before_step"] = step.name
                    break

                await step_task.collect_step(interaction.trace, runtime)
                rewards = await asyncio.wait_for(
                    step_task.verify_step(runtime), step.timeout.scoring
                )
                interaction.trace.info.setdefault("harbor_steps", []).append(
                    {"name": step.name, "rewards": rewards}
                )
                for key, value in rewards.items():
                    interaction.trace.record_metric(
                        f"harbor_step/{step.name}/{key}", value
                    )
                failure = min_reward_failure(rewards, step.min_reward)
                if failure is not None:
                    interaction.trace.info["harbor_min_reward_failure"] = failure
                    break

    async def finalize(self, task: Task, episode: Episode) -> None:
        if not isinstance(task, HarborTask) or not task.data.steps:
            return
        if self.config.resume_trajectory:
            # The session task scores the aggregate once when its interaction closes.
            return

        traces = [
            trace
            for trace in episode.traces
            if getattr(trace.task.data, "current_step", None) is not None
        ]
        results = [
            {
                "name": trace.task.data.current_step,
                "rewards": trace.info.get("harbor_step_rewards"),
            }
            for trace in traces
        ]
        aggregate = aggregate_step_rewards(
            results, task.data.multi_step_reward_strategy
        )
        for trace in traces:
            step_name = trace.task.data.current_step
            for key, value in (trace.info.get("harbor_step_rewards") or {}).items():
                trace.record_metric(f"harbor_step/{step_name}/{key}", value)
            trace.info["harbor_trial_rewards"] = aggregate
            # Every step is one trainable trace from the same Harbor trial. Give each
            # the trial reward while retaining its local verifier values as metrics.
            for key, value in aggregate.items():
                trace.record_reward(key, value)
