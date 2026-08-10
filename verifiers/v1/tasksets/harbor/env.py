"""Harbor's taskset-level control flow and separate-verifier grading.

Ordinary shared-verifier tasks take the single-agent path. Ordinary separate
verifiers grade in a fresh box. ``[[steps]]`` tasks run their ordered agent and
verifier phases in one shared agent runtime, with either fresh agent traces or one
resumed interaction.
"""

import asyncio
import logging
from contextlib import AsyncExitStack

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.runtimes import RuntimeConfig, provision_runtime
from verifiers.v1.tasksets.harbor.taskset import (
    HarborTask,
    aggregate_step_rewards,
    min_reward_failure,
    run_healthcheck,
    verifier_box_data,
)
from verifiers.v1.utils.artifacts import restore
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.retries import backoff

logger = logging.getLogger(__name__)


class HarborEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()
    """The policy under evaluation/training."""
    verifier_runtime: RuntimeConfig | None = None
    """Optional runtime policy for separate verifier boxes."""
    verifier_retries: int = Field(2, ge=0)
    """Extra infrastructure attempts for a separate verifier box."""
    resume_trajectory: bool = False
    """Continue one harness interaction across steps instead of fresh runs."""


class HarborEnv(vf.Env[HarborEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, HarborTask):
            raise TypeError(
                f"the harbor env runs harbor tasks; got {type(task).__name__}"
            )
        if task.data.steps:
            await self._run_multi_step(task, agents)
            return
        if task.data.verifier is None:
            await agents.agent.run(task)
            return
        # Resolve before the solve so an impossible separate-verifier pairing
        # does not cost a full agent run.
        self._verifier_config(task)
        await agents.agent.run(task.graded_elsewhere())

    async def _run_multi_step(self, task: HarborTask, agents: vf.Agents) -> None:
        async with agents.agent.provision(task) as runtime:
            if task.data.environment_healthcheck is not None:
                await run_healthcheck(runtime, task.data.environment_healthcheck)
            if self.config.resume_trajectory:
                await self._run_resumed(task, agents, runtime)
            else:
                await self._run_fresh(task, agents, runtime)

    async def _run_fresh(
        self, task: HarborTask, agents: vf.Agents, runtime: vf.Runtime
    ) -> None:
        for index, step in enumerate(task.data.steps):
            step_task = task.for_step(step, first=index == 0)
            run_task = step_task
            verifier_config = None
            if step_task.data.verifier is not None:
                verifier_config = self._verifier_config(step_task)
                run_task = step_task.graded_elsewhere()
            trace = await agents.agent.run(run_task, runtime=runtime)
            if not trace.ok:
                break
            if verifier_config is not None:
                grader = HarborTask(verifier_box_data(step_task.data))
                scores = await self._grade(verifier_config, grader, trace)
                rewards = self._step_rewards(scores)
                self._record_step_rewards(trace, step.name, rewards)
            else:
                rewards = trace.info.get("harbor_step_rewards")
            failure = min_reward_failure(rewards, step.min_reward)
            if failure is not None:
                trace.info["harbor_min_reward_failure"] = failure
                break

    async def _run_resumed(
        self, task: HarborTask, agents: vf.Agents, runtime: vf.Runtime
    ) -> None:
        first = task.data.steps[0]
        session_task = task.for_step(first, first=True, resume_session=True)
        # Per-step waits below enforce independent agent budgets. The rollout-level
        # deadline must not spend step one's budget across every resumed turn.
        session_task.data = session_task.data.model_copy(
            update={
                "timeout": vf.TaskTimeout(
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
                    await asyncio.wait_for(step_task.setup(runtime), step.timeout.setup)
                segment = await asyncio.wait_for(
                    interaction.turn(None if index == 0 else step.prompt),
                    step.timeout.agent,
                )
                if segment.terminated:
                    interaction.trace.info["harbor_stopped_before_step"] = step.name
                    break

                await asyncio.wait_for(
                    step_task.collect_step(interaction.trace, runtime),
                    step.timeout.finalize,
                )
                if step_task.data.verifier is not None:
                    config = self._verifier_config(step_task)
                    grader = HarborTask(verifier_box_data(step_task.data))
                    scores = await self._grade(config, grader, interaction.trace)
                    rewards = self._step_rewards(scores)
                else:
                    await step_task._stage_tests(runtime)
                    rewards = await asyncio.wait_for(
                        step_task._step_graded(runtime), step.timeout.scoring
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

    def _verifier_config(self, task: HarborTask) -> RuntimeConfig:
        base = (
            self.config.verifier_runtime
            if self.config.verifier_runtime is not None
            else self.config.agent.runtime
        )
        return resolve_runtime_config(base, HarborTask(verifier_box_data(task.data)))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        if not isinstance(task, HarborTask):
            return
        if task.data.steps:
            if not self.config.resume_trajectory:
                self._finalize_steps(task, episode)
            return
        if task.data.verifier is None:
            return
        solution = episode.traces[0]
        if not solution.ok:
            return
        grader = HarborTask(verifier_box_data(task.data))
        scores = await self._grade(self._verifier_config(task), grader, solution)
        self._record_scores(solution, scores)

    def _finalize_steps(self, task: HarborTask, episode: vf.Episode) -> None:
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
            for key, value in aggregate.items():
                trace.record_reward(key, value)

    async def _grade(
        self, config: RuntimeConfig, grader: HarborTask, solution: vf.Trace
    ) -> float | dict[str, float]:
        last: Exception | None = None
        for attempt in range(self.config.verifier_retries + 1):
            if attempt:
                delay = backoff(attempt - 1)
                logger.warning(
                    "harbor verifier attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    self.config.verifier_retries + 1,
                    last,
                    delay,
                )
                await asyncio.sleep(delay)
            try:
                async with AsyncExitStack() as boxes:
                    async with asyncio.timeout(grader.data.timeout.scoring):
                        box = await boxes.enter_async_context(provision_runtime(config))
                        await box.prepare_setup()
                        await restore(box, solution.state.artifacts)
                        await grader._stage_tests(box, wipe=True)
                        await box.prepare_execution([])
                        if grader.data.current_step is not None:
                            scores = await grader._step_graded(box)
                        else:
                            scores = await grader._graded(box, solution)
                    return scores
            except Exception as error:  # noqa: BLE001 - retry infrastructure failures
                last = error
        assert last is not None
        raise last

    @staticmethod
    def _step_rewards(scores: float | dict[str, float]) -> dict[str, float]:
        if isinstance(scores, dict):
            return scores
        return {"reward": scores}

    @staticmethod
    def _record_step_rewards(
        trace: vf.Trace, step_name: str, rewards: dict[str, float]
    ) -> None:
        trace.info["harbor_step"] = step_name
        trace.info["harbor_step_rewards"] = rewards
        for name, value in rewards.items():
            trace.record_reward(name, value)

    @staticmethod
    def _record_scores(trace: vf.Trace, scores: float | dict[str, float]) -> None:
        items = scores.items() if isinstance(scores, dict) else [("solved", scores)]
        for name, value in items:
            trace.record_reward(name, value)
