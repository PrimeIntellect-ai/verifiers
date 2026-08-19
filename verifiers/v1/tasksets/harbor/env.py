"""The harbor taskset's own env: the single solver seat, plus separate-verifier
grading for tasks that declare ``[verifier].environment_mode = "separate"``.

The default env for harbor runs (the taskset package exports it). A shared-verifier
task runs exactly as under the single-agent env: one `agent` trace, graded in the
box it worked in. A separate-verifier task is graded by `finalize` instead: the
solver's declared artifacts travel (collected by its task `finalize` while its box
is alive), a fresh box is provisioned from the task's verifier declaration,
`tests/` is staged there, and the verifier's rewards land on the solver's trace.
No second agent is involved — the verifier is the task's own `tests/test.sh`.
"""

import asyncio
import logging
from contextlib import AsyncExitStack

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.runtimes import RuntimeConfig, provision_runtime
from verifiers.v1.tasksets.harbor.taskset import (
    HarborTask,
    verifier_box_data,
)
from verifiers.v1.utils.artifacts import restore
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.retries import backoff

logger = logging.getLogger(__name__)


class HarborEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()
    """The one seat — the policy under evaluation/training; pin
    `--env.agent.harness.*` to choose its program or runtime."""
    verifier_runtime: RuntimeConfig | None = None
    """Where a separate-verifier task grades. None derives the grading box from
    the solver's runtime policy; set it (e.g. `--env.verifier-runtime.type prime
    --env.verifier-runtime.vm true`) when the verifier needs different placement
    than the agent."""
    verifier_retries: int = Field(2, ge=0)
    """Extra attempts at provisioning-and-grading the separate box before the
    episode fails. Grading is deterministic; what these retry is the
    infrastructure around it (image pulls, provisioning)."""


class HarborEnv(vf.Env[HarborEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, HarborTask):
            raise TypeError(
                f"the harbor env runs harbor tasks; got {type(task).__name__}"
            )
        if task.data.verifier is None:
            await agents.agent.run(task)
            return
        # Resolve the verifier's box before the solve, so an impossible pairing
        # (e.g. a restricted Prime verifier without vm=true) costs nothing
        # rather than a full agent run.
        self._verifier_config(task)
        await agents.agent.run(task.graded_elsewhere())

    def _verifier_config(self, task: HarborTask) -> RuntimeConfig:
        base = (
            self.config.verifier_runtime
            if self.config.verifier_runtime is not None
            else self.config.agent.runtime
        )
        return resolve_runtime_config(base, HarborTask(verifier_box_data(task.data)))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        """Grade a separate-verifier task in its own box, onto the solver's trace.

        Provision a fresh box from the task's verifier declaration, restore the
        solver's collected artifacts, stage `tests/`, run the verifier, and record
        its rewards (and any extra reward.json keys as metrics) on the solver's
        trace. Infrastructure failures retry per `verifier_retries`; the last one
        fails the episode — a grading box that can't be reached must never read
        as reward 0."""
        if not isinstance(task, HarborTask) or task.data.verifier is None:
            return
        solution = episode.traces[0]
        if solution.stop_condition == "agent_timeout":
            return
        if not solution.ok:
            return
        grader = HarborTask(verifier_box_data(task.data))
        scores = await self._grade(self._verifier_config(task), grader, solution)
        items = scores.items() if isinstance(scores, dict) else [("solved", scores)]
        for name, value in items:
            solution.record_reward(name, value)

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
                # The scoring deadline covers provisioning and grading, but not the
                # box's teardown: a score already in hand must not be discarded
                # because the teardown ran out the clock.
                async with AsyncExitStack() as boxes:
                    async with asyncio.timeout(grader.data.timeout.scoring):
                        box = await boxes.enter_async_context(provision_runtime(config))
                        await box.prepare_setup()
                        # Artifacts first, tests second: an artifact entry pointing
                        # into /tests must not survive staging, which wipes and
                        # rebuilds that directory.
                        await restore(box, solution.state.artifacts)
                        await grader._stage_tests(box, wipe=True)
                        await box.prepare_execution([])
                        scores = await grader._graded(box, solution)
                    return scores
            except Exception as e:  # noqa: BLE001 - each attempt's failure is retried
                last = e
        assert last is not None
        raise last
