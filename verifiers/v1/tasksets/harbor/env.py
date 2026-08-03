"""The harbor taskset's own env: one solver seat, plus a verifier seat for tasks
that declare ``[verifier].environment_mode = "separate"``.

The default env for harbor runs (the taskset package exports it). A shared-verifier
task runs exactly as under the single-agent env: one `agent` trace, graded in the
box it worked in. A separate-verifier task is graded by the `verifier` seat
instead: the solver's declared artifacts travel (collected by its `finalize` while
its box is alive), the seat provisions a fresh box from the task's verifier
declaration, and `finalize` records the verifier's rewards onto the solver's
trace. The verifier seat runs no program (the `noop` harness) and never calls the
model — grading is the task's own `tests/test.sh`.
"""

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor.taskset import HarborTask, HarborVerifierTask
from verifiers.v1.utils.compile import resolve_runtime_config, validate_pairing
from verifiers.v1.utils.loaders import harness_config_type


class HarborEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()
    """The solver seat — the policy under evaluation/training; pin
    `--env.agent.harness.*` to choose its program or runtime."""
    verifier: vf.AgentConfig = vf.AgentConfig()
    """The verifier seat's placement: where a separate-verifier task grades.
    Its runtime defaults to the solver's policy (`--env.verifier.runtime.*`
    overrides — e.g. `vm true` for a network-restricted Prime verifier); its
    harness is always the program-less `noop`, and its model is never called."""


class HarborEnv(vf.Env[HarborEnvConfig]):
    def __init__(self, config: HarborEnvConfig) -> None:
        # The verifier seat is not an agent: no program, no model calls. Its one
        # configurable dimension is placement, defaulting to the solver's policy.
        updates: dict = {"harness": harness_config_type("noop")(id="noop")}
        if "runtime" not in config.verifier.model_fields_set:
            updates["runtime"] = config.agent.runtime
        config.verifier = config.verifier.model_copy(update=updates)
        super().__init__(config)

    async def setup(self, agents: vf.Agents) -> None:
        # The verifier grades the policy; its (empty) exchange never trains it.
        agents.verifier.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, HarborTask) or task.data.verifier is None:
            await agents.agent.run(task)
            return
        data = HarborVerifierTask.data_for(task.data)
        # Resolve the verifier's box before the solve, so an impossible pairing
        # (e.g. a restricted Prime verifier without vm=true) costs nothing
        # rather than a full agent run.
        config = resolve_runtime_config(
            agents.verifier.runtime_config, HarborTask(data)
        )
        validate_pairing(self._harnesses["verifier"], HarborVerifierTask, config)
        solution = await agents.agent.run(task.graded_elsewhere())
        if not solution.ok:
            return
        await agents.verifier.run(HarborVerifierTask(data, solution.state.artifacts))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        by_agent = {t.agent.name: t for t in episode.traces}
        verifier = by_agent.get("verifier")
        if verifier is None:
            # Shared grading, or the solve failed (the episode is already not ok).
            return
        solution = by_agent["agent"]
        for name, reward in verifier.rewards.items():
            solution.record_reward(name, reward.score, reward.weight)
        solution.record_metrics(verifier.metrics)
