"""ComposableEnv — a CliAgentEnv that delegates to a TaskSet + Harness.

Subclasses ``CliAgentEnv`` and overrides its hooks to delegate to the
``TaskSet`` (what to solve) and ``Harness`` (how the agent runs).

Task / Harness contract
-----------------------

The task and harness each own different concerns.  ComposableEnv connects them:

**Task owns** (via TaskSet / SandboxTaskSet):
- Instruction text: ``get_instruction(info) -> str``
- Docker image + resources: ``get_sandbox_spec(info) -> SandboxSpec``
- Working directory: ``get_workdir(info) -> str`` (exported as ``AGENT_WORKDIR``)
- Sandbox setup: ``setup(sandbox_client, sandbox_id, state)``
- Evaluation: ``evaluate(...)``
- Environment variables: ``get_env_vars() -> dict``

**Harness owns** (via Harness dataclass):
- How to install the agent: ``install_script``
- How to run the agent: ``run_command``
- Where the agent reads instruction: ``instruction_path`` (default ``/opencode/prompt.txt``)
- Where the agent reads system prompt: ``system_prompt_path``
- System prompt content: ``system_prompt``
- Fallback sandbox resources: ``sandbox_spec`` (when task doesn't need sandbox)

**ComposableEnv connects them**:
1. Writes task's instruction text → harness's instruction_path
2. Writes harness's system prompt → harness's system_prompt_path
3. Harness's ``run_command`` reads from these paths

ComposableEnv exports the task's working directory as ``AGENT_WORKDIR`` for
harnesses that need a per-instance workdir while still using a static
``run_command``.
"""

from __future__ import annotations

import logging
from typing import Any

import verifiers as vf
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
from verifiers.envs.experimental.composable.harness import Harness
from verifiers.envs.experimental.composable.task import TaskSet
from verifiers.types import State

logger = logging.getLogger(__name__)


class ComposableEnv(CliAgentEnv):
    """CliAgentEnv that delegates to a TaskSet and a Harness.

    For SandboxTaskSet: uses task's SandboxSpec for image, task's setup/evaluate.
    For plain TaskSet: uses harness default image, task evaluate takes no sandbox args.

    Parameters
    ----------
    taskset:
        A ``TaskSet`` or ``SandboxTaskSet``.
    harness:
        A ``Harness`` — the agent configuration.
    """

    def __init__(
        self,
        taskset: TaskSet,
        harness: Harness,
        **kwargs: Any,
    ):
        kwargs["dataset"] = taskset.get_dataset()
        if "rubric" not in kwargs:
            kwargs["rubric"] = taskset.get_rubric()
        super().__init__(run_command=harness.run_command, **kwargs)

        self.taskset = taskset
        self.harness = harness

    # -- CliAgentEnv hooks --------------------------------------------------

    def _get_spec(self, state: State) -> Any:
        """Get SandboxSpec, cached on state to avoid redundant calls."""
        cached = state.get("_sandbox_spec")
        if cached is not None:
            return cached
        info = state.get("info") or {}
        spec = self.taskset.get_sandbox_spec(info)
        state["_sandbox_spec"] = spec
        return spec

    async def get_docker_image(self, state: State) -> str:
        spec = self._get_spec(state)
        if spec:
            return spec.image
        if self.harness.sandbox_spec:
            return self.harness.sandbox_spec.image
        return self.docker_image

    def get_sandbox_resources(self, state: State) -> dict[str, Any]:
        """Per-instance resources from SandboxSpec, or harness defaults."""
        spec = self._get_spec(state) or self.harness.sandbox_spec
        if spec:
            return {
                "cpu_cores": spec.cpu_cores,
                "memory_gb": spec.memory_gb,
                "disk_size_gb": spec.disk_size_gb,
                "gpu_count": spec.gpu_count,
                "gpu_type": spec.gpu_type,
                "vm": spec.gpu_count > 0,
                "timeout_minutes": spec.timeout_minutes,
            }
        return super().get_sandbox_resources(state)

    async def build_env_vars(self, state: State) -> dict[str, str]:
        env_vars = await super().build_env_vars(state)
        info = state.get("info") or {}
        task_env_vars = self.taskset.get_env_vars()
        if task_env_vars:
            conflicts = (
                self.PROTECTED_ENV_VARS | {"AGENT_WORKDIR"}
            ) & task_env_vars.keys()
            if conflicts:
                raise ValueError(
                    f"TaskSet.get_env_vars() must not override protected keys: {conflicts}."
                )
            env_vars.update(task_env_vars)
        env_vars["AGENT_WORKDIR"] = self.taskset.get_workdir(info)
        return env_vars

    async def post_sandbox_setup(self, state: State) -> None:
        """Task setup → upload instruction → upload system prompt → install agent."""
        sandbox_id = state["sandbox_id"]

        # Populate sandbox context in state (once, used by setup/evaluate/validate)
        state["sandbox_client"] = self.sandbox_client
        spec = self._get_spec(state)
        if spec:
            state["test_timeout"] = spec.timeout_minutes * 60
        elif self.harness.sandbox_spec:
            state["test_timeout"] = self.harness.sandbox_spec.timeout_minutes * 60
        else:
            state["test_timeout"] = 900

        # 1. Task setup
        await self.taskset.setup(state)

        # 2. Create parent dirs for instruction + system prompt in one roundtrip
        dirs = {self.harness.instruction_path.rsplit("/", 1)[0]}
        if self.harness.system_prompt:
            dirs.add(self.harness.system_prompt_path.rsplit("/", 1)[0])
        await self.sandbox_client.execute_command(
            sandbox_id, f"mkdir -p {' '.join(dirs)}", timeout=10
        )

        # 3. Upload instruction to harness-declared path
        info = state.get("info") or {}
        instruction = self.taskset.get_instruction(info)
        if instruction.strip():
            await self.upload_content(
                sandbox_id, instruction, self.harness.instruction_path
            )

        # 4. Upload system prompt to harness-declared path
        if self.harness.system_prompt:
            await self.upload_content(
                sandbox_id, self.harness.system_prompt, self.harness.system_prompt_path
            )

        # 5. Install agent binary
        if self.harness.install_script:
            self.logger.debug(f"Installing agent in sandbox {sandbox_id}")
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                self.harness.install_script,
                timeout=300,
            )
            if result.exit_code != 0:
                output = (result.stdout or "") + (result.stderr or "")
                raise vf.SandboxError(
                    f"Agent install failed (exit={result.exit_code}): {output[:500]}"
                )

    async def post_rollout(self, state: State) -> None:
        """Collect agent logs after the agent finishes.

        Scoring is handled entirely by the rubric (via ``score_rollout``),
        not here.  Use ``keep_sandbox_for_scoring=True`` so the sandbox
        stays alive for the rubric to run tests / read files.
        """
        sandbox_id = state.get("sandbox_id")
        if sandbox_id and self.harness.log_path and "agent_logs" not in state:
            try:
                result = await self.sandbox_client.execute_command(
                    sandbox_id,
                    f"cat {self.harness.log_path} 2>/dev/null || echo '<no logs>'",
                    working_dir=None,
                )
                state["agent_logs"] = (result.stdout or "").strip()
            except Exception as e:
                self.logger.warning(f"Failed to collect agent logs: {e}")

        await super().post_rollout(state)
