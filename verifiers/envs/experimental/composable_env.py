"""ComposableEnv — a CliAgentEnv that delegates to a Task.

Subclasses ``CliAgentEnv`` and overrides its hooks to delegate to the
``Task`` protocol.  This gives you the full interception machinery
(tunnel, HTTP proxy, background job polling, streaming) for free,
while the Task provides the docker image, sandbox setup, and evaluation.

Usage::

    from tasksets.swe import R2ETaskSet

    taskset = R2ETaskSet()
    env = ComposableEnv(
        taskset=taskset,
        run_command='opencode run "$(cat /task/instruction.md)"',
        install_script="curl -fsSL ... | bash",
    )
"""

from __future__ import annotations

import logging
from typing import Any

import verifiers as vf
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv, CliAgentMonitorRubric
from verifiers.envs.experimental.sandbox_mixin import SandboxMonitorRubric
from verifiers.envs.experimental.task import TaskSet
from verifiers.types import State

logger = logging.getLogger(__name__)


class ComposableRubric(SandboxMonitorRubric):
    """Rubric that reads the reward pre-computed by Task.evaluate().

    Extends ``SandboxMonitorRubric`` so sandbox OOM/timeout metrics are
    included in the same rubric.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_reward_func(self.spec_reward)

    async def spec_reward(self, state: State) -> float:
        """Return the reward computed by the Task during rollout."""
        return float(state.get("reward") or 0.0)


class ComposableEnv(CliAgentEnv):
    """CliAgentEnv that delegates image/setup/evaluation to a Task.

    Inherits all of CliAgentEnv's machinery: HTTP interception server,
    tunnel management, background job polling, streaming support, TITO
    caching, tool normalization, etc.

    The only new thing is the ``task`` parameter.  The Task provides:

    * ``get_image(info)`` → docker image per instance
    * ``get_env_vars()`` → extra env vars
    * ``setup(sandbox_client, sandbox_id, state)`` → sandbox preparation
    * ``evaluate(sandbox_client, sandbox_id, state)`` → reward computation

    Parameters
    ----------
    taskset:
        A ``TaskSet`` — the collection of tasks to solve.
    run_command:
        Shell command to start the agent binary in the sandbox.
    install_script:
        Optional shell command to install the agent binary during
        ``post_sandbox_setup``.
    test_timeout:
        Timeout in seconds for ``TaskSpec.evaluate()``.
    """

    def __init__(
        self,
        taskset: TaskSet,
        run_command: str,
        *,
        install_script: str | None = None,
        system_prompt_path: str | None = None,
        test_timeout: int = 900,
        **kwargs: Any,
    ):
        kwargs["dataset"] = taskset.get_dataset()
        if "rubric" not in kwargs:
            kwargs["rubric"] = ComposableRubric()
        super().__init__(run_command=run_command, **kwargs)

        self.spec = taskset.spec
        self.install_script = install_script
        self.system_prompt_path = system_prompt_path
        self.test_timeout = test_timeout

        self.add_rubric(CliAgentMonitorRubric())

    # -- CliAgentEnv hooks --------------------------------------------------

    async def get_docker_image(self, state: State) -> str:
        """Delegate to Task for per-instance docker images."""
        info = state.get("info") or {}
        try:
            return self.spec.get_image(info)
        except Exception:
            return self.docker_image

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Merge base env vars with Task-provided env vars."""
        env_vars = await super().build_env_vars(state)
        env_vars.update(self.spec.get_env_vars())
        # Set AGENT_WORKDIR from task if available
        info = state.get("info") or {}
        try:
            env_vars.setdefault("AGENT_WORKDIR", self.spec.get_workdir(info))
        except Exception:
            pass
        return env_vars

    async def post_sandbox_setup(self, state: State) -> None:
        """Run Task setup, upload instruction, then install agent binary."""
        sandbox_id = state["sandbox_id"]

        # 1. Task setup (repo checkout, venv links, proof file, etc.)
        await self.spec.setup(self.sandbox_client, sandbox_id, state)

        # 2. Upload task instruction for the agent to read
        info = state.get("info") or {}
        try:
            prompt = self.spec.get_prompt(info)
            instruction = ""
            for msg in prompt:
                content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
                if content and getattr(msg, "role", msg.get("role") if isinstance(msg, dict) else "") == "user":
                    instruction += str(content)
            if instruction:
                await self.sandbox_client.execute_command(sandbox_id, "mkdir -p /task", timeout=10)
                await self.sandbox_client.execute_command(
                    sandbox_id,
                    f"cat > /task/instruction.md << 'INSTRUCTION_EOF'\n{instruction}\nINSTRUCTION_EOF",
                    timeout=30,
                )
        except Exception as e:
            self.logger.warning(f"Failed to upload instruction: {e}")

        # 3. Upload system prompt if provided
        if self.system_prompt_path:
            await self.sandbox_client.execute_command(sandbox_id, "mkdir -p /opencode", timeout=10)
            await self.sandbox_client.upload_file(
                sandbox_id, "/opencode/system.txt", str(self.system_prompt_path),
            )

        # 4. Agent install (optional)
        if self.install_script:
            self.logger.debug(f"Installing agent in sandbox {sandbox_id}")
            result = await self.sandbox_client.execute_command(
                sandbox_id, self.install_script, timeout=300,
            )
            if result.exit_code != 0:
                output = (result.stdout or "") + (result.stderr or "")
                raise vf.SandboxError(
                    f"Agent install failed (exit={result.exit_code}): {output[:500]}"
                )

    async def post_rollout(self, state: State) -> None:
        """Run Task evaluation after the agent finishes."""
        await super().post_rollout(state)

        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return

        # Skip evaluation on infra errors
        if state.get("error") and isinstance(state["error"], vf.InfraError):
            state["reward"] = 0.0
            return

        # Store run_background_job so Task.evaluate() can run tests
        state["_run_background_job"] = self.run_background_job

        try:
            reward = await self.spec.evaluate(
                self.sandbox_client, sandbox_id, state,
            )
            if isinstance(reward, dict):
                state["role_rewards"] = reward
                state["reward"] = sum(reward.values()) / len(reward) if reward else 0.0
            else:
                state["reward"] = float(reward)
        except Exception as e:
            self.logger.warning(f"Task evaluation failed: {e}")
            state["reward"] = 0.0
