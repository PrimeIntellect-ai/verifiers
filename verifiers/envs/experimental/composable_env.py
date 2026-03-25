"""ComposableEnv — a CliAgentEnv that delegates to a Task.

Subclasses ``CliAgentEnv`` and overrides its hooks to delegate to the
``Task`` protocol.  This gives you the full interception machinery
(tunnel, HTTP proxy, background job polling, streaming) for free,
while the Task provides the docker image, sandbox setup, and evaluation.

Usage::

    task = SweTaskAdapter(R2EGymTask())
    env = ComposableEnv(
        task=task,
        run_command='opencode run "$(cat /task/instruction.md)"',
        install_script="curl -fsSL ... | bash",
    )

    # or with a TaskSet
    task = LeanTaskSet("minif2f")
    env = ComposableEnv(task=task, run_command="opencode run ...")
"""

from __future__ import annotations

import logging
from typing import Any

import verifiers as vf
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv, CliAgentMonitorRubric
from verifiers.envs.experimental.sandbox_mixin import SandboxMonitorRubric
from verifiers.envs.experimental.task import Task
from verifiers.types import State

logger = logging.getLogger(__name__)


class ComposableRubric(SandboxMonitorRubric):
    """Rubric that reads the reward pre-computed by Task.evaluate().

    Extends ``SandboxMonitorRubric`` so sandbox OOM/timeout metrics are
    included in the same rubric.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_reward_func(self.task_reward)

    async def task_reward(self, state: State) -> float:
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
    task:
        A ``Task`` or ``TaskSet`` that provides what to solve.
    run_command:
        Shell command to start the agent binary in the sandbox.
    install_script:
        Optional shell command to install the agent binary during
        ``post_sandbox_setup``.
    test_timeout:
        Timeout in seconds for ``task.evaluate()``.
    """

    def __init__(
        self,
        task: Task,
        run_command: str,
        *,
        install_script: str | None = None,
        test_timeout: int = 900,
        **kwargs: Any,
    ):
        # Auto-extract dataset from TaskSet if not provided
        if "dataset" not in kwargs and hasattr(task, "get_dataset"):
            kwargs["dataset"] = task.get_dataset()
        # Inject ComposableRubric unless user provided one
        if "rubric" not in kwargs:
            kwargs["rubric"] = ComposableRubric()
        super().__init__(run_command=run_command, **kwargs)

        self.task = task
        self.install_script = install_script
        self.test_timeout = test_timeout

        self.add_rubric(CliAgentMonitorRubric())

    # -- CliAgentEnv hooks --------------------------------------------------

    async def get_docker_image(self, state: State) -> str:
        """Delegate to Task for per-instance docker images."""
        info = state.get("info") or {}
        try:
            return self.task.get_image(info)
        except Exception:
            return self.docker_image

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Merge base env vars with Task-provided env vars."""
        env_vars = await super().build_env_vars(state)
        env_vars.update(self.task.get_env_vars())
        return env_vars

    async def post_sandbox_setup(self, state: State) -> None:
        """Run Task setup, then install agent binary if configured."""
        sandbox_id = state["sandbox_id"]

        # Task setup (repo checkout, venv links, proof file, etc.)
        await self.task.setup(self.sandbox_client, sandbox_id, state)

        # Agent install (optional)
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

        # Store run_background_job in state for SweTaskAdapter compat
        state["_run_background_job"] = self.run_background_job

        try:
            reward = await self.task.evaluate(
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
