"""ComposableEnv — a CliAgentEnv that delegates to a TaskSpec + Harness.

Subclasses ``CliAgentEnv`` and overrides its hooks to delegate to the
``TaskSpec`` (what to solve) and ``Harness`` (how the agent runs).

Usage::

    from swe_tasksets import R2ETaskSet
    from opencode_agent import opencode_harness
    from verifiers.envs.experimental.composable import ComposableEnv

    taskset = R2ETaskSet()
    harness = opencode_harness(system_prompt="You are a coding agent...")
    env = ComposableEnv(taskset=taskset, harness=harness)
"""

from __future__ import annotations

import logging
from typing import Any

import verifiers as vf
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
from verifiers.envs.experimental.composable.harness import Harness
from verifiers.envs.experimental.composable.task import TaskSet
from verifiers.envs.experimental.sandbox_mixin import SandboxMonitorRubric
from verifiers.types import State

logger = logging.getLogger(__name__)


class ComposableRubric(SandboxMonitorRubric):
    """Rubric that reads the reward pre-computed by TaskSpec.evaluate()."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_reward_func(self.task_reward)

    async def task_reward(self, state: State) -> float:
        return float(state.get("reward") or 0.0)


class ComposableEnv(CliAgentEnv):
    """CliAgentEnv that delegates to a TaskSpec (via TaskSet) and a Harness.

    The TaskSpec provides: docker image, sandbox setup, evaluation.
    The Harness provides: install script, run command, system prompt, paths.
    ComposableEnv connects them.

    Parameters
    ----------
    taskset:
        A ``TaskSet`` — the collection of tasks to solve.
    harness:
        A ``Harness`` — the agent configuration.
    test_timeout:
        Timeout in seconds for ``TaskSpec.evaluate()``.
    """

    def __init__(
        self,
        taskset: TaskSet,
        harness: Harness,
        *,
        test_timeout: int = 900,
        **kwargs: Any,
    ):
        kwargs["dataset"] = taskset.get_dataset()
        if "rubric" not in kwargs:
            kwargs["rubric"] = ComposableRubric()
        super().__init__(run_command=harness.run_command, **kwargs)

        self.spec = taskset.spec
        self.harness = harness
        self.test_timeout = test_timeout

        # Note: CliAgentMonitorRubric is already added by CliAgentEnv.__init__

    # -- CliAgentEnv hooks --------------------------------------------------

    async def get_docker_image(self, state: State) -> str:
        info = state.get("info") or {}
        try:
            return self.spec.get_image(info)
        except Exception:
            return self.docker_image

    async def build_env_vars(self, state: State) -> dict[str, str]:
        env_vars = await super().build_env_vars(state)
        env_vars.update(self.spec.get_env_vars())
        info = state.get("info") or {}
        try:
            env_vars.setdefault("AGENT_WORKDIR", self.spec.get_workdir(info))
        except Exception:
            pass
        return env_vars

    async def post_sandbox_setup(self, state: State) -> None:
        """Task setup → upload instruction → upload system prompt → install agent."""
        sandbox_id = state["sandbox_id"]

        # 1. Task setup (repo checkout, venv links, proof file, etc.)
        await self.spec.setup(self.sandbox_client, sandbox_id, state)

        # 2. Upload instruction to harness-declared path
        info = state.get("info") or {}
        try:
            prompt = self.spec.get_prompt(info)
            instruction = ""
            for msg in prompt:
                content = (
                    getattr(msg, "content", "")
                    if not isinstance(msg, dict)
                    else msg.get("content", "")
                )
                if content:
                    instruction += str(content) + "\n"
            if instruction.strip():
                parent = self.harness.instruction_path.rsplit("/", 1)[0]
                await self.sandbox_client.execute_command(
                    sandbox_id, f"mkdir -p {parent}", timeout=10
                )
                await self.sandbox_client.execute_command(
                    sandbox_id,
                    f"cat > {self.harness.instruction_path} << 'INSTRUCTION_EOF'\n{instruction.strip()}\nINSTRUCTION_EOF",
                    timeout=30,
                )
        except Exception as e:
            self.logger.warning(f"Failed to upload instruction: {e}")

        # 3. Upload system prompt to harness-declared path
        if self.harness.system_prompt:
            try:
                parent = self.harness.system_prompt_path.rsplit("/", 1)[0]
                await self.sandbox_client.execute_command(
                    sandbox_id, f"mkdir -p {parent}", timeout=10
                )
                await self.sandbox_client.execute_command(
                    sandbox_id,
                    f"cat > {self.harness.system_prompt_path} << 'SYSPROMPT_EOF'\n{self.harness.system_prompt}\nSYSPROMPT_EOF",
                    timeout=30,
                )
            except Exception as e:
                self.logger.warning(f"Failed to upload system prompt: {e}")

        # 4. Install agent binary
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
        """Run TaskSpec evaluation after the agent finishes."""
        await super().post_rollout(state)

        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return

        if state.get("error") and isinstance(state["error"], vf.InfraError):
            state["reward"] = 0.0
            return

        state["_run_background_job"] = self.run_background_job

        try:
            reward = await self.spec.evaluate(self.sandbox_client, sandbox_id, state)
            if isinstance(reward, dict):
                state["role_rewards"] = reward
                state["reward"] = sum(reward.values()) / len(reward) if reward else 0.0
            else:
                state["reward"] = float(reward)
        except Exception as e:
            self.logger.warning(f"Task evaluation failed: {e}")
            state["reward"] = 0.0
