"""Adapter from ``swe_tasks.SweTask`` to the verifiers ``Task`` protocol.

This bridges the ``SweTask`` protocol (defined in the ``swe-tasks`` package
in research-environments) with the generic ``Task`` protocol used by
``ComposableEnv``.  The mapping is intentionally thin:

.. code-block:: text

    SweTask                    →  Task
    ─────────────────────────     ──────────────────────
    get_dataset()              →  get_dataset()
    get_instruction(info)      →  get_prompt(info)   [wrapped in Messages]
    get_docker_image(info)     →  get_image(info)
    get_agent_workdir(info)    →  get_workdir(info)
    get_env_vars()             →  get_env_vars()
    setup_sandbox(...)         →  setup(...)
    run_tests + calculate_reward  →  evaluate(...)
    apply_gold_patch(...)      →  apply_gold_patch(...)
"""

from __future__ import annotations

import logging
from typing import Any

from verifiers.envs.experimental.task import TaskSet
from verifiers.types import Messages, State, SystemMessage, UserMessage

logger = logging.getLogger(__name__)


class SweTaskAdapter:
    """Wrap a ``SweTask`` as a verifiers ``Task``.

    Parameters
    ----------
    swe_task:
        Any object implementing the ``SweTask`` protocol (from the
        ``swe-tasks`` package).
    test_timeout:
        Timeout in seconds for test execution inside ``evaluate()``.
    system_prompt:
        Optional system prompt to prepend to the instruction.
    """

    def __init__(
        self,
        swe_task: Any,
        test_timeout: int = 900,
        system_prompt: str | None = None,
    ):
        self._swe_task = swe_task
        self._test_timeout = test_timeout
        self._system_prompt = system_prompt

    # -- Task protocol implementation ---------------------------------------

    def get_dataset(self) -> Any:
        return self._swe_task.get_dataset()

    def get_prompt(self, info: dict) -> Messages:
        """Build prompt Messages from the task instruction."""
        instruction = self._swe_task.get_instruction(info)
        messages: Messages = []
        if self._system_prompt:
            messages.append(SystemMessage(content=self._system_prompt))
        messages.append(UserMessage(content=instruction))
        return messages

    def get_image(self, info: dict) -> str:
        return self._swe_task.get_docker_image(info)

    def get_workdir(self, info: dict) -> str:
        return self._swe_task.get_agent_workdir(info)

    def get_env_vars(self) -> dict[str, str]:
        return self._swe_task.get_env_vars()

    async def setup(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: State,
    ) -> None:
        await self._swe_task.setup_sandbox(sandbox_client, sandbox_id, state)

    async def evaluate(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: State,
    ) -> float:
        """Run tests and compute reward.

        Uses ``_run_background_job`` from state (injected by ComposableEnv)
        to execute long-running test commands.
        """
        run_background_job = state.get("_run_background_job")
        if run_background_job is None:
            raise RuntimeError(
                "SweTaskAdapter.evaluate() requires state['_run_background_job']. "
                "Ensure ComposableEnv stores it during rollout."
            )

        info = state.get("info") or {}
        try:
            test_output = await self._swe_task.run_tests(
                sandbox_client,
                sandbox_id,
                state,
                run_background_job,
                self._test_timeout,
            )
            state["test_output"] = test_output
            reward = self._swe_task.calculate_reward(test_output, info)
        except Exception as e:
            logger.warning(f"Test execution failed: {e}")
            state["test_output"] = f"ERROR: {e}"
            reward = 0.0

        return float(reward)

    async def apply_gold_patch(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: State,
    ) -> None:
        await self._swe_task.apply_gold_patch(sandbox_client, sandbox_id, state)

    # -- TaskSet convenience ------------------------------------------------

    def to_taskset(self, name: str = "") -> TaskSet:
        """Create a ``TaskSet`` from this adapter's dataset."""
        return TaskSet(
            task=self,
            dataset=self.get_dataset(),
            name=name or type(self._swe_task).__name__,
        )
