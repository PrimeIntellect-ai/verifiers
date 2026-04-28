"""SolveEnv — gold-patch validation for any SandboxTaskSet.

Creates sandboxes, calls ``taskset.validate_instance(state)`` (which applies
the gold patch + runs tests), and reports the result as a reward.

No agent, no inference — just infrastructure validation.

Recommended invocation::

    vf-eval rlm-swe \
        --state-columns reason,attempts,elapsed_s,test_output_tail \
        # ...env-specific flags

The state columns above match the row shape produced by
:meth:`SandboxTaskSet.validate` so eval output is directly comparable.

::

    from verifiers.envs.experimental.composable import SolveEnv
    from swe_tasksets import R2EGymTaskSet

    taskset = R2EGymTaskSet()
    env = SolveEnv(taskset=taskset)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import verifiers as vf
from prime_sandboxes import CreateSandboxRequest
from verifiers.envs.experimental.sandbox_mixin import SandboxMixin, SandboxMonitorRubric
from verifiers.types import Messages, State

from .task import SandboxTaskSet, _classify_validate_outcome

logger = logging.getLogger(__name__)


class SolveRubric(SandboxMonitorRubric):
    """Reads ``state["reward"]`` set during setup. Inherits OOM/timeout metrics."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_reward_func(self.solve_reward, weight=1.0)

    async def solve_reward(self, state: vf.State, **kwargs: Any) -> float:
        return state.get("reward", 0.0)


class SolveEnv(SandboxMixin, vf.MultiTurnEnv):
    """Gold-patch solve: create sandbox, validate instance, score.

    Lifecycle:
    - setup_state(): create sandbox + taskset.setup + validate_instance
    - @vf.stop: immediately complete (no multi-turn loop)
    - @vf.cleanup: delete sandbox

    State columns surfaced (mirroring :meth:`SandboxTaskSet.validate` rows):
    - ``reward``: 1.0 on pass, 0.0 otherwise
    - ``reason``: one of ``pass``, ``test_failed``, ``gold_apply_failed``,
      ``setup_failed``, ``sandbox_error``, ``billing_error``, ``timeout``
    - ``attempts``: incremented on each ``setup_state`` entry (surfaces retries)
    - ``elapsed_s``: wall-clock seconds spent in ``validate_instance``
    - ``test_output_tail``: trailing chars of ``state["test_output"]``

    Suggested ``vf-eval --state-columns reason,attempts,elapsed_s,test_output_tail``.
    """

    def __init__(
        self,
        taskset: SandboxTaskSet,
        dataset: Any = None,
        test_timeout: int = 900,
        cpu_cores: int | None = None,
        memory_gb: int | None = None,
        disk_size_gb: int | None = None,
        labels: list[str] | None = None,
        timeout_seconds: float = 1800.0,
        test_output_tail_chars: int = 2000,
        **sandbox_kwargs: Any,
    ):
        self.taskset = taskset
        self.test_timeout = test_timeout
        self._cpu_cores = cpu_cores
        self._memory_gb = memory_gb
        self._disk_size_gb = disk_size_gb
        self.labels = labels or ["solve"]
        self.timeout_seconds = timeout_seconds
        self.test_output_tail_chars = test_output_tail_chars

        dataset = dataset or taskset.get_dataset()
        rubric = SolveRubric()
        super().__init__(dataset=dataset, rubric=rubric)
        self.init_sandbox_client(**sandbox_kwargs)

    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> Messages:
        raise NotImplementedError("SolveEnv does not use multi-turn interaction")

    # --- Lifecycle ---

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        state["attempts"] = state.get("attempts", 0) + 1
        info = state["info"]
        spec = self.taskset.get_sandbox_spec(info)

        request = CreateSandboxRequest(
            name=f"solve-{state.get('example_id', 'unknown')}",
            docker_image=spec.image,
            cpu_cores=self._cpu_cores or spec.cpu_cores,
            memory_gb=self._memory_gb or spec.memory_gb,
            disk_size_gb=self._disk_size_gb or spec.disk_size_gb,
            gpu_count=spec.gpu_count,
            gpu_type=spec.gpu_type,
            vm=spec.gpu_count > 0,
            timeout_minutes=math.ceil(self.timeout_seconds / 60),
            labels=self.labels,
        )
        await self.create_sandbox(state, request)

        t0 = time.monotonic()
        valid = False
        exc: BaseException | None = None
        try:
            valid = await self.taskset.validate_instance(state)
            state["reward"] = float(valid)
            state["completion"] = f"reward={float(valid)}"
        except Exception as e:
            exc = e
            state["error"] = vf.SandboxError(f"Validation failed: {repr(e)}")
            state["reward"] = 0.0
        finally:
            state["elapsed_s"] = time.monotonic() - t0

        reason, tail = _classify_validate_outcome(
            valid,
            exc,
            state,
            is_sandbox=True,
            test_output_tail_chars=self.test_output_tail_chars,
        )
        state["reason"] = reason
        if tail:
            state["test_output_tail"] = tail

        return state

    async def post_sandbox_setup(self, state: State) -> None:
        """Inject sandbox context into state, then run taskset setup."""
        state["sandbox_client"] = self.sandbox_client
        state["test_timeout"] = self.test_timeout
        state["run_background_job"] = self.run_background_job
        await self.taskset.setup(state)

    @vf.stop
    async def solve_completed(self, state: State) -> bool:
        return True

    @vf.cleanup
    async def destroy_sandbox(self, state: State) -> None:
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            await self.delete_sandbox(sandbox_id)
