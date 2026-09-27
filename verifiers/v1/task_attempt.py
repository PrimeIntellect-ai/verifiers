"""One task world, shared by agent runs and owned independently of them."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from contextlib import AsyncExitStack
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Self

from verifiers.v1.configs.verifier import VerifierConfig
from verifiers.v1.errors import TaskError, boundary
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
    provision_runtime,
)
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.artifacts import collect, restore
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.decorators import invoke
from verifiers.v1.utils.retries import backoff

if TYPE_CHECKING:
    from verifiers.v1.rollout import RolloutTimeouts
    from verifiers.v1.task import Task
    from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


class TaskAttempt:
    """A task's live services and grading lifecycle.

    Enter to provision services, pass the attempt to agent.run/interaction, then
    grade the selected trace. World preparation runs once on entry; each agent
    keeps its own trace state and setup. Exiting always frees services.
    An attempt is single-use and can be graded once, after all agents have closed.
    """

    def __init__(
        self,
        task: Task,
        *,
        placement: RuntimeConfig,
        runtime: Runtime | None = None,
        timeouts: RolloutTimeouts | None = None,
        verifier: VerifierConfig | None = None,
        collect_artifacts: bool = False,
    ) -> None:
        from verifiers.v1.agent import resolve_rollout_timeouts
        from verifiers.v1.configs.agent import TimeoutConfig

        self.task = task
        self.placement = placement
        self.timeouts = timeouts or resolve_rollout_timeouts(TimeoutConfig(), task)
        self.verifier = verifier if verifier is not None else task.config.verifier
        self.services: dict[str, Runtime] = {}
        self._borrowed = runtime
        self._collect_artifacts = collect_artifacts
        self._resources = AsyncExitStack()
        self._entered = False
        self._closed = False
        self._graded = False
        self._grading = False
        self._running: dict[str, Runtime] = {}
        self._traces: set[str] = set()

    @property
    def runtime(self) -> Runtime:
        if not self._entered or self._closed:
            raise RuntimeError("task attempt is not open")
        return self.services["main"]

    def check_open(self) -> None:
        if not self._entered or self._closed or self._graded or self._grading:
            raise RuntimeError("task attempt is not open for agent execution")

    async def __aenter__(self) -> Self:
        if self._entered or self._closed:
            raise RuntimeError("task attempt is single-use")
        self._entered = True
        try:
            if self.verifier is not None:
                if self.task.config.judges:
                    raise ValueError(
                        "isolated verification requires deterministic task scoring"
                    )
                self.verifier_config(self.verifier_task())
            async with asyncio.timeout(self.timeouts.setup):
                await self.provision()
                await self.runtime.prepare_setup()
                async with boundary(TaskError, "task world preparation"):
                    await self.task.prepare(self.runtime)
            return self
        except BaseException:
            await self.close()
            raise

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def provision(self) -> None:
        """Provision the task world. Adapters may populate several services."""
        if self._borrowed is not None:
            if self._borrowed.stopped:
                raise ValueError("cannot open a task in a stopped runtime")
            runtime = self._borrowed.with_env(self.task.runtime_env())
        else:
            config = resolve_runtime_config(self.placement, self.task)
            runtime = await self._resources.enter_async_context(
                provision_runtime(config, env=self.task.runtime_env())
            )
        self.services["main"] = runtime
        if self._collect_artifacts and isinstance(runtime.config, SubprocessConfig):
            raise TypeError("artifact collection requires a container runtime")

    async def attach(self, trace: Trace, runtime: Runtime) -> None:
        """Initialize one agent session without sharing or replacing its trace state."""
        self.check_open()
        # Trusted setup temporarily opens egress. Never open it underneath an
        # already executing agent in the same restricted runtime (with_env views
        # share the runtime's storage). Distinct runtimes can run concurrently.
        if runtime.network_restricted and any(
            box.__dict__ is runtime.__dict__ for box in self._running.values()
        ):
            raise ValueError("overlapping agents need distinct restricted runtimes")
        self._running[trace.id] = runtime
        self._traces.add(trace.id)
        await runtime.prepare_setup()
        async with boundary(TaskError, "task session setup"):
            await invoke(self.task.setup, {"trace": trace, "runtime": runtime})

    def release(self, trace: Trace) -> None:
        self._running.pop(trace.id, None)

    async def finalize(self, trace: Trace) -> None:
        await invoke(self.task.finalize, {"trace": trace, "runtime": self.runtime})
        if (
            self._collect_artifacts or self.verifier is not None
        ) and not trace.state.artifacts:
            trace.state.artifacts = await collect(
                self.runtime,
                self.task.data.artifacts,
                max_bytes=self.task.data.artifact_max_bytes,
            )

    def verifier_task(self) -> Task:
        task = copy.deepcopy(self.task)
        task.scoring_deferred = False
        return task

    def verifier_config(self, task: Task) -> RuntimeConfig:
        assert self.verifier is not None
        base = self.verifier.runtime or self.placement
        config = resolve_runtime_config(base, task)
        if self.verifier.runtime is not None and "image" in base.model_fields_set:
            config = config.model_copy(update={"image": base.image})
        if isinstance(config, SubprocessConfig):
            raise TypeError("isolated verification requires a container runtime")
        relative = [
            a.source
            for a in self.task.data.artifacts
            if not PurePosixPath(a.source).is_absolute()
        ]
        solver = self.services.get("main")
        solver_config = (
            solver.config
            if solver is not None
            else resolve_runtime_config(self.placement, self.task)
        )
        solver_workdir = PurePosixPath(getattr(solver_config, "workdir", "/") or "/app")
        if relative and solver_workdir != PurePosixPath(config.workdir or "/app"):
            raise ValueError(
                "isolated verification requires matching workdirs for relative artifacts"
            )
        return config

    async def grade(self, trace: Trace) -> Trace:
        """Finalize and score the chosen solution, updating that same trace in place."""
        self.check_open()
        if trace.id not in self._traces:
            raise ValueError("trace does not belong to this task attempt")
        if self._running:
            raise RuntimeError("close every agent run before grading the task attempt")
        if not trace.ok:
            return trace
        self._grading = True
        try:
            trace.timing.finalize.start = time.time()
            async with (
                boundary(TaskError, "task finalize"),
                asyncio.timeout(self.timeouts.finalize),
            ):
                await self.finalize(trace)
            trace.timing.finalize.end = time.time()
            trace.timing.scoring.start = time.time()
            async with boundary(TaskError, "task scoring"):
                if self.verifier is None:
                    async with asyncio.timeout(self.timeouts.scoring):
                        await self.task.score(trace, self.runtime)
                else:
                    # Resolve inherited images while services are still inspectable.
                    grader = self.verifier_task()
                    config = self.verifier_config(grader)
                    await self.stop_services()
                    await self.grade_isolated(grader, config, trace)
        except asyncio.CancelledError:
            trace.record_error(TaskError("task grading was cancelled"))
            trace.ok = False
            raise
        except Exception as error:
            trace.record_error(error)
            trace.ok = False
            raise
        finally:
            self._graded = True
            now = time.time()
            if not trace.timing.finalize.end:
                trace.timing.finalize.end = now
            if trace.timing.scoring.start:
                trace.timing.scoring.end = now
            trace.notify()
        return trace

    @property
    def verifier_attempt_timeout(self) -> float | None:
        """Optional budget covering the complete fresh verifier attempt."""
        return None

    async def grade_isolated(
        self, task: Task, config: RuntimeConfig, trace: Trace
    ) -> None:
        assert self.verifier is not None
        last: Exception | None = None
        for attempt in range(self.verifier.retries + 1):
            if attempt:
                delay = backoff(attempt - 1)
                logger.warning(
                    "retrying isolated verifier after %s in %.1fs", last, delay
                )
                await asyncio.sleep(delay)
            try:
                verifier_task = copy.deepcopy(task)
                solution = copy.deepcopy(trace)
                async with (
                    AsyncExitStack() as boxes,
                    asyncio.timeout(self.verifier_attempt_timeout),
                ):
                    async with asyncio.timeout(self.timeouts.setup):
                        runtime = await boxes.enter_async_context(
                            provision_runtime(
                                config,
                                env=verifier_task.runtime_env()
                                if self.verifier.env is None
                                else self.verifier.env,
                            )
                        )
                        await runtime.prepare_setup()
                        artifacts = dict(solution.state.artifacts)
                        await verifier_task.prepare(runtime)
                        await invoke(
                            verifier_task.setup, {"trace": solution, "runtime": runtime}
                        )
                        await restore(runtime, artifacts)
                        await invoke(
                            verifier_task.stage_verifier,
                            {"trace": solution, "runtime": runtime},
                        )
                        await runtime.prepare_execution([])
                    async with asyncio.timeout(self.timeouts.scoring):
                        await verifier_task.score(solution, runtime)
                # Keep episode/live-view references to the original trace valid.
                trace.state = solution.state
                trace.metrics = solution.metrics
                trace.rewards = solution.rewards
                trace.info = solution.info
                trace.extra_usage = solution.extra_usage
                return
            except Exception as error:  # noqa: BLE001 - retry the entire fresh verifier
                last = error
        assert last is not None
        raise last

    async def stop_services(self) -> None:
        await run_shielded(self._resources.aclose())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.stop_services()
