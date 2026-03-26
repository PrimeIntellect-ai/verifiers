"""TaskSpec and TaskSet — WHAT to solve.

A **TaskSpec** defines the shared behavior for a problem type: docker image,
sandbox setup, prompt construction, and evaluation logic.  You write one
TaskSpec per domain (SWE, Lean, Math, Harbor, etc.).

A **TaskSet** is a collection of Tasks backed by a TaskSpec.  This is what
you hand to ``ComposableEnv`` or a training loop — it looks like an HF
Dataset but also knows how to setup sandboxes and evaluate results.

::

    from tasksets.swe import R2ETaskSet     # 4578 tasks
    from tasksets.lean import LeanTaskSet   # 244 tasks

    r2e = R2ETaskSet()
    lean = LeanTaskSet("minif2f")
    subset = r2e.filter(lambda ex: ...).take(100)
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from verifiers.types import Messages, State


# ---------------------------------------------------------------------------
# TaskSpec — shared behavior for a problem type
# ---------------------------------------------------------------------------


@runtime_checkable
class TaskSpec(Protocol):
    """Protocol defining HOW to handle a problem type.

    Implementations provide the per-instance docker image, sandbox
    preparation, prompt construction, and evaluation logic.
    One TaskSpec per domain (R2EGymTaskSpec, LeanTaskSpec, etc.).
    """

    needs_sandbox: bool

    def get_prompt(self, info: dict) -> Messages: ...
    def get_image(self, info: dict) -> str: ...
    def get_workdir(self, info: dict) -> str: ...
    def get_env_vars(self) -> dict[str, str]: ...
    async def setup(self, sandbox_client: Any, sandbox_id: str, state: State) -> None: ...
    async def evaluate(self, sandbox_client: Any, sandbox_id: str, state: State) -> float | dict[str, float]: ...
    def get_extra_tools(self) -> list: ...

    async def validate(self, sandbox_client: Any, sandbox_id: str, state: State) -> bool:
        """Verify this instance is solvable.

        Applies the known-correct solution and checks that evaluation
        passes.  Domain-specific: SWE applies the gold patch and runs
        tests, Lean compiles the ground-truth proof, Harbor runs solve.sh.
        Returns True if the instance is valid, False if broken.
        Tasks without a gold solution should return True (assume valid).
        """
        ...


# ---------------------------------------------------------------------------
# TaskSet — a collection of problem instances
# ---------------------------------------------------------------------------


class TaskSet:
    """A collection of Tasks backed by a ``TaskSpec``.

    Each row in the dataset is one Task (one problem instance).
    The TaskSpec provides the shared behavior (image, setup, evaluate).
    """

    def __init__(self, spec: TaskSpec, dataset: Any, name: str = ""):
        self.spec = spec
        self._dataset = dataset
        self.name = name

    @property
    def needs_sandbox(self) -> bool:
        return getattr(self.spec, "needs_sandbox", True)

    def get_dataset(self) -> Any:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    # -- TaskSpec delegation ------------------------------------------------

    def get_prompt(self, info: dict) -> Messages:
        return self.spec.get_prompt(info)

    def get_image(self, info: dict) -> str:
        return self.spec.get_image(info)

    def get_workdir(self, info: dict) -> str:
        return self.spec.get_workdir(info)

    def get_env_vars(self) -> dict[str, str]:
        return self.spec.get_env_vars()

    def get_extra_tools(self) -> list:
        if hasattr(self.spec, "get_extra_tools"):
            return self.spec.get_extra_tools() or []
        return []

    async def setup(self, sandbox_client: Any, sandbox_id: str, state: State) -> None:
        return await self.spec.setup(sandbox_client, sandbox_id, state)

    async def evaluate(self, sandbox_client: Any, sandbox_id: str, state: State) -> float | dict[str, float]:
        return await self.spec.evaluate(sandbox_client, sandbox_id, state)

    async def validate(self, sandbox_client: Any, sandbox_id: str, state: State) -> bool:
        return await self.spec.validate(sandbox_client, sandbox_id, state)

    # -- Combinators --------------------------------------------------------

    def filter(self, predicate: Callable[[dict], bool]) -> TaskSet:
        filtered = self._dataset.filter(predicate)
        return TaskSet(spec=self.spec, dataset=filtered, name=self.name)

    def take(self, n: int) -> TaskSet:
        sliced = self._dataset.select(range(min(n, len(self._dataset))))
        return TaskSet(spec=self.spec, dataset=sliced, name=self.name)

    async def validate_taskset(
        self,
        n: int | None = None,
        concurrency: int = 10,
        cpu_cores: int = 2,
        memory_gb: int = 4,
        disk_size_gb: int = 2,
        timeout_minutes: int = 15,
        test_timeout: int = 900,
    ) -> list[dict]:
        """Validate instances by applying gold solutions and checking evaluation.

        Creates sandboxes, runs ``spec.validate()`` on each instance, tears
        down sandboxes.  Returns a list of ``{index, valid, elapsed, error}``
        dicts.

        Parameters
        ----------
        n:
            Number of instances to validate.  None = all.
        concurrency:
            Max concurrent sandboxes.
        """
        import asyncio
        import logging
        import time

        from prime_sandboxes import CreateSandboxRequest
        from verifiers.utils.threaded_sandbox_client import ThreadedAsyncSandboxClient

        logger = logging.getLogger(__name__)
        client = ThreadedAsyncSandboxClient(max_workers=concurrency)
        sem = asyncio.Semaphore(concurrency)
        ds = self.get_dataset()
        total = min(n, len(ds)) if n else len(ds)

        async def run_background_job(state: dict, cmd: str, timeout: int, working_dir: str | None = None) -> Any:
            sid = state["sandbox_id"]
            job = await client.start_background_job(sid, cmd, working_dir=working_dir)
            for _ in range(0, timeout + 3, 3):
                status = await client.get_background_job(sid, job)
                if status.completed:
                    return status
                await asyncio.sleep(3)
            raise TimeoutError(f"Background job timed out after {timeout}s")

        async def validate_one(i: int) -> dict:
            row = ds[i]
            info = row["info"]
            state: dict = {"info": info, "_run_background_job": run_background_job}

            async with sem:
                image = self.spec.get_image(info)
                sb = await client.create(CreateSandboxRequest(
                    name=f"validate-{i}",
                    docker_image=image,
                    cpu_cores=cpu_cores,
                    memory_gb=memory_gb,
                    disk_size_gb=disk_size_gb,
                    timeout_minutes=timeout_minutes,
                ))
                state["sandbox_id"] = sb.id
                await client.wait_for_creation(sb.id, max_attempts=120)

                t0 = time.time()
                try:
                    await self.spec.setup(client, sb.id, state)
                    valid = await self.spec.validate(client, sb.id, state)
                    elapsed = time.time() - t0
                    logger.info(f"[{i}] valid={valid} ({elapsed:.0f}s)")
                    return {"index": i, "valid": valid, "elapsed": elapsed, "error": None}
                except Exception as e:
                    elapsed = time.time() - t0
                    logger.warning(f"[{i}] ERROR: {e} ({elapsed:.0f}s)")
                    return {"index": i, "valid": False, "elapsed": elapsed, "error": str(e)}
                finally:
                    await client.delete(sb.id)

        logger.info(f"Validating {total} instances from {self.name} (concurrency={concurrency})")
        t0 = time.time()
        results = await asyncio.gather(*[validate_one(i) for i in range(total)])
        elapsed = time.time() - t0

        passed = sum(1 for r in results if r["valid"])
        failed = sum(1 for r in results if not r["valid"])
        logger.info(f"Validation complete: {passed}/{total} valid, {failed} failed ({elapsed:.0f}s)")

        return results

    def __repr__(self) -> str:
        return f"TaskSet(name={self.name!r}, len={len(self)})"
