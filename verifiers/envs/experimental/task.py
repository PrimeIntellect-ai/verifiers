"""TaskSpec and TaskSet — WHAT to solve.

Overview
--------

A **TaskSpec** defines the shared behavior for a problem type: docker image,
sandbox setup, prompt construction, and evaluation logic.  You write one
TaskSpec per domain (e.g. ``R2EGymTask``, ``LeanTask``, ``MathTask``).

A **TaskSet** is a collection of problem instances backed by a TaskSpec.
This is what you hand to ``ComposableEnv`` or a training loop.

Quick start::

    from swe_tasksets import R2ETaskSet
    from lean_tasksets import LeanTaskSet

    # Create tasksets
    swe = R2ETaskSet()                          # 4578 SWE instances
    lean = LeanTaskSet("minif2f")               # 244 Lean theorems

    # Access individual tasks
    task = swe[0]                               # one Task instance
    task.prompt                                 # the problem statement
    task.info                                   # metadata
    task.get_image()                            # docker image for this instance

    # Slice them
    small = swe.take(50)                        # first 50
    filtered = swe.filter(lambda ex: ...)       # custom filter

    # Validate (are the gold solutions correct?)
    results = await swe.take(10).validate_taskset(concurrency=5)
    # → [{index: 0, valid: True, elapsed: 12.0}, ...]

    # Train with an agent
    env = ComposableEnv(taskset=swe, run_command="opencode run ...")

TaskSpec protocol
-----------------

To create a new task type, implement these methods::

    class MyTaskSpec:
        needs_sandbox = True  # False for pure-LLM tasks (math, QA)

        def get_prompt(self, info: dict) -> Messages:
            '''What the agent sees.'''

        def get_image(self, info: dict) -> str:
            '''Docker image for this instance.'''

        def get_workdir(self, info: dict) -> str:
            '''Working directory inside the sandbox.'''

        def get_env_vars(self) -> dict[str, str]:
            '''Environment variables for the sandbox.'''

        async def setup(self, sandbox_client, sandbox_id, state) -> None:
            '''Prepare the sandbox (install deps, write files, etc).'''

        async def evaluate(self, sandbox_client, sandbox_id, state) -> float:
            '''Score the result (run tests, check answer, etc). Returns 0.0-1.0.'''

        async def validate(self, sandbox_client, sandbox_id, state) -> bool:
            '''Verify this instance is solvable (apply gold solution, evaluate).'''

        def get_extra_tools(self) -> list:
            '''Domain-specific tools (e.g. compile_proof for Lean).'''

TaskSet
-------

A TaskSet wraps a TaskSpec + an HF Dataset::

    taskset = TaskSet(spec=my_spec, dataset=my_dataset, name="my-tasks")

    len(taskset)                    # number of instances
    taskset.get_dataset()           # the HF Dataset
    taskset.spec                    # the TaskSpec

    # Combinators
    taskset.take(100)               # first 100 instances
    taskset.filter(predicate)       # filter by predicate

    # Validation
    await taskset.validate_taskset(n=10, concurrency=5)
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
    One TaskSpec per domain (R2EGymTask, LeanTask, etc.).
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
# Task — one problem instance
# ---------------------------------------------------------------------------


class Task:
    """A single problem instance: data + a reference to its TaskSpec.

    Usually created via ``TaskSet[i]`` rather than directly::

        taskset = R2ETaskSet()
        task = taskset[0]
        task.prompt          # Messages
        task.info            # metadata dict
        task.spec            # the TaskSpec that handles this type
        task.get_image()     # delegates to spec
    """

    def __init__(self, spec: TaskSpec, prompt: Messages, info: dict, answer: str = ""):
        self.spec = spec
        self.prompt = prompt
        self.info = info
        self.answer = answer

    def get_image(self) -> str:
        return self.spec.get_image(self.info)

    def get_workdir(self) -> str:
        return self.spec.get_workdir(self.info)

    def __repr__(self) -> str:
        return f"Task(spec={type(self.spec).__name__}, info_keys={list(self.info.keys())})"


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
        """Return the underlying HF Dataset."""
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, i: int) -> Task:
        """Return the i-th Task instance."""
        row = self._dataset[i]
        prompt = self.spec.get_prompt(row.get("info") or {})
        return Task(
            spec=self.spec,
            prompt=prompt,
            info=row.get("info") or {},
            answer=row.get("answer", ""),
        )

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
        """Return a new TaskSet with only examples matching *predicate*."""
        filtered = self._dataset.filter(predicate)
        return TaskSet(spec=self.spec, dataset=filtered, name=self.name)

    def take(self, n: int) -> TaskSet:
        """Return a new TaskSet with the first *n* examples."""
        sliced = self._dataset.select(range(min(n, len(self._dataset))))
        return TaskSet(spec=self.spec, dataset=sliced, name=self.name)

    # -- Validation ---------------------------------------------------------

    async def validate_taskset(
        self,
        n: int | None = None,
        concurrency: int = 10,
        cpu_cores: int = 2,
        memory_gb: int = 4,
        disk_size_gb: int = 2,
        timeout_minutes: int = 15,
    ) -> list[dict]:
        """Validate instances by applying gold solutions and checking evaluation.

        Creates sandboxes, runs ``spec.validate()`` on each instance, tears
        down sandboxes.  Returns a list of dicts::

            [{"index": 0, "valid": True, "elapsed": 12.3, "error": None}, ...]

        Example::

            taskset = R2ETaskSet().take(100)
            results = await taskset.validate_taskset(concurrency=10)
            valid = [r for r in results if r["valid"]]
            print(f"{len(valid)}/{len(results)} instances are solvable")

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
