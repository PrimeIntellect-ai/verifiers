"""Task and TaskSet — WHAT to solve.

A **Task** is a single, fully self-contained problem instance.  It knows
its prompt, its docker image, how to set up a sandbox, and how to
evaluate a solution.  No arguments needed — everything is bound.

A **TaskSet** is a collection of Tasks backed by an HF Dataset.
It defines the shared behavior (image resolution, setup, evaluation)
and produces fully-bound Task instances.

::

    from swe_tasksets import R2ETaskSet

    taskset = R2ETaskSet()               # 4578 tasks
    task = taskset[0]                     # one fully-bound task
    task.prompt                           # the problem statement
    task.image                            # docker image for THIS instance
    await task.setup(sandbox_client, sandbox_id)
    await task.evaluate(sandbox_client, sandbox_id, state)

    # Slice, filter, validate
    small = taskset.take(50)
    results = await taskset.validate(concurrency=5)

To create a new task type, subclass TaskSet::

    class MyTaskSet(TaskSet):
        needs_sandbox = True

        def _get_prompt(self, info: dict) -> Messages: ...
        def _get_image(self, info: dict) -> str: ...
        def _get_workdir(self, info: dict) -> str: ...
        def _get_env_vars(self) -> dict[str, str]: ...
        async def _setup(self, sandbox_client, sandbox_id, state) -> None: ...
        async def _evaluate(self, sandbox_client, sandbox_id, state) -> float: ...
        async def _validate(self, sandbox_client, sandbox_id, state) -> bool: ...
"""

from __future__ import annotations

from typing import Any, Callable

from verifiers.types import Messages, State


class Task:
    """A single, fully-bound problem instance.

    All methods are pre-bound to this instance's data — no ``info`` arg needed.

    Created via ``TaskSet[i]``, not directly::

        task = taskset[0]
        task.prompt         # Messages
        task.image          # docker image string
        task.info           # raw metadata dict
        task.answer         # ground truth (if available)
    """

    def __init__(
        self,
        taskset: TaskSet,
        prompt: Messages,
        info: dict,
        answer: str = "",
    ):
        self._taskset = taskset
        self.prompt = prompt
        self.info = info
        self.answer = answer

    @property
    def image(self) -> str:
        return self._taskset._get_image(self.info)

    @property
    def workdir(self) -> str:
        return self._taskset._get_workdir(self.info)

    async def setup(self, sandbox_client: Any, sandbox_id: str, state: State | None = None) -> None:
        s = state if state is not None else {"info": self.info}
        return await self._taskset._setup(sandbox_client, sandbox_id, s)

    async def evaluate(self, sandbox_client: Any, sandbox_id: str, state: State | None = None) -> float:
        s = state if state is not None else {"info": self.info}
        return await self._taskset._evaluate(sandbox_client, sandbox_id, s)

    async def validate(self, sandbox_client: Any, sandbox_id: str, state: State | None = None) -> bool:
        s = state if state is not None else {"info": self.info}
        return await self._taskset._validate(sandbox_client, sandbox_id, s)

    def __repr__(self) -> str:
        return f"Task(taskset={self._taskset.name!r}, info_keys={list(self.info.keys())})"


class TaskSet:
    """A collection of Tasks with shared behavior.

    Subclass this to create a new task type.  Override the ``_``-prefixed
    methods to define how tasks are handled.  The public API (used by
    ComposableEnv) delegates to these.

    Parameters
    ----------
    dataset:
        An HF ``Dataset`` with at least ``info`` and ``answer`` columns.
    name:
        Human-readable name for this task set.
    """

    needs_sandbox: bool = True

    def __init__(self, dataset: Any, name: str = ""):
        self._dataset = dataset
        self.name = name

    # -- Override these in subclasses ----------------------------------------

    def _get_prompt(self, info: dict) -> Messages:
        raise NotImplementedError

    def _get_image(self, info: dict) -> str:
        raise NotImplementedError

    def _get_workdir(self, info: dict) -> str:
        return "/app"

    def _get_env_vars(self) -> dict[str, str]:
        return {}

    async def _setup(self, sandbox_client: Any, sandbox_id: str, state: State) -> None:
        pass

    async def _evaluate(self, sandbox_client: Any, sandbox_id: str, state: State) -> float:
        return 0.0

    async def _validate(self, sandbox_client: Any, sandbox_id: str, state: State) -> bool:
        return True

    def _get_extra_tools(self) -> list:
        return []

    # -- Public API (used by ComposableEnv) ----------------------------------

    def get_dataset(self) -> Any:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> Task:
        row = self._dataset[i]
        info = row.get("info") or {}
        return Task(
            taskset=self,
            prompt=self._get_prompt(info),
            info=info,
            answer=row.get("answer", ""),
        )

    # -- Delegators for ComposableEnv (pass info from state) -----------------

    def get_prompt(self, info: dict) -> Messages:
        return self._get_prompt(info)

    def get_image(self, info: dict) -> str:
        return self._get_image(info)

    def get_workdir(self, info: dict) -> str:
        return self._get_workdir(info)

    def get_env_vars(self) -> dict[str, str]:
        return self._get_env_vars()

    def get_extra_tools(self) -> list:
        return self._get_extra_tools()

    async def setup(self, sandbox_client: Any, sandbox_id: str, state: State) -> None:
        return await self._setup(sandbox_client, sandbox_id, state)

    async def evaluate(self, sandbox_client: Any, sandbox_id: str, state: State) -> float:
        return await self._evaluate(sandbox_client, sandbox_id, state)

    async def validate_instance(self, sandbox_client: Any, sandbox_id: str, state: State) -> bool:
        return await self._validate(sandbox_client, sandbox_id, state)

    # -- Combinators --------------------------------------------------------

    def filter(self, predicate: Callable[[dict], bool]) -> TaskSet:
        filtered = self._dataset.filter(predicate)
        new = self._clone(filtered)
        return new

    def take(self, n: int) -> TaskSet:
        sliced = self._dataset.select(range(min(n, len(self._dataset))))
        return self._clone(sliced)

    def _clone(self, dataset: Any) -> TaskSet:
        """Create a copy of this TaskSet with a different dataset."""
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone._dataset = dataset
        return clone

    # -- Validation ---------------------------------------------------------

    async def validate(
        self,
        n: int | None = None,
        concurrency: int = 10,
        cpu_cores: int = 2,
        memory_gb: int = 4,
        disk_size_gb: int = 2,
        timeout_minutes: int = 15,
    ) -> list[dict]:
        """Validate instances by applying gold solutions and checking evaluation.

        Creates sandboxes, runs ``_validate()`` on each instance, tears
        down sandboxes.

        Returns a list of ``{"index": int, "valid": bool, "elapsed": float, "error": str|None}``.

        Example::

            results = await taskset.take(10).validate(concurrency=5)
        """
        import asyncio
        import logging
        import time

        from prime_sandboxes import CreateSandboxRequest
        from verifiers.utils.threaded_sandbox_client import ThreadedAsyncSandboxClient

        logger = logging.getLogger(__name__)
        client = ThreadedAsyncSandboxClient(max_workers=min(max(1, concurrency // 8), 50))
        sem = asyncio.Semaphore(concurrency)
        ds = self.get_dataset()
        total = min(n, len(ds)) if n else len(ds)

        async def validate_one(i: int) -> dict:
            row = ds[i]
            info = row["info"]
            state: dict = {"info": info}

            async with sem:
                image = self._get_image(info)
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
                    await self._setup(client, sb.id, state)
                    valid = await self._validate(client, sb.id, state)
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
        logger.info(f"Validation: {passed}/{total} valid ({elapsed:.0f}s)")
        return results

    def __repr__(self) -> str:
        return f"TaskSet(name={self.name!r}, len={len(self)})"
