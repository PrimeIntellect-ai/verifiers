"""Task and TaskSet — WHAT to solve.

A **Task** is pure data + hooks for a single problem type: docker image,
sandbox setup, prompt construction, and evaluation.

A **TaskSet** is a collection of problem instances backed by a Task.  It
produces an HF ``Dataset`` and delegates per-instance methods (image,
setup, evaluate) to the underlying Task.

TaskSets can come from different sources:

* HuggingFace datasets (uniform — all instances share image/setup/eval)
* Harbor directories (heterogeneous — each task has its own image/tests)
* Programmatic generation (Absolute Zero — generated at runtime)
* Merging multiple TaskSets

::

    # Uniform (all instances share same image/eval)
    taskset = SweTaskSet(R2EGymTask())

    # Heterogeneous (each task has own image/tests)
    taskset = HarborTaskSet("path/to/tasks/")

    # Mix
    combined = TaskSet.merge(swe_tasks, lean_tasks)
    subset = combined.filter(lambda ex: ex["info"].get("difficulty") == "easy").take(50)
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from verifiers.types import Messages, State


# ---------------------------------------------------------------------------
# Task protocol — one problem *type*
# ---------------------------------------------------------------------------


@runtime_checkable
class Task(Protocol):
    """Protocol describing WHAT to solve.

    Implementations provide the per-instance docker image, sandbox
    preparation, prompt construction, and evaluation logic.
    The Task never drives execution — that is the Agent's job.
    """

    needs_sandbox: bool
    """Whether this task requires a sandbox (docker image, setup, etc.)."""

    def get_prompt(self, info: dict) -> Messages:
        """Build the prompt messages the agent will see."""
        ...

    def get_image(self, info: dict) -> str:
        """Return the fully-qualified Docker image for this instance.
        Only called when ``needs_sandbox`` is True."""
        ...

    def get_workdir(self, info: dict) -> str:
        """Return the working directory inside the sandbox."""
        ...

    def get_env_vars(self) -> dict[str, str]:
        """Return task-specific environment variables."""
        ...

    async def setup(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> None:
        """Prepare the sandbox after creation."""
        ...

    async def evaluate(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> float | dict[str, float]:
        """Score the result.  Returns scalar or per-role dict."""
        ...

    def get_extra_tools(self) -> list:
        """Return domain-specific tools the agent may use.

        Each tool is a ``(callable, args_to_skip)`` tuple or a plain callable.
        ComposableEnv injects these into ReActAgent automatically.
        """
        ...

    async def apply_gold_patch(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> None:
        """Apply the known-correct solution.  Optional."""
        ...


# ---------------------------------------------------------------------------
# TaskSet — a collection of problem instances
# ---------------------------------------------------------------------------


class TaskSet:
    """A collection of problem instances backed by a ``Task``.

    Wraps an HF ``Dataset`` and a ``Task`` that knows how to handle each
    instance.  ``ComposableEnv`` accepts a ``TaskSet`` directly.

    Parameters
    ----------
    task:
        The Task implementation that handles setup/evaluate/etc.
    dataset:
        An HF ``Dataset`` with at least ``question``, ``info``, ``answer``.
    name:
        Human-readable name for this task set.
    """

    def __init__(self, task: Task, dataset: Any, name: str = ""):
        self.task = task
        self._dataset = dataset
        self.name = name

    @property
    def needs_sandbox(self) -> bool:
        return getattr(self.task, "needs_sandbox", True)

    # -- Dataset access -----------------------------------------------------

    def get_dataset(self) -> Any:
        """Return the HF Dataset."""
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    # -- Task protocol delegation -------------------------------------------
    # These delegate to self.task so TaskSet can be used anywhere Task is.

    def get_prompt(self, info: dict) -> Messages:
        return self.task.get_prompt(info)

    def get_image(self, info: dict) -> str:
        return self.task.get_image(info)

    def get_workdir(self, info: dict) -> str:
        return self.task.get_workdir(info)

    def get_env_vars(self) -> dict[str, str]:
        return self.task.get_env_vars()

    def get_extra_tools(self) -> list:
        if hasattr(self.task, "get_extra_tools"):
            return self.task.get_extra_tools() or []
        return []

    async def setup(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> None:
        return await self.task.setup(sandbox_client, sandbox_id, state)

    async def evaluate(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> float | dict[str, float]:
        return await self.task.evaluate(sandbox_client, sandbox_id, state)

    async def apply_gold_patch(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> None:
        return await self.task.apply_gold_patch(sandbox_client, sandbox_id, state)

    # -- Combinators --------------------------------------------------------

    def filter(self, predicate: Callable[[dict], bool]) -> TaskSet:
        """Return a new TaskSet with only examples matching *predicate*.

        *predicate* receives a single dataset row (dict) and returns bool.
        """
        filtered = self._dataset.filter(predicate)
        return TaskSet(task=self.task, dataset=filtered, name=self.name)

    def take(self, n: int) -> TaskSet:
        """Return a new TaskSet with the first *n* examples."""
        sliced = self._dataset.select(range(min(n, len(self._dataset))))
        return TaskSet(task=self.task, dataset=sliced, name=self.name)

    @staticmethod
    def merge(*tasksets: TaskSet) -> MergedTaskSet:
        """Merge multiple TaskSets into one.

        Each instance remembers which Task it came from, so per-instance
        image/setup/evaluate still work correctly even across heterogeneous
        sources.
        """
        return MergedTaskSet(list(tasksets))

    def __repr__(self) -> str:
        return f"TaskSet(name={self.name!r}, len={len(self)})"


class MergedTaskSet:
    """A TaskSet formed by concatenating multiple TaskSets.

    Each instance carries a ``_taskset_idx`` in its ``info`` dict so that
    per-instance methods (image, setup, evaluate) can be routed to the
    correct underlying Task.
    """

    def __init__(self, tasksets: list[TaskSet]):
        self._tasksets = tasksets
        self._task_map: dict[int, Task] = {}
        self._dataset: Any = None
        self._build()

    @property
    def needs_sandbox(self) -> bool:
        return any(ts.needs_sandbox for ts in self._tasksets)

    def _build(self) -> None:
        from datasets import concatenate_datasets

        tagged_datasets = []
        for idx, ts in enumerate(self._tasksets):
            self._task_map[idx] = ts.task
            ds = ts.get_dataset()
            # Tag each row with the taskset index
            ds = ds.map(
                lambda ex, i=idx: {
                    "info": {**(ex.get("info") or {}), "_taskset_idx": i}
                }
            )
            tagged_datasets.append(ds)
        self._dataset = concatenate_datasets(tagged_datasets)

    def _resolve_task(self, info: dict) -> Task:
        idx = info.get("_taskset_idx", 0)
        return self._task_map[idx]

    # -- Dataset access -----------------------------------------------------

    def get_dataset(self) -> Any:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    # -- Task protocol delegation (routes per-instance) ---------------------

    def get_prompt(self, info: dict) -> Messages:
        return self._resolve_task(info).get_prompt(info)

    def get_image(self, info: dict) -> str:
        return self._resolve_task(info).get_image(info)

    def get_workdir(self, info: dict) -> str:
        return self._resolve_task(info).get_workdir(info)

    def get_env_vars(self) -> dict[str, str]:
        # Merged tasksets don't have a single set of env vars
        return {}

    def get_extra_tools(self) -> list:
        # Can't resolve statically for merged sets — tools vary per instance
        return []

    async def setup(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> None:
        info = state.get("info") or {}
        return await self._resolve_task(info).setup(sandbox_client, sandbox_id, state)

    async def evaluate(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> float | dict[str, float]:
        info = state.get("info") or {}
        return await self._resolve_task(info).evaluate(sandbox_client, sandbox_id, state)

    async def apply_gold_patch(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> None:
        info = state.get("info") or {}
        return await self._resolve_task(info).apply_gold_patch(sandbox_client, sandbox_id, state)

    # -- Combinators --------------------------------------------------------

    def filter(self, predicate: Callable[[dict], bool]) -> MergedTaskSet:
        result = MergedTaskSet.__new__(MergedTaskSet)
        result._tasksets = self._tasksets
        result._task_map = self._task_map
        result._dataset = self._dataset.filter(predicate)
        return result

    def take(self, n: int) -> MergedTaskSet:
        result = MergedTaskSet.__new__(MergedTaskSet)
        result._tasksets = self._tasksets
        result._task_map = self._task_map
        result._dataset = self._dataset.select(range(min(n, len(self._dataset))))
        return result

    def __repr__(self) -> str:
        names = [ts.name for ts in self._tasksets]
        return f"MergedTaskSet(sources={names}, len={len(self)})"
