"""Task and TaskSet — WHAT to solve.

A **Task** is pure data + hooks for a single problem type: docker image,
sandbox setup, prompt construction, and evaluation.

A **TaskSet** is a collection of problem instances backed by a Task.  It
produces an HF ``Dataset`` and delegates per-instance methods (image,
setup, evaluate) to the underlying Task.

::

    from tasksets.swe import R2ETaskSet
    from tasksets.lean import LeanTaskSet

    r2e = R2ETaskSet()
    lean = LeanTaskSet("minif2f")
    subset = r2e.filter(lambda ex: ...).take(100)
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
    The Task never drives execution — that is the agent's job.
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
        ComposableEnv injects these into the agent automatically.
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
        """Return a new TaskSet with only examples matching *predicate*."""
        filtered = self._dataset.filter(predicate)
        return TaskSet(task=self.task, dataset=filtered, name=self.name)

    def take(self, n: int) -> TaskSet:
        """Return a new TaskSet with the first *n* examples."""
        sliced = self._dataset.select(range(min(n, len(self._dataset))))
        return TaskSet(task=self.task, dataset=sliced, name=self.name)

    def __repr__(self) -> str:
        return f"TaskSet(name={self.name!r}, len={len(self)})"
