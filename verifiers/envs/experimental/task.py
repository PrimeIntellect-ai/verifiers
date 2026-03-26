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
    async def apply_gold_patch(self, sandbox_client: Any, sandbox_id: str, state: State) -> None: ...


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

    async def apply_gold_patch(self, sandbox_client: Any, sandbox_id: str, state: State) -> None:
        return await self.spec.apply_gold_patch(sandbox_client, sandbox_id, state)

    # -- Combinators --------------------------------------------------------

    def filter(self, predicate: Callable[[dict], bool]) -> TaskSet:
        filtered = self._dataset.filter(predicate)
        return TaskSet(spec=self.spec, dataset=filtered, name=self.name)

    def take(self, n: int) -> TaskSet:
        sliced = self._dataset.select(range(min(n, len(self._dataset))))
        return TaskSet(spec=self.spec, dataset=sliced, name=self.name)

    def __repr__(self) -> str:
        return f"TaskSet(name={self.name!r}, len={len(self)})"
