"""Task, TaskSet, SandboxTaskSet, and SandboxSpec — WHAT to solve.

A **Task** is a single, fully-bound problem instance.

A **TaskSet** is a collection of Tasks — no sandbox needed.
Subclass for math, QA, or any pure-LLM task.

A **SandboxTaskSet** extends TaskSet with sandbox capabilities.
Subclass for SWE, Lean, Harbor, or anything needing a sandbox.

A **SandboxSpec** describes sandbox requirements (image, CPU, memory, etc.).

::

    # No sandbox (math)
    class MathTaskSet(TaskSet):
        def get_instruction(self, info) -> str: ...
        async def evaluate(self, state) -> float: ...

    # With sandbox (SWE)
    class R2EGymTaskSet(SandboxTaskSet):
        def get_instruction(self, info) -> str: ...
        def get_sandbox_spec(self, info) -> SandboxSpec: ...
        async def evaluate(self, sandbox_client, sandbox_id, state) -> float: ...
"""

from __future__ import annotations

import importlib
import importlib.resources as resources
from dataclasses import dataclass
from importlib.abc import Traversable
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from verifiers.types import Messages, State


def _module_package_name(module: ModuleType) -> str | None:
    """Return the package name for a module, or None if not in a package."""
    if hasattr(module, "__path__"):
        return module.__name__
    package_name = getattr(module, "__package__", None)
    return package_name or None


def discover_sibling_dir(taskset_cls: type, dirname: str) -> Traversable | Path | None:
    """Find a sibling directory relative to a TaskSet's defining module.

    Looks for a directory named *dirname* next to the module that defines
    *taskset_cls*.  Works with installed packages (via ``importlib.resources``)
    and plain filesystem paths.

    Returns ``None`` if no such directory exists or is empty.
    """
    module = importlib.import_module(taskset_cls.__module__)

    package_name = _module_package_name(module)
    if package_name:
        try:
            candidate = resources.files(package_name) / dirname
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
        except Exception:
            pass

    module_file = getattr(module, "__file__", None)
    if module_file:
        candidate_path = Path(module_file).resolve().parent / dirname
        if candidate_path.is_dir() and any(candidate_path.iterdir()):
            return candidate_path
    return None


@dataclass
class SandboxSpec:
    """Sandbox requirements for a task instance."""

    image: str = "python:3.11-slim"
    cpu_cores: int = 4
    memory_gb: int = 4
    disk_size_gb: int = 10
    gpu_count: int = 0
    gpu_type: str | None = None
    timeout_minutes: int = 60


class Task:
    """A single, fully-bound problem instance.

    Created via ``TaskSet[i]``::

        task = taskset[0]
        task.prompt          # Messages
        task.sandbox_spec    # SandboxSpec or None
        task.info            # raw metadata dict
    """

    def __init__(
        self, taskset: TaskSet, prompt: Messages, info: dict, answer: str = ""
    ):
        self._taskset = taskset
        self.prompt = prompt
        self.info = info
        self.answer = answer

    @property
    def sandbox_spec(self) -> SandboxSpec | None:
        if isinstance(self._taskset, SandboxTaskSet):
            return self._taskset.get_sandbox_spec(self.info)
        return None

    @property
    def image(self) -> str | None:
        spec = self.sandbox_spec
        return spec.image if spec else None

    @property
    def workdir(self) -> str:
        if isinstance(self._taskset, SandboxTaskSet):
            return self._taskset.get_workdir(self.info)
        return "/app"

    def __repr__(self) -> str:
        spec = self.sandbox_spec
        sandbox_info = f"image={spec.image!r}" if spec else "no sandbox"
        return f"Task(taskset={self._taskset.name!r}, {sandbox_info})"


class TaskSet:
    """A collection of Tasks. No sandbox needed.

    Subclass for pure-LLM tasks (math, QA, reasoning).

    Override:
        get_instruction(info) -> str
        get_rubric() -> vf.Rubric
        validate_instance(state) -> bool  (optional)
    """

    def __init__(self, dataset: Any, name: str = ""):
        self._dataset = dataset
        self.name = name

    # -- Override these ------------------------------------------------------

    def get_instruction(self, info: dict) -> str:
        """Plain text instruction for the agent.

        This is the primary method to override. Used for:
        - The instruction file written to the sandbox (e.g. /task/instruction.md)
        - Building Task.prompt (wrapped as a user message)
        """
        raise NotImplementedError

    def get_rubric(self) -> Any:
        """Return the rubric for scoring this taskset.

        The rubric owns all scoring logic — running tests, reading files,
        computing rewards.  Use ``keep_sandbox_for_scoring=True`` on the
        environment so the sandbox stays alive for the rubric.
        The rubric should have a ``@vf.cleanup`` handler to delete the
        sandbox when done.
        """
        raise NotImplementedError

    def get_sandbox_spec(self, info: dict) -> SandboxSpec | None:
        """Sandbox requirements. Return None if no sandbox needed (default)."""
        return None

    def get_workdir(self, info: dict) -> str:
        return "/app"

    def get_env_vars(self) -> dict[str, str]:
        return {}

    def get_skills_dir(self) -> Traversable | Path | None:
        """Return the taskset's skills directory, or None.

        By default, auto-discovers a sibling ``skills/`` directory next
        to the module that defines this taskset class.  Override to
        disable (return ``None``) or point to a different location.
        """
        return discover_sibling_dir(type(self), "skills")

    def get_upload_dirs(self) -> dict[str, Traversable | Path]:
        """Directories to upload to the sandbox before agent install.

        Returns a mapping of ``{logical_name: local_source}`` where
        *logical_name* is a short label (e.g. ``"skills"``) and
        *local_source* is a ``Path`` or ``importlib.abc.Traversable``
        pointing to a local directory.

        The *logical_name* is resolved to a sandbox path by the harness's
        ``upload_dir_mapping``.  Only directories whose logical name
        appears in the mapping are uploaded.

        By default, includes the skills directory from
        :meth:`get_skills_dir` under the ``"skills"`` key.  Override to
        add additional directories or disable skills upload.
        """
        dirs: dict[str, Traversable | Path] = {}
        skills = self.get_skills_dir()
        if skills is not None:
            dirs["skills"] = skills
        return dirs

    async def setup(self, state: State) -> None:
        pass

    async def validate_instance(self, state: State) -> bool:
        return True

    # -- Public API ----------------------------------------------------------

    def get_dataset(self) -> Any:
        """Return the dataset with a ``prompt`` column built from get_instruction.

        This pre-builds the prompt so the base Environment doesn't need a
        ``question`` column.
        """
        ds = self._dataset
        if "prompt" not in ds.column_names:

            def add_prompt(row: dict) -> dict:
                info = row.get("info") or {}
                instruction = self.get_instruction(info)
                return {"prompt": [{"role": "user", "content": instruction}]}

            ds = ds.map(add_prompt)
        return ds

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> Task:
        row = self._dataset[i]
        info = row.get("info") or {}
        from verifiers.types import UserMessage

        instruction = self.get_instruction(info)
        return Task(
            taskset=self,
            prompt=[UserMessage(content=instruction)],
            info=info,
            answer=row.get("answer", ""),
        )

    # -- Combinators ---------------------------------------------------------

    def filter(self, predicate: Callable[[dict], bool]) -> TaskSet:
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone._dataset = self._dataset.filter(predicate)
        return clone

    def take(self, n: int) -> TaskSet:
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone._dataset = self._dataset.select(range(min(n, len(self._dataset))))
        return clone

    # -- Validation ----------------------------------------------------------

    async def validate(
        self,
        n: int | None = None,
        concurrency: int = 10,
    ) -> list[dict]:
        """Validate instances. For sandbox tasks, creates sandboxes and runs validate_instance."""
        import asyncio
        import logging
        import time

        logger = logging.getLogger(__name__)
        ds = self.get_dataset()
        total = min(n, len(ds)) if n is not None else len(ds)
        is_sandbox = isinstance(self, SandboxTaskSet)

        if not is_sandbox:
            # No sandbox needed — run validate_instance concurrently
            sem = asyncio.Semaphore(concurrency)

            async def _validate_simple(i: int) -> dict:
                row = ds[i]
                state: State = {  # type: ignore[assignment]
                    "info": row.get("info") or {},
                    "answer": row.get("answer", ""),
                }
                async with sem:
                    t0 = time.time()
                    try:
                        valid = await self.validate_instance(state)
                        return {
                            "index": i,
                            "valid": valid,
                            "elapsed": time.time() - t0,
                            "error": None,
                        }
                    except Exception as e:
                        return {
                            "index": i,
                            "valid": False,
                            "elapsed": time.time() - t0,
                            "error": str(e),
                        }

            return await asyncio.gather(*[_validate_simple(i) for i in range(total)])

        # Sandbox path — lazy imports only needed here
        from prime_sandboxes import CreateSandboxRequest
        from verifiers.utils.threaded_sandbox_client import ThreadedAsyncSandboxClient

        client = ThreadedAsyncSandboxClient(
            max_workers=min(max(1, concurrency // 8), 50)
        )
        sem = asyncio.Semaphore(concurrency)
        assert isinstance(self, SandboxTaskSet)

        async def validate_one(i: int) -> dict:
            row = ds[i]
            info = row.get("info") or {}
            state: State = {  # type: ignore[assignment]
                "info": info,
                "answer": row.get("answer", ""),
            }

            async with sem:
                spec = self.get_sandbox_spec(info)
                sb = None
                t0 = time.time()
                try:
                    sb = await client.create(
                        CreateSandboxRequest(
                            name=f"validate-{i}",
                            docker_image=spec.image,
                            cpu_cores=spec.cpu_cores,
                            memory_gb=spec.memory_gb,
                            disk_size_gb=spec.disk_size_gb,
                            gpu_count=spec.gpu_count,
                            gpu_type=spec.gpu_type,
                            vm=spec.gpu_count > 0,
                            timeout_minutes=spec.timeout_minutes,
                        )
                    )
                    state["sandbox_id"] = sb.id
                    state["sandbox_client"] = client
                    state["test_timeout"] = spec.timeout_minutes * 60
                    await client.wait_for_creation(sb.id, max_attempts=120)
                    await self.setup(state)
                    valid = await self.validate_instance(state)
                    elapsed = time.time() - t0
                    logger.info(f"[{i}] valid={valid} ({elapsed:.0f}s)")
                    return {
                        "index": i,
                        "valid": valid,
                        "elapsed": elapsed,
                        "error": None,
                    }
                except Exception as e:
                    elapsed = time.time() - t0
                    logger.warning(f"[{i}] ERROR: {e} ({elapsed:.0f}s)")
                    return {
                        "index": i,
                        "valid": False,
                        "elapsed": elapsed,
                        "error": str(e),
                    }
                finally:
                    if sb is not None:
                        await client.delete(sb.id)

        logger.info(
            f"Validating {total} instances from {self.name} (concurrency={concurrency})"
        )
        t0 = time.time()
        try:
            results = await asyncio.gather(*[validate_one(i) for i in range(total)])
        finally:
            client.teardown()
        elapsed = time.time() - t0
        passed = sum(1 for r in results if r["valid"])
        rate = passed / total if total else 0
        logger.info(f"Validation: {passed}/{total} valid ({rate:.1%}, {elapsed:.0f}s)")
        return results

    def __repr__(self) -> str:
        return f"TaskSet(name={self.name!r}, len={len(self)})"


class SandboxTaskSet(TaskSet):
    """A TaskSet that requires sandboxes.

    Subclass for SWE, Lean, Harbor, or anything needing a sandbox.

    Override:
        get_instruction(info) -> str
        get_sandbox_spec(info) -> SandboxSpec
        get_rubric() -> vf.Rubric
        setup(state) -> None
        validate_instance(state) -> bool  (optional)
        get_workdir(info) -> str  (optional, default "/app")
        get_env_vars() -> dict  (optional)
        get_skills_dir() -> Path | None  (optional, auto-discovered by default)
        get_upload_dirs() -> dict  (optional, dirs to upload before install)

    All methods receive ``state`` which contains sandbox context:
    - ``state["sandbox_client"]`` — the async sandbox client
    - ``state["sandbox_id"]`` — current sandbox ID
    - ``state["test_timeout"]`` — evaluation timeout (default 900)
    """

    default_workdir: str = "/app"
    """Default working directory. Expose to harness via ``taskset.default_workdir``."""

    # -- Override these ------------------------------------------------------

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        raise NotImplementedError

    def get_workdir(self, info: dict) -> str:
        return self.default_workdir

    def get_env_vars(self) -> dict[str, str]:
        return {}

    async def setup(self, state: State) -> None:
        pass

    async def validate_instance(self, state: State) -> bool:
        return True
