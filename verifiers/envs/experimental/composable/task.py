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

from verifiers.envs.experimental.composable._filter import _resolve_filter_fn
from verifiers.types import DatasetBuilder, Messages, State


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
    # If None, lifetime is derived by SandboxMixin.compute_sandbox_timeout_minutes.
    timeout_minutes: int | None = None


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

    def __init__(
        self,
        dataset: Any | DatasetBuilder,
        name: str = "",
        filter_fn: str | None = None,
    ):
        """
        Args:
            dataset: The dataset backing this taskset, or a ``DatasetBuilder``
                (zero-arg callable returning the dataset).
            name: Human-readable taskset name.
            filter_fn: Optional Python expression string (e.g. a lambda) that
                evaluates to a ``Callable[[dict], bool]`` and is applied to
                the post-``_process_example`` rows of ``dataset`` (i.e. the
                ``{"question", "info", "answer", ...}`` shape published by
                concrete tasksets). Rows for which it returns truthy are
                kept. The string is ``eval()``'d with restricted builtins,
                but it is still ``eval`` of user input — intended for local
                ``vf-eval`` runs, not untrusted inputs.
        """
        self.name = name
        # Cache the raw expression (not the callable) for reproducibility /
        # debugging; the resolved predicate isn't pickle-safe and the
        # realized dataset already carries the filtered state.
        self._filter_fn_src = filter_fn
        self.dataset_source: DatasetBuilder = (
            dataset if callable(dataset) else (lambda ds=dataset: ds)
        )
        self._built_dataset: Any = None

    @property
    def dataset(self) -> Any:
        if self._built_dataset is None:
            ds = self.dataset_source()
            if self._filter_fn_src is not None:
                ds = ds.filter(_resolve_filter_fn(self._filter_fn_src))
            self._built_dataset = ds
        return self._built_dataset

    @dataset.setter
    def dataset(self, value: Any) -> None:
        self._built_dataset = value

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
        ds = self.dataset
        if "prompt" not in ds.column_names:

            def add_prompt(row: dict) -> dict:
                info = row.get("info") or {}
                instruction = self.get_instruction(info)
                return {"prompt": [{"role": "user", "content": instruction}]}

            ds = ds.map(add_prompt)
        return ds

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> Task:
        row = self.dataset[i]
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
        clone.dataset = self.dataset.filter(predicate)
        return clone

    def take(self, n: int) -> TaskSet:
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.dataset = self.dataset.select(range(min(n, len(self.dataset))))
        return clone

    # -- Validation ----------------------------------------------------------

    async def validate(
        self,
        n: int | None = None,
        concurrency: int = 10,
        *,
        out_path: str | Path | None = None,
        max_retries: int = 0,
        resume: bool = False,
        test_output_tail_chars: int = 2000,
    ) -> list[dict]:
        """Validate instances with streaming progress and crash-safe output.

        Results stream via ``asyncio.as_completed`` and — if ``out_path`` is
        given — each row is appended to a JSONL file as soon as it finishes,
        so a crash or Ctrl-C keeps partial work. Progress is shown via tqdm
        with a live pass-rate.

        Parameters
        ----------
        n:
            Optional cap on how many instances to validate (first ``n`` rows).
        concurrency:
            Max number of in-flight instances (and, for sandbox tasksets,
            max live sandboxes) at a time.
        out_path:
            Optional path to a JSONL file. Each completed row is appended
            immediately (off the event loop via ``asyncio.to_thread``).
        max_retries:
            Number of times to retry an instance on ``vf.InfraError``
            (sandbox create/exec failures, tunnel drops). Each retry uses a
            fresh sandbox. Bumps the ``attempts`` field on the result.
        resume:
            If True and ``out_path`` exists, indices already recorded in
            the JSONL are skipped and new results are appended. Their
            rows are returned in the result list unchanged (so downstream
            analysis sees the union of old + new). If False (default),
            ``out_path`` is truncated at start.
        test_output_tail_chars:
            Number of trailing characters of ``state["test_output"]`` to
            store in each result's ``test_output_tail``. Default 2000,
            which is large enough to capture both halves of a two-run
            eval script (FAIL_TO_PASS + PASS_TO_PASS summaries) for SWE
            tasksets. Bump higher for verbose test suites, lower to save
            disk.

        Result schema per row
        ---------------------
        ``index``, ``instance_id``, ``repo``, ``valid``, ``reason``,
        ``attempts``, ``elapsed``, ``error``, ``error_type``, ``test_output_tail``.

        ``reason`` values: ``pass``, ``test_failed``, ``gold_apply_failed``,
        ``setup_failed``, ``sandbox_error``, ``billing_error``, ``timeout``.
        """
        import asyncio
        import json
        import logging
        import time

        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            tqdm = None  # type: ignore[assignment]

        import verifiers as vf

        logger = logging.getLogger(__name__)
        ds = self.get_dataset()
        total = min(n, len(ds)) if n is not None else len(ds)
        is_sandbox = isinstance(self, SandboxTaskSet)

        if resume and out_path is None:
            raise ValueError(
                "resume=True requires out_path; nothing to resume from without a JSONL sink."
            )

        def _read_prior_rows(path: Path) -> tuple[list[dict], set[int]]:
            """Parse a prior JSONL and return ``(prior_rows, completed_indices)``.

            Only rows whose ``index`` is in ``[0, total)`` are kept — resuming
            with a smaller ``n`` than a prior run shouldn't surface
            out-of-range rows in the returned results or summary stats.
            Malformed / non-JSON lines and rows without an integer ``index``
            are silently skipped. Logs a one-line resume summary when any
            prior rows are found.
            """
            rows: list[dict] = []
            indices: set[int] = set()
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row.get("index"), int) and row["index"] < total:
                        rows.append(row)
                        indices.add(row["index"])
            if rows:
                logger.info(
                    "Resuming: %d rows already in %s, will skip",
                    len(rows),
                    path,
                )
            return rows, indices

        out_path_p = Path(out_path) if out_path is not None else None
        prior_rows: list[dict] = []
        completed_indices: set[int] = set()
        if out_path_p is not None:
            out_path_p.parent.mkdir(parents=True, exist_ok=True)
            if resume and out_path_p.exists():
                prior_rows, completed_indices = _read_prior_rows(out_path_p)
            else:
                # truncate so repeated runs don't mix with old output
                out_path_p.write_text("")
        write_lock = asyncio.Lock()

        def _write_line(path: Path, line: str) -> None:
            with path.open("a") as f:
                f.write(line)

        async def _append_jsonl(row: dict) -> None:
            if out_path_p is None:
                return
            line = json.dumps(row, default=str) + "\n"
            async with write_lock:
                await asyncio.to_thread(_write_line, out_path_p, line)

        # Sandbox-specific exception types — imported lazily so pure-LLM
        # tasksets don't pull in prime_sandboxes. Dispatch is via isinstance
        # so pytest output containing the substring "timeout" never looks
        # like a wall-clock timeout.
        _sb_timeout_types: tuple[type, ...] = (asyncio.TimeoutError, TimeoutError)
        _sb_infra_types: tuple[type, ...] = ()
        _sb_billing_types: tuple[type, ...] = ()
        if is_sandbox:
            from prime_sandboxes import (
                APIError,
                APITimeoutError,
                CommandTimeoutError,
                DownloadTimeoutError,
                PaymentRequiredError,
                SandboxImagePullError,
                SandboxNotRunningError,
                SandboxTimeoutError,
                UploadTimeoutError,
            )

            _sb_timeout_types = _sb_timeout_types + (
                APITimeoutError,
                CommandTimeoutError,
                DownloadTimeoutError,
                SandboxTimeoutError,
                UploadTimeoutError,
            )
            _sb_infra_types = (
                APIError,
                SandboxImagePullError,
                SandboxNotRunningError,
            )
            _sb_billing_types = (PaymentRequiredError,)

        def _classify(
            valid: bool, exc: BaseException | None, state: dict
        ) -> tuple[str, str | None]:
            """Return (reason, test_output_tail).

            Classification is exception-class driven. A failed
            ``validate_instance`` with no exception is always
            ``test_failed`` — the run completed, pytest reported a
            non-pass result, and whatever is in ``state["test_output"]``
            is a test log, not a typed signal.
            """
            test_output = state.get("test_output") if isinstance(state, dict) else None
            tail = None
            if isinstance(test_output, str) and test_output:
                tail = test_output[-test_output_tail_chars:]
            if valid:
                return "pass", tail
            if exc is not None:
                if isinstance(exc, _sb_billing_types):
                    return "billing_error", tail
                if isinstance(exc, _sb_timeout_types):
                    return "timeout", tail
                if isinstance(exc, vf.InfraError) or isinstance(exc, _sb_infra_types):
                    return "sandbox_error", tail
                # Gold-patch apply raises RuntimeError from within our own
                # code (_apply_patch_file / _apply_gold_patch), so a narrow
                # message check is the cleanest signal short of adding a
                # dedicated exception class.
                msg = str(exc).lower()
                if (
                    "apply failed" in msg
                    or "patch failed" in msg
                    or "no gold patch" in msg
                ):
                    return "gold_apply_failed", tail
                return "setup_failed", tail
            # exc is None and validate_instance returned False — the run
            # completed, the test result was non-pass. Don't try to guess
            # a finer-grained reason from the test log.
            return "test_failed", tail

        def _row_info(i: int) -> tuple[dict, str | None, str | None]:
            row = ds[i]
            info = row.get("info") or {}
            return info, info.get("instance_id"), info.get("repo")

        sem = asyncio.Semaphore(concurrency)

        if not is_sandbox:

            async def _validate_once(i: int) -> tuple[bool, BaseException | None, dict]:
                info, _, _ = _row_info(i)
                row = ds[i]
                state: State = {  # type: ignore[assignment]
                    "info": info,
                    "answer": row.get("answer", ""),
                }
                try:
                    valid = await self.validate_instance(state)
                    return valid, None, state
                except Exception as e:  # noqa: BLE001
                    return False, e, state

        else:
            from prime_sandboxes import CreateSandboxRequest
            from verifiers.utils.threaded_sandbox_client import (
                ThreadedAsyncSandboxClient,
            )

            client = ThreadedAsyncSandboxClient(
                max_workers=min(max(1, concurrency // 8), 50)
            )
            assert isinstance(self, SandboxTaskSet)

            async def _validate_once(i: int) -> tuple[bool, BaseException | None, dict]:
                info, _, _ = _row_info(i)
                row = ds[i]
                state: State = {  # type: ignore[assignment]
                    "info": info,
                    "answer": row.get("answer", ""),
                }
                spec = self.get_sandbox_spec(info)
                # validate() runs without a SandboxMixin, so resolve
                # spec.timeout_minutes=None (its "auto-derive at rollout
                # time" sentinel) to a concrete fallback for both the
                # SDK call and the in-test wall-clock cap.
                timeout_minutes = spec.timeout_minutes or 60
                sb = None
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
                            timeout_minutes=timeout_minutes,
                        )
                    )
                    state["sandbox_id"] = sb.id
                    state["sandbox_client"] = client
                    state["test_timeout"] = timeout_minutes * 60
                    await client.wait_for_creation(sb.id, max_attempts=120)
                    await self.setup(state)
                    valid = await self.validate_instance(state)
                    return valid, None, state
                except Exception as e:  # noqa: BLE001
                    return False, e, state
                finally:
                    # Shield cleanup from outer cancellation so a Ctrl-C /
                    # billing-fail-fast doesn't leak live sandboxes, and
                    # catch BaseException (not Exception) because
                    # asyncio.CancelledError is a BaseException in 3.9+.
                    if sb is not None:
                        try:
                            await asyncio.shield(client.delete(sb.id))
                        except BaseException:  # noqa: BLE001
                            pass

        async def validate_one(i: int) -> dict:
            async with sem:
                info, instance_id, repo = _row_info(i)
                t0 = time.time()
                attempts = 0
                last_valid = False
                last_exc: BaseException | None = None
                reason = "test_failed"
                tail: str | None = None

                # primary attempts with InfraError retry
                for _attempt in range(1 + max_retries):
                    attempts += 1
                    valid, exc, state = await _validate_once(i)
                    last_valid, last_exc = valid, exc
                    reason, tail = _classify(valid, exc, state)
                    if valid or reason != "sandbox_error":
                        break  # only InfraError triggers retry

                elapsed = time.time() - t0
                result = {
                    "index": i,
                    "instance_id": instance_id,
                    "repo": repo,
                    "valid": bool(last_valid),
                    "reason": reason,
                    "attempts": attempts,
                    "elapsed": elapsed,
                    "error": str(last_exc) if last_exc is not None else None,
                    "error_type": type(last_exc).__name__
                    if last_exc is not None
                    else None,
                    "test_output_tail": tail,
                }
                return result

        todo_indices = [i for i in range(total) if i not in completed_indices]
        skipped = len(completed_indices)
        logger.info(
            f"Validating {len(todo_indices)} instances from {self.name} "
            f"(concurrency={concurrency}, max_retries={max_retries}, "
            f"skipped={skipped} from prior run)"
        )
        t0 = time.time()
        results: list[dict] = list(prior_rows)
        tasks = [asyncio.create_task(validate_one(i)) for i in todo_indices]
        passed = sum(1 for r in prior_rows if r.get("valid"))
        try:
            bar = (
                tqdm(
                    total=len(todo_indices),
                    desc="validate",
                    dynamic_ncols=True,
                )
                if tqdm is not None
                else None
            )
            try:
                for fut in asyncio.as_completed(tasks):
                    r = await fut
                    results.append(r)
                    await _append_jsonl(r)
                    if r["valid"]:
                        passed += 1
                    rate = passed / len(results)
                    logger.info(
                        "[%d] %s reason=%s attempts=%d elapsed=%.0fs "
                        "instance=%s (running pass-rate %d/%d = %.1f%%)",
                        r["index"],
                        "PASS" if r["valid"] else "FAIL",
                        r["reason"],
                        r["attempts"],
                        r["elapsed"],
                        r["instance_id"],
                        passed,
                        len(results),
                        100 * rate,
                    )
                    if bar is not None:
                        bar.update(1)
                        bar.set_postfix_str(
                            f"pass={passed}/{len(results)} ({100 * rate:.1f}%)"
                        )
                    if r["reason"] == "billing_error":
                        pending = sum(1 for t in tasks if not t.done())
                        logger.error(
                            "Aborting validate(): billing error at index %d "
                            "(instance=%s). Cancelling %d pending tasks.",
                            r["index"],
                            r["instance_id"],
                            pending,
                        )
                        break
            finally:
                if bar is not None:
                    bar.close()
                # cancel any still-pending tasks (billing fail-fast, or if
                # the for-loop exits for any other reason) — mirrors the
                # KeyboardInterrupt handler below.
                pending_tasks = [t for t in tasks if not t.done()]
                if pending_tasks:
                    for t in pending_tasks:
                        t.cancel()
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.warning("Validation interrupted; cancelling pending tasks")
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if is_sandbox:
                client.teardown()

        elapsed = time.time() - t0
        denom = len(results) or 1
        rate = passed / denom
        logger.info(
            f"Validation: {passed}/{len(results)} valid ({rate:.1%}) "
            f"[new={len(results) - len(prior_rows)} skipped={skipped}] "
            f"in {elapsed:.0f}s"
        )
        # reason breakdown
        by_reason: dict[str, int] = {}
        for r in results:
            by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
        logger.info("Reason breakdown: %s", dict(sorted(by_reason.items())))
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
