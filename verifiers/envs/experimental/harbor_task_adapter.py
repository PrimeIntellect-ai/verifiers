"""Adapter from Harbor task directories to the verifiers Task/TaskSet protocol.

Harbor tasks are self-contained directories with:

* ``task.toml`` — config (docker image, timeouts, metadata)
* ``instruction.md`` — problem statement
* ``tests/test.sh`` — test execution script (writes reward to /logs/verifier/reward.txt)
* ``solution/solve.sh`` — reference solution

This adapter converts them into the Task protocol so they can be used with
``ComposableEnv`` and any Agent.

Usage::

    # Single task directory
    taskset = HarborTaskSet("path/to/tasks/hello-world")

    # Directory of tasks (each subdirectory is a task)
    taskset = HarborTaskSet("path/to/tasks/")

    # With ComposableEnv
    env = ComposableEnv(task=taskset, agent=opencode_agent)
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from verifiers.envs.experimental.task import TaskSet
from verifiers.types import Messages, State, UserMessage

logger = logging.getLogger(__name__)


def _read_toml(path: Path) -> dict:
    """Read a TOML file, using tomllib (3.11+) or tomli as fallback."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)


def _tar_directory(src_dir: Path, prefix: str = "") -> bytes:
    """Create a tar.gz archive of a directory in memory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in src_dir.rglob("*"):
            if p.is_file():
                arcname = f"{prefix}/{p.relative_to(src_dir)}" if prefix else str(p.relative_to(src_dir))
                tar.add(str(p), arcname=arcname)
    return buf.getvalue()


class HarborTask:
    """Task implementation for a single Harbor task directory.

    Implements the ``Task`` protocol.
    """

    def __init__(self, task_dir: str | Path, default_image: str = "python:3.11-slim"):
        self.task_dir = Path(task_dir)
        self.default_image = default_image

        # Load config
        toml_path = self.task_dir / "task.toml"
        self.config = _read_toml(toml_path) if toml_path.exists() else {}

        # Load instruction
        instruction_path = self.task_dir / "instruction.md"
        self.instruction = instruction_path.read_text() if instruction_path.exists() else ""

    def get_prompt(self, info: dict) -> Messages:
        return [UserMessage(content=self.instruction)]

    def get_image(self, info: dict) -> str:
        return self.config.get("environment", {}).get("docker_image", self.default_image)

    def get_workdir(self, info: dict) -> str:
        return "/app"

    def get_env_vars(self) -> dict[str, str]:
        return {
            "HARBOR_TASK_DIR": "/task",
            "HARBOR_INSTRUCTION_PATH": "/task/instruction.md",
        }

    async def setup(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> None:
        """Upload instruction.md and task.toml to /task/ in the sandbox."""
        # Create directories
        await sandbox_client.execute_command(
            sandbox_id, "mkdir -p /task /logs/verifier /oracle /tests /app", timeout=10,
        )

        # TAR and upload task files (instruction.md + task.toml)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name in ["instruction.md", "task.toml"]:
                src = self.task_dir / name
                if src.exists():
                    tar.add(str(src), arcname=f"task/{name}")
        tar_bytes = buf.getvalue()

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            f.write(tar_bytes)
            tmp_path = f.name

        try:
            await sandbox_client.upload_file(sandbox_id, "/tmp/harbor_task.tar.gz", tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        await sandbox_client.execute_command(
            sandbox_id,
            "tar -xzf /tmp/harbor_task.tar.gz -C / && rm /tmp/harbor_task.tar.gz",
            timeout=30,
        )

    async def evaluate(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> float:
        """Upload tests + solution, run test.sh, read reward."""
        run_background_job = state.get("_run_background_job")

        # Upload solution/ → /oracle/ and tests/ → /tests/
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            solution_dir = self.task_dir / "solution"
            if solution_dir.exists():
                for p in solution_dir.rglob("*"):
                    if p.is_file():
                        tar.add(str(p), arcname=f"oracle/{p.relative_to(solution_dir)}")
            tests_dir = self.task_dir / "tests"
            if tests_dir.exists():
                for p in tests_dir.rglob("*"):
                    if p.is_file():
                        tar.add(str(p), arcname=f"tests/{p.relative_to(tests_dir)}")

        tar_bytes = buf.getvalue()
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            f.write(tar_bytes)
            tmp_path = f.name

        try:
            await sandbox_client.upload_file(sandbox_id, "/tmp/harbor_tests.tar.gz", tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        await sandbox_client.execute_command(
            sandbox_id,
            "tar -xzf /tmp/harbor_tests.tar.gz -C / && rm /tmp/harbor_tests.tar.gz && chmod +x /tests/test.sh",
            timeout=30,
        )

        # Run test.sh
        test_timeout = int(self.config.get("verifier", {}).get("timeout_sec", 300))
        if run_background_job:
            await run_background_job(
                state, "cd /tests && bash test.sh", test_timeout,
            )
        else:
            await sandbox_client.execute_command(
                sandbox_id, "cd /tests && bash test.sh", timeout=test_timeout,
            )

        # Read reward
        reward_result = await sandbox_client.execute_command(
            sandbox_id, "cat /logs/verifier/reward.txt 2>/dev/null || cat /logs/verifier/reward.json 2>/dev/null || echo 0", timeout=10,
        )
        reward_str = (reward_result.stdout or "0").strip()

        try:
            return float(reward_str)
        except ValueError:
            try:
                return float(json.loads(reward_str).get("reward", 0))
            except Exception:
                logger.warning(f"Failed to parse reward: {reward_str!r}")
                return 0.0

    async def apply_gold_patch(
        self, sandbox_client: Any, sandbox_id: str, state: State,
    ) -> None:
        """Upload and run the reference solution."""
        solution_dir = self.task_dir / "solution"
        if not solution_dir.exists():
            return

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for p in solution_dir.rglob("*"):
                if p.is_file():
                    tar.add(str(p), arcname=f"oracle/{p.relative_to(solution_dir)}")

        tar_bytes = buf.getvalue()
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            f.write(tar_bytes)
            tmp_path = f.name

        try:
            await sandbox_client.upload_file(sandbox_id, "/tmp/harbor_solution.tar.gz", tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        await sandbox_client.execute_command(
            sandbox_id,
            "tar -xzf /tmp/harbor_solution.tar.gz -C / && rm /tmp/harbor_solution.tar.gz && chmod +x /oracle/solve.sh",
            timeout=30,
        )
        result = await sandbox_client.execute_command(
            sandbox_id, "cd /app && bash /oracle/solve.sh", timeout=600,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Gold patch failed (exit={result.exit_code}): "
                f"{(result.stdout or '')[:200]} {(result.stderr or '')[:200]}"
            )


def HarborTaskSet(
    path: str | Path,
    task_names: list[str] | None = None,
    default_image: str = "python:3.11-slim",
) -> TaskSet:
    """Create a ``TaskSet`` from a Harbor task directory or collection.

    Parameters
    ----------
    path:
        Either a single task directory (has ``task.toml``) or a parent
        directory containing multiple task subdirectories.
    task_names:
        If given, only include these task names from the collection.
    default_image:
        Default Docker image when not specified in task.toml.

    Returns
    -------
    TaskSet
        A TaskSet where each instance delegates to its own HarborTask.
    """
    from datasets import Dataset

    path = Path(path)

    # Single task directory
    if (path / "task.toml").exists():
        task = HarborTask(path, default_image)
        ds = Dataset.from_dict({
            "question": [task.instruction],
            "info": [{"task_dir": str(path), "task_name": path.name, "config": task.config}],
            "answer": [""],
        })
        return TaskSet(task=task, dataset=ds, name=path.name)

    # Collection: each subdirectory with task.toml is a task
    tasks: list[HarborTask] = []
    rows: list[dict] = []

    for subdir in sorted(path.iterdir()):
        if not subdir.is_dir():
            continue
        if not (subdir / "task.toml").exists():
            continue
        if task_names and subdir.name not in task_names:
            continue

        task = HarborTask(subdir, default_image)
        tasks.append(task)
        rows.append({
            "question": task.instruction,
            "info": {"task_dir": str(subdir), "task_name": subdir.name, "config": task.config},
            "answer": "",
        })

    if not rows:
        raise ValueError(f"No Harbor tasks found in {path}")

    ds = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "info": [r["info"] for r in rows],
        "answer": [r["answer"] for r in rows],
    })

    # For heterogeneous collections, use a delegating task that routes
    # per-instance based on info["task_dir"]
    delegating_task = _HarborDelegatingTask(
        {t.task_dir.name: t for t in tasks}, default_image
    )
    return TaskSet(task=delegating_task, dataset=ds, name=path.name)


class _HarborDelegatingTask:
    """Internal: routes Task methods to the correct HarborTask per instance."""

    def __init__(self, tasks_by_name: dict[str, HarborTask], default_image: str):
        self._tasks = tasks_by_name
        self._default_image = default_image

    def _resolve(self, info: dict) -> HarborTask:
        name = info.get("task_name", "")
        if name in self._tasks:
            return self._tasks[name]
        # Fallback: try task_dir
        task_dir = info.get("task_dir", "")
        for t in self._tasks.values():
            if str(t.task_dir) == task_dir:
                return t
        raise KeyError(f"No HarborTask found for {info}")

    def get_prompt(self, info: dict) -> Messages:
        return self._resolve(info).get_prompt(info)

    def get_image(self, info: dict) -> str:
        return self._resolve(info).get_image(info)

    def get_workdir(self, info: dict) -> str:
        return self._resolve(info).get_workdir(info)

    def get_env_vars(self) -> dict[str, str]:
        return {"HARBOR_TASK_DIR": "/task", "HARBOR_INSTRUCTION_PATH": "/task/instruction.md"}

    async def setup(self, sandbox_client: Any, sandbox_id: str, state: State) -> None:
        info = state.get("info") or {}
        return await self._resolve(info).setup(sandbox_client, sandbox_id, state)

    async def evaluate(self, sandbox_client: Any, sandbox_id: str, state: State) -> float:
        info = state.get("info") or {}
        return await self._resolve(info).evaluate(sandbox_client, sandbox_id, state)

    async def apply_gold_patch(self, sandbox_client: Any, sandbox_id: str, state: State) -> None:
        info = state.get("info") or {}
        return await self._resolve(info).apply_gold_patch(sandbox_client, sandbox_id, state)
