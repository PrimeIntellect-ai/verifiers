from __future__ import annotations

import errno
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from verifiers.envs.experimental.composable.tasksets.harbor import HarborDatasetTaskSet

DEFAULT_REPO_URL = "https://github.com/harbor-framework/terminal-bench-2.git"
DEFAULT_GIT_REF = "main"
DEFAULT_TASKS_SUBDIR = "."
DEFAULT_AGENT_WORKDIR = "/app"


class TerminalBench2TaskSet(HarborDatasetTaskSet):
    """Terminal-Bench 2 as Harbor-native task directories."""

    default_workdir = DEFAULT_AGENT_WORKDIR

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        repo_url: str = DEFAULT_REPO_URL,
        git_ref: str = DEFAULT_GIT_REF,
        tasks_subdir: str = DEFAULT_TASKS_SUBDIR,
        task_ids: list[str] | None = None,
        tasks: list[str] | None = None,
        max_examples: int = -1,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        name: str = "terminal-bench/terminal-bench-2",
    ):
        dataset_path = self.resolve_dataset_path(
            dataset_path=dataset_path,
            repo_url=repo_url,
            git_ref=git_ref,
            tasks_subdir=tasks_subdir,
            cache_dir=cache_dir,
            force_download=force_download,
        )
        task_names = self.select_task_names(
            dataset_path=dataset_path,
            requested_tasks=task_ids or tasks,
            max_examples=max_examples,
        )
        super().__init__(
            dataset_path=dataset_path,
            task_names=task_names,
            agent_workdir=agent_workdir,
            name=name,
        )

    def resolve_dataset_path(
        self,
        *,
        dataset_path: str | Path | None,
        repo_url: str,
        git_ref: str,
        tasks_subdir: str,
        cache_dir: str | Path | None,
        force_download: bool,
    ) -> Path:
        if dataset_path is not None:
            return Path(dataset_path).expanduser()

        subdir_path = Path(tasks_subdir)
        if subdir_path.is_absolute() or ".." in subdir_path.parts:
            raise ValueError("tasks_subdir must stay within the Terminal-Bench repo.")

        root = Path(cache_dir or Path.home() / ".cache" / "verifiers")
        safe_ref = re.sub(r"[^A-Za-z0-9_.-]+", "-", git_ref).strip("-") or "head"
        source_key = hashlib.sha256(f"{repo_url}\n{tasks_subdir}".encode()).hexdigest()
        ref_key = hashlib.sha256(git_ref.encode()).hexdigest()
        target = (
            root / "terminal-bench-2" / source_key[:16] / f"{safe_ref}-{ref_key[:16]}"
        )
        if force_download and target.exists():
            shutil.rmtree(target)
        if target.exists():
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="terminal_bench_2_clone_") as tmp:
            clone_dir = Path(tmp) / "repo"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "--no-checkout",
                    repo_url,
                    str(clone_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(clone_dir), "fetch", "--depth=1", "origin", git_ref],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(clone_dir), "checkout", "--detach", "FETCH_HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            clone_root = clone_dir.resolve()
            source = (clone_dir / subdir_path).resolve()
            if not source.is_relative_to(clone_root):
                raise ValueError(
                    "tasks_subdir must stay within the Terminal-Bench repo."
                )
            with tempfile.TemporaryDirectory(
                prefix=f".{target.name}.", dir=target.parent
            ) as tmp_target:
                staged_target = Path(tmp_target) / "dataset"
                shutil.copytree(
                    source, staged_target, ignore=shutil.ignore_patterns(".git")
                )
                try:
                    staged_target.rename(target)
                except OSError as e:
                    if e.errno in {errno.EEXIST, errno.ENOTEMPTY} and target.exists():
                        return target
                    raise
        return target

    def select_task_names(
        self,
        *,
        dataset_path: Path,
        requested_tasks: list[str] | None,
        max_examples: int,
    ) -> list[str] | None:
        if requested_tasks:
            names = requested_tasks
        elif max_examples > -1:
            names = [
                task_dir.name
                for task_dir in sorted(dataset_path.iterdir())
                if self.is_harbor_task_dir(task_dir)
            ]
        else:
            return None

        selected_names = names[:max_examples] if max_examples > -1 else names
        if not selected_names:
            raise ValueError("No Terminal-Bench tasks matched the requested filters.")
        return selected_names

    def is_harbor_task_dir(self, task_dir: Path) -> bool:
        return (
            task_dir.is_dir()
            and (task_dir / "task.toml").exists()
            and (task_dir / "instruction.md").exists()
        )


TerminalBenchTaskSet = TerminalBench2TaskSet
