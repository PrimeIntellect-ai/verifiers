from __future__ import annotations

import atexit
import math
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset

import verifiers as vf
from verifiers.envs.sandbox_env import SandboxEnv

from terminal_bench.dataset import Dataset as TBDataset
from terminal_bench.handlers.trial_handler import Task, TaskPaths

_DEFAULT_DATASET_NAME = "terminal-bench-core"
_DEFAULT_DATASET_VERSION = "head"
_CONTAINER_TASK_ROOT = "/workspace/task"
_CONTAINER_TESTS_DIR = "/tests"
_CONTAINER_LOGS_DIR = "/var/log/tbench"
_REMOTE_ARCHIVE_PATH = "/tmp/terminal_bench_task.tar.gz"


def _extract_base_image(task_dir: Path) -> str:
    dockerfile_path = task_dir / "Dockerfile"
    if dockerfile_path.exists():
        for line in dockerfile_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("FROM"):
                parts = stripped.split(maxsplit=1)
                if len(parts) == 2 and parts[1]:
                    return parts[1]
    # Fall back to the image used by the canonical hello-world task
    return "ghcr.io/laude-institute/t-bench/python-3-13:20250620"


def _create_task_archive(task_root: Path) -> Path:
    temp_file = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    archive_path = Path(temp_file.name)
    temp_file.close()
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for child in task_root.iterdir():
            archive.add(child, arcname=child.name)
    return archive_path


class TerminalBenchSandboxEnv(SandboxEnv):
    def __init__(
        self,
        task_id: str,
        task_paths: TaskPaths,
        task_config: Task,
        docker_image: str,
        **kwargs: Any,
    ) -> None:
        self.task_id = task_id
        self.task_paths = task_paths
        self.task_config = task_config
        self.container_task_root = _CONTAINER_TASK_ROOT
        self.container_tests_dir = _CONTAINER_TESTS_DIR
        self.container_logs_dir = _CONTAINER_LOGS_DIR
        self._archive_path = _create_task_archive(self.task_paths.input_path)
        atexit.register(self._cleanup_archive)

        environment_vars = dict(kwargs.pop("environment_vars", {}))
        environment_vars.setdefault("TEST_DIR", self.container_tests_dir)
        environment_vars.setdefault("T_BENCH_TEST_DIR", self.container_tests_dir)
        environment_vars.setdefault(
            "T_BENCH_CONTAINER_LOGS_PATH", self.container_logs_dir
        )

        timeout_minutes = max(1, math.ceil(task_config.max_agent_timeout_sec / 60))
        start_command = kwargs.pop(
            "start_command",
            "bash -lc 'mkdir -p /workspace/task /tests /var/log/tbench /app && tail -f /dev/null'",
        )

        super().__init__(
            sandbox_name=f"terminal-bench-{task_id}",
            docker_image=docker_image,
            start_command=start_command,
            timeout_minutes=timeout_minutes,
            environment_vars=environment_vars,
            **kwargs,
        )

    def _cleanup_archive(self) -> None:
        try:
            if self._archive_path.exists():
                self._archive_path.unlink()
        except OSError:
            pass

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        sandbox_id = state["sandbox_id"]
        await self.sandbox_client.upload_file(
            sandbox_id,
            _REMOTE_ARCHIVE_PATH,
            str(self._archive_path),
        )
        await self.sandbox_client.execute_command(
            sandbox_id,
            f"mkdir -p {self.container_task_root} {self.container_tests_dir} {self.container_logs_dir} /app",
        )
        await self.sandbox_client.execute_command(
            sandbox_id,
            f"tar -xzf {_REMOTE_ARCHIVE_PATH} -C {self.container_task_root}",
        )
        await self.sandbox_client.execute_command(
            sandbox_id,
            f"rm -f {_REMOTE_ARCHIVE_PATH}",
        )
        await self.sandbox_client.execute_command(
            sandbox_id,
            (
                f"if [ -d {self.container_task_root}/tests ]; then "
                f"rm -rf {self.container_tests_dir} && mkdir -p {self.container_tests_dir} && "
                f"cp -r {self.container_task_root}/tests/. {self.container_tests_dir}; "
                f"else mkdir -p {self.container_tests_dir}; fi"
            ),
        )
        await self.sandbox_client.execute_command(
            sandbox_id,
            f"if [ -f {self.container_task_root}/run-tests.sh ]; then chmod +x {self.container_task_root}/run-tests.sh; fi",
        )
        state["terminal_bench"] = {
            "task_root": self.container_task_root,
            "tests_dir": self.container_tests_dir,
            "test_exit_code": None,
            "test_stdout": "",
            "test_stderr": "",
            "reward": 0.0,
            "post_rollout_complete": False,
        }
        return state

    async def post_rollout(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> None:
        tb_state = state.get("terminal_bench")
        if tb_state is None:
            tb_state = {}
            state["terminal_bench"] = tb_state
        if tb_state.get("post_rollout_complete"):
            return

        sandbox_id = state.get("sandbox_id")
        if sandbox_id is None:
            return

        if not self.task_paths.run_tests_path.exists():
            self.logger.error(
                "Task %s is missing run-tests.sh; marking reward as 0.0", self.task_id
            )
            tb_state.update(
                {
                    "test_exit_code": None,
                    "test_stdout": "",
                    "test_stderr": "",
                    "reward": 0.0,
                    "post_rollout_complete": True,
                }
            )
            return

        try:
            timeout = int(math.ceil(self.task_config.max_test_timeout_sec))
            timeout_arg = timeout if timeout > 0 else None
            response = await self.sandbox_client.execute_command(
                sandbox_id,
                f"cd {self.container_task_root} && bash ./run-tests.sh",
                env={
                    "TEST_DIR": self.container_tests_dir,
                    "T_BENCH_TEST_DIR": self.container_tests_dir,
                    "T_BENCH_CONTAINER_LOGS_PATH": self.container_logs_dir,
                },
                timeout=timeout_arg,
            )
            reward = 1.0 if response.exit_code == 0 else 0.0
            tb_state.update(
                {
                    "test_exit_code": response.exit_code,
                    "test_stdout": response.stdout,
                    "test_stderr": response.stderr,
                    "reward": reward,
                    "post_rollout_complete": True,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to run Terminal-Bench tests: %s", exc)
            tb_state.update(
                {
                    "test_exit_code": None,
                    "test_stdout": "",
                    "test_stderr": str(exc),
                    "reward": 0.0,
                    "post_rollout_complete": True,
                }
            )


def _cached_reward(
    prompt: vf.Messages,
    completion: vf.Messages,
    answer: str,
    state: vf.State,
    **_: Any,
) -> float:
    tb_state = state.get("terminal_bench") or {}
    reward = tb_state.get("reward")
    return float(reward) if reward is not None else 0.0


def load_environment(
    task_id: str = "hello-world",
    dataset_name: str = _DEFAULT_DATASET_NAME,
    dataset_version: str = _DEFAULT_DATASET_VERSION,
    registry_url: str | None = None,
    local_registry_path: str | Path | None = None,
    max_turns: int = 20,
    **kwargs: Any,
) -> vf.Environment:
    local_registry = None
    if local_registry_path is not None:
        local_registry = Path(local_registry_path)

    tb_dataset = TBDataset(
        name=dataset_name,
        version=dataset_version,
        task_ids=[task_id],
        registry_url=registry_url,
        local_registry_path=local_registry,
    )
    if not tb_dataset.tasks:
        raise ValueError(
            f"Task {task_id} not found in dataset {dataset_name}=={dataset_version}"
        )

    task_dir = tb_dataset.tasks[0]
    task_paths = TaskPaths(task_dir)
    task_config = Task.from_yaml(task_paths.task_config_path)
    docker_image = _extract_base_image(task_dir)

    system_prompt = (
        "You are an autonomous software agent operating inside a persistent Linux "
        "sandbox that mirrors a Terminal-Bench task. Use the `bash` tool to run "
        "shell commands; state persists across calls."
    )
    user_prompt = (
        f"Terminal-Bench task `{task_id}`.\n\n"
        f"{task_config.instruction.strip()}\n\n"
        f"Task files are staged at {_CONTAINER_TASK_ROOT}. Tests will automatically "
        f"run with `bash ./run-tests.sh` after you finish. You can run them manually "
        f"with `cd {_CONTAINER_TASK_ROOT} && TEST_DIR={_CONTAINER_TESTS_DIR} bash ./run-tests.sh`."
    )

    dataset = Dataset.from_list(
        [
            {
                "question": user_prompt,
                "answer": "",
                "task": task_id,
                "info": {
                    "task_id": task_id,
                    "tests_dir": _CONTAINER_TESTS_DIR,
                    "task_root": _CONTAINER_TASK_ROOT,
                },
            }
        ]
    )

    parser = kwargs.pop("parser", vf.Parser())
    rubric = kwargs.pop("rubric", vf.Rubric(funcs=[_cached_reward]))

    env = TerminalBenchSandboxEnv(
        task_id=task_id,
        task_paths=task_paths,
        task_config=task_config,
        docker_image=docker_image,
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        **kwargs,
    )
    return env
