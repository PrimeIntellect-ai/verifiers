from __future__ import annotations

import json
import logging
import math
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.tasksets.base import SandboxSpec, Task, TaskSet
from verifiers.utils.import_utils import load_toml

if TYPE_CHECKING:
    from verifiers.types import State


logger = logging.getLogger(__name__)


def _parse_capacity_gb(
    *,
    gb_value: int | float | None = None,
    mb_value: int | float | None = None,
    text_value: str | None = None,
    default: int,
) -> int:
    if gb_value is not None:
        return max(1, int(math.ceil(float(gb_value))))
    if mb_value is not None:
        return max(1, int(math.ceil(float(mb_value) / 1024.0)))
    if text_value is None:
        return default

    normalized = text_value.strip().upper()
    if normalized.endswith("GB"):
        return max(1, int(math.ceil(float(normalized[:-2]))))
    if normalized.endswith("G"):
        return max(1, int(math.ceil(float(normalized[:-1]))))
    if normalized.endswith("MB"):
        return max(1, int(math.ceil(float(normalized[:-2]) / 1024.0)))
    if normalized.endswith("M"):
        return max(1, int(math.ceil(float(normalized[:-1]) / 1024.0)))
    return max(1, int(math.ceil(float(normalized))))


class HarborTaskSet(TaskSet, Task):
    """Harbor task loader with Harbor-style setup and verifier execution."""

    def __init__(
        self,
        dataset_path: str | Path,
        tasks: list[str] | None = None,
        agent_workdir: str = "/app",
        docker_image: str = "python:3.11-slim",
    ):
        Task.__init__(self)
        self.dataset_path = Path(dataset_path)
        self.task_names = tasks
        self.agent_workdir = agent_workdir
        self.docker_image = docker_image
        self.dataset = self._load_dataset()

    def get_dataset(self) -> Dataset:
        return self.dataset

    def get_task(self, state: State) -> Task:
        return self

    def _load_dataset(self) -> Dataset:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        tasks: list[dict[str, Any]] = []
        for task_dir in sorted(self.dataset_path.iterdir()):
            if not task_dir.is_dir():
                continue
            if self.task_names and task_dir.name not in self.task_names:
                continue

            task_toml = task_dir / "task.toml"
            instruction_md = task_dir / "instruction.md"
            if not task_toml.exists() or not instruction_md.exists():
                logger.warning(
                    "Skipping %s: missing task.toml or instruction.md",
                    task_dir.name,
                )
                continue

            with open(task_toml, "rb") as f:
                config = load_toml(f)

            tasks.append(
                {
                    "example_id": len(tasks),
                    "task": task_dir.name,
                    "question": instruction_md.read_text().strip(),
                    "answer": "",
                    "info": {
                        "task_dir": str(task_dir),
                        "task_name": task_dir.name,
                        "config": config,
                    },
                }
            )

        if not tasks:
            raise ValueError(f"No valid Harbor tasks found in {self.dataset_path}")

        return Dataset.from_list(tasks)

    async def prompt(self, state: State):
        return [{"role": "user", "content": state.get("question", "")}]

    async def get_sandbox_spec(self, state: State) -> SandboxSpec:
        info = state.get("info") or {}
        config = info.get("config", {}) if isinstance(info, dict) else {}
        environment = config.get("environment", {})
        agent = config.get("agent", {})
        verifier = config.get("verifier", {})

        timeout_seconds = max(
            float(agent.get("timeout_sec", 3600.0)),
            float(verifier.get("timeout_sec", 3600.0)),
        )
        return SandboxSpec(
            docker_image=environment.get("docker_image") or self.docker_image,
            start_command=environment.get("start_command", "tail -f /dev/null"),
            cpu_cores=int(environment.get("cpus", environment.get("cpu_cores", 1))),
            memory_gb=_parse_capacity_gb(
                gb_value=environment.get("memory_gb"),
                mb_value=environment.get("memory_mb"),
                text_value=environment.get("memory"),
                default=2,
            ),
            disk_size_gb=_parse_capacity_gb(
                gb_value=environment.get("storage_gb"),
                mb_value=environment.get("storage_mb"),
                text_value=environment.get("storage"),
                default=10,
            ),
            gpu_count=int(environment.get("gpus", environment.get("gpu_count", 0))),
            timeout_minutes=max(1, int(math.ceil(timeout_seconds / 60.0))),
            labels=[],
        )

    async def build_env_vars(self, state: State) -> dict[str, str]:
        return {
            "HARBOR_TASK_NAME": state.get("task", ""),
            "HARBOR_TASK_DIR": "/task",
            "HARBOR_INSTRUCTION_PATH": "/task/instruction.md",
            "AGENT_WORKDIR": self.agent_workdir,
        }

    async def setup(self, env: Any, state: State) -> None:
        info = state.get("info") or {}
        task_dir = Path(info["task_dir"])
        await self._upload_task_assets(
            env=env,
            sandbox_id=state["sandbox_id"],
            task_dir=task_dir,
        )
        state["harbor_task_dir"] = str(task_dir)

    async def post_rollout(self, env: Any, state: State) -> None:
        if isinstance(state.get("error"), vf.InfraError):
            state["task_reward"] = 0.0
            return

        sandbox_id = state.get("sandbox_id")
        task_dir_str = state.get("harbor_task_dir")
        if not sandbox_id or not task_dir_str:
            state["task_reward"] = 0.0
            return

        task_dir = Path(task_dir_str)
        try:
            await self._upload_test_assets(
                env=env,
                sandbox_id=sandbox_id,
                task_dir=task_dir,
            )
            result = await env.run_background_job(
                state,
                "bash test.sh",
                timeout=self._get_test_timeout_seconds(state),
                working_dir="/tests",
                poll_interval=5,
            )
            state["test_exit_code"] = getattr(result, "exit_code", 0)
            reward_result = await env.sandbox_client.execute_command(
                sandbox_id,
                "if [ -s /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; "
                "elif [ -s /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; fi",
                working_dir=None,
            )
        except Exception as e:
            if state.get("error") is None:
                state["error"] = vf.SandboxError(str(e))
            state["task_reward"] = 0.0
            return

        reward_output = (getattr(reward_result, "stdout", "") or "").strip()
        state["task_reward"] = self._parse_reward_output(reward_output)

    def _get_test_timeout_seconds(self, state: State) -> int:
        info = state.get("info") or {}
        config = info.get("config", {}) if isinstance(info, dict) else {}
        verifier = config.get("verifier", {})
        return int(float(verifier.get("timeout_sec", 300.0)))

    def _parse_reward_output(self, reward_output: str) -> float:
        if not reward_output:
            return 0.0
        try:
            return float(reward_output)
        except ValueError:
            try:
                data = json.loads(reward_output)
            except json.JSONDecodeError:
                return 0.0
            return float(data.get("reward", 0.0))

    async def _upload_task_assets(
        self,
        *,
        env: Any,
        sandbox_id: str,
        task_dir: Path,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(task_dir / "instruction.md", arcname="task/instruction.md")
                tar.add(task_dir / "task.toml", arcname="task/task.toml")

            remote_tar = "/tmp/harbor_task.tar.gz"
            await env.sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
            await env.sandbox_client.execute_command(
                sandbox_id,
                f"mkdir -p /task /logs/verifier {self.agent_workdir} && "
                f"tar -xzf {remote_tar} -C / && rm {remote_tar}",
                working_dir=None,
            )
        finally:
            tar_path.unlink(missing_ok=True)

    async def _upload_test_assets(
        self,
        *,
        env: Any,
        sandbox_id: str,
        task_dir: Path,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                solution_dir = task_dir / "solution"
                tests_dir = task_dir / "tests"

                if solution_dir.exists():
                    for item in solution_dir.iterdir():
                        tar.add(item, arcname=f"oracle/{item.name}")

                if tests_dir.exists():
                    for item in tests_dir.iterdir():
                        tar.add(item, arcname=f"tests/{item.name}")

            remote_tar = "/tmp/harbor_tests.tar.gz"
            await env.sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
            await env.sandbox_client.execute_command(
                sandbox_id,
                f"mkdir -p /oracle /tests && tar -xzf {remote_tar} -C / && rm {remote_tar}",
                working_dir=None,
                timeout=900,
            )
        finally:
            tar_path.unlink(missing_ok=True)
