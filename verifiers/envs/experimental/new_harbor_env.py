"""Harbor environment using SandboxManager for resource lifecycle."""

import json
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset
from prime_sandboxes import AsyncSandboxClient

import verifiers as vf
from verifiers.envs.experimental.new_cli_agent_env import NewCliAgentEnv
from verifiers.utils.import_utils import load_toml

logger = logging.getLogger(__name__)


class NewHarborEnv(NewCliAgentEnv):
    """Harbor task environment with managed sandbox lifecycle."""

    def __init__(
        self,
        run_command: str,
        dataset_path: str | Path,
        tasks: list[str] | None = None,
        agent_workdir: str = "/app",
        **kwargs: Any,
    ):
        self.dataset_path = Path(dataset_path)
        self.task_names = tasks
        self.agent_workdir = agent_workdir

        dataset = self._load_harbor_dataset()
        rubric = vf.Rubric(funcs=[self._harbor_reward], weights=[1.0])

        super().__init__(run_command=run_command, dataset=dataset, rubric=rubric, **kwargs)

    def _load_harbor_dataset(self) -> Dataset:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        tasks = []
        for task_dir in sorted(self.dataset_path.iterdir()):
            if not task_dir.is_dir():
                continue
            if self.task_names and task_dir.name not in self.task_names:
                continue

            task_toml = task_dir / "task.toml"
            instruction_md = task_dir / "instruction.md"

            if not task_toml.exists() or not instruction_md.exists():
                continue

            with open(task_toml, "rb") as f:
                config = load_toml(f)

            instruction = instruction_md.read_text().strip()
            tasks.append({
                "example_id": len(tasks),
                "task": task_dir.name,
                "prompt": [{"role": "user", "content": instruction}],
                "info": {
                    "task_dir": str(task_dir),
                    "docker_image": config.get("environment", {}).get("docker_image"),
                    "config": config,
                },
            })

        if not tasks:
            raise ValueError(f"No valid Harbor tasks found in {self.dataset_path}")
        return Dataset.from_list(tasks)

    async def get_docker_image(self, state: vf.State) -> str:
        info = state.get("info") or {}
        return info.get("docker_image") or self.docker_image

    async def build_env_vars(self, state: vf.State) -> dict[str, str]:
        env_vars = await super().build_env_vars(state)
        env_vars.setdefault("HARBOR_TASK_NAME", state.get("task", ""))
        env_vars.setdefault("HARBOR_TASK_DIR", "/task")
        if self.agent_workdir:
            env_vars.setdefault("AGENT_WORKDIR", self.agent_workdir)
        return env_vars

    async def post_sandbox_setup(self, state: vf.State, sandbox_client: AsyncSandboxClient) -> None:
        info = state.get("info", {}) or {}
        task_dir = Path(info.get("task_dir", ""))
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        # Upload instruction only (tests uploaded after agent completes)
        await self._upload_task_instruction(sandbox_client, state["sandbox_id"], task_dir)
        state["harbor_config"] = info.get("config", {})
        state["harbor_task_dir"] = str(task_dir)

    async def _upload_task_instruction(self, client: AsyncSandboxClient, sandbox_id: str, task_dir: Path) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tar_path = Path(tmp.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                instruction = task_dir / "instruction.md"
                task_toml = task_dir / "task.toml"
                if instruction.exists():
                    tar.add(instruction, arcname="task/instruction.md")
                if task_toml.exists():
                    tar.add(task_toml, arcname="task/task.toml")

            await client.upload_file(sandbox_id, "/tmp/task.tar.gz", str(tar_path))
            await client.execute_command(
                sandbox_id,
                f"mkdir -p /task /logs/verifier {self.agent_workdir} && tar -xzf /tmp/task.tar.gz -C / && rm /tmp/task.tar.gz",
            )
        finally:
            tar_path.unlink(missing_ok=True)

    async def post_rollout(self, state: vf.State):
        state["reward"] = await self._compute_reward(state)

    async def _harbor_reward(self, state: vf.State, **kwargs) -> float:
        return state.get("reward", 0.0)

    async def _compute_reward(self, state: vf.State) -> float:
        sandbox_id = state.get("sandbox_id")
        task_dir_str = state.get("harbor_task_dir", "")
        if not sandbox_id or not task_dir_str:
            return 0.0

        task_dir = Path(task_dir_str)
        if not task_dir.exists():
            return 0.0

        client = AsyncSandboxClient()
        try:
            # Upload test assets
            await self._upload_test_assets(client, sandbox_id, task_dir)

            # Run tests
            await client.execute_command(sandbox_id, "bash test.sh", working_dir="/tests")

            # Get reward
            result = await client.execute_command(
                sandbox_id,
                "if [ -s /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; "
                "elif [ -s /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; fi",
            )
            stdout = getattr(result, "stdout", "") or ""
            stdout = stdout.strip()
            if stdout:
                try:
                    return float(stdout)
                except ValueError:
                    return float(json.loads(stdout).get("reward", 0.0))
            return 0.0
        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            return 0.0

    async def _upload_test_assets(self, client: AsyncSandboxClient, sandbox_id: str, task_dir: Path) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tar_path = Path(tmp.name)

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

            await client.upload_file(sandbox_id, "/tmp/tests.tar.gz", str(tar_path))
            await client.execute_command(
                sandbox_id,
                "mkdir -p /oracle /tests && tar -xzf /tmp/tests.tar.gz -C / && rm /tmp/tests.tar.gz",
                timeout=900,
            )
        finally:
            tar_path.unlink(missing_ok=True)
