import json
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[unresolved-import]
except ImportError:
    import tomli as tomllib  # type: ignore[unresolved-import]
from datasets import Dataset
from prime_sandboxes import AsyncSandboxClient

import verifiers as vf

logger = logging.getLogger(__name__)


class HarborEnv(vf.CliAgentEnv):
    """CliAgentEnv subclass that loads Harbor-format tasks."""

    def __init__(
        self,
        run_command: str,
        dataset_path: str | Path,
        tasks: list[str] | None = None,
        agent_workdir: str = "/app",
        docker_image: str = "python:3.11-slim",
        **kwargs,
    ):
        self.dataset_path = Path(dataset_path)
        self.task_names = tasks
        self.agent_workdir = agent_workdir

        kwargs["docker_image"] = docker_image

        dataset = self.load_harbor_dataset()
        rubric = vf.Rubric(funcs=[self.harbor_reward], weights=[1.0])

        super().__init__(
            run_command=run_command, dataset=dataset, rubric=rubric, **kwargs
        )

    def load_harbor_dataset(self) -> Dataset:
        """Load Harbor tasks from dataset directory into a Dataset with prompts."""
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
                logger.warning(
                    f"Skipping {task_dir.name}: missing task.toml or instruction.md"
                )
                continue

            with open(task_toml, "rb") as f:
                config = tomllib.load(f)

            instruction = instruction_md.read_text().strip()

            messages = [{"role": "user", "content": instruction}]

            task_entry = {
                "example_id": len(tasks),
                "task": task_dir.name,
                "prompt": messages,
                "info": {
                    "task_dir": str(task_dir),
                    "docker_image": config.get("environment", {}).get("docker_image"),
                    "config": config,
                },
            }

            tasks.append(task_entry)

        if not tasks:
            raise ValueError(f"No valid Harbor tasks found in {self.dataset_path}")

        logger.info(f"Loaded {len(tasks)} Harbor tasks from {self.dataset_path}")
        return Dataset.from_list(tasks)

    async def get_docker_image(self, state: vf.State) -> str:
        """Get Docker image from task info, falling back to default."""
        task_info: dict[str, Any] = state.get("info") or {}
        return task_info.get("docker_image") or self.docker_image

    async def build_env_vars(self, state: vf.State) -> dict[str, str]:
        """Build env vars with Harbor-specific additions."""
        env_vars = await super().build_env_vars(state)
        env_vars.setdefault("HARBOR_TASK_NAME", state.get("task", ""))
        env_vars.setdefault("HARBOR_TASK_DIR", "/task")
        env_vars.setdefault("HARBOR_INSTRUCTION_PATH", "/task/instruction.md")
        if self.agent_workdir:
            env_vars.setdefault("AGENT_WORKDIR", self.agent_workdir)
        return env_vars

    async def post_sandbox_setup(
        self, state: vf.State, sandbox_client: AsyncSandboxClient
    ) -> None:
        """Upload Harbor task assets after sandbox creation."""
        task_info: dict[str, Any] = state.get("info", {}) or {}
        task_dir_str = task_info.get("task_dir", "")
        if not task_dir_str:
            raise ValueError("task_dir not set in task info")
        task_dir = Path(task_dir_str)
        config = task_info.get("config", {})

        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        await self.prepare_harbor_task(sandbox_client, state["sandbox_id"], task_dir)
        state["harbor_config"] = config
        state["harbor_task_dir"] = str(task_dir)

    async def prepare_harbor_task(
        self, sandbox_client: AsyncSandboxClient, sandbox_id: str, task_dir: Path
    ) -> None:
        """Upload task instruction only (oracle/tests uploaded after agent completes)."""
        instruction_path = task_dir / "instruction.md"
        task_toml_path = task_dir / "task.toml"

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                if instruction_path.exists():
                    tar.add(instruction_path, arcname="task/instruction.md")

                if task_toml_path.exists():
                    tar.add(task_toml_path, arcname="task/task.toml")

            remote_tar = "/tmp/harbor_task.tar.gz"
            await sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
            await sandbox_client.execute_command(
                sandbox_id,
                f"mkdir -p /task /logs/verifier {self.agent_workdir} && "
                f"tar -xzf {remote_tar} -C / && rm {remote_tar}",
                working_dir=None,
            )
            logger.debug(f"Uploaded task instruction for {task_dir.name}")
        finally:
            tar_path.unlink(missing_ok=True)

    async def upload_test_assets(
        self, sandbox_client: AsyncSandboxClient, sandbox_id: str, task_dir: Path
    ) -> None:
        """Upload oracle/tests after agent completes, right before running tests."""
        solution_dir = task_dir / "solution"
        tests_dir = task_dir / "tests"

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                if solution_dir.exists():
                    for item in solution_dir.iterdir():
                        tar.add(item, arcname=f"oracle/{item.name}")

                if tests_dir.exists():
                    for item in tests_dir.iterdir():
                        tar.add(item, arcname=f"tests/{item.name}")

            remote_tar = "/tmp/harbor_tests.tar.gz"
            await sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
            await sandbox_client.execute_command(
                sandbox_id,
                f"mkdir -p /oracle /tests && tar -xzf {remote_tar} -C / && rm {remote_tar}",
                working_dir=None,
            )
            logger.debug(f"Uploaded test assets for {task_dir.name}")
        finally:
            tar_path.unlink(missing_ok=True)

    async def post_rollout(self, state: vf.State):
        """Run Harbor tests to compute reward before sandbox destruction."""
        await super().post_rollout(state)
        state["reward"] = await self.compute_reward(state)

    async def harbor_reward(self, state: vf.State, **kwargs) -> float:
        return state.get("reward", 0.0)

    async def compute_reward(self, state: vf.State) -> float:
        """
        Execute Harbor tests (tests/test.sh) inside the sandbox to compute reward.
        Uploads oracle/tests first (they don't exist during agent execution).
        Prioritizes /logs/verifier/reward.txt, falling back to reward.json.
        """
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            logger.error("No sandbox_id in state")
            return 0.0

        task_dir_str = state.get("harbor_task_dir", "")
        if not task_dir_str:
            logger.error("harbor_task_dir not set in state")
            return 0.0
        task_dir = Path(task_dir_str)
        if not task_dir.exists():
            logger.error(f"Task directory not found: {task_dir}")
            return 0.0

        sandbox_client = AsyncSandboxClient()
        try:
            # Upload test assets now that agent has completed
            await self.upload_test_assets(sandbox_client, sandbox_id, task_dir)

            logger.info(f"Running Harbor tests for task {state.get('task')}")
            await sandbox_client.execute_command(
                sandbox_id, "bash test.sh", working_dir="/tests"
            )

            reward_result = await sandbox_client.execute_command(
                sandbox_id,
                "if [ -s /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; "
                "elif [ -s /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; fi",
                working_dir=None,
            )
            stdout_val = getattr(reward_result, "stdout", "")
            if stdout_val is None:
                reward_val = ""
            elif isinstance(stdout_val, str):
                reward_val = stdout_val.strip()
            else:
                reward_val = str(stdout_val).strip()
            if reward_val:
                try:
                    # Try as plain float first (reward.txt format)
                    value = float(reward_val)
                    logger.info(f"Reward from reward.txt: {value}")
                    return value
                except ValueError:
                    # Fall back to JSON (reward.json format)
                    data = json.loads(reward_val)
                    value = float(data.get("reward", 0.0))
                    logger.info(f"Reward from reward.json: {value}")
                    return value

            logger.warning("No reward.txt or reward.json produced by Harbor tests")
            return 0.0
        except Exception as e:
            logger.error(f"Error computing Harbor reward: {e}")
            return 0.0
