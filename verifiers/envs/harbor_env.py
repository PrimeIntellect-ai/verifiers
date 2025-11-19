"""
Supports Harbor task format with:
- instruction.md: Task instruction
- task.toml: Task configuration
- environment/: Environment definition (Dockerfile)
- solution/: Solution scripts
- tests/: Test scripts (outputs reward.txt or reward.json)
"""

import json
import logging
import tarfile
import tempfile
from pathlib import Path

import tomli
from datasets import Dataset

import verifiers as vf
from verifiers.envs.sandbox_env import SandboxEnv

logger = logging.getLogger(__name__)


class HarborEnv(SandboxEnv):
    """
    Environment for Harbor-format tasks using Prime sandboxes.

    Each Harbor task is loaded as a separate example in the dataset.
    The agent is given the task instruction and has access to a bash tool
    to complete the task. After the agent finishes, tests are run to verify
    completion and compute the reward.
    """

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        tasks: list[str] | None = None,
        timeout_minutes: int = 60,
        cpu_cores: int = 2,
        memory_gb: int = 4,
        disk_size_gb: int = 10,
        **kwargs,
    ):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.task_names = tasks
        self.timeout_minutes = timeout_minutes
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb

        dataset = self._load_harbor_dataset()
        rubric = vf.Rubric(funcs=[self.harbor_reward], weights=[1.0])

        super().__init__(
            dataset=dataset,
            rubric=rubric,
            timeout_minutes=timeout_minutes,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            **kwargs,
        )

    def _load_harbor_dataset(self) -> Dataset:
        """Load Harbor tasks from dataset directory into HuggingFace Dataset."""
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        tasks = []
        task_dirs = sorted(self.dataset_path.iterdir())

        for task_dir in task_dirs:
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
                config = tomli.load(f)

            instruction = instruction_md.read_text().strip()

            docker_image = config.get("environment", {}).get("docker_image")
            if not docker_image:
                logger.warning(
                    f"Skipping {task_dir.name}: no docker_image in task.toml. "
                    f"Run harbor_build.py first to build and push images."
                )
                continue

            task_entry = {
                "example_id": len(tasks),
                "task": task_dir.name,
                "question": instruction,
                "answer": "",  # Harbor tasks don't have reference answers
                "info": {
                    "task_dir": str(task_dir),
                    "docker_image": docker_image,
                    "config": config,
                },
            }

            tasks.append(task_entry)

        if not tasks:
            raise ValueError(f"No valid Harbor tasks found in {self.dataset_path}")

        logger.info(f"Loaded {len(tasks)} Harbor tasks from {self.dataset_path}")

        return Dataset.from_list(tasks)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """
        Creates the sandbox, uploads solution and test files,
        and prepares the environment for the agent.
        """
        task_info = state.get("info", {})
        task_dir = Path(task_info.get("task_dir", ""))
        docker_image = task_info.get("docker_image")
        config = task_info.get("config", {})

        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        if docker_image:
            self.sandbox_request.docker_image = docker_image

        state = await super().setup_state(state, **kwargs)

        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            raise RuntimeError("Sandbox not created in parent setup_state")

        await self.sandbox_client.wait_for_creation(sandbox_id)

        # Upload solution folder to /oracle
        solution_dir = task_dir / "solution"
        if solution_dir.exists():
            await self._upload_directory(sandbox_id, solution_dir, "/oracle")
            logger.debug(f"Uploaded solution to /oracle in sandbox {sandbox_id}")

        # Upload tests folder to /tests
        tests_dir = task_dir / "tests"
        if tests_dir.exists():
            await self._upload_directory(sandbox_id, tests_dir, "/tests")
            logger.debug(f"Uploaded tests to /tests in sandbox {sandbox_id}")

        # Create /logs/verifier directory for test outputs (run from root)
        await self.bash("mkdir -p /logs/verifier", sandbox_id, working_dir=None)

        # Store task config in state for reward function
        state["harbor_config"] = config
        state["harbor_task_dir"] = str(task_dir)

        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        """Inject working_dir=/app for agent bash calls."""
        updated_args = super().update_tool_args(
            tool_name, tool_args, messages, state, **kwargs
        )

        # For bash commands, inject working_dir=/app for agent calls
        if tool_name == "bash" and "working_dir" not in updated_args:
            updated_args["working_dir"] = "/app"

        return updated_args

    async def _upload_directory(
        self, sandbox_id: str, local_dir: Path, remote_path: str
    ):
        """Upload a directory to the sandbox using tar."""
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                for item in local_dir.iterdir():
                    tar.add(item, arcname=item.name)

            # Upload tar to sandbox (upload_file params are sandbox_id, remote_path, local_path)
            remote_tar = f"/tmp/upload_{local_dir.name}.tar.gz"
            await self.sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))

            # Extract in sandbox (run from root to access /tmp and create directories)
            await self.bash(
                f"mkdir -p {remote_path} && tar -xzf {remote_tar} -C {remote_path} && rm {remote_tar}",
                sandbox_id,
                working_dir=None,
            )
        finally:
            tar_path.unlink(missing_ok=True)

    async def post_rollout(self, state: vf.State):
        reward = await self._compute_reward(state)
        state["harbor_reward_value"] = reward

    async def harbor_reward(self, state: vf.State, **kwargs) -> float:
        return state.get("harbor_reward_value", 0.0)

    async def _compute_reward(self, state: vf.State) -> float:
        """
        Compute reward by running Harbor tests.

        Harbor tests should output either:
        - /logs/verifier/reward.txt: Single number (0 or 1 typically)
        - /logs/verifier/reward.json: JSON with "reward" field

        Returns:
            float: Reward value (typically 0 or 1)
        """
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            logger.error("No sandbox_id in state")
            return 0.0

        try:
            logger.info(f"Running tests for task {state.get('task')}")
            result = await self.bash("bash test.sh", sandbox_id, working_dir="/tests")
            logger.debug(f"Test script output: {result}")

            # Try to read reward.txt first
            reward_txt_result = await self.bash(
                "cat /logs/verifier/reward.txt 2>/dev/null || echo ''",
                sandbox_id,
                working_dir=None,
            )

            if reward_txt_result and reward_txt_result.strip():
                reward_value = float(reward_txt_result.strip())
                logger.info(f"Reward from reward.txt: {reward_value}")
                return reward_value

            # Fall back to reward.json
            reward_json_result = await self.bash(
                "cat /logs/verifier/reward.json 2>/dev/null || echo ''",
                sandbox_id,
                working_dir=None,
            )

            if reward_json_result and reward_json_result.strip():
                reward_data = json.loads(reward_json_result)
                reward_value = float(reward_data.get("reward", 0.0))
                logger.info(f"Reward from reward.json: {reward_value}")
                return reward_value

            logger.warning(
                f"No reward.txt or reward.json found for task {state.get('task')}"
            )
            return 0.0

        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            return 0.0
