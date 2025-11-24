import asyncio
import json
import logging
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Any

import tomli
from datasets import Dataset
from prime_sandboxes import (
    AsyncSandboxClient,
    CreateSandboxRequest,
)

import verifiers as vf
from verifiers.envs.cli_agent_env import CliAgentEnv

logger = logging.getLogger(__name__)


class HarborCliAgentEnv(CliAgentEnv):
    """CliAgentEnv subclass that loads Harbor-format tasks."""

    def __init__(
        self,
        dataset_path: str | Path,
        tasks: list[str] | None = None,
        agent_workdir: str = "/app",
        default_docker_image: str | None = None,
        **kwargs,
    ):
        self.dataset_path = Path(dataset_path)
        self.task_names = tasks
        self.agent_workdir = agent_workdir
        self.default_docker_image = default_docker_image

        if default_docker_image and "docker_image" not in kwargs:
            kwargs["docker_image"] = default_docker_image

        dataset = self._load_harbor_dataset()
        rubric = vf.Rubric(funcs=[self.harbor_reward], weights=[1.0])

        super().__init__(dataset=dataset, rubric=rubric, **kwargs)

    def _load_harbor_dataset(self) -> Dataset:
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
                config = tomli.load(f)

            instruction = instruction_md.read_text().strip()

            docker_image = (
                config.get("environment", {}).get("docker_image")
                or self.default_docker_image
            )
            if not docker_image:
                logger.warning(
                    f"Skipping {task_dir.name}: no environment.docker_image in task.toml "
                    "and no default_docker_image provided. "
                    "Run harbor_build.py first to build/push images."
                )
                continue

            # TODO: remove this prompt
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an autonomous agent inside a Harbor sandbox. "
                        f"Work in {self.agent_workdir}, follow the user instruction, write /tmp/vf_complete "
                        "when finished, and do not tamper with /tests or /oracle."
                    ),
                },
                {"role": "user", "content": instruction},
            ]

            task_entry = {
                "example_id": len(tasks),
                "task": task_dir.name,
                "prompt": messages,
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

    async def setup_state(self, state: vf.State) -> vf.State:
        """Create sandbox per rollout with Harbor assets uploaded."""
        # Skip CliAgentEnv.setup_state (needs per-task docker image); call MultiTurnEnv.setup_state
        state = await super(CliAgentEnv, self).setup_state(state)  # type: ignore[misc]

        task_info: dict[str, Any] = state.get("info", {}) or {}
        task_dir = Path(task_info.get("task_dir", ""))
        config = task_info.get("config", {})
        docker_image = task_info.get("docker_image") or self.docker_image

        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        await self._ensure_interception_server()

        tunnel_url: str | None = None
        if self.interception_host is None:
            tunnel_url = await self._get_tunnel_url()
            state["interception_base_url"] = f"{tunnel_url}/rollout/{rollout_id}/v1"
        else:
            state["interception_base_url"] = (
                f"http://{self.interception_host}:{self.interception_port}/rollout/{rollout_id}/v1"
            )

        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = state["interception_base_url"]
        model = state.get("model")
        if model:
            env_vars["OPENAI_MODEL"] = model
        env_vars.setdefault("HARBOR_TASK_NAME", state.get("task", ""))
        env_vars.setdefault("HARBOR_TASK_DIR", "/task")
        env_vars.setdefault("HARBOR_INSTRUCTION_PATH", "/task/instruction.md")
        if self.agent_workdir:
            env_vars.setdefault("AGENT_WORKDIR", self.agent_workdir)

        sandbox_client = AsyncSandboxClient()
        sandbox_request = CreateSandboxRequest(
            name=f"harbor-cli-agent-{rollout_id}",
            docker_image=docker_image,
            start_command=self.start_command,
            cpu_cores=self.cpu_cores,
            memory_gb=self.memory_gb,
            disk_size_gb=self.disk_size_gb,
            gpu_count=self.gpu_count,
            timeout_minutes=self.timeout_minutes,
            environment_vars=env_vars,
            team_id=self.team_id,
            advanced_configs=self.advanced_configs,
        )

        logger.debug(
            f"Creating Harbor sandbox with OPENAI_BASE_URL={env_vars.get('OPENAI_BASE_URL')} "
            f"docker_image={docker_image}"
        )
        sandbox = await sandbox_client.create(sandbox_request)
        state["sandbox_id"] = sandbox.id
        await sandbox_client.wait_for_creation(sandbox.id)

        await self._prepare_harbor_task(sandbox_client, sandbox.id, task_dir)
        state["harbor_config"] = config
        state["harbor_task_dir"] = str(task_dir)

        request_id_queue: asyncio.Queue[str] = asyncio.Queue()
        state["request_id_queue"] = request_id_queue
        state["current_request_id"] = None
        state["tunnel_url"] = tunnel_url if self.interception_host is None else None
        self.active_rollouts[rollout_id] = {
            "request_id_queue": request_id_queue,
            "current_request_id": None,
        }

        return state

    async def _prepare_harbor_task(
        self, sandbox_client: AsyncSandboxClient, sandbox_id: str, task_dir: Path
    ) -> None:
        """Upload solution/tests and make log directory."""
        solution_dir = task_dir / "solution"
        tests_dir = task_dir / "tests"
        instruction_path = task_dir / "instruction.md"
        task_toml_path = task_dir / "task.toml"

        if solution_dir.exists():
            await self._upload_directory(
                sandbox_client, sandbox_id, solution_dir, "/oracle"
            )
            logger.debug(f"Uploaded solution for {task_dir.name} to /oracle")

        if tests_dir.exists():
            await self._upload_directory(
                sandbox_client, sandbox_id, tests_dir, "/tests"
            )
            logger.debug(f"Uploaded tests for {task_dir.name} to /tests")

        mkdir_cmd = f"mkdir -p /logs/verifier /task {self.agent_workdir}"
        await sandbox_client.execute_command(sandbox_id, mkdir_cmd, working_dir=None)

        if instruction_path.exists():
            await sandbox_client.upload_file(
                sandbox_id, "/task/instruction.md", str(instruction_path)
            )
        if task_toml_path.exists():
            await sandbox_client.upload_file(
                sandbox_id, "/task/task.toml", str(task_toml_path)
            )

    async def _upload_directory(
        self,
        sandbox_client: AsyncSandboxClient,
        sandbox_id: str,
        local_dir: Path,
        remote_path: str,
    ) -> None:
        """Tar + upload a directory into the sandbox."""
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                for item in local_dir.iterdir():
                    tar.add(item, arcname=item.name)

            remote_tar = f"/tmp/upload_{local_dir.name}.tar.gz"
            await sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
            await sandbox_client.execute_command(
                sandbox_id,
                f"mkdir -p {remote_path} && tar -xzf {remote_tar} -C {remote_path} && rm {remote_tar}",
                working_dir=None,
            )
        finally:
            tar_path.unlink(missing_ok=True)

    async def post_rollout(self, state: vf.State):
        """Run Harbor tests to compute reward before sandbox destruction."""
        reward = await self._compute_reward(state)
        state["harbor_reward_value"] = reward
        state["reward"] = reward

    async def harbor_reward(self, state: vf.State, **kwargs) -> float:
        return state.get("harbor_reward_value", 0.0)

    async def _compute_reward(self, state: vf.State) -> float:
        """
        Execute Harbor tests (tests/test.sh) inside the sandbox to compute reward.
        Prioritizes /logs/verifier/reward.txt, falling back to reward.json.
        """
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            logger.error("No sandbox_id in state")
            return 0.0

        sandbox_client = AsyncSandboxClient()
        try:
            logger.info(f"Running Harbor tests for task {state.get('task')}")
            await sandbox_client.execute_command(
                sandbox_id, "bash test.sh", working_dir="/tests"
            )

            reward_txt = await sandbox_client.execute_command(
                sandbox_id,
                "cat /logs/verifier/reward.txt 2>/dev/null || echo ''",
                working_dir=None,
            )
            reward_txt_val = self._stdout_text(reward_txt)
            if reward_txt_val:
                value = float(reward_txt_val)
                logger.info(f"Reward from reward.txt: {value}")
                return value

            reward_json = await sandbox_client.execute_command(
                sandbox_id,
                "cat /logs/verifier/reward.json 2>/dev/null || echo ''",
                working_dir=None,
            )
            reward_json_val = self._stdout_text(reward_json)
            if reward_json_val:
                data = json.loads(reward_json_val)
                value = float(data.get("reward", 0.0))
                logger.info(f"Reward from reward.json: {value}")
                return value

            logger.warning("No reward.txt or reward.json produced by Harbor tests")
            return 0.0
        except Exception as e:
            logger.error(f"Error computing Harbor reward: {e}")
            return 0.0

    @staticmethod
    def _stdout_text(result: Any) -> str:
        """Extract trimmed stdout from a Sandbox command result."""
        stdout_val = getattr(result, "stdout", "")
        if stdout_val is None:
            return ""
        if isinstance(stdout_val, str):
            return stdout_val.strip()
        return str(stdout_val).strip()
