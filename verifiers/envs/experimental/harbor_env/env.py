import io
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset

import verifiers as vf
from verifiers.harbor import (
    HarborStep,
    harbor_step_info,
    harbor_step_summary,
    harbor_task_prompt,
    load_harbor_steps,
    make_harbor_tar,
    parse_harbor_rewards,
    run_harbor_steps,
)
from verifiers.utils.import_utils import load_toml

from .mcp import HarborMCPHealthcheck, HarborMCPMixin

logger = logging.getLogger(__name__)


class HarborEnv(HarborMCPMixin, vf.CliAgentEnv):
    """CliAgentEnv subclass that loads Harbor-format tasks."""

    def __init__(
        self,
        run_command: str,
        dataset_path: str | Path,
        tasks: list[str] | None = None,
        agent_workdir: str = "/app",
        docker_image: str = "python:3.11-slim",
        mcp_launch_commands: dict[str, str] | None = None,
        mcp_healthcheck: HarborMCPHealthcheck | None = None,
        **kwargs,
    ):
        self.dataset_path = Path(dataset_path)
        self.task_names = tasks
        self.agent_workdir = agent_workdir
        self.mcp_launch_commands = (
            mcp_launch_commands if mcp_launch_commands is not None else {}
        )
        self.mcp_healthcheck = (
            mcp_healthcheck if mcp_healthcheck is not None else HarborMCPHealthcheck()
        )

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

            if not task_toml.exists():
                logger.warning(f"Skipping {task_dir.name}: missing task.toml")
                continue

            with open(task_toml, "rb") as f:
                config = load_toml(f)

            try:
                steps = load_harbor_steps(task_dir, config)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Skipping {task_dir.name}: invalid Harbor steps: {e}")
                continue
            if not instruction_md.exists() and not steps:
                logger.warning(
                    f"Skipping {task_dir.name}: missing instruction.md or valid steps"
                )
                continue

            instruction = harbor_task_prompt(task_dir, steps)
            messages = [{"role": "user", "content": instruction}]

            task_entry = {
                "example_id": len(tasks),
                "prompt": messages,
                "info": {
                    "task_name": task_dir.name,
                    "task_dir": str(task_dir),
                    "docker_image": config.get("environment", {}).get("docker_image"),
                    "config": config,
                    "steps": [harbor_step_info(step) for step in steps],
                    "multi_step_reward_strategy": config.get(
                        "multi_step_reward_strategy", "mean"
                    ),
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
        task_info: dict[str, Any] = state.get("info") or {}
        env_vars.setdefault("HARBOR_TASK_NAME", task_info.get("task_name", ""))
        env_vars.setdefault("HARBOR_TASK_DIR", "/task")
        env_vars.setdefault("HARBOR_INSTRUCTION_PATH", "/task/instruction.md")
        if self.agent_workdir:
            env_vars.setdefault("AGENT_WORKDIR", self.agent_workdir)

        config: dict[str, Any] = task_info.get("config", {}) or {}
        for key, value in (await self.mcp_agent_env_vars(config, state)).items():
            env_vars.setdefault(key, value)
        return env_vars

    async def post_sandbox_setup(self, state: vf.State) -> None:
        """Upload Harbor task assets and start declared MCP servers."""
        task_info: dict[str, Any] = state.get("info", {}) or {}
        task_dir_str = task_info.get("task_dir", "")
        if not task_dir_str:
            raise ValueError("task_dir not set in task info")
        task_dir = Path(task_dir_str)
        config = task_info.get("config", {})

        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        sandbox_id = state["sandbox_id"]
        await self.prepare_harbor_task(sandbox_id, task_dir)
        state["harbor_task_dir"] = str(task_dir)

        await self.pre_mcp_setup(state)
        await self.start_mcp_servers(sandbox_id, config, state)

    async def pre_mcp_setup(self, state: vf.State) -> None:
        """Hook for installing dependencies or uploading code needed by MCP servers."""
        return None

    async def prepare_harbor_task(self, sandbox_id: str, task_dir: Path) -> None:
        """Upload task instruction only (oracle/tests uploaded after agent completes)."""
        instruction_path = task_dir / "instruction.md"
        task_toml_path = task_dir / "task.toml"

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                if instruction_path.exists():
                    tar.add(instruction_path, arcname="task/instruction.md")
                else:
                    if task_toml_path.exists():
                        with open(task_toml_path, "rb") as f:
                            config = load_toml(f)
                    else:
                        config = {}
                    instruction = harbor_task_prompt(
                        task_dir, load_harbor_steps(task_dir, config)
                    )
                    info = tarfile.TarInfo(name="task/instruction.md")
                    data = instruction.encode()
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))

                if task_toml_path.exists():
                    tar.add(task_toml_path, arcname="task/task.toml")

            remote_tar = "/tmp/harbor_task.tar.gz"
            await self.sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
            await self.sandbox_client.execute_command(
                sandbox_id,
                f"mkdir -p /task /logs/verifier {self.agent_workdir} && "
                f"tar -xzf {remote_tar} -C / && rm {remote_tar}",
                working_dir=None,
            )
            logger.debug(f"Uploaded task instruction for {task_dir.name}")
        finally:
            tar_path.unlink(missing_ok=True)

    async def upload_test_assets(
        self,
        sandbox_id: str,
        task_dir: Path,
        tests_dir: Path | None = None,
        solution_dir: Path | None = None,
        shared_tests_dir: Path | None = None,
    ) -> None:
        """Upload oracle/tests after agent completes, right before running tests.

        Multi-step Harbor tasks merge root tests first, then step tests so the
        step-local test.sh overrides the shared fallback/helper files.
        """
        tests_dir = task_dir / "tests" if tests_dir is None else tests_dir
        solution_dir = task_dir / "solution" if solution_dir is None else solution_dir

        directories = []
        if shared_tests_dir is not None:
            directories.append((shared_tests_dir, "tests"))
        directories.extend([(tests_dir, "tests"), (solution_dir, "oracle")])

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)
            tmp_file.write(make_harbor_tar(directories))

        try:
            remote_tar = "/tmp/harbor_tests.tar.gz"
            await self.sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
            await self.sandbox_client.execute_command(
                sandbox_id,
                f"mkdir -p /oracle /tests && tar -xzf {remote_tar} -C / && rm {remote_tar}",
                working_dir=None,
                timeout=900,
            )
            logger.debug(f"Uploaded test assets for {task_dir.name}")
        finally:
            tar_path.unlink(missing_ok=True)

    async def post_rollout(self, state: vf.State):
        """Run Harbor tests to compute reward before sandbox destruction."""
        await super().post_rollout(state)
        if isinstance(state.get("error"), vf.InfraError):
            logger.debug(f"Skipping Harbor tests due to prior error: {state['error']}")
            state["reward"] = 0.0
            return
        state["reward"] = await self.compute_reward(state)

    async def harbor_reward(self, state: vf.State, **kwargs) -> float:
        return state.get("reward", 0.0)

    async def compute_step_reward(
        self,
        state: vf.State,
        task_dir: Path,
        verifier_dir: Path,
        step: HarborStep | None,
    ) -> dict[str, float]:
        sandbox_id = state["sandbox_id"]
        await self.with_retry(self.upload_test_assets)(
            sandbox_id,
            task_dir,
            tests_dir=verifier_dir / "tests",
            solution_dir=verifier_dir / "solution",
            shared_tests_dir=task_dir / "tests" if verifier_dir != task_dir else None,
        )

        logger.info(
            "Running Harbor tests for task %s step %s",
            state.get("task"),
            step.name if step is not None else "single",
        )
        timeout = int((step.scoring_timeout if step is not None else None) or 300)
        results = await self.run_background_job(
            state,
            "bash test.sh",
            timeout=timeout,
            working_dir="/tests",
            poll_interval=5,
        )
        if getattr(results, "exit_code", 0) != 0:
            logger.warning(
                f"Harbor tests exit_code={results.exit_code} "
                f"stdout_len={len(getattr(results, 'stdout', '') or '')} "
                f"stderr_len={len(getattr(results, 'stderr', '') or '')}"
            )

        reward_result = await self.with_retry(self.sandbox_client.execute_command)(
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
        return parse_harbor_rewards(reward_val)

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

        task_info: dict[str, Any] = state.get("info") or {}
        steps = [
            HarborStep.model_validate(step) for step in task_info.get("steps") or []
        ]
        try:
            if steps:

                async def run_step(step: HarborStep) -> dict[str, float]:
                    return await self.compute_step_reward(
                        state, task_dir, step.task_dir, step
                    )

                strategy = task_info.get("multi_step_reward_strategy", "mean")
                step_results = await run_harbor_steps(steps, run_step)
                step_records, aggregate = harbor_step_summary(step_results, strategy)
                state["harbor_steps"] = step_records
                state["harbor_multi_step_reward"] = aggregate
                return float(aggregate.get("reward", 0.0))

            rewards = await self.compute_step_reward(state, task_dir, task_dir, None)
            if rewards:
                value = float(rewards.get("reward", 0.0))
                logger.info(f"Reward from Harbor verifier: {value}")
                return value
        except Exception as e:
            if state.get("error") is None:
                state["error"] = vf.SandboxError(str(e))
            logger.error(f"Error computing Harbor reward: {e}")
            return 0.0

        logger.warning("No reward.txt or reward.json produced by Harbor tests")
        return 0.0
