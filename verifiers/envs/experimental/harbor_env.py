import json
import logging
import math
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import tenacity as tc

try:
    import tomllib  # type: ignore[unresolved-import]
except ImportError:
    import tomli as tomllib  # type: ignore[unresolved-import]
from datasets import Dataset
from prime_sandboxes import AsyncSandboxClient

import verifiers as vf
from verifiers.envs.experimental.sandbox_mixin import ThreadedAsyncSandboxClient

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
        verifier_timeout_seconds: float = -1,  # -1 = use task.toml [verifier].timeout_sec
        capture_episode_artifacts: bool = False,
        **kwargs,
    ):
        self.dataset_path = Path(dataset_path)
        self.task_names = tasks
        self.agent_workdir = agent_workdir
        self.verifier_timeout_seconds = verifier_timeout_seconds
        self.capture_episode_artifacts = capture_episode_artifacts

        kwargs["docker_image"] = docker_image

        dataset = self.load_harbor_dataset()
        rubric = vf.Rubric(funcs=[self.harbor_reward], weights=[1.0])

        super().__init__(
            run_command=run_command, dataset=dataset, rubric=rubric, **kwargs
        )

    async def _list_episode_dirs(self, sandbox_id: str) -> list[str]:
        """List Terminus episode directories under /logs."""
        try:
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                "ls -1 /logs 2>/dev/null | grep '^episode-' || true",
                working_dir=None,
                timeout=30,
            )
        except Exception as e:
            logger.warning(f"Failed to list /logs/agent for {sandbox_id}: {e}")
            return []

        stdout = getattr(result, "stdout", "") or ""
        episodes = [line.strip() for line in stdout.splitlines() if line.strip()]

        def _sort_key(name: str):
            try:
                return int(name.split("-", 1)[1])
            except Exception:
                return name

        return sorted(episodes, key=_sort_key)

    async def _download_text_file(
        self, sandbox_id: str, remote_path: str, timeout: int = 60
    ) -> str | None:
        """Download a sandbox file and return its text content (or None if missing)."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            local_path = Path(tmp_file.name)

        try:
            await self.sandbox_client.download_file(
                sandbox_id, remote_path, str(local_path), timeout=timeout
            )
        except Exception as e:
            logger.debug(f"Skipping {remote_path} for {sandbox_id}: {e}")
            local_path.unlink(missing_ok=True)
            return None

        try:
            return local_path.read_text(encoding="utf-8", errors="replace")
        finally:
            local_path.unlink(missing_ok=True)

    async def capture_terminus_artifacts(self, state: vf.State) -> None:
        """Capture Terminus per-episode artifacts from /logs/agent into state."""
        if not self.capture_episode_artifacts:
            return

        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return

        episodes = await self._list_episode_dirs(sandbox_id)
        prompts: dict[str, str | None] = {}
        responses: dict[str, str | None] = {}
        debugs: dict[str, str | None] = {}

        for episode in episodes:
            base = f"/logs/{episode}"
            prompts[episode] = await self._download_text_file(
                sandbox_id, f"{base}/prompt.txt"
            )
            responses[episode] = await self._download_text_file(
                sandbox_id, f"{base}/response.txt"
            )
            debugs[episode] = await self._download_text_file(
                sandbox_id, f"{base}/debug.json"
            )

        trajectory = await self._download_text_file(sandbox_id, "/logs/trajectory.json")

        state["terminus_episode_prompts"] = prompts
        state["terminus_episode_responses"] = responses
        state["terminus_episode_debug"] = debugs
        state["terminus_trajectory_json"] = trajectory

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

    async def get_timeout_seconds(self, state: vf.State) -> float:
        """Get timeout: user override > task.toml > default. Use -1 for per-task config."""
        if self.timeout_seconds > 0:
            return self.timeout_seconds  # User override
        config = (state.get("info") or {}).get("config", {})
        timeout = config.get("agent", {}).get("timeout_sec")
        return float(timeout) if timeout else 3600.0  # task.toml or default

    async def get_verifier_timeout_seconds(self, state: vf.State) -> float:
        """Get verifier timeout: user override > task.toml > default. Use -1 for per-task config."""
        if self.verifier_timeout_seconds > 0:
            return self.verifier_timeout_seconds  # User override
        config = (state.get("info") or {}).get("config", {})
        timeout = config.get("verifier", {}).get("timeout_sec")
        return float(timeout) if timeout else 300.0  # task.toml or default

    async def get_sandbox_timeout_minutes(self, state: vf.State) -> int:
        """Get sandbox lifetime: user override > per-task agent+verifier (no buffer)."""
        if self.timeout_minutes > 0:
            return self.timeout_minutes  # User override
        agent_timeout = await self.get_timeout_seconds(state)
        verifier_timeout = await self.get_verifier_timeout_seconds(state)
        total_seconds = agent_timeout + verifier_timeout
        return max(1, math.ceil(total_seconds / 60.0))

    async def get_sandbox_request(
        self, state: vf.State, env_vars: dict[str, str], docker_image: str
    ):
        """Build request: user override > task.toml > CreateSandboxRequest default. Use -1 for per-task config."""
        request = await super().get_sandbox_request(state, env_vars, docker_image)
        config = (state.get("info") or {}).get("config", {})
        env_config = config.get("environment", {})

        updates = {}
        # Only apply task.toml values if user didn't explicitly override (-1 = use per-task)
        if self.cpu_cores < 0 and (cpus := env_config.get("cpus")):
            updates["cpu_cores"] = int(cpus)
        if self.memory_gb < 0 and (mem := env_config.get("memory")):
            updates["memory_gb"] = (
                int(str(mem).rstrip("gG"))
                if str(mem).upper().endswith("G")
                else int(mem)
            )
        if self.disk_size_gb < 0 and (storage := env_config.get("storage")):
            updates["disk_size_gb"] = (
                int(str(storage).rstrip("gG"))
                if str(storage).upper().endswith("G")
                else int(storage)
            )

        if self.timeout_minutes < 0:
            updates["timeout_minutes"] = await self.get_sandbox_timeout_minutes(state)

        return request.model_copy(update=updates) if updates else request

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
        self,
        state: vf.State,
        sandbox_client: AsyncSandboxClient | ThreadedAsyncSandboxClient,
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
        self,
        sandbox_client: AsyncSandboxClient | ThreadedAsyncSandboxClient,
        sandbox_id: str,
        task_dir: Path,
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
                timeout=900,
            )
            logger.debug(f"Uploaded test assets for {task_dir.name}")
        finally:
            tar_path.unlink(missing_ok=True)

    async def post_rollout(self, state: vf.State):
        """Run Harbor tests to compute reward before sandbox destruction."""
        await super().post_rollout(state)
        await self.capture_terminus_artifacts(state)
        if isinstance(state.get("error"), vf.InfraError):
            logger.debug(f"Skipping Harbor tests due to prior error: {state['error']}")
            state["reward"] = 0.0
            return
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

        def _is_retryable(exc: Exception) -> bool:
            msg = str(exc).lower()
            return (
                "sandbox_not_ready" in msg
                or "sandbox is not ready" in msg
                or "sandbox is no longer running" in msg
            )

        # Use tenacity for transient sandbox readiness issues (seen as HTTP 409s).
        retry_upload = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(6),
            wait=tc.wait_exponential_jitter(
                initial=1.0, exp_base=2, max=20.0, jitter=0.2
            ),
            retry=tc.retry_if_exception(_is_retryable),
            before_sleep=tc.before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ).wraps

        retry_short = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(3),
            wait=tc.wait_exponential_jitter(
                initial=1.0, exp_base=2, max=10.0, jitter=0.2
            ),
            retry=tc.retry_if_exception(_is_retryable),
            before_sleep=tc.before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ).wraps

        try:
            # Upload test assets now that agent has completed (retry if sandbox not ready)
            await retry_upload(self.upload_test_assets)(
                sandbox_client, sandbox_id, task_dir
            )

            logger.info(f"Running Harbor tests for task {state.get('task')}")
            verifier_timeout = await self.get_verifier_timeout_seconds(state)
            results = await self.run_background_job(
                state,
                "bash test.sh",
                timeout=int(verifier_timeout),
                working_dir="/tests",
                poll_interval=5,
                sandbox_client=sandbox_client,
                start_retry=retry_short,
                poll_retry=retry_short,
            )
            if getattr(results, "exit_code", 0) != 0:
                logger.warning(
                    f"Harbor tests exit_code={results.exit_code} "
                    f"stdout_len={len(getattr(results, 'stdout', '') or '')} "
                    f"stderr_len={len(getattr(results, 'stderr', '') or '')}"
                )

            reward_result = await retry_short(sandbox_client.execute_command)(
                sandbox_id,
                "if [ -s /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; "
                "elif [ -s /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; fi",
                working_dir=None,
            )
        except Exception as e:
            if state.get("error") is None:
                state["error"] = vf.SandboxError(str(e))
            logger.error(f"Error computing Harbor reward: {e}")
            return 0.0

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
                try:
                    data = json.loads(reward_val)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid reward.json: {e}")
                    return 0.0
                value = float(data.get("reward", 0.0))
                logger.info(f"Reward from reward.json: {value}")
                return value

        logger.warning("No reward.txt or reward.json produced by Harbor tests")
        return 0.0
