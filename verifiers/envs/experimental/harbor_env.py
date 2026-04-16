import asyncio
import json
import logging
import shlex
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from datasets import Dataset

import verifiers as vf
from verifiers.utils.import_utils import load_toml

logger = logging.getLogger(__name__)


_NETWORK_TRANSPORTS = frozenset({"streamable-http", "http", "sse"})


@dataclass
class HarborMCPServer:
    """An MCP server entry parsed from `[[environment.mcp_servers]]` in task.toml."""

    name: str
    transport: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_network(self) -> bool:
        return self.transport in _NETWORK_TRANSPORTS


@dataclass
class HarborMCPLauncher:
    """How to start a network-transport MCP server inside the sandbox."""

    command: str
    user: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    bind_host: str = "127.0.0.1"
    transport_env_var: str = "MCP_TRANSPORT"
    port_env_var: str = "MCP_PORT"
    host_env_var: str = "MCP_BIND_HOST"


class HarborEnv(vf.CliAgentEnv):
    """CliAgentEnv subclass that loads Harbor-format tasks."""

    def __init__(
        self,
        run_command: str,
        dataset_path: str | Path,
        tasks: list[str] | None = None,
        agent_workdir: str = "/app",
        docker_image: str = "python:3.11-slim",
        mcp_launchers: dict[str, HarborMCPLauncher] | None = None,
        mcp_health_check_retries: int = 30,
        mcp_health_check_interval: float = 2.0,
        **kwargs,
    ):
        self.dataset_path = Path(dataset_path)
        self.task_names = tasks
        self.agent_workdir = agent_workdir
        self.mcp_launchers = mcp_launchers or {}
        self.mcp_health_check_retries = mcp_health_check_retries
        self.mcp_health_check_interval = mcp_health_check_interval

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
                config = load_toml(f)

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

        config: dict[str, Any] = (state.get("info") or {}).get("config", {}) or {}
        for server in self._parse_mcp_servers(config):
            url = self._mcp_agent_url(server)
            if url is None:
                continue
            key = f"HARBOR_MCP_{server.name.upper().replace('-', '_')}_URL"
            env_vars.setdefault(key, url)
        return env_vars

    async def post_sandbox_setup(self, state: vf.State) -> None:
        """Upload Harbor task assets and start framework-managed MCP servers."""
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
        state["harbor_config"] = config
        state["harbor_task_dir"] = str(task_dir)

        await self._start_task_mcp_servers(sandbox_id, config, state)

    async def _start_task_mcp_servers(
        self, sandbox_id: str, config: dict[str, Any], state: vf.State
    ) -> None:
        """Start every MCP server declared in `task.toml`."""
        servers = self._parse_mcp_servers(config)
        state["harbor_mcp_servers"] = servers
        state.setdefault("harbor_mcp_started", [])
        if not servers:
            return

        await self._patch_mcp_etc_hosts(sandbox_id, servers)

        for server in servers:
            if not server.is_network:
                continue
            launcher = self.mcp_launchers.get(server.name)
            if launcher is None:
                logger.debug(
                    "MCP server %r has transport %r but no launcher registered "
                    "— assuming externally managed",
                    server.name,
                    server.transport,
                )
                continue
            await self._start_mcp_server(
                sandbox_id, server, launcher, state, phase="agent"
            )

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

    def _parse_mcp_servers(self, config: dict[str, Any]) -> list[HarborMCPServer]:
        """Normalize `[[environment.mcp_servers]]` into typed entries."""
        raw_list = (config.get("environment") or {}).get("mcp_servers") or []
        servers: list[HarborMCPServer] = []
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                continue
            servers.append(
                HarborMCPServer(
                    name=str(name),
                    transport=str(entry.get("transport", "stdio")),
                    command=entry.get("command"),
                    args=list(entry.get("args") or []),
                    url=entry.get("url"),
                    raw=dict(entry),
                )
            )
        return servers

    def _mcp_agent_url(self, server: HarborMCPServer) -> str | None:
        """Rewrite the task.toml URL to point at 127.0.0.1 inside the sandbox."""
        if not server.is_network or not server.url:
            return None
        parsed = urlparse(server.url)
        port = self._mcp_url_port(server)
        netloc = f"127.0.0.1:{port}" if port is not None else "127.0.0.1"
        return urlunparse(parsed._replace(netloc=netloc))

    @staticmethod
    def _mcp_url_port(server: HarborMCPServer) -> int | None:
        """Extract the port a network MCP server is reachable on."""
        if not server.url:
            return None
        parsed = urlparse(server.url)
        if parsed.port is not None:
            return parsed.port
        if parsed.scheme == "https":
            return 443
        if parsed.scheme in ("http", ""):
            return 80
        return None

    def _mcp_extra_env(
        self, server: HarborMCPServer, state: vf.State, phase: str
    ) -> dict[str, str]:
        """Override hook for env vars injected into a managed MCP process."""
        return {}

    @staticmethod
    def _mcp_pid_file(name: str) -> str:
        return f"/tmp/harbor-mcp-{name}.pid"

    @staticmethod
    def _mcp_log_file(name: str) -> str:
        return f"/tmp/harbor-mcp-{name}.log"

    def _mcp_start_cmd(
        self,
        server: HarborMCPServer,
        launcher: HarborMCPLauncher,
        port: int,
        extra_env: dict[str, str],
    ) -> str:
        """Build the shell command to launch one MCP server."""
        env_pairs: dict[str, str] = {}
        env_pairs[launcher.transport_env_var] = server.transport
        env_pairs[launcher.port_env_var] = str(port)
        env_pairs[launcher.host_env_var] = launcher.bind_host
        env_pairs.update(launcher.env)
        env_pairs.update(extra_env)

        env_prefix = " ".join(
            f"{k}={shlex.quote(str(v))}" for k, v in env_pairs.items()
        )

        pid_file = self._mcp_pid_file(server.name)
        log_file = self._mcp_log_file(server.name)
        cd_prefix = f"cd {shlex.quote(launcher.cwd)} && " if launcher.cwd else ""
        inner = (
            f"{cd_prefix}{env_prefix} "
            f"nohup {launcher.command} > {log_file} 2>&1 & "
            f"echo $! > {pid_file}"
        )

        if launcher.user:
            return f"su -s /bin/sh {shlex.quote(launcher.user)} -c {shlex.quote(inner)}"
        return inner

    @staticmethod
    def _mcp_stop_cmd(name: str) -> str:
        """Idempotent SIGTERM-then-SIGKILL by PID file. No-ops if not running."""
        pid_file = HarborEnv._mcp_pid_file(name)
        return (
            f"if [ -f {pid_file} ]; then "
            f'pid="$(cat {pid_file})"; '
            f'kill "$pid" 2>/dev/null || true; '
            f"for _ in 1 2 3 4 5; do "
            f'kill -0 "$pid" 2>/dev/null || break; '
            f"sleep 1; "
            f"done; "
            f'kill -9 "$pid" 2>/dev/null || true; '
            f"rm -f {pid_file}; "
            f"fi"
        )

    async def _patch_mcp_etc_hosts(
        self, sandbox_id: str, servers: list[HarborMCPServer]
    ) -> None:
        """Ensure task.toml service-name URLs resolve to 127.0.0.1."""
        hosts: set[str] = set()
        for server in servers:
            if not server.url:
                continue
            host = urlparse(server.url).hostname
            if not host:
                continue
            if host in ("localhost", "127.0.0.1", "::1"):
                continue
            hosts.add(host)
        if not hosts:
            return

        # Compose one shell line per host. `grep -qw` (whole-word match) prevents
        # duplicate entries if post_sandbox_setup is ever re-run for the same
        # sandbox. Hostnames are passed through `shlex.quote` since they
        # originate from task.toml.
        statements = " && ".join(
            f"(grep -qw {shlex.quote(h)} /etc/hosts || "
            f"echo {shlex.quote(f'127.0.0.1 {h}')} >> /etc/hosts)"
            for h in sorted(hosts)
        )
        await self.sandbox_client.execute_command(
            sandbox_id, statements, working_dir=None
        )

    async def _start_mcp_server(
        self,
        sandbox_id: str,
        server: HarborMCPServer,
        launcher: HarborMCPLauncher,
        state: vf.State,
        *,
        phase: str = "agent",
    ) -> None:
        """Start a single MCP server and wait for it to accept connections."""
        if not server.is_network:
            raise ValueError(
                f"MCP server {server.name!r} has stdio transport; the framework "
                "does not manage its lifecycle"
            )
        port = self._mcp_url_port(server)
        if port is None:
            raise ValueError(
                f"MCP server {server.name!r} has no port in its URL {server.url!r}"
            )

        extra_env = self._mcp_extra_env(server, state, phase) or {}
        cmd = self._mcp_start_cmd(server, launcher, port, extra_env)

        await self.sandbox_client.execute_command(sandbox_id, cmd, working_dir=None)
        await self._wait_for_mcp_server(sandbox_id, server.name, port)

        started: list[str] = state.setdefault("harbor_mcp_started", [])
        if server.name not in started:
            started.append(server.name)
        logger.info(
            "MCP server %r ready on port %d (phase=%s)", server.name, port, phase
        )

    async def _stop_mcp_server(
        self, sandbox_id: str, name: str, state: vf.State | None = None
    ) -> None:
        """Stop a single MCP server and remove it from the rollout's started list."""
        await self.sandbox_client.execute_command(
            sandbox_id, self._mcp_stop_cmd(name), working_dir=None
        )
        if state is not None:
            started = state.get("harbor_mcp_started") or []
            if name in started:
                started.remove(name)
                state["harbor_mcp_started"] = started

    async def _wait_for_mcp_server(self, sandbox_id: str, name: str, port: int) -> None:
        """Poll until the server is accepting TCP connections on `localhost:port`."""
        health_cmd = f"bash -c 'exec 3<>/dev/tcp/127.0.0.1/{port}' 2>/dev/null"
        for attempt in range(self.mcp_health_check_retries):
            result = await self.sandbox_client.execute_command(
                sandbox_id, health_cmd, working_dir=None, timeout=10
            )
            if getattr(result, "exit_code", 1) == 0:
                logger.debug(
                    "MCP server %r healthy on port %d after %d checks",
                    name,
                    port,
                    attempt + 1,
                )
                return
            await asyncio.sleep(self.mcp_health_check_interval)

        log_tail = await self.sandbox_client.execute_command(
            sandbox_id,
            f"tail -n 40 {self._mcp_log_file(name)} 2>/dev/null || true",
            working_dir=None,
        )
        raise vf.SandboxError(
            f"MCP server {name!r} on port {port} failed health check after "
            f"{self.mcp_health_check_retries} attempts. "
            f"Recent log tail:\n{(getattr(log_tail, 'stdout', '') or '').strip()}"
        )

    @vf.cleanup
    async def stop_mcp_servers(self, state: vf.State) -> None:
        """Stop every framework-managed MCP server started for this rollout."""
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return
        started = list(state.get("harbor_mcp_started") or [])
        for name in started:
            try:
                await self.sandbox_client.execute_command(
                    sandbox_id, self._mcp_stop_cmd(name), working_dir=None
                )
            except Exception as e:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Failed to stop MCP server %r: %s", name, e)
        state["harbor_mcp_started"] = []

    async def upload_test_assets(self, sandbox_id: str, task_dir: Path) -> None:
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

        try:
            await self.with_retry(self.upload_test_assets)(sandbox_id, task_dir)

            logger.info(f"Running Harbor tests for task {state.get('task')}")
            results = await self.run_background_job(
                state,
                "bash test.sh",
                timeout=300,
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
                value = float(reward_val)
                logger.info(f"Reward from reward.txt: {value}")
                return value
            except ValueError:
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
