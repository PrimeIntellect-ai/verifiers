from __future__ import annotations

import asyncio
import json
import logging
import math
import shlex
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import verifiers as vf
from tasksets.base import SandboxSpec, SandboxTaskSet
from verifiers.envs.composable_tools import TaskTools
from verifiers.utils.import_utils import load_toml

logger = logging.getLogger(__name__)

SANDBOX_TIMEOUT_BUFFER_SECONDS = 900.0
NETWORK_TRANSPORTS = frozenset({"streamable-http", "http", "sse"})
MCP_EXTRA_CONFIG_KEYS = (
    "enabled",
    "startup_timeout_sec",
    "disabled_tools",
    "enabled_tools",
    "env_vars",
    "bearer_token_env_var",
    "oauth_resource",
    "env",
    "headers",
    "http_headers",
    "tools",
    "cwd",
    "workingDirectory",
    "connectionTimeoutMs",
)


@dataclass
class HarborMCPHealthcheck:
    command: str | None = None
    retries: int = 30
    interval_sec: float = 2.0
    start_period_sec: float = 0.0
    timeout_sec: float = 10.0


@dataclass
class HarborMCPServer:
    name: str
    transport: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_network(self) -> bool:
        return self.transport in NETWORK_TRANSPORTS

    def to_harness_config(self, *, url: str | None = None) -> dict[str, Any]:
        config = {"name": self.name, "transport": self.transport, **self.extra}
        if self.command is not None:
            config["command"] = self.command
        if self.args:
            config["args"] = list(self.args)
        if url or self.url:
            config["url"] = url or self.url
        return config


def parse_mcp_servers(config: dict[str, Any]) -> list[HarborMCPServer]:
    raw_servers = (config.get("environment") or {}).get("mcp_servers") or []
    servers: list[HarborMCPServer] = []
    for entry in raw_servers:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue

        transport = str(entry.get("transport") or "sse")
        command = entry.get("command")
        url = entry.get("url")
        name = str(entry["name"])
        if transport in NETWORK_TRANSPORTS and not url:
            raise ValueError(
                f"MCP server {name!r}: 'url' is required for transport {transport!r}"
            )
        if transport == "stdio" and not command:
            raise ValueError(
                f"MCP server {name!r}: 'command' is required for transport 'stdio'"
            )

        extra = {key: entry[key] for key in MCP_EXTRA_CONFIG_KEYS if key in entry}
        servers.append(
            HarborMCPServer(
                name=name,
                transport=transport,
                command=command,
                args=list(entry.get("args") or []),
                url=url,
                extra=extra,
            )
        )
    return servers


def mcp_url_port(server: HarborMCPServer) -> int | None:
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


def mcp_agent_url(server: HarborMCPServer) -> str | None:
    if not server.is_network or not server.url:
        return None
    parsed = urlparse(server.url)
    port = mcp_url_port(server)
    host = f"127.0.0.1:{port}" if port is not None else "127.0.0.1"
    at_idx = parsed.netloc.rfind("@")
    userinfo = parsed.netloc[: at_idx + 1] if at_idx >= 0 else ""
    return urlunparse(parsed._replace(netloc=f"{userinfo}{host}"))


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


def _parse_env_vars(environment: dict[str, Any]) -> dict[str, str]:
    raw = (
        environment.get("environment_vars")
        or environment.get("env_vars")
        or environment.get("env")
        or {}
    )
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def build_harbor_sandbox_spec(
    config: dict[str, Any],
    *,
    default_docker_image: str = "python:3.11-slim",
) -> SandboxSpec:
    environment = config.get("environment", {})
    agent = config.get("agent", {})
    verifier = config.get("verifier", {})
    timeout_seconds = (
        max(
            float(agent.get("timeout_sec", 3600.0)),
            float(verifier.get("timeout_sec", 3600.0)),
        )
        + SANDBOX_TIMEOUT_BUFFER_SECONDS
    )

    return SandboxSpec(
        image=environment.get("docker_image") or default_docker_image,
        start_command=environment.get("start_command") or "tail -f /dev/null",
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
        gpu_type=environment.get("gpu_type"),
        vm=environment.get("vm"),
        timeout_minutes=max(1, int(math.ceil(timeout_seconds / 60.0))),
        environment_vars=_parse_env_vars(environment),
    )


class HarborRubric(vf.Rubric):
    def __init__(self, taskset: HarborTaskSet, **kwargs: Any):
        super().__init__(**kwargs)
        self.taskset = taskset
        self.add_reward_func(self.solved)

    async def solved(self, state: vf.State, info: dict, **kwargs: Any) -> float:
        if isinstance(state.get("error"), vf.InfraError):
            return 0.0

        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            return 0.0

        test_output = await self.taskset.run_tests(
            sandbox_client=sandbox_client,
            sandbox_id=sandbox_id,
            state=state,
            test_timeout=int(state.get("test_timeout", 900)),
        )
        state["test_output"] = test_output
        return self.taskset.calculate_reward(test_output, info)

    @vf.cleanup
    async def cleanup_sandbox(self, state: vf.State) -> None:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            await self.taskset.stop_mcp_servers(state)
            await sandbox_client.delete(sandbox_id)


class HarborTaskSet(SandboxTaskSet):
    """Harbor-format task directory collection."""

    default_workdir = "/app"

    def __init__(
        self,
        dataset_path: str | Path,
        tasks: list[str] | None = None,
        agent_workdir: str = "/app",
        docker_image: str = "python:3.11-slim",
        name: str | None = None,
        mcp_launch_commands: dict[str, str] | None = None,
        mcp_healthcheck: HarborMCPHealthcheck | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.task_names = tasks
        self.agent_workdir = agent_workdir
        self.docker_image = docker_image
        self.mcp_launch_commands = dict(mcp_launch_commands or {})
        self.mcp_healthcheck = mcp_healthcheck or HarborMCPHealthcheck()
        super().__init__(
            dataset=self._load_dataset(),
            name=name or f"harbor/{self.dataset_path.name}",
        )

    def _load_dataset(self) -> Any:
        from datasets import Dataset

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        rows: list[dict[str, Any]] = []
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

            with open(task_toml, "rb") as file:
                config = load_toml(file)

            extra_info: dict[str, Any] = {}
            info_json = task_dir / "info.json"
            if info_json.exists():
                extra_info = json.loads(info_json.read_text())

            rows.append(
                {
                    "example_id": len(rows),
                    "task": task_dir.name,
                    "answer": "",
                    "info": {
                        "task_dir": str(task_dir),
                        "task_name": task_dir.name,
                        "config": config,
                        "docker_image": config.get("environment", {}).get(
                            "docker_image"
                        ),
                        **extra_info,
                    },
                }
            )

        if not rows:
            raise ValueError(f"No valid Harbor tasks found in {self.dataset_path}")
        return Dataset.from_list(rows)

    def get_instruction(self, info: dict) -> str:
        return (Path(info["task_dir"]) / "instruction.md").read_text().strip()

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return build_harbor_sandbox_spec(
            dict(info.get("config") or {}),
            default_docker_image=self.docker_image,
        )

    def get_workdir(self, info: dict) -> str:
        return self.agent_workdir

    def get_env_vars(self, info: dict | None = None) -> dict[str, str]:
        task_name = ""
        if info:
            task_name = str(info.get("task_name") or Path(info["task_dir"]).name)
        return {
            "HARBOR_TASK_NAME": task_name,
            "HARBOR_TASK_DIR": "/task",
            "HARBOR_INSTRUCTION_PATH": "/task/instruction.md",
        }

    def get_tools(self, info: dict) -> TaskTools:
        config = dict(info.get("config") or {})
        return TaskTools(
            mcp_servers=[
                server.to_harness_config() for server in parse_mcp_servers(config)
            ]
        )

    async def prepare_tools(self, state: vf.State, tools: TaskTools) -> TaskTools:
        info = state.get("info") or {}
        servers = parse_mcp_servers(dict(info.get("config") or {}))
        if not servers:
            return tools

        sandbox_id = state["sandbox_id"]
        resolved_servers: list[dict[str, Any] | str] = []
        env_vars = dict(tools.env_vars)
        managed_servers: list[tuple[HarborMCPServer, str]] = []
        for server in servers:
            resolved_url = server.url
            if server.is_network and server.url:
                command = await self.mcp_launch_command(server, state)
                if command is not None:
                    managed_servers.append((server, command))
                    resolved_url = mcp_agent_url(server)
                env_vars[f"HARBOR_MCP_{self._mcp_env_name(server.name)}_URL"] = (
                    resolved_url or server.url
                )
            resolved_servers.append(server.to_harness_config(url=resolved_url))

        if managed_servers:
            await self.start_mcp_servers(sandbox_id, managed_servers, state)
        return TaskTools(mcp_servers=resolved_servers, env_vars=env_vars)

    async def mcp_launch_command(
        self, server: HarborMCPServer, state: vf.State
    ) -> str | None:
        return self.mcp_launch_commands.get(server.name)

    async def start_mcp_servers(
        self,
        sandbox_id: str,
        servers: list[tuple[HarborMCPServer, str]],
        state: vf.State,
    ) -> None:
        await self._patch_mcp_etc_hosts(
            sandbox_id, [server for server, _ in servers], state
        )
        tasks = [
            asyncio.create_task(
                self._start_mcp_server(sandbox_id, server, command, state),
                name=f"harbor-mcp-start-{server.name}",
            )
            for server, command in servers
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def stop_mcp_servers(self, state: vf.State) -> None:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        jobs = state.get("harbor_mcp_jobs") or {}
        if not sandbox_client or not sandbox_id:
            return
        for name in list(jobs):
            try:
                await sandbox_client.execute_command(
                    sandbox_id, self._mcp_stop_cmd(name), working_dir=None
                )
            except Exception as e:
                logger.debug("Failed to stop MCP server %r: %s", name, e)
        state["harbor_mcp_jobs"] = {}

    async def _start_mcp_server(
        self,
        sandbox_id: str,
        server: HarborMCPServer,
        command: str,
        state: vf.State,
    ) -> None:
        port = mcp_url_port(server)
        if port is None:
            raise ValueError(
                f"MCP server {server.name!r} has no port in its URL {server.url!r}"
            )

        sandbox_client = state["sandbox_client"]
        job = await sandbox_client.start_background_job(
            sandbox_id=sandbox_id,
            command=self._mcp_start_cmd(server.name, command),
            working_dir=None,
        )
        jobs: dict[str, Any] = state.setdefault("harbor_mcp_jobs", {})
        jobs[server.name] = job
        await self._wait_for_mcp_server(sandbox_id, server.name, port, job, state)

    async def _wait_for_mcp_server(
        self,
        sandbox_id: str,
        name: str,
        port: int,
        job: Any,
        state: vf.State,
    ) -> None:
        healthcheck = self.mcp_healthcheck
        health_cmd = (
            healthcheck.command.format(port=port)
            if healthcheck.command
            else self._default_mcp_health_cmd(port)
        )
        probe_timeout = max(1, int(healthcheck.timeout_sec))
        loop_time = asyncio.get_event_loop().time
        start_period_end = loop_time() + healthcheck.start_period_sec
        consecutive_failures = 0

        while True:
            sandbox_client = state["sandbox_client"]
            status = await sandbox_client.get_background_job(sandbox_id, job)
            if getattr(status, "completed", False):
                stderr = (getattr(status, "stderr", "") or "").strip()
                raise vf.SandboxError(
                    f"MCP server {name!r} on port {port} exited before "
                    f"becoming healthy. Stderr:\n{stderr}"
                )

            result = await sandbox_client.execute_command(
                sandbox_id, health_cmd, working_dir=None, timeout=probe_timeout
            )
            if getattr(result, "exit_code", 1) == 0:
                return
            if loop_time() < start_period_end:
                await asyncio.sleep(healthcheck.interval_sec)
                continue

            consecutive_failures += 1
            if consecutive_failures >= healthcheck.retries:
                raise vf.SandboxError(
                    f"MCP server {name!r} on port {port} failed health check "
                    f"after {healthcheck.retries} consecutive retries."
                )
            await asyncio.sleep(healthcheck.interval_sec)

    async def _patch_mcp_etc_hosts(
        self,
        sandbox_id: str,
        servers: list[HarborMCPServer],
        state: vf.State,
    ) -> None:
        hosts: set[str] = set()
        for server in servers:
            if server.url:
                host = urlparse(server.url).hostname
                if host and host not in {"localhost", "127.0.0.1", "::1"}:
                    hosts.add(host)
        if not hosts:
            return

        statements = " && ".join(
            f"(grep -qxF {shlex.quote(f'127.0.0.1 {host}')} /etc/hosts || "
            f"echo {shlex.quote(f'127.0.0.1 {host}')} >> /etc/hosts)"
            for host in sorted(hosts)
        )
        result = await state["sandbox_client"].execute_command(
            sandbox_id, statements, working_dir=None
        )
        if getattr(result, "exit_code", 0) != 0:
            stderr = (getattr(result, "stderr", "") or "").strip()
            raise vf.SandboxError(
                f"Failed to alias MCP hostnames {sorted(hosts)} in /etc/hosts. "
                f"Stderr:\n{stderr}"
            )

    @staticmethod
    def _mcp_env_name(name: str) -> str:
        return name.upper().replace("-", "_")

    @staticmethod
    def _mcp_pid_file(name: str) -> str:
        return f"/tmp/harbor-mcp-{name}.pid"

    def _mcp_start_cmd(self, name: str, command: str) -> str:
        pid_file = shlex.quote(self._mcp_pid_file(name))
        return f"setsid sh -c {shlex.quote(command)} & echo $! > {pid_file}; wait"

    def _mcp_stop_cmd(self, name: str) -> str:
        pid_file = shlex.quote(self._mcp_pid_file(name))
        return f'kill -9 -"$(cat {pid_file} 2>/dev/null)" 2>/dev/null; rm -f {pid_file}'

    @staticmethod
    def _default_mcp_health_cmd(port: int) -> str:
        port_hex = f"{port:04X}"
        return (
            f'awk \'$4 == "0A" && $2 ~ /:{port_hex}$/ '
            f"{{ok=1}} END {{exit !ok}}' "
            f"/proc/net/tcp /proc/net/tcp6 2>/dev/null"
        )

    def get_agent_timeout_seconds(self, info: dict) -> float | None:
        timeout = (info.get("config") or {}).get("agent", {}).get("timeout_sec")
        return float(timeout) if timeout is not None else None

    def get_test_timeout_seconds(self, info: dict) -> float | None:
        timeout = (info.get("config") or {}).get("verifier", {}).get("timeout_sec")
        return float(timeout) if timeout is not None else None

    def get_rubric(self) -> vf.Rubric:
        return HarborRubric(self)

    async def setup(self, state: vf.State) -> None:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        task_dir = Path(state["info"]["task_dir"])

        remote_tar = "/tmp/harbor_task.tar.gz"
        await self.upload_archive(
            sandbox_client=sandbox_client,
            sandbox_id=sandbox_id,
            remote_tar=remote_tar,
            items=[
                (task_dir / "instruction.md", "task/instruction.md"),
                (task_dir / "task.toml", "task/task.toml"),
            ],
            extract_command=(
                f"mkdir -p /task /logs/verifier /logs/agent "
                f"{shlex.quote(self.agent_workdir)} && "
                f"tar -xzf {shlex.quote(remote_tar)} -C / && "
                f"rm {shlex.quote(remote_tar)}"
            ),
        )
        await self.setup_environment(state)

    async def setup_environment(self, state: vf.State) -> None:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        task_dir = Path(state["info"]["task_dir"])
        environment_dir = task_dir / "environment"
        setup_script = environment_dir / "setup.sh"
        if not setup_script.exists():
            return

        remote_tar = "/tmp/harbor_environment.tar.gz"
        await self.upload_archive(
            sandbox_client=sandbox_client,
            sandbox_id=sandbox_id,
            remote_tar=remote_tar,
            items=[(environment_dir, "environment")],
            extract_command=(
                f"rm -rf /environment && tar -xzf {shlex.quote(remote_tar)} "
                f"-C / && rm {shlex.quote(remote_tar)}"
            ),
        )

        config = state["info"].get("config") or {}
        timeout = int(
            math.ceil(
                float((config.get("environment") or {}).get("build_timeout_sec", 900))
            )
        )
        result = await sandbox_client.run_background_job(
            sandbox_id=sandbox_id,
            command="bash /environment/setup.sh && rm -rf /environment",
            working_dir=None,
            timeout=max(1, timeout),
        )
        if result.exit_code != 0:
            output = (result.stdout or "") + (result.stderr or "")
            raise vf.SandboxError(
                f"Harbor task setup failed (exit={result.exit_code}): {output[:1000]}"
            )

    async def run_tests(
        self,
        *,
        sandbox_client: Any,
        sandbox_id: str,
        state: vf.State,
        test_timeout: int,
    ) -> str:
        task_dir = Path(state["info"]["task_dir"])
        solution_dir = task_dir / "solution"
        tests_dir = task_dir / "tests"

        remote_tar = "/tmp/harbor_tests.tar.gz"
        items: list[tuple[Path, str]] = []
        if solution_dir.exists():
            items.extend(
                (item, f"oracle/{item.name}") for item in solution_dir.iterdir()
            )
        if tests_dir.exists():
            items.extend((item, f"tests/{item.name}") for item in tests_dir.iterdir())
        await self.upload_archive(
            sandbox_client=sandbox_client,
            sandbox_id=sandbox_id,
            remote_tar=remote_tar,
            items=items,
            extract_command=(
                f"mkdir -p /oracle /tests && "
                f"tar -xzf {shlex.quote(remote_tar)} -C / && "
                f"rm {shlex.quote(remote_tar)}"
            ),
            timeout=900,
        )

        result = await sandbox_client.run_background_job(
            sandbox_id=sandbox_id,
            command="bash test.sh",
            working_dir="/tests",
            timeout=max(1, int(math.ceil(test_timeout))),
        )
        state["test_exit_code"] = getattr(result, "exit_code", 0)
        state["test_stdout"] = getattr(result, "stdout", "") or ""
        state["test_stderr"] = getattr(result, "stderr", "") or ""

        reward_result = await sandbox_client.execute_command(
            sandbox_id,
            "if [ -s /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; "
            "elif [ -s /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; fi",
            working_dir=None,
        )
        return (getattr(reward_result, "stdout", "") or "").strip()

    def calculate_reward(self, test_output: str, info: dict) -> float:
        if not test_output:
            return 0.0
        try:
            return float(test_output)
        except ValueError:
            pass
        try:
            data = json.loads(test_output)
            return float(data.get("reward", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    async def apply_gold_patch(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: vf.State,
    ) -> None:
        task_dir = Path(state["info"]["task_dir"])
        solve_sh = task_dir / "solution" / "solve.sh"
        if not solve_sh.exists():
            raise RuntimeError(f"No solution/solve.sh in {task_dir}")

        remote_tar = "/tmp/harbor_oracle.tar.gz"
        await self.upload_archive(
            sandbox_client=sandbox_client,
            sandbox_id=sandbox_id,
            remote_tar=remote_tar,
            items=[(item, f"oracle/{item.name}") for item in solve_sh.parent.iterdir()],
            extract_command=(
                f"mkdir -p /oracle && "
                f"tar -xzf {shlex.quote(remote_tar)} -C / && "
                f"rm {shlex.quote(remote_tar)}"
            ),
        )

        result = await sandbox_client.execute_command(
            sandbox_id,
            "bash /oracle/solve.sh",
            working_dir=self.agent_workdir,
            timeout=120,
        )
        if result.exit_code != 0:
            stderr = (result.stderr or "")[:500]
            raise RuntimeError(
                f"solve.sh failed: exit_code={result.exit_code} stderr={stderr}"
            )

    async def validate_instance(self, state: vf.State) -> bool:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await self.apply_gold_patch(sandbox_client, sandbox_id, state)
        test_output = await self.run_tests(
            sandbox_client=sandbox_client,
            sandbox_id=sandbox_id,
            state=state,
            test_timeout=int(state.get("test_timeout", 900)),
        )
        return self.calculate_reward(test_output, state.get("info") or {}) > 0

    async def upload_archive(
        self,
        *,
        sandbox_client: Any,
        sandbox_id: str,
        remote_tar: str,
        items: list[tuple[Path, str]],
        extract_command: str,
        timeout: int = 900,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                for path, arcname in items:
                    tar.add(path, arcname=arcname)

            await sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
            await sandbox_client.execute_command(
                sandbox_id,
                extract_command,
                working_dir=None,
                timeout=timeout,
            )
        finally:
            tar_path.unlink(missing_ok=True)


HarborDatasetTaskSet = HarborTaskSet
HarborTaskset = HarborTaskSet
