from __future__ import annotations

import json
import logging
import math
import shlex
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from datasets import Dataset

from verifiers.envs.experimental.channels import (
    ChannelMap,
    SandboxResources,
    SandboxSeed,
)
from verifiers.envs.experimental.task import Task
from verifiers.envs.experimental.taskset import Taskset
from verifiers.errors import InfraError
from verifiers.rubrics.rubric import Rubric
from verifiers.utils.error_utils import error_type_name

if TYPE_CHECKING:
    from verifiers.envs.experimental.resources import Resources

NETWORK_MCP_TRANSPORTS = {"streamable-http", "http", "sse"}
logger = logging.getLogger(__name__)


class HarborRubric(Rubric):
    """Run Harbor verifier scripts against the rollout sandbox."""

    def __init__(self):
        super().__init__(funcs=[self.harbor_reward], weights=[1.0])

    async def harbor_reward(
        self, state, task: Task | None = None, resources: Resources | None = None
    ) -> float:
        error = state.get("error")
        if isinstance(error, InfraError) or error_type_name(error) == "InfraError":
            state["harbor_reward"] = 0.0
            return 0.0
        sandbox_id = state.get("sandbox_id")
        sandbox_runtime_obj = (
            resources.get("sandbox_runtime") if resources is not None else None
        )
        if not sandbox_id or not isinstance(sandbox_runtime_obj, SandboxResources):
            return 0.0
        sandbox_runtime = sandbox_runtime_obj
        sandbox_client = sandbox_runtime.client
        if task is not None and resources is not None:
            await self._upload_verifier_assets(resources.harness, sandbox_id, task)
        test_script = "/tests/test.sh"
        try:
            result = await sandbox_runtime.with_retry(sandbox_client.execute_command)(
                sandbox_id,
                f"test -f {test_script} && bash {test_script}",
                working_dir="/tests",
                timeout=600,
            )
            state["harbor_verifier_stdout"] = result.stdout
            state["harbor_verifier_stderr"] = result.stderr
            state["harbor_verifier_exit_code"] = result.exit_code
        except Exception as e:
            logger.warning(f"Harbor verifier failed before reading reward: {e}")
            state["harbor_verifier_error"] = repr(e)

        reward = await self._read_reward(sandbox_runtime, sandbox_client, sandbox_id)
        state["harbor_reward"] = reward
        return reward

    async def _upload_verifier_assets(
        self, harness, sandbox_id: str, task: Task
    ) -> None:
        task_dir = Path(task.info["task_dir"])
        solution_dir = task_dir / "solution"
        tests_dir = task_dir / "tests"
        if solution_dir.exists():
            await harness.upload_path(sandbox_id, solution_dir, "/oracle")
        if tests_dir.exists():
            await harness.upload_path(sandbox_id, tests_dir, "/tests")

    async def _read_reward(
        self, sandbox_runtime, sandbox_client, sandbox_id: str
    ) -> float:
        for path in ("/logs/verifier/reward.txt", "/logs/verifier/reward.json"):
            try:
                result = await sandbox_runtime.with_retry(sandbox_client.read_file)(
                    sandbox_id, path, timeout=10
                )
            except Exception:
                continue
            content = getattr(result, "content", "") or ""
            if not content.strip():
                continue
            if path.endswith(".json"):
                payload = json.loads(content)
                return float(payload.get("reward", payload.get("score", 0.0)))
            return float(content.strip())
        return 0.0

    async def cleanup(self, state):
        await super().cleanup(state)
        sandbox_id = state.get("sandbox_id")
        resources = self.class_objects.get("resources")
        sandbox_runtime = (
            resources.get("sandbox_runtime") if resources is not None else None
        )
        if not sandbox_id or sandbox_runtime is None:
            return
        if not state.get("sandbox_retained_for_scoring"):
            return
        try:
            await sandbox_runtime.delete(sandbox_id)
        except Exception as e:
            logger.warning(f"Failed to delete Harbor scoring sandbox {sandbox_id}: {e}")


class HarborTaskset(Taskset):
    """Structured taskset adapter for Harbor task directories."""

    def __init__(
        self,
        path: str | Path,
        tasks: list[str] | None = None,
        name: str | None = None,
        rubric: Rubric | None = None,
        tools: list[object] | None = None,
        workdir: str = "/app",
        mcp_launch_commands: dict[str, str] | None = None,
        mcp_healthcheck_command: str | None = None,
        mcp_healthcheck_retries: int = 30,
        mcp_healthcheck_interval_sec: float = 2.0,
        mcp_healthcheck_start_period_sec: float = 0.0,
    ):
        self.path = Path(path)
        self.task_names = set(tasks) if tasks else None
        self.workdir = workdir
        self.mcp_launch_commands = dict(mcp_launch_commands or {})
        self.mcp_healthcheck_command = mcp_healthcheck_command
        self.mcp_healthcheck_retries = mcp_healthcheck_retries
        self.mcp_healthcheck_interval_sec = mcp_healthcheck_interval_sec
        self.mcp_healthcheck_start_period_sec = mcp_healthcheck_start_period_sec
        super().__init__(
            source=self._discover,
            rubric=rubric or HarborRubric(),
            tools=tools,
            name=name,
        )

    def channels(self, task: Task | None = None) -> ChannelMap:
        channels = dict(super().channels(task))
        task_sandbox = channels.pop("sandbox", None)
        sandbox: dict[str, object] = {"scoring": True}
        if task is not None:
            sandbox["spec"] = self._sandbox_seed(task)
        if isinstance(task_sandbox, Mapping):
            sandbox.update(task_sandbox)
        elif task_sandbox is not None:
            sandbox["spec"] = task_sandbox
        channels["sandbox"] = sandbox
        return channels

    def _sandbox_seed(self, task: Task) -> SandboxSeed:
        config = task.info.get("config") or {}
        environment = config.get("environment", {}) or {}
        env_vars = {
            str(k): str(v) for k, v in (environment.get("env", {}) or {}).items()
        }
        env_vars.setdefault("HARBOR_TASK_NAME", task.info["task_name"])
        env_vars.setdefault("HARBOR_TASK_DIR", "/task")
        env_vars.setdefault("HARBOR_INSTRUCTION_PATH", "/task/instruction.md")
        env_vars.update(self._mcp_env_vars(environment))
        task_dir = Path(task.info["task_dir"])
        return SandboxSeed(
            image=environment.get("docker_image"),
            cpu_cores=environment.get("cpus"),
            memory_gb=self._memory_gb(environment.get("memory_mb")),
            disk_size_gb=self._optional_int(environment.get("disk_size_gb")),
            timeout_minutes=self._optional_int(environment.get("timeout_minutes")),
            network_access=environment.get("allow_internet"),
            environment_vars=env_vars,
            files={
                str(task_dir / "instruction.md"): "/task/instruction.md",
                str(task_dir / "task.toml"): "/task/task.toml",
            },
            setup_commands=self._setup_commands(environment),
            labels=[f"harbor-task:{task.info['task_name']}"],
        )

    def _discover(self) -> Dataset:
        if not self.path.exists():
            raise FileNotFoundError(f"Harbor taskset path not found: {self.path}")
        rows: list[dict[str, Any]] = []
        for task_dir in sorted(self.path.iterdir()):
            if not task_dir.is_dir():
                continue
            if self.task_names and task_dir.name not in self.task_names:
                continue
            task_toml = task_dir / "task.toml"
            instruction_md = task_dir / "instruction.md"
            if not task_toml.exists() or not instruction_md.exists():
                continue
            with task_toml.open("rb") as f:
                config = tomllib.load(f)
            instruction = instruction_md.read_text().strip()
            rows.append(
                {
                    "example_id": len(rows),
                    "prompt": [{"role": "user", "content": instruction}],
                    "info": {
                        "task_name": task_dir.name,
                        "task_dir": str(task_dir),
                        "instruction": instruction,
                        "workdir": self.workdir,
                        "config": config,
                    },
                }
            )
        if not rows:
            raise ValueError(f"No valid Harbor tasks found in {self.path}")
        return Dataset.from_list(rows)

    def _memory_gb(self, memory_mb: Any) -> int | None:
        if memory_mb is None:
            return None
        return max(1, math.ceil(int(memory_mb) / 1024))

    def _optional_int(self, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    def _setup_commands(self, environment: dict[str, Any]) -> list[str]:
        commands: list[str] = [
            f"mkdir -p /task /logs/verifier /oracle /tests {shlex.quote(self.workdir)}"
        ]
        for server in environment.get("mcp_servers", []) or []:
            if not isinstance(server, dict):
                continue
            if str(server.get("transport", "sse")) not in NETWORK_MCP_TRANSPORTS:
                continue
            command = self._mcp_launch_command(server)
            if command is None:
                continue
            host_patch = self._mcp_hosts_patch(server)
            if host_patch:
                commands.append(host_patch)
            name = str(server.get("name") or "mcp")
            commands.append(self._mcp_start_cmd(name, command))
            port = self._mcp_url_port(server)
            if port is not None:
                commands.append(self._mcp_wait_cmd(name, port))
        healthcheck = environment.get("healthcheck")
        if isinstance(healthcheck, dict) and healthcheck.get("command"):
            commands.append(self._healthcheck_cmd(healthcheck))
        return commands

    def _mcp_launch_command(self, server: dict[str, Any]) -> str | None:
        name = str(server.get("name") or "")
        if name in self.mcp_launch_commands:
            return self.mcp_launch_commands[name]
        command = server.get("command")
        if command is None:
            return None
        env = server.get("env", {}) or {}
        env_prefix = " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in env.items()
        )
        args = " ".join(shlex.quote(str(arg)) for arg in server.get("args", []))
        return f"{env_prefix} {command} {args}".strip()

    def _mcp_start_cmd(self, name: str, command: str) -> str:
        pid_file = shlex.quote(self._mcp_pid_file(name))
        log_path = shlex.quote(f"/tmp/{name}.log")
        return (
            f"setsid sh -c {shlex.quote(command)} > {log_path} 2>&1 & "
            f"echo $! > {pid_file}"
        )

    def _mcp_wait_cmd(self, name: str, port: int) -> str:
        health_cmd = (
            self.mcp_healthcheck_command.format(port=port)
            if self.mcp_healthcheck_command
            else self._default_mcp_health_cmd(port)
        )
        start_sleep = (
            f"sleep {self.mcp_healthcheck_start_period_sec}; "
            if self.mcp_healthcheck_start_period_sec > 0
            else ""
        )
        retries = max(1, int(self.mcp_healthcheck_retries))
        interval = max(0.0, float(self.mcp_healthcheck_interval_sec))
        log_path = shlex.quote(f"/tmp/{name}.log")
        return (
            f"{start_sleep}for i in $(seq 1 {retries}); do "
            f"{health_cmd} && exit 0; "
            f"sleep {interval}; "
            "done; "
            f"echo {shlex.quote(f'MCP server {name} failed health check')} >&2; "
            f"tail -c 2000 {log_path} >&2 || true; "
            "exit 1"
        )

    def _healthcheck_cmd(self, healthcheck: dict[str, Any]) -> str:
        command = str(healthcheck["command"])
        retries = max(1, int(healthcheck.get("retries", 1)))
        interval = max(0.0, float(healthcheck.get("interval_sec", 1.0)))
        start_period = max(0.0, float(healthcheck.get("start_period_sec", 0.0)))
        start_sleep = f"sleep {start_period}; " if start_period > 0 else ""
        return (
            f"{start_sleep}for i in $(seq 1 {retries}); do "
            f"{command} && exit 0; "
            f"sleep {interval}; "
            "done; "
            "exit 1"
        )

    def _mcp_env_vars(self, environment: dict[str, Any]) -> dict[str, str]:
        env_vars: dict[str, str] = {}
        for server in environment.get("mcp_servers", []) or []:
            if not isinstance(server, dict) or not server.get("name"):
                continue
            if str(server.get("transport", "sse")) not in NETWORK_MCP_TRANSPORTS:
                continue
            url = server.get("url")
            if not url:
                continue
            key = f"HARBOR_MCP_{str(server['name']).upper().replace('-', '_')}_URL"
            env_vars[key] = (
                self._sandbox_local_url(url)
                if self._mcp_launch_command(server) is not None
                else str(url)
            )
        return env_vars

    def _mcp_url_port(self, server: dict[str, Any]) -> int | None:
        url = server.get("url")
        if not url:
            return None
        parsed = urlparse(str(url))
        if parsed.port is not None:
            return parsed.port
        if parsed.scheme == "https":
            return 443
        if parsed.scheme in ("http", ""):
            return 80
        return None

    def _mcp_hosts_patch(self, server: dict[str, Any]) -> str | None:
        url = server.get("url")
        if not url:
            return None
        host = urlparse(str(url)).hostname
        if not host or host in {"localhost", "127.0.0.1", "::1"}:
            return None
        entry = shlex.quote(f"127.0.0.1 {host}")
        return f"(grep -qxF {entry} /etc/hosts || echo {entry} >> /etc/hosts)"

    def _mcp_pid_file(self, name: str) -> str:
        return f"/tmp/harbor-mcp-{name}.pid"

    def _default_mcp_health_cmd(self, port: int) -> str:
        port_hex = f"{port:04X}"
        return (
            f'awk \'$4 == "0A" && $2 ~ /:{port_hex}$/ '
            f"{{ok=1}} END {{exit !ok}}' "
            f"/proc/net/tcp /proc/net/tcp6 2>/dev/null"
        )

    def _sandbox_local_url(self, url: Any) -> str:
        text = str(url)
        parsed = urlparse(text)
        if not parsed.hostname:
            return text
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        netloc = f"127.0.0.1:{port}"
        if "@" in parsed.netloc:
            userinfo = parsed.netloc.rsplit("@", 1)[0]
            netloc = f"{userinfo}@{netloc}"
        return parsed._replace(netloc=netloc).geturl()
