from __future__ import annotations

import asyncio
import shlex
from typing import Any

from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

from verifiers.utils.mcp_utils.models import MCPServerConfig
from verifiers.utils.mcp_utils.transports.streaming_http import StreamingHTTPTransport

PROCESS_START_TIMEOUT_SECONDS = 10.0
PROCESS_POLL_INTERVAL_SECONDS = 0.5


class SandboxTransport(StreamingHTTPTransport):
    _client: AsyncSandboxClient | None = None

    @classmethod
    def get_client(cls) -> AsyncSandboxClient:
        if cls._client is None:
            cls._client = AsyncSandboxClient()
        return cls._client

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        sandbox_image: str,
        sandbox_start_command: str,
        sandbox_environment_vars: dict[str, str],
        sandbox_cpu_cores: int,
        sandbox_memory_gb: int,
        sandbox_disk_size_gb: int,
        sandbox_timeout_minutes: int,
        port_to_expose: int,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.sandbox_request = CreateSandboxRequest(
            name=f"mcp-{config.name}",
            docker_image=sandbox_image,
            start_command=sandbox_start_command,
            cpu_cores=sandbox_cpu_cores,
            memory_gb=sandbox_memory_gb,
            disk_size_gb=sandbox_disk_size_gb,
            timeout_minutes=sandbox_timeout_minutes,
            environment_vars={
                **sandbox_environment_vars,
                **(config.env or {}),
            },
        )
        self.sandbox_id: str | None = None
        self.port_to_expose = port_to_expose
        self.log_file = "/tmp/mcp.log"
        self.pid_file = f"/tmp/mcp-{config.name}.pid"
        super().__init__(
            config,
            url="",
            timeout=timeout,
            max_retries=max_retries,
        )

    async def connect(self):
        startup_error: Exception | None = None
        try:
            await self.create_sandbox()
            await self.run_setup_commands()
            await self.start_mcp_server()
            await self.expose_port()
            return await super().connect()
        except Exception as exc:
            startup_error = exc
            raise await self.build_startup_error(exc) from exc
        finally:
            if startup_error is not None:
                try:
                    await self.disconnect()
                except Exception:
                    pass

    async def create_sandbox(self) -> str:
        client = self.get_client()
        sandbox = await client.create(self.sandbox_request)
        self.sandbox_id = sandbox.id
        await client.wait_for_creation(self.sandbox_id)
        return self.sandbox_id

    async def run_command(self, command: str, *, timeout: int | None = None) -> Any:
        if self.sandbox_id is None:
            raise RuntimeError("Sandbox not created yet")
        client = self.get_client()
        if timeout is None:
            return await client.execute_command(self.sandbox_id, command)
        return await client.execute_command(self.sandbox_id, command, timeout=timeout)

    async def run_command_checked(
        self,
        command: str,
        *,
        context: str,
        timeout: int | None = None,
    ) -> Any:
        result = await self.run_command(command, timeout=timeout)
        exit_code = getattr(result, "exit_code", 0)
        if exit_code not in (0, None):
            raise RuntimeError(self._format_command_failure(result, context))
        return result

    def _format_command_failure(self, result: Any, context: str) -> str:
        exit_code = getattr(result, "exit_code", None)
        stdout = (getattr(result, "stdout", "") or "").strip()
        stderr = (getattr(result, "stderr", "") or "").strip()
        detail = f"{context} failed"
        if exit_code is not None:
            detail += f" with exit code {exit_code}"
        if stdout:
            detail += f"\nstdout:\n{stdout}"
        if stderr:
            detail += f"\nstderr:\n{stderr}"
        return detail

    async def run_setup_commands(self) -> None:
        if not self.config.setup_commands:
            return
        for command in self.config.setup_commands:
            await self.run_command_checked(command, context="sandbox setup command")

    async def start_mcp_server(self) -> None:
        command_parts = [self.config.command or "", *(self.config.args or [])]
        command = shlex.join([part for part in command_parts if part]).strip()
        if not command:
            raise RuntimeError(
                f"Sandbox transport for '{self.config.name}' requires a launch command"
            )
        launch_command = (
            f"rm -f {shlex.quote(self.pid_file)} && "
            f"nohup {command} >{shlex.quote(self.log_file)} 2>&1 & "
            f"echo $! > {shlex.quote(self.pid_file)}"
        )
        await self.run_command_checked(
            f"sh -lc {shlex.quote(launch_command)}",
            context="start MCP server",
        )
        await self.wait_for_process_start()

    async def wait_for_process_start(self) -> None:
        deadline = asyncio.get_running_loop().time() + PROCESS_START_TIMEOUT_SECONDS
        check_command = (
            f"test -s {shlex.quote(self.pid_file)} && "
            f"kill -0 \"$(cat {shlex.quote(self.pid_file)})\""
        )

        while True:
            result = await self.run_command(
                f"sh -lc {shlex.quote(check_command)}",
                timeout=5,
            )
            exit_code = getattr(result, "exit_code", 0)
            if exit_code in (0, None):
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    "MCP server process exited before becoming ready"
                    + self._startup_log_suffix()
                )
            await asyncio.sleep(PROCESS_POLL_INTERVAL_SECONDS)

    async def expose_port(self) -> str:
        if self.sandbox_id is None:
            raise RuntimeError("Sandbox not created yet")
        client = self.get_client()
        exposed = await client.expose(self.sandbox_id, self.port_to_expose)
        self.url = f"{exposed.url}/mcp"
        return self.url

    async def read_log_tail(self) -> str | None:
        try:
            result = await self.run_command(
                f"sh -lc {shlex.quote(f'tail -n 50 {self.log_file} 2>/dev/null || true')}",
                timeout=5,
            )
        except Exception:
            return None

        stdout = (getattr(result, "stdout", "") or "").strip()
        stderr = (getattr(result, "stderr", "") or "").strip()
        if stdout:
            return stdout
        if stderr:
            return stderr
        return None

    def _startup_log_suffix(self, logs: str | None = None) -> str:
        if not logs:
            return ""
        return f"\nStartup log tail:\n{logs}"

    async def build_startup_error(self, exc: Exception) -> RuntimeError:
        logs = await self.read_log_tail()
        return RuntimeError(
            f"Failed to start sandbox MCP server '{self.config.name}': {exc}"
            f"{self._startup_log_suffix(logs)}"
        )

    async def disconnect(self) -> None:
        disconnect_error: Exception | None = None
        try:
            await super().disconnect()
        except Exception as exc:
            disconnect_error = exc

        if self.sandbox_id is not None:
            client = self.get_client()
            sandbox_id = self.sandbox_id
            self.sandbox_id = None
            try:
                await client.delete(sandbox_id)
            except Exception as exc:
                if disconnect_error is None:
                    disconnect_error = exc

        if disconnect_error is not None:
            raise disconnect_error
