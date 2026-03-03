from __future__ import annotations

import asyncio
import shlex
from typing import Any
from urllib.parse import urlparse

from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

from verifiers.utils.mcp_utils.models import MCPServerConfig
from verifiers.utils.mcp_utils.transports.streaming_http import StreamingHTTPTransport

SERVICE_READY_TIMEOUT_SECONDS = 10.0
SERVICE_POLL_INTERVAL_SECONDS = 0.5


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
        super().__init__(
            config,
            url="",
            timeout=timeout,
            max_retries=max_retries,
        )

    async def connect(self):
        try:
            await self.create_sandbox()
            await self.run_setup_commands()
            await self.start_mcp_server()
            await self.expose_port()
            await self.wait_for_service_ready()
            return await super().connect()
        except Exception as exc:
            startup_error = await self.build_startup_error(exc)
            try:
                await self.disconnect()
            except Exception as cleanup_exc:
                raise self._combine_startup_and_cleanup_errors(
                    startup_error,
                    cleanup_exc,
                ) from exc
            raise startup_error from exc

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
        # Keep the launch path simple and let HTTP readiness confirm the service.
        launch_command = f"nohup {command} >{shlex.quote(self.log_file)} 2>&1 &"
        await self.run_command_checked(
            f"sh -lc {shlex.quote(launch_command)}",
            context="start MCP server",
        )

    async def wait_for_service_ready(self) -> None:
        parsed_url = urlparse(self.url)
        host = parsed_url.hostname
        if host is None:
            raise RuntimeError(f"Sandbox transport produced an invalid URL: {self.url}")
        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
        use_tls = parsed_url.scheme == "https"
        deadline = asyncio.get_running_loop().time() + SERVICE_READY_TIMEOUT_SECONDS
        last_error: Exception | None = None

        while True:
            writer = None
            try:
                # Sandbox launchers can fork before the final MCP server is ready, so
                # use endpoint reachability as the readiness signal instead of a PID.
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port, ssl=use_tls),
                    timeout=self.timeout,
                )
                writer.close()
                await writer.wait_closed()
                return
            except Exception as exc:
                last_error = exc
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(
                        f"MCP server endpoint did not become reachable at {self.url}: "
                        f"{last_error}"
                    ) from exc
                await asyncio.sleep(SERVICE_POLL_INTERVAL_SECONDS)
            finally:
                if writer is not None and not writer.is_closing():
                    writer.close()
                    await writer.wait_closed()

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

    def _combine_startup_and_cleanup_errors(
        self,
        startup_error: RuntimeError,
        cleanup_exc: Exception,
    ) -> RuntimeError:
        return RuntimeError(
            f"{startup_error}\nCleanup after startup failure also failed: {cleanup_exc}"
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
