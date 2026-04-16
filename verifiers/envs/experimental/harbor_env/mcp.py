"""MCP server lifecycle for Harbor-format tasks."""

from __future__ import annotations

import asyncio
import logging
import shlex
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

import verifiers as vf

logger = logging.getLogger(__name__)


NETWORK_TRANSPORTS: frozenset[str] = frozenset({"streamable-http", "http", "sse"})
"""Transports where the server listens on a URL (vs. stdio)."""


@dataclass
class HarborMCPServer:
    """An MCP server entry parsed from [[environment.mcp_servers]] in task.toml."""

    name: str
    transport: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None

    @property
    def is_network(self) -> bool:
        return self.transport in NETWORK_TRANSPORTS


@dataclass
class HarborMCPHealthcheck:
    """Readiness probe applied to the MCP server."""

    command: str | None = None
    retries: int = 30
    interval_sec: float = 2.0
    start_period_sec: float = 0.0
    timeout_sec: float = 10.0


def parse_mcp_servers(config: dict[str, Any]) -> list[HarborMCPServer]:
    """Parse [[environment.mcp_servers]]."""
    raw_list = (config.get("environment") or {}).get("mcp_servers") or []
    servers: list[HarborMCPServer] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue

        transport = str(entry.get("transport", "sse"))
        command = entry.get("command")
        url = entry.get("url")

        if transport in NETWORK_TRANSPORTS and not url:
            raise ValueError(
                f"MCP server {name!r}: 'url' is required for transport {transport!r}"
            )
        if transport == "stdio" and not command:
            raise ValueError(
                f"MCP server {name!r}: 'command' is required for transport 'stdio'"
            )

        servers.append(
            HarborMCPServer(
                name=str(name),
                transport=transport,
                command=command,
                args=list(entry.get("args") or []),
                url=url,
            )
        )
    return servers


def mcp_url_port(server: HarborMCPServer) -> int | None:
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


def mcp_agent_url(server: HarborMCPServer) -> str | None:
    """Rewrite the task.toml URL to point at 127.0.0.1 inside the sandbox."""
    if not server.is_network or not server.url:
        return None
    parsed = urlparse(server.url)
    port = mcp_url_port(server)
    netloc = f"127.0.0.1:{port}" if port is not None else "127.0.0.1"
    return urlunparse(parsed._replace(netloc=netloc))


class HarborMCPMixin:
    """Mixin to get framework-managed MCP servers."""

    mcp_launch_commands: dict[str, str]
    mcp_healthcheck: HarborMCPHealthcheck

    async def mcp_launch_command(
        self, server: HarborMCPServer, state: vf.State
    ) -> str | None:
        """Shell command to launch `server`, or `None` to skip it."""
        return self.mcp_launch_commands.get(server.name)

    async def mcp_agent_env_vars(
        self, config: dict[str, Any], state: vf.State
    ) -> dict[str, str]:
        """Publish `HARBOR_MCP_<NAME>_URL` for every declared network server."""
        env_vars: dict[str, str] = {}
        for server in parse_mcp_servers(config):
            if not server.is_network or not server.url:
                continue
            command = await self.mcp_launch_command(server, state)
            url = mcp_agent_url(server) if command is not None else server.url
            if url is None:
                continue
            key = f"HARBOR_MCP_{server.name.upper().replace('-', '_')}_URL"
            env_vars[key] = url
        return env_vars

    async def start_mcp_servers(
        self, sandbox_id: str, config: dict[str, Any], state: vf.State
    ) -> None:
        """Start every framework-managed MCP server declared in ``config``."""
        framework_managed: list[tuple[HarborMCPServer, str]] = []
        for server in parse_mcp_servers(config):
            if not server.is_network:
                continue
            command = await self.mcp_launch_command(server, state)
            if command is None:
                logger.debug(
                    "MCP server %r has no launch command — assuming externally managed",
                    server.name,
                )
                continue
            framework_managed.append((server, command))

        await self._patch_mcp_etc_hosts(sandbox_id, [s for s, _ in framework_managed])
        for server, command in framework_managed:
            await self._start_mcp_server(sandbox_id, server, command, state)

    @vf.cleanup(priority=1)
    async def stop_mcp_servers(self, state: vf.State) -> None:
        """Stop every framework-managed MCP server for this rollout."""
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return
        jobs: dict[str, Any] = state.get("harbor_mcp_jobs") or {}
        for name in list(jobs):
            try:
                await self.sandbox_client.execute_command(  # type: ignore[attr-defined]
                    sandbox_id, self._mcp_stop_cmd(name), working_dir=None
                )
            except Exception as e:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Failed to stop MCP server %r: %s", name, e)
        state["harbor_mcp_jobs"] = {}

    async def _start_mcp_server(
        self,
        sandbox_id: str,
        server: HarborMCPServer,
        command: str,
        state: vf.State,
    ) -> None:
        """Start a single MCP server and wait for it to accept connections."""
        port = mcp_url_port(server)
        if port is None:
            raise ValueError(
                f"MCP server {server.name!r} has no port in its URL {server.url!r}"
            )

        job = await self.sandbox_client.start_background_job(  # type: ignore[attr-defined]
            sandbox_id=sandbox_id,
            command=self._mcp_start_cmd(server.name, command),
            working_dir=None,
        )
        jobs: dict[str, Any] = state.setdefault("harbor_mcp_jobs", {})
        jobs[server.name] = job

        await self._wait_for_mcp_server(sandbox_id, server.name, port, job)
        logger.info("MCP server %r ready on port %d", server.name, port)

    async def _wait_for_mcp_server(
        self, sandbox_id: str, name: str, port: int, job: Any
    ) -> None:
        """Poll until the server is ready on localhost:port."""
        hc = self.mcp_healthcheck
        health_cmd = (
            hc.command.format(port=port)
            if hc.command
            else self._default_mcp_health_cmd(port)
        )
        probe_timeout = max(1, int(hc.timeout_sec))

        loop_time = asyncio.get_event_loop().time
        start_time = loop_time()
        start_period_end = start_time + hc.start_period_sec
        consecutive_failures = 0

        while True:
            status = await self.sandbox_client.get_background_job(  # type: ignore[attr-defined]
                sandbox_id, job
            )
            if getattr(status, "completed", False):
                stderr = (getattr(status, "stderr", "") or "").strip()
                exit_code = getattr(status, "exit_code", None)
                raise vf.SandboxError(
                    f"MCP server {name!r} on port {port} exited before "
                    f"becoming healthy (exit_code={exit_code}). "
                    f"Stderr:\n{stderr}"
                )

            result = await self.sandbox_client.execute_command(  # type: ignore[attr-defined]
                sandbox_id, health_cmd, working_dir=None, timeout=probe_timeout
            )
            if getattr(result, "exit_code", 1) == 0:
                logger.debug(
                    "MCP server %r healthy on port %d (after %.2fs)",
                    name,
                    port,
                    loop_time() - start_time,
                )
                return

            in_start_period = loop_time() < start_period_end
            logger.debug(
                "MCP server %r health probe failed (rc=%s, in_start_period=%s, "
                "consecutive_failures=%s)",
                name,
                getattr(result, "exit_code", None),
                in_start_period,
                consecutive_failures,
            )
            if in_start_period:
                await asyncio.sleep(hc.interval_sec)
                continue

            consecutive_failures += 1
            if consecutive_failures >= hc.retries:
                log_tail = ""
                try:
                    status = await self.sandbox_client.get_background_job(  # type: ignore[attr-defined]
                        sandbox_id, job
                    )
                    log_tail = (getattr(status, "stderr", "") or "").strip()[-2000:]
                except Exception as e:  # noqa: BLE001 — best-effort
                    logger.debug("Failed to fetch MCP log tail for %r: %s", name, e)
                raise vf.SandboxError(
                    f"MCP server {name!r} on port {port} failed health check "
                    f"after {hc.retries} consecutive retries. "
                    f"Recent log tail:\n{log_tail}"
                )
            await asyncio.sleep(hc.interval_sec)

    async def _patch_mcp_etc_hosts(
        self, sandbox_id: str, servers: list[HarborMCPServer]
    ) -> None:
        """Alias each task.toml MCP hostname to 127.0.0.1 via `/etc/hosts`."""
        hosts: set[str] = set()
        for server in servers:
            if not server.url:
                continue
            host = urlparse(server.url).hostname
            if not host or host in ("localhost", "127.0.0.1", "::1"):
                continue
            hosts.add(host)
        if not hosts:
            return

        statements = " && ".join(
            f"(grep -qxF {shlex.quote(f'127.0.0.1 {h}')} /etc/hosts || "
            f"echo {shlex.quote(f'127.0.0.1 {h}')} >> /etc/hosts)"
            for h in sorted(hosts)
        )
        await self.sandbox_client.execute_command(  # type: ignore[attr-defined]
            sandbox_id, statements, working_dir=None
        )

    @staticmethod
    def _mcp_pid_file(name: str) -> str:
        return f"/tmp/harbor-mcp-{name}.pid"

    def _mcp_start_cmd(self, name: str, command: str) -> str:
        """Background `command` and record its PID via `$!` so stop can SIGKILL it."""
        pid_file = shlex.quote(self._mcp_pid_file(name))
        return f"{command} & echo $! > {pid_file}; wait"

    def _mcp_stop_cmd(self, name: str) -> str:
        """SIGKILL the recorded PID and remove the pidfile."""
        pid_file = shlex.quote(self._mcp_pid_file(name))
        return f'kill -9 "$(cat {pid_file} 2>/dev/null)" 2>/dev/null; rm -f {pid_file}'

    @staticmethod
    def _default_mcp_health_cmd(port: int) -> str:
        """Exit 0 iff some TCP listener is in LISTEN state on `port` (awk on `/proc/net/tcp{,6}`)."""
        port_hex = f"{port:04X}"
        return (
            f'awk \'$4 == "0A" && $2 ~ /:{port_hex}$/ '
            f"{{ok=1}} END {{exit !ok}}' "
            f"/proc/net/tcp /proc/net/tcp6 2>/dev/null"
        )
