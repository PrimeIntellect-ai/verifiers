"""Framework-managed MCP server lifecycle for Harbor-format tasks.

Harbor's `task.toml` can declare MCP servers under `[[environment.mcp_servers]]`:

```toml
[[environment.mcp_servers]]
name = "mcp-server"
transport = "streamable-http"
url = "http://mcp-server:8000/mcp"

# Optional sub-table consumed only by HarborEnv. Harbor itself ignores it.
[environment.mcp_servers.launch]
command = ".venv/bin/python -u server.py"
cwd = "/opt/mcp-server"
user = "environment"
env = { MCP_CALLER_UID = "10001" }
phases = ["agent"]           # when the server should be running
```

In native Harbor, network-transport servers are provided by the task's own
`docker-compose.yaml` sidecars. Prime sandboxes are single-container, so we
emulate that by starting the server *inside* the same sandbox via
`sandbox_client.start_background_job` (which handles daemonization, stdout/
stderr capture, and exit-code tracking for us). Because the sandbox client
runs commands as root, we can `su` to any user even on a `nosuid` filesystem
— the same effect as Harbor's setuid wrapper, without needing setuid
binaries.

This module intentionally avoids assuming any particular interpreter inside
the sandbox image: health checks use bash `/dev/tcp`, and the launch command
is user-supplied.
"""

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

DEFAULT_PHASE = "agent"
"""Phase used when a task.toml entry doesn't specify one."""


@dataclass
class HarborMCPServer:
    """An MCP server entry parsed from `[[environment.mcp_servers]]` in task.toml."""

    name: str
    transport: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    phases: list[str] = field(default_factory=lambda: [DEFAULT_PHASE])
    launch: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_network(self) -> bool:
        return self.transport in NETWORK_TRANSPORTS

    def active_in(self, phase: str) -> bool:
        return phase in self.phases


@dataclass
class HarborMCPLauncher:
    """How to start a network-transport MCP server inside the sandbox.

    `command` is executed via `nohup` under the sandbox's default shell.
    When `user` is set, the command is wrapped in `su -s /bin/sh <user> -c`
    (sandbox commands run as root, so this works even on nosuid filesystems).
    """

    command: str
    user: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    bind_host: str = "127.0.0.1"
    transport_env_var: str = "MCP_TRANSPORT"
    port_env_var: str = "MCP_PORT"
    host_env_var: str = "MCP_BIND_HOST"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarborMCPLauncher":
        """Build a launcher from a task.toml `launch` sub-table."""
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in allowed}
        if "env" in kwargs and kwargs["env"] is None:
            kwargs["env"] = {}
        return cls(**kwargs)


def parse_mcp_servers(config: dict[str, Any]) -> list[HarborMCPServer]:
    """Normalize `[[environment.mcp_servers]]` into typed entries.

    Accepts an optional per-entry `launch` sub-table and `phases` list. Both
    are HarborEnv extensions — Harbor core ignores them.
    """
    raw_list = (config.get("environment") or {}).get("mcp_servers") or []
    servers: list[HarborMCPServer] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue

        phases_raw = entry.get("phases")
        if isinstance(phases_raw, list) and phases_raw:
            phases = [str(p) for p in phases_raw]
        else:
            phases = [DEFAULT_PHASE]

        launch = entry.get("launch")
        if launch is not None and not isinstance(launch, dict):
            launch = None

        servers.append(
            HarborMCPServer(
                name=str(name),
                transport=str(entry.get("transport", "stdio")),
                command=entry.get("command"),
                args=list(entry.get("args") or []),
                url=entry.get("url"),
                phases=phases,
                launch=dict(launch) if launch else None,
                raw=dict(entry),
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
    """Mix into a `SandboxMixin`-using env to get framework-managed MCP servers.

    Subclasses can override :meth:`mcp_launcher_for` to provide dynamic
    launcher configuration (e.g. injecting phase-dependent env vars). The
    default implementation reads launcher config from the task.toml `launch`
    sub-table, falling back to the `mcp_launchers` constructor dict.
    """

    # Wired up by __init__ on the concrete class. Declared here so static
    # type checkers and IDEs resolve them.
    mcp_launchers: dict[str, HarborMCPLauncher]
    mcp_health_check_retries: int
    mcp_health_check_interval: float

    # --------------------------------------------------------------------- #
    # Subclass hooks                                                        #
    # --------------------------------------------------------------------- #

    def mcp_launcher_for(
        self, server: HarborMCPServer, state: vf.State, phase: str
    ) -> HarborMCPLauncher | None:
        """Return the launcher for `server` in `phase`, or None to skip it.

        Default precedence:
          1. task.toml `[environment.mcp_servers.launch]` sub-table
          2. `mcp_launchers={<name>: HarborMCPLauncher(...)}` constructor arg
          3. None (treat as externally managed)

        Subclasses override this to compute env vars at start time (e.g.
        BTB sets `MCP_CALLER_UID` per phase).
        """
        if server.launch:
            return HarborMCPLauncher.from_dict(server.launch)
        return self.mcp_launchers.get(server.name)

    def mcp_extra_env(
        self, server: HarborMCPServer, state: vf.State, phase: str
    ) -> dict[str, str]:
        """Per-call env-var overrides merged into the launched MCP process.

        Takes precedence over `HarborMCPLauncher.env`. Typical use: inject a
        phase-specific `MCP_CALLER_UID`, or propagate secrets from the host.
        """
        return {}

    def mcp_agent_env_vars(
        self, config: dict[str, Any], phase: str = DEFAULT_PHASE
    ) -> dict[str, str]:
        """`HARBOR_MCP_<NAME>_URL` env vars for servers reachable in `phase`.

        The agent's run_command can template these (e.g. OpenCode config):
        a server declared as `http://mcp-server:8000/mcp` is published as
        `HARBOR_MCP_MCP_SERVER_URL=http://127.0.0.1:8000/mcp`.
        """
        env_vars: dict[str, str] = {}
        for server in parse_mcp_servers(config):
            if not server.active_in(phase):
                continue
            url = mcp_agent_url(server)
            if url is None:
                continue
            key = f"HARBOR_MCP_{server.name.upper().replace('-', '_')}_URL"
            env_vars[key] = url
        return env_vars

    async def start_mcp_servers_for_phase(
        self,
        sandbox_id: str,
        config: dict[str, Any],
        state: vf.State,
        phase: str = DEFAULT_PHASE,
    ) -> None:
        """Start every MCP server whose `phases` include `phase`."""
        servers = parse_mcp_servers(config)
        state["harbor_mcp_servers"] = servers
        state.setdefault("harbor_mcp_started", [])
        if not servers:
            return

        await self._patch_mcp_etc_hosts(sandbox_id, servers)

        for server in servers:
            if not server.is_network or not server.active_in(phase):
                continue
            launcher = self.mcp_launcher_for(server, state, phase)
            if launcher is None:
                logger.debug(
                    "MCP server %r has transport %r but no launcher — "
                    "assuming externally managed",
                    server.name,
                    server.transport,
                )
                continue
            await self._start_mcp_server(
                sandbox_id, server, launcher, state, phase=phase
            )

    async def restart_mcp_for_phase(
        self,
        sandbox_id: str,
        state: vf.State,
        phase: str,
    ) -> None:
        """Stop servers not in `phase` and start servers that are.

        Idempotent: a server that's already running and still applies to
        `phase` is left alone. Typical use:

        ```python
        async def compute_reward(self, state):
            await self.restart_mcp_for_phase(state["sandbox_id"], state, "verifier")
            ...
        ```
        """
        servers: list[HarborMCPServer] = state.get("harbor_mcp_servers") or []
        started: list[str] = list(state.get("harbor_mcp_started") or [])
        by_name = {s.name: s for s in servers}

        for name in started:
            server = by_name.get(name)
            if server is None or not server.active_in(phase):
                await self._stop_mcp_server(sandbox_id, name, state)

        for server in servers:
            if not server.is_network or not server.active_in(phase):
                continue
            if server.name in (state.get("harbor_mcp_started") or []):
                continue
            launcher = self.mcp_launcher_for(server, state, phase)
            if launcher is None:
                continue
            await self._start_mcp_server(
                sandbox_id, server, launcher, state, phase=phase
            )

    @vf.cleanup
    async def stop_mcp_servers(self, state: vf.State) -> None:
        """Stop every framework-managed MCP server for this rollout."""
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return
        started = list(state.get("harbor_mcp_started") or [])
        for name in started:
            try:
                await self.sandbox_client.execute_command(  # type: ignore[attr-defined]
                    sandbox_id, self._mcp_stop_cmd(name), working_dir=None
                )
            except Exception as e:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Failed to stop MCP server %r: %s", name, e)
        state["harbor_mcp_started"] = []

    # --------------------------------------------------------------------- #
    # Internals                                                             #
    # --------------------------------------------------------------------- #

    async def _start_mcp_server(
        self,
        sandbox_id: str,
        server: HarborMCPServer,
        launcher: HarborMCPLauncher,
        state: vf.State,
        *,
        phase: str,
    ) -> None:
        """Start a single MCP server and wait for it to accept connections.

        Uses :meth:`sandbox_client.start_background_job` so the SDK owns
        detachment, stdout/stderr capture, and exit-code tracking. We keep a
        separate PID file so we can stop the process at phase transitions
        (the SDK has no cancel API at time of writing).
        """
        if not server.is_network:
            raise ValueError(
                f"MCP server {server.name!r} has stdio transport; the framework "
                "does not manage its lifecycle"
            )
        port = mcp_url_port(server)
        if port is None:
            raise ValueError(
                f"MCP server {server.name!r} has no port in its URL {server.url!r}"
            )

        extra_env = self.mcp_extra_env(server, state, phase) or {}
        cmd = self._mcp_start_cmd(server, launcher, port, extra_env)

        job = await self.sandbox_client.start_background_job(  # type: ignore[attr-defined]
            sandbox_id=sandbox_id,
            command=cmd,
            working_dir=None,
        )
        jobs: dict[str, Any] = state.setdefault("harbor_mcp_jobs", {})
        jobs[server.name] = job

        await self._wait_for_mcp_server(sandbox_id, server.name, port, job)

        started: list[str] = state.setdefault("harbor_mcp_started", [])
        if server.name not in started:
            started.append(server.name)
        logger.info(
            "MCP server %r ready on port %d (phase=%s)", server.name, port, phase
        )

    async def _stop_mcp_server(
        self, sandbox_id: str, name: str, state: vf.State | None = None
    ) -> None:
        """Stop a single MCP server and untrack it."""
        await self.sandbox_client.execute_command(  # type: ignore[attr-defined]
            sandbox_id, self._mcp_stop_cmd(name), working_dir=None
        )
        if state is not None:
            started = state.get("harbor_mcp_started") or []
            if name in started:
                started.remove(name)
                state["harbor_mcp_started"] = started
            jobs = state.get("harbor_mcp_jobs") or {}
            jobs.pop(name, None)
            state["harbor_mcp_jobs"] = jobs

    async def _wait_for_mcp_server(
        self,
        sandbox_id: str,
        name: str,
        port: int,
        job: Any | None = None,
    ) -> None:
        """Poll until the server is accepting TCP connections on `localhost:port`.

        If a `BackgroundJob` handle is provided, the job's completion status is
        checked in the same loop — a crashed server bails out immediately with
        its real stderr rather than waiting for the full retry budget.

        Uses bash's `/dev/tcp` so we don't assume Python in the sandbox image.
        """
        health_cmd = f"bash -c 'exec 3<>/dev/tcp/127.0.0.1/{port}' 2>/dev/null"
        for attempt in range(self.mcp_health_check_retries):
            if job is not None:
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

        log_tail = ""
        if job is not None:
            try:
                status = await self.sandbox_client.get_background_job(  # type: ignore[attr-defined]
                    sandbox_id, job
                )
                log_tail = (getattr(status, "stderr", "") or "").strip()[-2000:]
            except Exception as e:  # noqa: BLE001 — best-effort log retrieval
                logger.debug("Failed to fetch MCP log tail for %r: %s", name, e)

        raise vf.SandboxError(
            f"MCP server {name!r} on port {port} failed health check after "
            f"{self.mcp_health_check_retries} attempts. "
            f"Recent log tail:\n{log_tail}"
        )

    async def _patch_mcp_etc_hosts(
        self, sandbox_id: str, servers: list[HarborMCPServer]
    ) -> None:
        """Ensure task.toml service-name URLs resolve to 127.0.0.1.

        Harbor tasks typically reference MCP servers by docker-compose service
        name (`http://mcp-server:8000/mcp`). There's no compose network in a
        Prime sandbox, so we alias the hostname to loopback.
        """
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

        # `grep -qw` (whole-word match) prevents duplicate entries across
        # repeated calls. Hostnames are shell-quoted because they originate
        # from task.toml.
        statements = " && ".join(
            f"(grep -qw {shlex.quote(h)} /etc/hosts || "
            f"echo {shlex.quote(f'127.0.0.1 {h}')} >> /etc/hosts)"
            for h in sorted(hosts)
        )
        await self.sandbox_client.execute_command(  # type: ignore[attr-defined]
            sandbox_id, statements, working_dir=None
        )

    def _mcp_start_cmd(
        self,
        server: HarborMCPServer,
        launcher: HarborMCPLauncher,
        port: int,
        extra_env: dict[str, str],
    ) -> str:
        """Build the shell command that launches one MCP server.

        Intended to be handed to :meth:`sandbox_client.start_background_job`,
        which already wraps the command in ``nohup sh -c '(...) > stdout 2> stderr; echo $? > exit'``.
        So here we just:

        1. ``cd`` into the launcher's working directory (if any).
        2. Record ``$$`` (the shell's PID) into a pidfile, so :meth:`_mcp_stop_cmd`
           can kill the process later. After ``exec`` the PID belongs to the
           target binary.
        3. ``exec`` the target command with inline env-var assignments. Inline
           assignments are required because env vars set via
           ``start_background_job(env=...)`` would be stripped by ``su``.
        """
        env_pairs: dict[str, str] = {
            launcher.transport_env_var: server.transport,
            launcher.port_env_var: str(port),
            launcher.host_env_var: launcher.bind_host,
        }
        env_pairs.update(launcher.env)
        env_pairs.update(extra_env)

        env_prefix = " ".join(
            f"{k}={shlex.quote(str(v))}" for k, v in env_pairs.items()
        )

        pid_file = shlex.quote(self._mcp_pid_file(server.name))
        cd_prefix = f"cd {shlex.quote(launcher.cwd)} && " if launcher.cwd else ""
        inner = (
            f"{cd_prefix}echo $$ > {pid_file} && {env_prefix} exec {launcher.command}"
        )

        if launcher.user:
            return f"su -s /bin/sh {shlex.quote(launcher.user)} -c {shlex.quote(inner)}"
        return inner

    @staticmethod
    def _mcp_stop_cmd(name: str) -> str:
        """Idempotent SIGTERM-then-SIGKILL by PID file. No-ops if not running."""
        pid_file = shlex.quote(HarborMCPMixin._mcp_pid_file(name))
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

    @staticmethod
    def _mcp_pid_file(name: str) -> str:
        return f"/tmp/harbor-mcp-{name}.pid"
