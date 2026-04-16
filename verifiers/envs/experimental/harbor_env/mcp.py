"""Framework-managed MCP server lifecycle for Harbor-format tasks.

task.toml stays pure Harbor. In native Harbor, network-transport MCP servers
come from a ``docker-compose.yaml`` sidecar, so the task file only needs to
advertise the server's URL — not how to start it. Harbor's
:class:`harbor.models.task.config.MCPServerConfig` has exactly five fields:
``name``, ``transport``, ``url``, ``command``, ``args``.

Prime sandboxes are single-container, so HarborEnv has to start the servers
itself. All of that "how to start" detail lives on the Python side, wired up
via :class:`HarborMCPLauncher` instances passed to
:meth:`HarborMCPMixin.__init__` (or returned from the
:meth:`HarborMCPMixin.mcp_launcher_for` hook for dynamic cases).

Typical usage::

    class MyHarborEnv(HarborEnv):
        def __init__(self, **kwargs):
            super().__init__(
                mcp_launchers={
                    "mcp-server": HarborMCPLauncher(
                        command=".venv/bin/python -u server.py",
                        cwd="/opt/mcp-server",
                        user="environment",
                    ),
                },
                **kwargs,
            )

The resulting MCP server process is launched via
``sandbox_client.start_background_job`` (which handles daemonization,
stdout/stderr capture, and exit-code tracking). Because the sandbox client
runs commands as root, we can ``su`` to any user even on a ``nosuid``
filesystem — the same effect as Harbor's setuid wrapper, without needing
setuid binaries.

Where behavior overlaps with Harbor proper, we deliberately mirror it:

* ``[[environment.mcp_servers]]`` field names + transport validation match
  :class:`harbor.models.task.config.MCPServerConfig`.
* Per-server readiness checking mirrors Harbor's
  :class:`HealthcheckConfig` / ``run_healthcheck`` semantics (start period,
  consecutive-failure retry budget, per-probe timeout) so task authors who
  know Harbor's health model don't have to learn a second one.

This module intentionally avoids assuming any particular interpreter inside
the sandbox image: the default readiness probe uses ``awk`` + ``/proc/net/tcp``,
and the launch command is user-supplied.
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
    """An MCP server entry parsed from `[[environment.mcp_servers]]` in task.toml.

    Fields mirror :class:`harbor.models.task.config.MCPServerConfig` exactly —
    HarborEnv does not extend Harbor's task.toml schema. How HarborEnv should
    *start* a declared server is a Python-side concern; see
    :class:`HarborMCPLauncher` and :meth:`HarborMCPMixin.mcp_launcher_for`.
    """

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
    """Per-MCP-server readiness check, mirroring Harbor's `HealthcheckConfig`.

    See ``harbor.models.task.config.HealthcheckConfig``. Field names,
    defaults, and semantics are deliberately the same so a task author who
    has read Harbor's docs doesn't have to learn a second model.

    One extension: ``command`` may be ``None``, meaning "use the framework's
    default probe" (a ``/proc/net/tcp`` LISTEN scan that depends only on
    ``awk``). Harbor's own `HealthcheckConfig.command` is required because
    Harbor always defers to the task author.

    The ``command`` string is templated with ``str.format(port=...)``, so it
    may reference ``{port}`` — handy for overrides like
    ``curl -fsS http://127.0.0.1:{port}/health``.
    """

    command: str | None = None
    interval_sec: float = 2.0
    timeout_sec: float = 10.0
    start_period_sec: float = 0.0
    start_interval_sec: float = 2.0
    retries: int = 30

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarborMCPHealthcheck":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class HarborMCPLauncher:
    """How to start a network-transport MCP server inside the sandbox.

    All HarborEnv-specific configuration for a task.toml-declared MCP server
    lives on this class. It is *never* serialized to task.toml — Harbor tasks
    stay pure Harbor.

    ``command`` is handed to ``sandbox_client.start_background_job`` (which
    wraps it in its own ``nohup sh -c`` shim). When ``user`` is set, the
    command is further wrapped in ``su -s /bin/sh <user> -c`` (sandbox
    commands run as root, so this works even on ``nosuid`` filesystems).

    The framework exports three env vars to the launched process so the
    server knows what to listen on: ``MCP_TRANSPORT`` (the transport string
    from task.toml), ``MCP_PORT`` (parsed from the URL), and
    ``MCP_BIND_HOST=127.0.0.1`` (single-container sandboxes never need to
    bind anything else). Servers that use different env-var names should
    bridge in ``env=`` (e.g. ``env={"SERVER_PORT": "$MCP_PORT"}``).

    ``phases`` restricts which rollout phases the server should be running in.
    Defaults to ``["agent", "verifier"]`` — i.e. up for the entire trial,
    matching native Harbor's docker-compose-sidecar behavior. Narrow it (e.g.
    to ``["agent"]``) if you want :meth:`HarborEnv.compute_reward` to stop
    the server before tests run.

    ``healthcheck`` controls readiness probing. See
    :class:`HarborMCPHealthcheck`. When ``None``, the mixin's
    ``default_mcp_healthcheck`` is used.
    """

    command: str
    user: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    phases: list[str] = field(default_factory=lambda: ["agent", "verifier"])
    healthcheck: HarborMCPHealthcheck | None = None

    def active_in(self, phase: str) -> bool:
        return phase in self.phases


def parse_mcp_servers(config: dict[str, Any]) -> list[HarborMCPServer]:
    """Normalize `[[environment.mcp_servers]]` into typed entries.

    Applies the same transport-field validation as Harbor's
    :class:`harbor.models.task.config.MCPServerConfig`: SSE/streamable-http
    transports require ``url``, stdio requires ``command``. Entries that
    violate these constraints raise ``ValueError``. Extra keys in the entry
    are ignored (matching pydantic's default ``extra="ignore"``).
    """
    raw_list = (config.get("environment") or {}).get("mcp_servers") or []
    servers: list[HarborMCPServer] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue

        transport = str(entry.get("transport", "stdio"))
        command = entry.get("command")
        url = entry.get("url")

        # Mirror Harbor's validate_transport_fields.
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
    """Mix into a `SandboxMixin`-using env to get framework-managed MCP servers.

    Subclasses can override :meth:`mcp_launcher_for` to provide dynamic
    launcher configuration (e.g. injecting phase-dependent env vars). The
    default implementation reads launcher config from the task.toml `launch`
    sub-table, falling back to the `mcp_launchers` constructor dict.
    """

    # Wired up by __init__ on the concrete class. Declared here so static
    # type checkers and IDEs resolve them.
    mcp_launchers: dict[str, HarborMCPLauncher]
    default_mcp_healthcheck: HarborMCPHealthcheck

    # --------------------------------------------------------------------- #
    # Subclass hooks                                                        #
    # --------------------------------------------------------------------- #

    def mcp_launcher_for(
        self, server: HarborMCPServer, state: vf.State, phase: str
    ) -> HarborMCPLauncher | None:
        """Return the launcher for `server` in `phase`, or None to skip it.

        Default: look up the constructor's ``mcp_launchers`` dict by server
        name. Servers with no matching launcher are treated as externally
        managed (e.g. Harbor-style docker-compose sidecars — not something
        HarborEnv starts).

        Subclasses override this for dynamic behavior — e.g. returning
        different launchers per phase, or computing env vars at start time.
        """
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
        self,
        config: dict[str, Any],
        state: vf.State,
        phase: str = DEFAULT_PHASE,
    ) -> dict[str, str]:
        """`HARBOR_MCP_<NAME>_URL` env vars for servers reachable in `phase`.

        The agent's run_command can template these (e.g. OpenCode config):
        a server declared as `http://mcp-server:8000/mcp` is published as
        `HARBOR_MCP_MCP_SERVER_URL=http://127.0.0.1:8000/mcp`.

        A server is considered "reachable in `phase`" if it either:

        * has a launcher whose ``phases`` include ``phase``, or
        * has no launcher at all (externally managed — assumed always up).

        ``state`` is plumbed through to :meth:`mcp_launcher_for` so subclass
        overrides can branch on rollout state when resolving launchers.
        """
        env_vars: dict[str, str] = {}
        for server in parse_mcp_servers(config):
            launcher = self.mcp_launcher_for(server, state, phase)
            if launcher is not None and not launcher.active_in(phase):
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
        """Start every framework-managed MCP server that's active in `phase`."""
        servers = parse_mcp_servers(config)
        state["harbor_mcp_servers"] = servers
        state.setdefault("harbor_mcp_started", [])
        if not servers:
            return

        await self._patch_mcp_etc_hosts(sandbox_id, servers)

        for server in servers:
            if not server.is_network:
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
            if not launcher.active_in(phase):
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

        # Stop everything currently running that doesn't belong in `phase`.
        for name in started:
            server = by_name.get(name)
            if server is None:
                await self._stop_mcp_server(sandbox_id, name, state)
                continue
            launcher = self.mcp_launcher_for(server, state, phase)
            if launcher is None or not launcher.active_in(phase):
                await self._stop_mcp_server(sandbox_id, name, state)

        # Start anything that's missing and should be up.
        for server in servers:
            if not server.is_network:
                continue
            if server.name in (state.get("harbor_mcp_started") or []):
                continue
            launcher = self.mcp_launcher_for(server, state, phase)
            if launcher is None or not launcher.active_in(phase):
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

        hc = launcher.healthcheck or self.default_mcp_healthcheck
        await self._wait_for_mcp_server(sandbox_id, server.name, port, job, hc)

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
        healthcheck: HarborMCPHealthcheck | None = None,
    ) -> None:
        """Poll until the server is ready on ``localhost:port``.

        Implements the same readiness semantics as Harbor's
        :meth:`harbor.environments.base.BaseEnvironment.run_healthcheck`:

        * During ``start_period_sec``, failures don't count toward the retry
          budget — we just sleep ``start_interval_sec`` and re-check. This
          gives slow-starting servers unlimited time to open their port as
          long as they keep producing "not ready yet."
        * After the start period, ``retries`` *consecutive* failures fail the
          check. Successive attempts are spaced by ``interval_sec``.
        * Each probe is bounded by ``timeout_sec``.

        If a ``BackgroundJob`` handle is provided, the job's completion is
        checked in the same loop — a crashed server bails out immediately
        with its real stderr, bypassing the retry budget entirely.

        The probe itself defaults to a ``/proc/net/tcp`` LISTEN scan
        (see :meth:`_default_mcp_health_cmd`). Tasks can override per-server
        with :attr:`HarborMCPHealthcheck.command`.
        """
        hc = healthcheck or self.default_mcp_healthcheck
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
                "MCP server %r health probe failed "
                "(rc=%s, in_start_period=%s, consecutive_failures=%s)",
                name,
                getattr(result, "exit_code", None),
                in_start_period,
                consecutive_failures,
            )
            if in_start_period:
                await asyncio.sleep(hc.start_interval_sec)
                continue

            consecutive_failures += 1
            if consecutive_failures >= hc.retries:
                log_tail = ""
                if job is not None:
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
            "MCP_TRANSPORT": server.transport,
            "MCP_PORT": str(port),
            "MCP_BIND_HOST": "127.0.0.1",
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

    def _mcp_stop_cmd(self, name: str) -> str:
        """Shell command to stop a previously launched MCP server.

        Default: SIGKILL by PID file. MCP daemons serving stateless RPC have
        nothing to flush on shutdown, and SIGKILL releases the listening
        socket instantly — which matters at phase transitions where we want
        to bind the same port seconds later.

        Overridable: a subclass can return any shell string. Typical override
        is a graceful shutdown for servers that do need a TERM window, e.g.::

            def _mcp_stop_cmd(self, name):
                pid_file = self._mcp_pid_file(name)
                return (
                    f'pid=$(cat {pid_file} 2>/dev/null); '
                    f'[ -n "$pid" ] && kill "$pid"; sleep 2; '
                    f'kill -9 "$pid" 2>/dev/null; rm -f {pid_file}'
                )
        """
        pid_file = shlex.quote(self._mcp_pid_file(name))
        # Reads PID, SIGKILL it (errors swallowed if missing/stale), unlink.
        return f'kill -9 "$(cat {pid_file} 2>/dev/null)" 2>/dev/null; rm -f {pid_file}'

    @staticmethod
    def _mcp_pid_file(name: str) -> str:
        return f"/tmp/harbor-mcp-{name}.pid"

    @staticmethod
    def _default_mcp_health_cmd(port: int) -> str:
        """Portable TCP LISTEN probe via `/proc/net/tcp{,6}`.

        `/proc/net/tcp` lines look like::

            sl  local_address rem_address   st …
             0: 0100007F:1F40 00000000:0000 0A …

        Column 4 is the connection state (`0A` = LISTEN) and the port in
        column 2 is uppercase hex. Exits 0 iff some listener's local port
        matches `port`. Works on every Linux with `/proc` mounted and busybox
        `awk` (i.e. ~every container image we care about).
        """
        port_hex = f"{port:04X}"
        # `awk` command is single-quoted; the only interpolations are the
        # (validated-integer) hex port, so there's no injection risk.
        return (
            f'awk \'$4 == "0A" && $2 ~ /:{port_hex}$/ '
            f"{{ok=1}} END {{exit !ok}}' "
            f"/proc/net/tcp /proc/net/tcp6 2>/dev/null"
        )
