from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from verifiers.envs.experimental.harbor_env import (
    DEFAULT_PHASE,
    HarborMCPHealthcheck,
    HarborMCPLauncher,
    HarborMCPMixin,
    HarborMCPServer,
    mcp_agent_url,
    mcp_url_port,
    parse_mcp_servers,
)


class TestParseMCPServers:
    def test_reads_basic_entries(self):
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "svc",
                        "transport": "streamable-http",
                        "url": "http://svc:8000/mcp",
                    },
                    {
                        "name": "stdio-srv",
                        "transport": "stdio",
                        "command": "/usr/bin/x",
                    },
                ]
            }
        }
        servers = parse_mcp_servers(cfg)
        assert [s.name for s in servers] == ["svc", "stdio-srv"]
        assert servers[0].is_network
        assert not servers[1].is_network

    def test_defaults_to_agent_phase(self):
        cfg = {
            "environment": {
                "mcp_servers": [{"name": "svc", "transport": "stdio", "command": "x"}]
            }
        }
        (server,) = parse_mcp_servers(cfg)
        assert server.phases == [DEFAULT_PHASE]

    def test_phases_are_normalized_to_strings(self):
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "svc",
                        "transport": "streamable-http",
                        "url": "http://svc:1/mcp",
                        "phases": ["agent", "verifier"],
                    }
                ]
            }
        }
        (server,) = parse_mcp_servers(cfg)
        assert server.active_in("agent")
        assert server.active_in("verifier")
        assert not server.active_in("other")

    def test_launch_subtable_round_trips(self):
        launch = {
            "command": ".venv/bin/python -u server.py",
            "cwd": "/opt/x",
            "user": "env-user",
            "env": {"K": "V"},
        }
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "svc",
                        "transport": "streamable-http",
                        "url": "http://svc:1/mcp",
                        "launch": launch,
                    }
                ]
            }
        }
        (server,) = parse_mcp_servers(cfg)
        assert server.launch == launch
        built = HarborMCPLauncher.from_dict(server.launch)
        assert built.command == launch["command"]
        assert built.cwd == launch["cwd"]
        assert built.user == launch["user"]
        assert built.env == launch["env"]

    def test_skips_entries_without_name(self):
        cfg = {"environment": {"mcp_servers": [{"transport": "stdio"}]}}
        assert parse_mcp_servers(cfg) == []


class TestURLHelpers:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("http://svc:8000/mcp", 8000),
            ("https://svc/mcp", 443),
            ("http://svc/mcp", 80),
        ],
    )
    def test_mcp_url_port(self, url: str, expected: int):
        s = HarborMCPServer(name="svc", transport="streamable-http", url=url)
        assert mcp_url_port(s) == expected

    def test_mcp_agent_url_rewrites_to_loopback(self):
        s = HarborMCPServer(
            name="svc", transport="streamable-http", url="http://svc:8000/mcp?x=1"
        )
        assert mcp_agent_url(s) == "http://127.0.0.1:8000/mcp?x=1"

    def test_mcp_agent_url_none_for_stdio(self):
        s = HarborMCPServer(name="svc", transport="stdio", command="x")
        assert mcp_agent_url(s) is None


# --------------------------------------------------------------------------- #
# Mixin behaviour                                                             #
# --------------------------------------------------------------------------- #


def _make_background_job(name: str) -> MagicMock:
    """Stand-in for `prime_sandboxes.BackgroundJob`."""
    job = MagicMock()
    job.job_id = f"job-{name}"
    job.stdout_log_file = f"/tmp/job_{name}.stdout.log"
    job.stderr_log_file = f"/tmp/job_{name}.stderr.log"
    job.exit_file = f"/tmp/job_{name}.exit"
    return job


class _DummyEnv(HarborMCPMixin):
    """Bare mixin host for unit testing; bypasses CliAgentEnv setup."""

    def __init__(self, *, mcp_launchers: dict[str, HarborMCPLauncher] | None = None):
        self.sandbox_client = MagicMock()
        # `execute_command` is used for health probes, /etc/hosts patches, stop cmds.
        self.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(exit_code=0, stdout="")
        )
        # `start_background_job` launches the MCP daemon.
        self.started_jobs: list[tuple[str, str]] = []

        async def _start_bg(sandbox_id, command, working_dir=None, env=None):
            # Derive a stable job id from the pid file referenced in the command;
            # fall back to a counter if the pattern isn't present.
            import re

            m = re.search(r"harbor-mcp-([^.]+)\.pid", command)
            name = m.group(1) if m else f"anon-{len(self.started_jobs)}"
            self.started_jobs.append((name, command))
            return _make_background_job(name)

        self.sandbox_client.start_background_job = AsyncMock(side_effect=_start_bg)
        # `get_background_job` — default to "still running"
        self.sandbox_client.get_background_job = AsyncMock(
            return_value=MagicMock(completed=False, exit_code=None, stderr="")
        )
        self.mcp_launchers = mcp_launchers or {}
        # Fast defaults so tests don't sleep.
        self.default_mcp_healthcheck = HarborMCPHealthcheck(
            interval_sec=0.0,
            start_interval_sec=0.0,
            timeout_sec=1.0,
            retries=1,
            start_period_sec=0.0,
        )


def _config_with_server(
    *,
    phases: Iterable[str] | None = None,
    launch: dict[str, Any] | None = None,
    name: str = "svc",
    port: int = 8000,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "transport": "streamable-http",
        "url": f"http://svc-host:{port}/mcp",
    }
    if phases is not None:
        entry["phases"] = list(phases)
    if launch is not None:
        entry["launch"] = launch
    return {"environment": {"mcp_servers": [entry]}}


class TestLauncherResolution:
    def test_task_toml_launch_beats_constructor(self):
        cfg_launch = {"command": "toml-cmd"}
        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="ctor-cmd")})
        (server,) = parse_mcp_servers(_config_with_server(launch=cfg_launch))
        launcher = env.mcp_launcher_for(server, state={}, phase="agent")
        assert launcher is not None
        assert launcher.command == "toml-cmd"

    def test_falls_back_to_constructor(self):
        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="ctor-cmd")})
        (server,) = parse_mcp_servers(_config_with_server())
        launcher = env.mcp_launcher_for(server, state={}, phase="agent")
        assert launcher is not None
        assert launcher.command == "ctor-cmd"

    def test_returns_none_when_unregistered(self):
        env = _DummyEnv()
        (server,) = parse_mcp_servers(_config_with_server())
        assert env.mcp_launcher_for(server, state={}, phase="agent") is None


class TestStartCommand:
    def test_su_wraps_command_when_user_set(self):
        env = _DummyEnv()
        server = HarborMCPServer(
            name="svc", transport="streamable-http", url="http://svc:8000/mcp"
        )
        launcher = HarborMCPLauncher(
            command=".venv/bin/python -u server.py",
            user="env-user",
            cwd="/opt/x",
            env={"STATIC": "1"},
        )
        cmd = env._mcp_start_cmd(server, launcher, 8000, {"EXTRA": "2"})
        assert "su -s /bin/sh env-user -c" in cmd
        assert "cd /opt/x" in cmd
        assert "MCP_TRANSPORT=streamable-http" in cmd
        assert "MCP_PORT=8000" in cmd
        assert "MCP_BIND_HOST=127.0.0.1" in cmd
        assert "STATIC=1" in cmd
        assert "EXTRA=2" in cmd
        # The SDK's start_background_job wraps in nohup for us, so our inner
        # command should exec directly (no nohup, no log redirection, no `&`).
        assert "exec .venv/bin/python -u server.py" in cmd
        assert "nohup" not in cmd
        assert ".log" not in cmd  # SDK owns stdout/stderr capture
        assert " & " not in cmd  # exec runs in the foreground of the sh -c body
        # PID must be recorded so _mcp_stop_cmd can kill the process later.
        assert "echo $$ > /tmp/harbor-mcp-svc.pid" in cmd

    def test_no_su_without_user(self):
        env = _DummyEnv()
        server = HarborMCPServer(
            name="svc", transport="streamable-http", url="http://svc:8000/mcp"
        )
        launcher = HarborMCPLauncher(command="server")
        cmd = env._mcp_start_cmd(server, launcher, 8000, {})
        assert "su -s" not in cmd
        assert "exec server" in cmd

    def test_extra_env_overrides_launcher_env(self):
        env = _DummyEnv()
        server = HarborMCPServer(
            name="svc", transport="streamable-http", url="http://svc:8000/mcp"
        )
        launcher = HarborMCPLauncher(command="x", env={"K": "old"})
        cmd = env._mcp_start_cmd(server, launcher, 8000, {"K": "new"})
        assert "K=new" in cmd
        assert "K=old" not in cmd

    def test_server_name_with_shell_metachars_is_quoted(self):
        """Server name is task-author-controlled; treat it like any other shell arg.

        Particularly matters for the no-user branch of `_mcp_start_cmd` (no outer
        `su -c 'shlex.quote(inner)'` wrapper) and every call site in
        `_mcp_stop_cmd` (which goes straight to `execute_command`).
        """
        env = _DummyEnv()
        # Name containing a shell-special sequence. The generated commands
        # must keep the `$(...)` literal, not let the outer sh -c evaluate it.
        server = HarborMCPServer(
            name="evil$(whoami)",
            transport="streamable-http",
            url="http://svc:8000/mcp",
        )
        launcher = HarborMCPLauncher(command="server")  # no user → no outer su quoting
        start = env._mcp_start_cmd(server, launcher, 8000, {})
        # The pid path must appear exactly once, already single-quoted.
        assert "'/tmp/harbor-mcp-evil$(whoami).pid'" in start
        # And it must NOT appear in unquoted form anywhere.
        unquoted = "/tmp/harbor-mcp-evil$(whoami).pid"
        assert start.count(unquoted) == start.count(f"'{unquoted}'")

        stop = env._mcp_stop_cmd("evil$(whoami)")
        # Both uses of the pid file in the stop command must be quoted.
        assert stop.count(f"'{unquoted}'") == 2
        assert stop.count(unquoted) == stop.count(f"'{unquoted}'")


class TestStopCommand:
    def test_default_is_sigkill_plus_rm(self):
        """Default: one SIGKILL, then unlink the pidfile. No graceful-shutdown theater."""
        env = _DummyEnv()
        cmd = env._mcp_stop_cmd("svc")
        assert "kill -9" in cmd
        assert "rm -f" in cmd
        # No TERM-then-KILL polling in the default.
        assert "kill -0" not in cmd
        assert "sleep" not in cmd
        # No lingering control-flow keywords — stays on one line.
        assert "for _" not in cmd
        assert "while" not in cmd

    def test_default_references_pidfile(self):
        env = _DummyEnv()
        cmd = env._mcp_stop_cmd("svc")
        assert "/tmp/harbor-mcp-svc.pid" in cmd

    def test_default_is_short(self):
        """Keep it one line — fewer round trips, easier to read in logs."""
        env = _DummyEnv()
        cmd = env._mcp_stop_cmd("svc")
        assert "\n" not in cmd
        # < 120 chars keeps it grep-friendly in sandbox exec logs.
        assert len(cmd) < 120

    def test_subclass_can_override_for_graceful_shutdown(self):
        """Users who need SIGTERM grace can override the method."""

        class GracefulEnv(_DummyEnv):
            def _mcp_stop_cmd(self, name):
                pid_file = self._mcp_pid_file(name)
                return (
                    f"pid=$(cat {pid_file} 2>/dev/null); "
                    f'[ -n "$pid" ] && kill "$pid"; sleep 5; '
                    f'kill -9 "$pid" 2>/dev/null; rm -f {pid_file}'
                )

        env = GracefulEnv()
        cmd = env._mcp_stop_cmd("svc")
        assert 'kill "$pid"' in cmd
        assert "sleep 5" in cmd


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_starts_only_servers_for_current_phase(self):
        env = _DummyEnv(
            mcp_launchers={
                "svc": HarborMCPLauncher(command="x"),
                "verifier-svc": HarborMCPLauncher(command="y"),
            }
        )
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "svc",
                        "transport": "streamable-http",
                        "url": "http://svc:8000/mcp",
                        "phases": ["agent"],
                    },
                    {
                        "name": "verifier-svc",
                        "transport": "streamable-http",
                        "url": "http://svc:9000/mcp",
                        "phases": ["verifier"],
                    },
                ]
            }
        }
        state: dict[str, Any] = {}
        await env.start_mcp_servers_for_phase("sbx", cfg, state, phase=DEFAULT_PHASE)
        assert state["harbor_mcp_started"] == ["svc"]

    @pytest.mark.asyncio
    async def test_restart_swaps_phase_servers(self):
        env = _DummyEnv(
            mcp_launchers={
                "agent-svc": HarborMCPLauncher(command="a"),
                "verifier-svc": HarborMCPLauncher(command="v"),
            }
        )
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "agent-svc",
                        "transport": "streamable-http",
                        "url": "http://x:8000/mcp",
                        "phases": ["agent"],
                    },
                    {
                        "name": "verifier-svc",
                        "transport": "streamable-http",
                        "url": "http://x:9000/mcp",
                        "phases": ["verifier"],
                    },
                ]
            }
        }
        state: dict[str, Any] = {}
        await env.start_mcp_servers_for_phase("sbx", cfg, state, phase="agent")
        assert state["harbor_mcp_started"] == ["agent-svc"]

        await env.restart_mcp_for_phase("sbx", state, phase="verifier")
        assert state["harbor_mcp_started"] == ["verifier-svc"]

    @pytest.mark.asyncio
    async def test_restart_leaves_multi_phase_server_alone(self):
        env = _DummyEnv(
            mcp_launchers={"svc": HarborMCPLauncher(command="x")},
        )
        cfg = _config_with_server(phases=["agent", "verifier"])
        state: dict[str, Any] = {}
        await env.start_mcp_servers_for_phase("sbx", cfg, state, phase="agent")
        env.sandbox_client.execute_command.reset_mock()
        env.sandbox_client.start_background_job.reset_mock()

        await env.restart_mcp_for_phase("sbx", state, phase="verifier")

        # Server applies to both phases → no new job started, no stop command.
        env.sandbox_client.start_background_job.assert_not_awaited()
        stop_cmds = [
            c.args[1]
            for c in env.sandbox_client.execute_command.call_args_list
            if "kill -9" in c.args[1] and "harbor-mcp-svc.pid" in c.args[1]
        ]
        assert stop_cmds == []
        assert state["harbor_mcp_started"] == ["svc"]

    @pytest.mark.asyncio
    async def test_externally_managed_server_is_skipped(self):
        env = _DummyEnv()  # no launchers
        cfg = _config_with_server()
        state: dict[str, Any] = {}
        await env.start_mcp_servers_for_phase("sbx", cfg, state, phase="agent")
        assert state["harbor_mcp_started"] == []

    @pytest.mark.asyncio
    async def test_stop_mcp_servers_cleanup_clears_state(self):
        env = _DummyEnv(
            mcp_launchers={"svc": HarborMCPLauncher(command="x")},
        )
        cfg = _config_with_server()
        state: dict[str, Any] = {"sandbox_id": "sbx"}
        await env.start_mcp_servers_for_phase("sbx", cfg, state, phase="agent")
        assert state["harbor_mcp_started"] == ["svc"]

        await env.stop_mcp_servers(state)
        assert state["harbor_mcp_started"] == []


class TestBackgroundJob:
    @pytest.mark.asyncio
    async def test_uses_start_background_job_not_execute_command(self):
        """Daemon launch should go through start_background_job, not execute_command."""
        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="python x")})
        cfg = _config_with_server()
        state: dict[str, Any] = {}
        await env.start_mcp_servers_for_phase("sbx", cfg, state, phase="agent")

        env.sandbox_client.start_background_job.assert_awaited_once()
        # The only execute_command calls should be /etc/hosts patch + the TCP
        # health probe — never the MCP start command itself.
        start_commands = [
            c.args[1]
            for c in env.sandbox_client.execute_command.call_args_list
            if "exec python x" in c.args[1]
        ]
        assert start_commands == []

    @pytest.mark.asyncio
    async def test_background_job_stored_in_state(self):
        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="x")})
        cfg = _config_with_server()
        state: dict[str, Any] = {}
        await env.start_mcp_servers_for_phase("sbx", cfg, state, phase="agent")

        jobs = state["harbor_mcp_jobs"]
        assert set(jobs.keys()) == {"svc"}
        assert jobs["svc"].job_id == "job-svc"

    @pytest.mark.asyncio
    async def test_stop_clears_job_handle(self):
        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="x")})
        cfg = _config_with_server()
        state: dict[str, Any] = {"sandbox_id": "sbx"}
        await env.start_mcp_servers_for_phase("sbx", cfg, state, phase="agent")
        assert "svc" in state["harbor_mcp_jobs"]

        await env._stop_mcp_server("sbx", "svc", state)
        assert "svc" not in state["harbor_mcp_jobs"]

    @pytest.mark.asyncio
    async def test_early_crash_bails_out_with_stderr(self):
        """If the daemon exits before the port opens, fail fast with its stderr."""
        import verifiers as vf

        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="x")})
        # Daemon reports "already exited with stack trace" on first poll.
        env.sandbox_client.get_background_job = AsyncMock(
            return_value=MagicMock(
                completed=True,
                exit_code=1,
                stderr="ImportError: missing widget",
            )
        )
        # TCP probe would still say "not listening", but we shouldn't need it.
        env.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(exit_code=1, stdout="")
        )
        # Large retry budget — we need to prove we bail on the 1st iteration.
        env.default_mcp_healthcheck = HarborMCPHealthcheck(
            retries=5, interval_sec=0.0, start_interval_sec=0.0, timeout_sec=1.0
        )

        cfg = _config_with_server()
        state: dict[str, Any] = {}
        with pytest.raises(vf.SandboxError, match="exited before becoming healthy"):
            await env.start_mcp_servers_for_phase("sbx", cfg, state, phase="agent")
        assert env.sandbox_client.get_background_job.await_count == 1


class TestHealthCheck:
    """Readiness probing — default `/proc/net/tcp` + user override."""

    def test_default_probe_uses_procfs_not_bash(self):
        cmd = HarborMCPMixin._default_mcp_health_cmd(8000)
        assert "bash" not in cmd
        assert "/dev/tcp" not in cmd
        assert "/proc/net/tcp" in cmd
        assert "/proc/net/tcp6" in cmd

    @pytest.mark.parametrize(
        "port,hex_expected",
        [(80, "0050"), (8000, "1F40"), (65535, "FFFF"), (1, "0001")],
    )
    def test_default_probe_encodes_port_as_uppercase_hex(self, port, hex_expected):
        cmd = HarborMCPMixin._default_mcp_health_cmd(port)
        assert f":{hex_expected}$" in cmd

    def test_default_probe_requires_listen_state(self):
        # State `0A` means LISTEN in /proc/net/tcp; we must filter on it so
        # we don't accept an ESTABLISHED connection FROM a coincidentally-
        # matching ephemeral port.
        cmd = HarborMCPMixin._default_mcp_health_cmd(8000)
        assert '$4 == "0A"' in cmd

    @pytest.mark.asyncio
    async def test_launcher_override_takes_precedence(self):
        env = _DummyEnv(
            mcp_launchers={
                "svc": HarborMCPLauncher(
                    command="x",
                    healthcheck=HarborMCPHealthcheck(
                        command="curl -fs http://127.0.0.1:{port}/health",
                        interval_sec=0.0,
                        start_interval_sec=0.0,
                        timeout_sec=1.0,
                        retries=1,
                    ),
                )
            }
        )
        cfg = _config_with_server()
        state: dict[str, Any] = {}
        await env.start_mcp_servers_for_phase("sbx", cfg, state, phase="agent")

        health_calls = [
            c.args[1]
            for c in env.sandbox_client.execute_command.call_args_list
            if "http://127.0.0.1" in c.args[1]
        ]
        assert health_calls == ["curl -fs http://127.0.0.1:8000/health"]
        # Default /proc/net/tcp probe must NOT have been issued.
        assert not any(
            "/proc/net/tcp" in c.args[1]
            for c in env.sandbox_client.execute_command.call_args_list
        )

    @pytest.mark.asyncio
    async def test_default_probe_issued_when_no_override(self):
        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="x")})
        cfg = _config_with_server()
        await env.start_mcp_servers_for_phase("sbx", cfg, {}, phase="agent")

        health_calls = [
            c.args[1]
            for c in env.sandbox_client.execute_command.call_args_list
            if "/proc/net/tcp" in c.args[1]
        ]
        assert len(health_calls) == 1
        assert ":1F40$" in health_calls[0]


class TestEtcHosts:
    @pytest.mark.asyncio
    async def test_patches_non_loopback_hosts(self):
        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="x")})
        cfg = _config_with_server()
        await env.start_mcp_servers_for_phase("sbx", cfg, {}, phase="agent")
        commands = [
            call.args[1] for call in env.sandbox_client.execute_command.call_args_list
        ]
        assert any("svc-host" in c and "/etc/hosts" in c for c in commands), (
            "expected /etc/hosts patch for service-name URL"
        )

    @pytest.mark.asyncio
    async def test_skips_loopback_hosts(self):
        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="x")})
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "svc",
                        "transport": "streamable-http",
                        "url": "http://localhost:8000/mcp",
                    }
                ]
            }
        }
        await env.start_mcp_servers_for_phase("sbx", cfg, {}, phase="agent")
        commands = [
            call.args[1] for call in env.sandbox_client.execute_command.call_args_list
        ]
        assert not any("/etc/hosts" in c for c in commands)


class TestEnvVarPublishing:
    def test_publishes_only_servers_active_in_phase(self):
        env = _DummyEnv()
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "only-agent",
                        "transport": "streamable-http",
                        "url": "http://x:1/mcp",
                        "phases": ["agent"],
                    },
                    {
                        "name": "only-verifier",
                        "transport": "streamable-http",
                        "url": "http://x:2/mcp",
                        "phases": ["verifier"],
                    },
                ]
            }
        }
        agent = env.mcp_agent_env_vars(cfg, phase="agent")
        verifier = env.mcp_agent_env_vars(cfg, phase="verifier")
        assert agent == {"HARBOR_MCP_ONLY_AGENT_URL": "http://127.0.0.1:1/mcp"}
        assert verifier == {"HARBOR_MCP_ONLY_VERIFIER_URL": "http://127.0.0.1:2/mcp"}


class TestHarborValidation:
    """Mirrors `harbor.models.task.config.MCPServerConfig.validate_transport_fields`."""

    def test_sse_requires_url(self):
        cfg = {
            "environment": {
                "mcp_servers": [{"name": "svc", "transport": "sse"}],
            }
        }
        with pytest.raises(ValueError, match="'url' is required for transport 'sse'"):
            parse_mcp_servers(cfg)

    def test_streamable_http_requires_url(self):
        cfg = {
            "environment": {
                "mcp_servers": [{"name": "svc", "transport": "streamable-http"}],
            }
        }
        with pytest.raises(
            ValueError, match="'url' is required for transport 'streamable-http'"
        ):
            parse_mcp_servers(cfg)

    def test_stdio_requires_command(self):
        cfg = {
            "environment": {
                "mcp_servers": [{"name": "svc", "transport": "stdio"}],
            }
        }
        with pytest.raises(
            ValueError, match="'command' is required for transport 'stdio'"
        ):
            parse_mcp_servers(cfg)


class TestHarborHealthcheckSemantics:
    """Mirrors `harbor.environments.base.run_healthcheck` retry/start-period logic."""

    @pytest.mark.asyncio
    async def test_launch_succeeds_when_probe_passes_first_try(self):
        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="x")})
        cfg = _config_with_server()
        await env.start_mcp_servers_for_phase("sbx", cfg, {}, phase="agent")
        # Exactly one probe attempt should have run.
        probes = [
            c
            for c in env.sandbox_client.execute_command.call_args_list
            if "/proc/net/tcp" in c.args[1]
        ]
        assert len(probes) == 1

    @staticmethod
    def _probe_count(env) -> int:
        """Count probe calls (exclude /etc/hosts patching etc.)."""
        return sum(
            1
            for c in env.sandbox_client.execute_command.call_args_list
            if "dummy" in c.args[1] or "/proc/net/tcp" in c.args[1]
        )

    @pytest.mark.asyncio
    async def test_failures_during_start_period_do_not_count(self):
        """Harbor parity: failures during `start_period_sec` never exhaust retries."""
        # Eight failures with retries=3 would normally exceed the budget. With a
        # start_period covering all of them, we should pass once the probe
        # flips to healthy.
        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="x")})
        env.default_mcp_healthcheck = HarborMCPHealthcheck(
            command="dummy {port}",
            retries=3,
            interval_sec=0.0,
            start_period_sec=1000.0,  # cover all failures below
            start_interval_sec=0.0,
            timeout_sec=1.0,
        )
        # /etc/hosts patch + 8 failures + 1 success.
        results = (
            [MagicMock(exit_code=0)]  # /etc/hosts patch
            + [MagicMock(exit_code=1)] * 8
            + [MagicMock(exit_code=0)]
        )
        env.sandbox_client.execute_command = AsyncMock(side_effect=list(results))

        cfg = _config_with_server()
        state: dict[str, Any] = {}
        await env.start_mcp_servers_for_phase("sbx", cfg, state, phase="agent")
        assert self._probe_count(env) == 9

    @pytest.mark.asyncio
    async def test_retry_budget_enforced_after_start_period(self):
        """Harbor parity: consecutive failures after start period fail at `retries`."""
        import verifiers as vf

        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="x")})
        env.default_mcp_healthcheck = HarborMCPHealthcheck(
            command="dummy {port}",
            retries=2,
            interval_sec=0.0,
            start_period_sec=0.0,  # no grace period
            start_interval_sec=0.0,
            timeout_sec=1.0,
        )
        env.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(exit_code=1)
        )

        cfg = _config_with_server()
        state: dict[str, Any] = {}
        with pytest.raises(vf.SandboxError, match="after 2 consecutive retries"):
            await env.start_mcp_servers_for_phase("sbx", cfg, state, phase="agent")
        # Exactly `retries` probes before giving up.
        assert self._probe_count(env) == 2

    @pytest.mark.asyncio
    async def test_per_probe_timeout_is_respected(self):
        env = _DummyEnv(mcp_launchers={"svc": HarborMCPLauncher(command="x")})
        env.default_mcp_healthcheck = HarborMCPHealthcheck(
            command="dummy {port}",
            retries=1,
            interval_sec=0.0,
            start_interval_sec=0.0,
            timeout_sec=7.5,
        )
        env.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(exit_code=0)
        )
        cfg = _config_with_server()
        await env.start_mcp_servers_for_phase("sbx", cfg, {}, phase="agent")

        probe_calls = [
            c
            for c in env.sandbox_client.execute_command.call_args_list
            if "dummy" in c.args[1]
        ]
        assert len(probe_calls) == 1
        # execute_command was called with timeout=int(7.5)=7.
        assert probe_calls[0].kwargs["timeout"] == 7


class TestHarborHealthcheckParsing:
    def test_from_dict_roundtrip(self):
        hc = HarborMCPHealthcheck.from_dict(
            {
                "command": "true {port}",
                "interval_sec": 1.5,
                "timeout_sec": 3.0,
                "start_period_sec": 10.0,
                "start_interval_sec": 1.0,
                "retries": 5,
                "ignored_field": "ignored",
            }
        )
        assert hc.command == "true {port}"
        assert hc.interval_sec == 1.5
        assert hc.timeout_sec == 3.0
        assert hc.start_period_sec == 10.0
        assert hc.start_interval_sec == 1.0
        assert hc.retries == 5

    def test_launcher_parses_nested_healthcheck_subtable(self):
        launcher = HarborMCPLauncher.from_dict(
            {
                "command": ".venv/bin/python server.py",
                "healthcheck": {
                    "command": "curl -fs http://127.0.0.1:{port}/health",
                    "retries": 10,
                    "start_period_sec": 5.0,
                },
            }
        )
        assert isinstance(launcher.healthcheck, HarborMCPHealthcheck)
        assert launcher.healthcheck.command == "curl -fs http://127.0.0.1:{port}/health"
        assert launcher.healthcheck.retries == 10
        assert launcher.healthcheck.start_period_sec == 5.0
