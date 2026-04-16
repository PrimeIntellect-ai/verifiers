from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from verifiers.envs.experimental.harbor_env import (
    HarborMCPHealthcheck,
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

    def test_skips_entries_without_name(self):
        cfg = {"environment": {"mcp_servers": [{"transport": "stdio"}]}}
        assert parse_mcp_servers(cfg) == []

    def test_default_transport_matches_harbor(self):
        """Harbor's MCPServerConfig.transport defaults to 'sse'. Stay consistent."""
        cfg = {
            "environment": {"mcp_servers": [{"name": "svc", "url": "http://svc:1/mcp"}]}
        }
        (server,) = parse_mcp_servers(cfg)
        assert server.transport == "sse"

    def test_does_not_leak_harbor_env_only_fields(self):
        """HarborEnv-specific keys must never land on HarborMCPServer —
        task.toml stays pure Harbor."""
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "svc",
                        "transport": "streamable-http",
                        "url": "http://svc:1/mcp",
                        "launch": {"command": "should-be-ignored"},
                        "phases": ["verifier"],
                    }
                ]
            }
        }
        (server,) = parse_mcp_servers(cfg)
        for forbidden in ("launch", "phases", "raw"):
            assert not hasattr(server, forbidden), (
                f"HarborMCPServer leaks `{forbidden}`; keep task.toml pure Harbor"
            )
        assert server.name == "svc"
        assert server.transport == "streamable-http"
        assert server.url == "http://svc:1/mcp"


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


class TestHarborValidation:
    """Mirrors `harbor.models.task.config.MCPServerConfig.validate_transport_fields`."""

    def test_sse_requires_url(self):
        cfg = {"environment": {"mcp_servers": [{"name": "svc", "transport": "sse"}]}}
        with pytest.raises(ValueError, match="'url' is required for transport 'sse'"):
            parse_mcp_servers(cfg)

    def test_streamable_http_requires_url(self):
        cfg = {
            "environment": {
                "mcp_servers": [{"name": "svc", "transport": "streamable-http"}]
            }
        }
        with pytest.raises(
            ValueError, match="'url' is required for transport 'streamable-http'"
        ):
            parse_mcp_servers(cfg)

    def test_stdio_requires_command(self):
        cfg = {"environment": {"mcp_servers": [{"name": "svc", "transport": "stdio"}]}}
        with pytest.raises(
            ValueError, match="'command' is required for transport 'stdio'"
        ):
            parse_mcp_servers(cfg)


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

    def __init__(
        self,
        *,
        mcp_launch_commands: dict[str, str] | None = None,
    ):
        self.sandbox_client = MagicMock()
        self.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(exit_code=0, stdout="")
        )
        self.started_jobs: list[tuple[str, str]] = []

        async def _start_bg(sandbox_id, command, working_dir=None, env=None):
            import re

            m = re.search(r"harbor-mcp-([^.]+)\.pid", command)
            name = m.group(1) if m else f"anon-{len(self.started_jobs)}"
            self.started_jobs.append((name, command))
            return _make_background_job(name)

        self.sandbox_client.start_background_job = AsyncMock(side_effect=_start_bg)
        self.sandbox_client.get_background_job = AsyncMock(
            return_value=MagicMock(completed=False, exit_code=None, stderr="")
        )
        self.mcp_launch_commands = mcp_launch_commands or {}
        # Fast defaults so tests don't sleep.
        self.mcp_healthcheck = HarborMCPHealthcheck(
            interval_sec=0.0,
            timeout_sec=1.0,
            retries=1,
            start_period_sec=0.0,
        )


def _config_with_server(*, name: str = "svc", port: int = 8000) -> dict[str, Any]:
    """Pure-Harbor task.toml fragment with one streamable-http MCP server."""
    return {
        "environment": {
            "mcp_servers": [
                {
                    "name": name,
                    "transport": "streamable-http",
                    "url": f"http://svc-host:{port}/mcp",
                }
            ]
        }
    }


class TestLaunchCommandResolution:
    @pytest.mark.asyncio
    async def test_returns_command_from_constructor_dict(self):
        env = _DummyEnv(mcp_launch_commands={"svc": "ctor-cmd"})
        (server,) = parse_mcp_servers(_config_with_server())
        cmd = await env.mcp_launch_command(server, state={})
        assert cmd == "ctor-cmd"

    @pytest.mark.asyncio
    async def test_returns_none_when_unregistered(self):
        """No entry → server is externally managed (e.g. a docker-compose sidecar)."""
        env = _DummyEnv()
        (server,) = parse_mcp_servers(_config_with_server())
        assert await env.mcp_launch_command(server, state={}) is None

    @pytest.mark.asyncio
    async def test_subclass_override_beats_constructor(self):
        """Subclasses can override `mcp_launch_command` for dynamic behavior."""

        class DynamicEnv(_DummyEnv):
            async def mcp_launch_command(self, server, state):
                return f"cmd-for-{state.get('rollout_id', 'unknown')}"

        env = DynamicEnv(mcp_launch_commands={"svc": "ctor-cmd"})
        (server,) = parse_mcp_servers(_config_with_server())
        assert (
            await env.mcp_launch_command(server, state={"rollout_id": "r1"})
            == "cmd-for-r1"
        )


class TestStartCommand:
    def test_records_real_process_pid_via_bang(self):
        """PID tracking must use `$!` (backgrounded child), NOT `$$` (outer shell).

        Inside the SDK's `(...)` subshell, `$$` is the outer shell's PID —
        killing it would leave the exec'd target orphaned.
        """
        env = _DummyEnv()
        cmd = env._mcp_start_cmd("svc", "python server.py")
        assert "echo $!" in cmd
        assert "echo $$" not in cmd
        # Must end with `wait` so the subshell's exit code reflects the target's.
        assert cmd.rstrip().endswith("wait")
        assert "/tmp/harbor-mcp-svc.pid" in cmd

    def test_user_command_is_not_exec_wrapped(self):
        """No `exec`, no `su`, no extra env prefix — just background + pidfile."""
        env = _DummyEnv()
        cmd = env._mcp_start_cmd("svc", "python -u /opt/x/server.py")
        assert "exec" not in cmd
        assert "su -s" not in cmd
        assert "python -u /opt/x/server.py" in cmd

    def test_server_name_with_shell_metachars_is_quoted(self):
        """Server name is task-author-controlled; treat it like any shell arg."""
        env = _DummyEnv()
        start = env._mcp_start_cmd("evil$(whoami)", "server")
        # Pid path appears exactly once, already single-quoted.
        assert "'/tmp/harbor-mcp-evil$(whoami).pid'" in start
        unquoted = "/tmp/harbor-mcp-evil$(whoami).pid"
        assert start.count(unquoted) == start.count(f"'{unquoted}'")

        stop = env._mcp_stop_cmd("evil$(whoami)")
        assert stop.count(f"'{unquoted}'") == 2
        assert stop.count(unquoted) == stop.count(f"'{unquoted}'")


class TestStopCommand:
    def test_default_is_sigkill_plus_rm(self):
        """Default: one SIGKILL, then unlink the pidfile."""
        env = _DummyEnv()
        cmd = env._mcp_stop_cmd("svc")
        assert "kill -9" in cmd
        assert "rm -f" in cmd
        assert "kill -0" not in cmd
        assert "sleep" not in cmd

    def test_default_references_pidfile(self):
        env = _DummyEnv()
        cmd = env._mcp_stop_cmd("svc")
        assert "/tmp/harbor-mcp-svc.pid" in cmd

    def test_default_is_short(self):
        """Keep it one line — fewer round trips, easier to grep in logs."""
        env = _DummyEnv()
        cmd = env._mcp_stop_cmd("svc")
        assert "\n" not in cmd
        assert len(cmd) < 120


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_starts_server_with_registered_launch_command(self):
        env = _DummyEnv(mcp_launch_commands={"svc": "python server.py"})
        state: dict[str, Any] = {}
        await env.start_mcp_servers("sbx", _config_with_server(), state)
        assert set(state["harbor_mcp_jobs"].keys()) == {"svc"}

    @pytest.mark.asyncio
    async def test_externally_managed_server_is_skipped(self):
        """Declared in task.toml but no launch command → treated as externally managed."""
        env = _DummyEnv()
        state: dict[str, Any] = {}
        await env.start_mcp_servers("sbx", _config_with_server(), state)
        assert state.get("harbor_mcp_jobs", {}) == {}
        env.sandbox_client.start_background_job.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_mcp_servers_cleanup_clears_state(self):
        env = _DummyEnv(mcp_launch_commands={"svc": "python server.py"})
        state: dict[str, Any] = {"sandbox_id": "sbx"}
        await env.start_mcp_servers("sbx", _config_with_server(), state)
        assert "svc" in state["harbor_mcp_jobs"]

        await env.stop_mcp_servers(state)
        assert state["harbor_mcp_jobs"] == {}

    @pytest.mark.asyncio
    async def test_stop_issues_kill_with_pidfile(self):
        env = _DummyEnv(mcp_launch_commands={"svc": "python server.py"})
        state: dict[str, Any] = {"sandbox_id": "sbx"}
        await env.start_mcp_servers("sbx", _config_with_server(), state)
        env.sandbox_client.execute_command.reset_mock()

        await env.stop_mcp_servers(state)

        stop_calls = [
            c.args[1]
            for c in env.sandbox_client.execute_command.call_args_list
            if "kill -9" in c.args[1]
        ]
        assert len(stop_calls) == 1
        assert "harbor-mcp-svc.pid" in stop_calls[0]

    @pytest.mark.asyncio
    async def test_stop_without_sandbox_id_is_a_noop(self):
        env = _DummyEnv()
        await env.stop_mcp_servers({})  # no sandbox_id
        env.sandbox_client.execute_command.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stdio_server_is_ignored(self):
        """Stdio transports have no network endpoint — mixin must not try to start them."""
        env = _DummyEnv(mcp_launch_commands={"svc": "python server.py"})
        cfg = {
            "environment": {
                "mcp_servers": [
                    {"name": "svc", "transport": "stdio", "command": "python x.py"}
                ]
            }
        }
        await env.start_mcp_servers("sbx", cfg, {})
        env.sandbox_client.start_background_job.assert_not_awaited()


class TestBackgroundJob:
    @pytest.mark.asyncio
    async def test_uses_start_background_job_not_execute_command(self):
        """Daemon launch goes through start_background_job, not execute_command."""
        env = _DummyEnv(mcp_launch_commands={"svc": "python x"})
        state: dict[str, Any] = {}
        await env.start_mcp_servers("sbx", _config_with_server(), state)

        env.sandbox_client.start_background_job.assert_awaited_once()
        start_commands = [
            c.args[1]
            for c in env.sandbox_client.execute_command.call_args_list
            if "python x &" in c.args[1]
        ]
        assert start_commands == []

    @pytest.mark.asyncio
    async def test_background_job_stored_in_state(self):
        env = _DummyEnv(mcp_launch_commands={"svc": "python x"})
        state: dict[str, Any] = {}
        await env.start_mcp_servers("sbx", _config_with_server(), state)

        jobs = state["harbor_mcp_jobs"]
        assert set(jobs.keys()) == {"svc"}
        assert jobs["svc"].job_id == "job-svc"

    @pytest.mark.asyncio
    async def test_early_crash_bails_out_with_stderr(self):
        """If the daemon exits before the port opens, fail fast with its stderr."""
        import verifiers as vf

        env = _DummyEnv(mcp_launch_commands={"svc": "python x"})
        env.sandbox_client.get_background_job = AsyncMock(
            return_value=MagicMock(
                completed=True,
                exit_code=1,
                stderr="ImportError: missing widget",
            )
        )
        env.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(exit_code=1, stdout="")
        )
        env.mcp_healthcheck = HarborMCPHealthcheck(
            retries=5, interval_sec=0.0, timeout_sec=1.0
        )

        with pytest.raises(vf.SandboxError, match="exited before becoming healthy"):
            await env.start_mcp_servers("sbx", _config_with_server(), {})
        # We should bail on the 1st poll, not burn the retry budget.
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
        cmd = HarborMCPMixin._default_mcp_health_cmd(8000)
        assert '$4 == "0A"' in cmd

    @pytest.mark.asyncio
    async def test_custom_healthcheck_command_templated_with_port(self):
        env = _DummyEnv(mcp_launch_commands={"svc": "python x"})
        env.mcp_healthcheck = HarborMCPHealthcheck(
            command="curl -fs http://127.0.0.1:{port}/health",
            interval_sec=0.0,
            timeout_sec=1.0,
            retries=1,
        )
        await env.start_mcp_servers("sbx", _config_with_server(), {})

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
        env = _DummyEnv(mcp_launch_commands={"svc": "python x"})
        await env.start_mcp_servers("sbx", _config_with_server(), {})

        health_calls = [
            c.args[1]
            for c in env.sandbox_client.execute_command.call_args_list
            if "/proc/net/tcp" in c.args[1]
        ]
        assert len(health_calls) == 1
        assert ":1F40$" in health_calls[0]

    @pytest.mark.asyncio
    async def test_probe_timeout_is_respected(self):
        env = _DummyEnv(mcp_launch_commands={"svc": "python x"})
        env.mcp_healthcheck = HarborMCPHealthcheck(
            command="dummy {port}",
            retries=1,
            interval_sec=0.0,
            timeout_sec=7.5,
        )
        env.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(exit_code=0)
        )
        await env.start_mcp_servers("sbx", _config_with_server(), {})

        probe_calls = [
            c
            for c in env.sandbox_client.execute_command.call_args_list
            if "dummy" in c.args[1]
        ]
        assert len(probe_calls) == 1
        # execute_command was called with timeout=int(7.5)=7.
        assert probe_calls[0].kwargs["timeout"] == 7


class TestHarborHealthcheckSemantics:
    """Mirrors `harbor.environments.base.run_healthcheck` retry/start-period logic."""

    @staticmethod
    def _probe_count(env) -> int:
        return sum(
            1
            for c in env.sandbox_client.execute_command.call_args_list
            if "dummy" in c.args[1] or "/proc/net/tcp" in c.args[1]
        )

    @pytest.mark.asyncio
    async def test_failures_during_start_period_do_not_count(self):
        """Failures during `start_period_sec` never exhaust retries."""
        env = _DummyEnv(mcp_launch_commands={"svc": "x"})
        env.mcp_healthcheck = HarborMCPHealthcheck(
            command="dummy {port}",
            retries=3,
            interval_sec=0.0,
            start_period_sec=1000.0,  # cover all failures below
            timeout_sec=1.0,
        )
        results = (
            [MagicMock(exit_code=0)]  # /etc/hosts patch
            + [MagicMock(exit_code=1)] * 8
            + [MagicMock(exit_code=0)]
        )
        env.sandbox_client.execute_command = AsyncMock(side_effect=list(results))

        await env.start_mcp_servers("sbx", _config_with_server(), {})
        assert self._probe_count(env) == 9

    @pytest.mark.asyncio
    async def test_retry_budget_enforced_after_start_period(self):
        """Consecutive failures after start period fail at `retries`."""
        import verifiers as vf

        env = _DummyEnv(mcp_launch_commands={"svc": "x"})
        env.mcp_healthcheck = HarborMCPHealthcheck(
            command="dummy {port}",
            retries=2,
            interval_sec=0.0,
            start_period_sec=0.0,
            timeout_sec=1.0,
        )
        env.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(exit_code=1)
        )

        with pytest.raises(vf.SandboxError, match="after 2 consecutive retries"):
            await env.start_mcp_servers("sbx", _config_with_server(), {})
        assert self._probe_count(env) == 2


class TestEnvVarPublishing:
    def test_publishes_url_for_every_declared_server(self):
        env = _DummyEnv()
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "one",
                        "transport": "streamable-http",
                        "url": "http://x:1/mcp",
                    },
                    {
                        "name": "two",
                        "transport": "streamable-http",
                        "url": "http://x:2/mcp",
                    },
                ]
            }
        }
        assert env.mcp_agent_env_vars(cfg) == {
            "HARBOR_MCP_ONE_URL": "http://127.0.0.1:1/mcp",
            "HARBOR_MCP_TWO_URL": "http://127.0.0.1:2/mcp",
        }

    def test_stdio_servers_are_not_published(self):
        """Stdio servers don't have URLs — nothing to publish."""
        env = _DummyEnv()
        cfg = {
            "environment": {
                "mcp_servers": [
                    {"name": "stdio-srv", "transport": "stdio", "command": "x"}
                ]
            }
        }
        assert env.mcp_agent_env_vars(cfg) == {}

    def test_server_name_is_normalized_to_upper_snake(self):
        env = _DummyEnv()
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "my-cool-server",
                        "transport": "streamable-http",
                        "url": "http://x:1/mcp",
                    }
                ]
            }
        }
        assert "HARBOR_MCP_MY_COOL_SERVER_URL" in env.mcp_agent_env_vars(cfg)


class TestEtcHosts:
    @pytest.mark.asyncio
    async def test_patches_non_loopback_hosts(self):
        env = _DummyEnv(mcp_launch_commands={"svc": "x"})
        await env.start_mcp_servers("sbx", _config_with_server(), {})
        commands = [
            call.args[1] for call in env.sandbox_client.execute_command.call_args_list
        ]
        assert any("svc-host" in c and "/etc/hosts" in c for c in commands), (
            "expected /etc/hosts patch for service-name URL"
        )

    @pytest.mark.asyncio
    async def test_skips_loopback_hosts(self):
        env = _DummyEnv(mcp_launch_commands={"svc": "x"})
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
        await env.start_mcp_servers("sbx", cfg, {})
        commands = [
            call.args[1] for call in env.sandbox_client.execute_command.call_args_list
        ]
        assert not any("/etc/hosts" in c for c in commands)

    @pytest.mark.asyncio
    async def test_patches_externally_managed_server_hosts_too(self):
        """Servers without launch commands still need /etc/hosts aliasing so
        the agent can reach them by the task.toml hostname."""
        env = _DummyEnv()  # no launch commands
        await env.start_mcp_servers("sbx", _config_with_server(), {})
        commands = [
            call.args[1] for call in env.sandbox_client.execute_command.call_args_list
        ]
        assert any("svc-host" in c and "/etc/hosts" in c for c in commands)
