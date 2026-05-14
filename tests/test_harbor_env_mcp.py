import asyncio
import time
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

    @pytest.mark.parametrize(
        "url,expected",
        [
            # Basic userinfo survives the host rewrite.
            (
                "http://user:pass@svc:8000/mcp",
                "http://user:pass@127.0.0.1:8000/mcp",
            ),
            # Username only (no password).
            ("http://user@svc:8000/mcp", "http://user@127.0.0.1:8000/mcp"),
            # Percent-encoded userinfo is preserved byte-for-byte — notice the
            # `@` inside the encoded password that would trip a naive split.
            (
                "http://user:p%40ss@svc:8000/mcp",
                "http://user:p%40ss@127.0.0.1:8000/mcp",
            ),
            # Query strings + userinfo.
            (
                "http://u:p@svc:8000/mcp?x=1&y=2",
                "http://u:p@127.0.0.1:8000/mcp?x=1&y=2",
            ),
        ],
    )
    def test_mcp_agent_url_preserves_basic_auth_userinfo(self, url: str, expected: str):
        """Dropping `user:pass@` would silently break auth-bearing MCP URLs."""
        s = HarborMCPServer(name="svc", transport="streamable-http", url=url)
        assert mcp_agent_url(s) == expected


class TestHarborValidation:
    """Mirrors `harbor.models.task.config.MCPServerConfig.validate_transport_fields`."""

    @pytest.mark.parametrize(
        "transport,expected_error",
        [
            ("sse", "'url' is required for transport 'sse'"),
            ("streamable-http", "'url' is required for transport 'streamable-http'"),
            ("stdio", "'command' is required for transport 'stdio'"),
        ],
    )
    def test_required_fields_per_transport(self, transport, expected_error):
        cfg = {
            "environment": {"mcp_servers": [{"name": "svc", "transport": transport}]}
        }
        with pytest.raises(ValueError, match=expected_error):
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


class TestStartStopCommands:
    def test_start_cmd_tracks_process_group_leader_pid(self):
        """Start command must capture `$!` (the backgrounded pgroup leader),
        not `$$` (the outer shell), and must end with `wait` so the recorded
        exit code reflects the launched daemon's.
        """
        cmd = _DummyEnv()._mcp_start_cmd("svc", "python -u /opt/x/server.py")
        assert "echo $!" in cmd
        assert "echo $$" not in cmd
        assert cmd.rstrip().endswith("wait")
        assert "/tmp/harbor-mcp-svc.pid" in cmd
        assert "python -u /opt/x/server.py" in cmd

    def test_start_cmd_wraps_in_setsid_for_process_group_semantics(self):
        """Wrapping the user's command in `setsid sh -c ...` is what makes
        `$!` a process-group leader, so `kill -9 -$PID` can reap the whole
        daemon tree on stop. Compound commands (e.g. `cd /x && python y.py`)
        must be preserved verbatim inside the sh -c payload so their own
        semantics are unchanged."""
        cmd = _DummyEnv()._mcp_start_cmd("svc", "cd /opt && python server.py")
        assert "setsid sh -c " in cmd
        assert "'cd /opt && python server.py'" in cmd

    def test_stop_cmd_is_one_line_sigkill_plus_rm(self):
        """Default: one SIGKILL to the process group, then unlink the
        pidfile — no poll/sleep loop."""
        cmd = _DummyEnv()._mcp_stop_cmd("svc")
        assert "kill -9" in cmd
        assert "rm -f" in cmd
        assert "/tmp/harbor-mcp-svc.pid" in cmd
        assert "kill -0" not in cmd
        assert "sleep" not in cmd
        assert "\n" not in cmd
        assert len(cmd) < 120

    def test_stop_cmd_targets_process_group_not_single_pid(self):
        """The `-` prefix on the `$(cat …)` expansion is what turns kill(1)
        into a process-group kill — without it, SIGKILL only lands on the
        wrapping shell and e.g. a `python` child spawned via `cd && python`
        leaks as an orphan."""
        cmd = _DummyEnv()._mcp_stop_cmd("svc")
        assert 'kill -9 -"$(cat' in cmd

    def test_server_name_with_shell_metachars_is_quoted(self):
        """Server name is task-author-controlled; every pidfile reference
        must appear only inside single-quoted spans."""
        env = _DummyEnv()
        unquoted = "/tmp/harbor-mcp-evil$(whoami).pid"
        quoted = f"'{unquoted}'"
        for cmd in (
            env._mcp_start_cmd("evil$(whoami)", "x"),
            env._mcp_stop_cmd("evil$(whoami)"),
        ):
            assert quoted in cmd
            # Every raw occurrence must be inside an already-quoted span.
            assert cmd.count(unquoted) == cmd.count(quoted)

    def test_launch_command_with_shell_metachars_is_quoted(self):
        """Same for the user's launch command: it's task-author-controlled,
        must land inside a single-quoted span once wrapped in `sh -c`."""
        env = _DummyEnv()
        evil_cmd = "python -c 'print(1)' && touch /pwned"
        quoted = f"'{evil_cmd}'".replace("'", "'\"'\"'")
        # shlex-quoted output contains the evil string only inside quotes.
        cmd = env._mcp_start_cmd("svc", evil_cmd)
        assert "setsid sh -c " in cmd
        # No unquoted `&& touch /pwned` outside a single-quoted span.
        assert cmd.count(evil_cmd) == 0 or quoted in cmd


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
    async def test_stop_issues_kill_and_clears_state(self):
        env = _DummyEnv(mcp_launch_commands={"svc": "python server.py"})
        state: dict[str, Any] = {"sandbox_id": "sbx"}
        await env.start_mcp_servers("sbx", _config_with_server(), state)
        assert "svc" in state["harbor_mcp_jobs"]
        env.sandbox_client.execute_command.reset_mock()

        await env.stop_mcp_servers(state)

        stop_calls = [
            c.args[1]
            for c in env.sandbox_client.execute_command.call_args_list
            if "kill -9" in c.args[1]
        ]
        assert len(stop_calls) == 1
        assert "harbor-mcp-svc.pid" in stop_calls[0]
        assert state["harbor_mcp_jobs"] == {}

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


class TestParallelStartup:
    """Harbor benchmarks can declare dozens of MCP servers; startup must fan
    out rather than serialize. These tests pin that behavior so a future
    regression back to a sequential loop gets caught."""

    @staticmethod
    def _parallel_cfg(names: list[str]) -> dict[str, Any]:
        return {
            "environment": {
                "mcp_servers": [
                    {
                        "name": n,
                        "transport": "streamable-http",
                        "url": f"http://{n}-host:{8000 + i}/mcp",
                    }
                    for i, n in enumerate(names)
                ]
            }
        }

    @pytest.mark.asyncio
    async def test_servers_start_concurrently_not_sequentially(self):
        """Three servers each settling for ~`SETTLE` seconds complete in
        ~`SETTLE` wall time under parallel startup, not ~3*SETTLE. The
        assertion is deliberately loose (2x) to survive CI noise while
        still catching a re-serialization regression."""
        SETTLE = 0.25
        env = _DummyEnv(mcp_launch_commands={"a": "cmd", "b": "cmd", "c": "cmd"})

        async def _exec(sandbox_id, command, **kwargs):
            if "/etc/hosts" in command:
                return MagicMock(exit_code=0, stdout="")
            await asyncio.sleep(SETTLE)
            return MagicMock(exit_code=0, stdout="")

        env.sandbox_client.execute_command = AsyncMock(side_effect=_exec)

        t0 = time.monotonic()
        await env.start_mcp_servers("sbx", self._parallel_cfg(["a", "b", "c"]), {})
        elapsed = time.monotonic() - t0

        assert elapsed < SETTLE * 2, (
            f"expected concurrent startup (~{SETTLE:.2f}s), got {elapsed:.2f}s "
            f"— start_mcp_servers has regressed to sequential"
        )

    @pytest.mark.asyncio
    async def test_all_servers_registered_in_state_under_parallel_startup(self):
        """Every framework-managed server lands in `state['harbor_mcp_jobs']`
        after parallel startup — dict mutations across tasks must not race."""
        names = [f"svc{i}" for i in range(8)]
        env = _DummyEnv(mcp_launch_commands={n: "cmd" for n in names})
        state: dict[str, Any] = {}
        await env.start_mcp_servers("sbx", self._parallel_cfg(names), state)

        assert set(state["harbor_mcp_jobs"].keys()) == set(names)
        assert env.sandbox_client.start_background_job.await_count == len(names)

    @pytest.mark.asyncio
    async def test_etc_hosts_patched_once_for_all_servers(self):
        """Even under parallel startup, `/etc/hosts` is patched a single time
        with the union of all framework-managed hostnames — not once per server
        (which would race on the file)."""
        env = _DummyEnv(mcp_launch_commands={"a": "cmd", "b": "cmd", "c": "cmd"})
        await env.start_mcp_servers("sbx", self._parallel_cfg(["a", "b", "c"]), {})

        etc_hosts_calls = [
            c.args[1]
            for c in env.sandbox_client.execute_command.call_args_list
            if "/etc/hosts" in c.args[1]
        ]
        assert len(etc_hosts_calls) == 1
        for host in ("a-host", "b-host", "c-host"):
            assert host in etc_hosts_calls[0]

    @pytest.mark.asyncio
    async def test_failure_cancels_in_flight_sibling_startups(self):
        """When one MCP fails healthcheck, pending siblings must be cancelled
        rather than burn through their retry budget on a rollout that's
        already dead. The cleanup path (`stop_mcp_servers`) is responsible
        for SIGKILLing any daemons that did make it up."""
        import verifiers as vf

        env = _DummyEnv(mcp_launch_commands={"bad": "cmd", "slow": "cmd"})
        cancelled = {"count": 0}

        async def _exec(sandbox_id, command, **kwargs):
            if "/etc/hosts" in command:
                return MagicMock(exit_code=0)
            # "bad" listens on port 8000 → hex 1F40. Fail its probe instantly
            # so gather surfaces its SandboxError while "slow" is still mid-sleep.
            if ":1F40" in command:
                return MagicMock(exit_code=1)
            try:
                await asyncio.sleep(2.0)
                return MagicMock(exit_code=0)
            except asyncio.CancelledError:
                cancelled["count"] += 1
                raise

        env.sandbox_client.execute_command = AsyncMock(side_effect=_exec)

        t0 = time.monotonic()
        with pytest.raises(vf.SandboxError):
            await env.start_mcp_servers("sbx", self._parallel_cfg(["bad", "slow"]), {})
        elapsed = time.monotonic() - t0

        assert cancelled["count"] == 1, (
            "slow sibling startup must be cancelled once bad fails"
        )
        # Sanity check: we bailed fast, not after `slow`'s 2s sleep.
        assert elapsed < 1.5, (
            f"fail-fast path took {elapsed:.2f}s — siblings were not cancelled"
        )


class TestBackgroundJob:
    @pytest.mark.asyncio
    async def test_launch_uses_background_job_and_stores_handle(self):
        """Daemon launch goes through start_background_job (not execute_command),
        and the returned handle is stashed on state for cleanup."""
        env = _DummyEnv(mcp_launch_commands={"svc": "python x"})
        state: dict[str, Any] = {}
        await env.start_mcp_servers("sbx", _config_with_server(), state)

        env.sandbox_client.start_background_job.assert_awaited_once()
        assert set(state["harbor_mcp_jobs"]) == {"svc"}
        assert state["harbor_mcp_jobs"]["svc"].job_id == "job-svc"

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

        # /etc/hosts patch (run before probing) must succeed; only the
        # healthcheck probe should fail.
        def _exec(sbx, command, **kwargs):
            if "/etc/hosts" in command:
                return MagicMock(exit_code=0, stdout="")
            return MagicMock(exit_code=1, stdout="")

        env.sandbox_client.execute_command = AsyncMock(side_effect=_exec)
        env.mcp_healthcheck = HarborMCPHealthcheck(
            retries=5, interval_sec=0.0, timeout_sec=1.0
        )

        with pytest.raises(vf.SandboxError, match="exited before becoming healthy"):
            await env.start_mcp_servers("sbx", _config_with_server(), {})
        # We should bail on the 1st poll, not burn the retry budget.
        assert env.sandbox_client.get_background_job.await_count == 1


class TestHealthCheck:
    """Readiness probing — default `/proc/net/tcp` + user override."""

    def test_default_probe_shape(self):
        """Portable awk on /proc/net/tcp{,6}, matching LISTEN state only,
        with no bash-ism dependency like /dev/tcp."""
        cmd = HarborMCPMixin._default_mcp_health_cmd(8000)
        assert "bash" not in cmd and "/dev/tcp" not in cmd
        assert "/proc/net/tcp" in cmd and "/proc/net/tcp6" in cmd
        assert '$4 == "0A"' in cmd  # LISTEN state

    @pytest.mark.parametrize(
        "port,hex_expected",
        [(80, "0050"), (8000, "1F40"), (65535, "FFFF"), (1, "0001")],
    )
    def test_default_probe_encodes_port_as_uppercase_hex(self, port, hex_expected):
        cmd = HarborMCPMixin._default_mcp_health_cmd(port)
        assert f":{hex_expected}$" in cmd

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

        # Only the probe should fail — keep the /etc/hosts patch succeeding.
        def _exec(sbx, command, **kwargs):
            if "/etc/hosts" in command:
                return MagicMock(exit_code=0)
            return MagicMock(exit_code=1)

        env.sandbox_client.execute_command = AsyncMock(side_effect=_exec)

        with pytest.raises(vf.SandboxError, match="after 2 consecutive retries"):
            await env.start_mcp_servers("sbx", _config_with_server(), {})
        assert self._probe_count(env) == 2


class TestEnvVarPublishing:
    @pytest.mark.asyncio
    async def test_framework_managed_url_is_rewritten_to_loopback(self):
        env = _DummyEnv(mcp_launch_commands={"one": "cmd1", "two": "cmd2"})
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
        assert await env.mcp_agent_env_vars(cfg, {}) == {
            "HARBOR_MCP_ONE_URL": "http://127.0.0.1:1/mcp",
            "HARBOR_MCP_TWO_URL": "http://127.0.0.1:2/mcp",
        }

    @pytest.mark.asyncio
    async def test_externally_managed_url_is_preserved_verbatim(self):
        """Remote managed endpoints in task.toml must reach the agent as-is —
        rewriting them to 127.0.0.1 would strand the agent on a closed port."""
        env = _DummyEnv()  # no launch command for "remote"
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "remote",
                        "transport": "streamable-http",
                        "url": "https://mcp.example.com/mcp",
                    }
                ]
            }
        }
        assert await env.mcp_agent_env_vars(cfg, {}) == {
            "HARBOR_MCP_REMOTE_URL": "https://mcp.example.com/mcp",
        }

    @pytest.mark.asyncio
    async def test_mix_of_managed_and_external_servers(self):
        env = _DummyEnv(mcp_launch_commands={"local": "cmd"})
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "local",
                        "transport": "streamable-http",
                        "url": "http://local-svc:9000/mcp",
                    },
                    {
                        "name": "remote",
                        "transport": "streamable-http",
                        "url": "https://mcp.example.com/mcp",
                    },
                ]
            }
        }
        assert await env.mcp_agent_env_vars(cfg, {}) == {
            "HARBOR_MCP_LOCAL_URL": "http://127.0.0.1:9000/mcp",
            "HARBOR_MCP_REMOTE_URL": "https://mcp.example.com/mcp",
        }

    @pytest.mark.asyncio
    async def test_stdio_servers_are_not_published(self):
        """Stdio servers don't have URLs — nothing to publish."""
        env = _DummyEnv()
        cfg = {
            "environment": {
                "mcp_servers": [
                    {"name": "stdio-srv", "transport": "stdio", "command": "x"}
                ]
            }
        }
        assert await env.mcp_agent_env_vars(cfg, {}) == {}

    @pytest.mark.asyncio
    async def test_server_name_is_normalized_to_upper_snake(self):
        env = _DummyEnv(mcp_launch_commands={"my-cool-server": "x"})
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
        assert "HARBOR_MCP_MY_COOL_SERVER_URL" in await env.mcp_agent_env_vars(cfg, {})


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
    async def test_does_not_patch_externally_managed_server_hosts(self):
        """Externally managed servers may be real remote endpoints — aliasing
        their hostname to 127.0.0.1 would make them unreachable."""
        env = _DummyEnv()  # no launch commands
        await env.start_mcp_servers("sbx", _config_with_server(), {})
        commands = [
            call.args[1] for call in env.sandbox_client.execute_command.call_args_list
        ]
        assert not any("/etc/hosts" in c for c in commands)

    @pytest.mark.asyncio
    async def test_patches_only_framework_managed_hosts_in_mixed_config(self):
        env = _DummyEnv(mcp_launch_commands={"local": "cmd"})
        cfg = {
            "environment": {
                "mcp_servers": [
                    {
                        "name": "local",
                        "transport": "streamable-http",
                        "url": "http://local-svc:9000/mcp",
                    },
                    {
                        "name": "remote",
                        "transport": "streamable-http",
                        "url": "https://mcp.example.com/mcp",
                    },
                ]
            }
        }
        await env.start_mcp_servers("sbx", cfg, {})
        etc_hosts_cmds = [
            call.args[1]
            for call in env.sandbox_client.execute_command.call_args_list
            if "/etc/hosts" in call.args[1]
        ]
        assert len(etc_hosts_cmds) == 1
        assert "local-svc" in etc_hosts_cmds[0]
        assert "mcp.example.com" not in etc_hosts_cmds[0]

    @pytest.mark.asyncio
    async def test_raises_setup_error_when_etc_hosts_write_fails(self):
        """Non-root image / read-only /etc makes `echo >> /etc/hosts` fail.
        We must surface that immediately, not let the daemon "come up healthy"
        on 127.0.0.1 while the agent has no way to resolve the hostname."""
        import verifiers as vf

        env = _DummyEnv(mcp_launch_commands={"svc": "cmd"})
        # /etc/hosts write fails with a classic Permission denied.
        env.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(
                exit_code=1, stderr="bash: /etc/hosts: Permission denied"
            )
        )

        with pytest.raises(vf.SandboxError, match="Failed to alias MCP hostnames"):
            await env.start_mcp_servers("sbx", _config_with_server(), {})
        # Critically, no daemon must be launched after a hosts-patch failure.
        env.sandbox_client.start_background_job.assert_not_awaited()
