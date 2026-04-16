from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from verifiers.envs.experimental.harbor_env import (
    DEFAULT_PHASE,
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


class _DummyEnv(HarborMCPMixin):
    """Bare mixin host for unit testing; bypasses CliAgentEnv setup."""

    def __init__(self, *, mcp_launchers: dict[str, HarborMCPLauncher] | None = None):
        self.sandbox_client = MagicMock()
        self.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(exit_code=0, stdout="")
        )
        self.mcp_launchers = mcp_launchers or {}
        self.mcp_health_check_retries = 1
        self.mcp_health_check_interval = 0.0


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
        assert "nohup .venv/bin/python -u server.py" in cmd

    def test_no_su_without_user(self):
        env = _DummyEnv()
        server = HarborMCPServer(
            name="svc", transport="streamable-http", url="http://svc:8000/mcp"
        )
        launcher = HarborMCPLauncher(command="server")
        assert "su -s" not in env._mcp_start_cmd(server, launcher, 8000, {})

    def test_extra_env_overrides_launcher_env(self):
        env = _DummyEnv()
        server = HarborMCPServer(
            name="svc", transport="streamable-http", url="http://svc:8000/mcp"
        )
        launcher = HarborMCPLauncher(command="x", env={"K": "old"})
        cmd = env._mcp_start_cmd(server, launcher, 8000, {"K": "new"})
        assert "K=new" in cmd
        assert "K=old" not in cmd


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

        await env.restart_mcp_for_phase("sbx", state, phase="verifier")

        # Server should not be stopped/started again.
        commands = [
            c.args[1] for c in env.sandbox_client.execute_command.call_args_list
        ]
        assert all("nohup" not in c for c in commands)
        assert all(
            "harbor-mcp-svc.pid" not in c or "if [ -f" not in c for c in commands
        )
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
