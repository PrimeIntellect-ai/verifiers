"""Runtime topology tests, including an optional local-Docker isolation proof."""

import asyncio
import subprocess
import uuid
from typing import Literal

import pytest

import verifiers.v1.runtimes.docker as docker_module
from verifiers.v1.interception.pool import InterceptionPool
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    DockerConfig,
    runtime_is_local,
    runtime_reaches_host_locally,
)
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.runtimes.docker import DockerRuntime


def result(stdout: str = "", stderr: str = "", exit_code: int = 0) -> ProgramResult:
    return ProgramResult(exit_code=exit_code, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize(
    ("network_access", "network"),
    [("full", "host"), ("interception", "bridge")],
)
async def test_docker_start_uses_setup_network(
    monkeypatch,
    network_access: Literal["full", "interception"],
    network: str,
) -> None:
    calls: list[tuple[str, ...]] = []

    async def fake_docker(*args: str) -> ProgramResult:
        calls.append(args)
        return result("29.0.0" if args[0] == "version" else "abcdef1234567890")

    monkeypatch.setattr(docker_module, "docker", fake_docker)
    runtime = DockerRuntime(DockerConfig(network_access=network_access), name="vf-test")
    await runtime.start()

    run = next(args for args in calls if args[0] == "run")
    assert run[run.index("--network") + 1] == network
    if network == "bridge":
        assert ("--cap-drop", "NET_ADMIN") == run[4:6]


def test_interception_docker_is_local_but_needs_a_host_tunnel() -> None:
    config = DockerConfig(network_access="interception")

    assert runtime_is_local(config)
    assert not runtime_reaches_host_locally(config)
    assert not InterceptionPool(config, multiplex=1).host_is_local


async def test_docker_seals_agent_behind_fixed_interception_relay(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    async def fake_docker(*args: str) -> ProgramResult:
        calls.append(args)
        return result("ok")

    monkeypatch.setattr(docker_module, "docker", fake_docker)
    runtime = DockerRuntime(DockerConfig(network_access="interception"), name="vf-test")
    runtime._container = runtime.name

    endpoint = await runtime.seal_agent_network(
        "https://rollout.tunnel.example/v1?dialect=responses"
    )

    assert endpoint == "http://vf-interception:8080/v1?dialect=responses"
    assert calls[0][:4] == ("network", "create", "--internal", "--opt")
    relay = next(args for args in calls if args[0] == "create")
    config = relay[-1]
    assert "nginx:1.28.3-alpine" in relay
    assert "proxy_pass https://rollout.tunnel.example;" in config
    assert "proxy_set_header Host rollout.tunnel.example;" in config
    assert "proxy_ssl_verify on;" in config
    assert (
        "network",
        "connect",
        "vf-test-network",
        "vf-test",
    ) in calls
    assert ("network", "disconnect", "bridge", "vf-test") in calls


async def test_docker_rejects_non_http_interception_endpoint(monkeypatch) -> None:
    async def unexpected_docker(*args: str) -> ProgramResult:
        raise AssertionError(f"unexpected docker call: {args}")

    monkeypatch.setattr(docker_module, "docker", unexpected_docker)
    runtime = DockerRuntime(DockerConfig(network_access="interception"), name="vf-test")
    runtime._container = runtime.name

    with pytest.raises(SandboxError, match="invalid interception endpoint"):
        await runtime.seal_agent_network("file:///tmp/interception.sock")


def test_docker_cleanup_removes_relay_container_and_network(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(docker_module.subprocess, "run", fake_run)
    runtime = DockerRuntime(DockerConfig(network_access="interception"), name="vf-test")
    runtime._container = "vf-test"
    runtime._relay = "vf-test-relay"
    runtime._network = "vf-test-network"

    runtime.cleanup()
    runtime.cleanup()

    assert calls == [
        ["docker", "rm", "--force", "vf-test-relay"],
        ["docker", "rm", "--force", "vf-test"],
        ["docker", "network", "rm", "vf-test-network"],
    ]


@pytest.mark.integration
@pytest.mark.docker
async def test_interception_network_allows_relay_and_blocks_direct_egress() -> None:
    version = await docker_module.docker("version", "--format", "{{.Server.Version}}")
    if version.exit_code != 0:
        pytest.skip("Docker daemon is not available")

    suffix = uuid.uuid4().hex[:12]
    upstream = f"vf-upstream-{suffix}"
    runtime = DockerRuntime(
        DockerConfig(network_access="interception"), name=f"vf-agent-{suffix}"
    )
    server = await docker_module.docker(
        "run",
        "--detach",
        "--network",
        "bridge",
        "--name",
        upstream,
        "python:3.11-slim",
        "python",
        "-m",
        "http.server",
        "8000",
    )
    if server.exit_code != 0:
        pytest.fail(f"test upstream failed to start: {server.stderr}")
    try:
        for _ in range(20):
            ready = await docker_module.docker(
                "exec",
                upstream,
                "python",
                "-c",
                "import socket; socket.create_connection(('127.0.0.1', 8000), 1)",
            )
            if ready.exit_code == 0:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("test upstream did not become ready")
        await runtime.start()
        inspected = await docker_module.docker(
            "inspect",
            "--format",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            upstream,
        )
        upstream_ip = inspected.stdout.strip()
        assert upstream_ip

        endpoint = await runtime.seal_agent_network(f"http://{upstream_ip}:8000/")
        allowed = await runtime.run(
            [
                "python",
                "-c",
                "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=3).read()",
                endpoint,
            ],
            {},
        )
        direct = await runtime.run(
            [
                "python",
                "-c",
                "import socket, sys; socket.create_connection((sys.argv[1], 8000), 1)",
                upstream_ip,
            ],
            {},
        )
        internet = await runtime.run(
            [
                "python",
                "-c",
                "import socket; socket.create_connection(('1.1.1.1', 80), 1)",
            ],
            {},
        )

        assert allowed.exit_code == 0, allowed.stderr
        assert direct.exit_code != 0
        assert internet.exit_code != 0
    finally:
        await runtime.stop()
        await docker_module.docker("rm", "--force", upstream)
