"""Compose keeps live shared volumes and network namespaces inside a Prime VM."""

import asyncio
import os

import httpx
import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.tasksets.harbor.taskset import (
    HarborConfig,
    HarborTask,
    parse_task,
)

IMAGE = "public.ecr.aws/docker/library/python@sha256:b701198cb708b68d9413ca67d6311bd97c9389a2c5fe4da0511fc652642366cd"

COMPOSE = f"""\
services:
  main:
    build: .
    working_dir: /app
    network_mode: service:world
    volumes: [world_logs:/logs/world:ro, ../payload:/payload:ro]
    depends_on:
      world:
        condition: service_healthy
  world:
    image: {IMAGE}
    command: [sh, -c, "echo 42 > /logs/world/answer && exec python -m http.server 8080 --directory /logs/world"]
    volumes: [world_logs:/logs/world]
    ports: ["8000:8080"]
    healthcheck:
      test: [CMD, python, -c, "import urllib.request; urllib.request.urlopen('http://localhost:8080/answer', timeout=2)"]
      interval: 1s
      retries: 30
volumes:
  world_logs: {{}}
"""


class ComposeVMHarness(vf.Harness):
    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        return await runtime.run(
            [
                "sh",
                "-c",
                (
                    'test "$(cat /logs/world/answer)" = 43 && '
                    "! echo changed > /logs/world/answer && "
                    "python -c \"import urllib.request; open('/app/answer', 'wb').write(urllib.request.urlopen('http://localhost:8080/answer').read())\" && "
                    "touch /app/solver-only"
                ),
            ],
            {},
        )


class ComposeVMTask(HarborTask):
    async def setup(self, runtime, deployment=None):
        await super().setup(runtime, deployment)
        if deployment is not None:
            assert set(deployment.target_backends) == {"main", "world"}
            # A write after startup must be visible through main's read-only mount.
            await deployment.target("world").write("/logs/world/answer", b"43\n")
            # Context upload and binary transfers must fit on disk, not host /tmp.
            payload = await runtime.read("/payload")
            await runtime.write("/app/copied", payload)
            assert await runtime.read("/app/copied") == payload
            await runtime.write("/app/probe", b"workspace")
            process = await runtime.exec(
                ["python", "-m", "http.server", "8000", "--directory", "/app"], {}
            )
            try:
                # The sidecar already owns host port 8000; expose must translate.
                url = await runtime.expose(8000)
                async with httpx.AsyncClient(timeout=10) as client:
                    async with asyncio.timeout(60):
                        while True:
                            try:
                                response = await client.get(f"{url}/probe")
                                if response.status_code == 200:
                                    assert response.content == b"workspace"
                                    break
                            except httpx.TransportError:
                                pass
                            await asyncio.sleep(1)
            finally:
                await process.terminate()
                await process.wait()


@pytest.mark.parametrize("compose_mode", ["vm", "auto"])
@pytest.mark.parametrize("scoring_mode", ["shared", "isolated"])
@pytest.mark.prime
@pytest.mark.skipif(not os.environ.get("PRIME_API_KEY"), reason="needs PRIME_API_KEY")
async def test_compose_vm(tmp_path, scoring_mode, compose_mode, monkeypatch):
    async def no_local_docker(*args, **kwargs):
        raise AssertionError("VM mode must not invoke evaluator Docker")

    if compose_mode == "vm":
        monkeypatch.setattr("verifiers.v1.runtimes.compose.cli", no_local_docker)
    # Incompressible data exceeds the VM's small /tmp filesystem in the shared case.
    (tmp_path / "payload").write_bytes(
        os.urandom((72 if scoring_mode == "shared" else 1) * 1024**2)
    )
    environment = tmp_path / "environment"
    environment.mkdir()
    if scoring_mode == "isolated":
        (environment / "docker-compose.yaml").write_text("include: [services.yaml]\n")
        (environment / "services.yaml").write_text(COMPOSE)
    else:
        (environment / "docker-compose.yaml").write_text(COMPOSE)
    (environment / "Dockerfile").write_text(f"FROM {IMAGE}\nWORKDIR /app\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "set -eu\n"
        'test "$(cat /app/answer)" = 43\n'
        + (
            'test "$(cat /evidence/answer)" = 43\ntest ! -e /app/solver-only\n'
            if scoring_mode == "isolated"
            else ""
        )
        + "echo 1 > /logs/verifier/reward.txt\n"
    )
    (tmp_path / "instruction.md").write_text("Read the shared world logs")
    (tmp_path / "task.toml").write_text(
        'version = "1.0"\n'
        'artifacts = [{source = "/app/answer"}, {source = "/evidence/answer", service = "world"}]\n'
        "[[verifier.collect]]\n"
        'service = "world"\n'
        'command = "mkdir -p /evidence && cp /logs/world/answer /evidence/answer"\n'
    )
    task = ComposeVMTask(parse_task(tmp_path, 0, HarborConfig()))
    config = vf.AgentConfig(
        model="unused",
        runtime=vf.PrimeConfig(cpu=2, memory=4, disk=10, labels=["vf-ci"]),
        deployment=vf.DeploymentConfig(compose_mode=compose_mode),
        scoring=vf.ScoringConfig(
            mode=scoring_mode, retries=vf.RetryConfig(max_retries=0)
        ),
        timeout=TimeoutConfig(setup=1200, finalize=60, scoring=600),
    )
    async with vf.Agent(config) as agent:
        agent.harness = ComposeVMHarness(vf.HarnessConfig())
        trace = await agent.run(task)
    assert trace.ok and trace.reward == 1, trace.errors
    assert trace.info["compose"]["mode"] == "vm"
    if compose_mode == "auto":
        assert "share volume" in trace.info["compose"]["reason"]
    else:
        assert trace.info["compose"]["reason"] == "compose_mode=vm"
