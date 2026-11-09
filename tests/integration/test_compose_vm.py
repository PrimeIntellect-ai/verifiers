"""Compose keeps live shared volumes and network namespaces inside a Prime VM."""

import os

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.tasksets.harbor.taskset import (
    CollectHook,
    HarborArtifact,
    HarborData,
    HarborTask,
)

COMPOSE = """\
services:
  main:
    build: .
    working_dir: /app
    network_mode: service:world
    volumes: [world_logs:/logs/world:ro]
    depends_on:
      world:
        condition: service_healthy
  world:
    image: python:3.11-slim
    command: [sh, -c, "echo 42 > /logs/world/answer && exec python -m http.server 8080 --directory /logs/world"]
    volumes: [world_logs:/logs/world]
    healthcheck:
      test: [CMD, python, -c, "import urllib.request; urllib.request.urlopen('http://localhost:8080/answer', timeout=2)"]
      interval: 1s
      retries: 30
volumes:
  world_logs: {}
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


@pytest.mark.parametrize("compose_mode", ["auto", "vm"])
@pytest.mark.parametrize("scoring_mode", ["shared", "isolated"])
@pytest.mark.prime
@pytest.mark.skipif(not os.environ.get("PRIME_API_KEY"), reason="needs PRIME_API_KEY")
async def test_compose_vm(tmp_path, compose_mode, scoring_mode):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "docker-compose.yaml").write_text(COMPOSE)
    (environment / "Dockerfile").write_text("FROM python:3.11-slim\nWORKDIR /app\n")
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
    task = ComposeVMTask(
        HarborData(
            prompt="Read the shared world logs",
            task_dir=str(tmp_path),
            artifacts=[
                HarborArtifact(source="/app/answer"),
                HarborArtifact(source="/evidence/answer", service="world"),
            ],
            collect=[
                CollectHook(
                    service="world",
                    command="mkdir -p /evidence && cp /logs/world/answer /evidence/answer",
                )
            ],
        )
    )
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
