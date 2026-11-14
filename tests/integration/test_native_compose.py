"""A task's Compose file runs on Prime as one native VM per service: health-gated
startup, task-file binds, names and LAN rules from Compose networks, an offline
internal service, and grading in place or in a fresh verifier."""

import os

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.tasksets.harbor.taskset import HarborArtifact, HarborData, HarborTask

# Pinned so the imported VM image stays cached across CI runs.
IMAGE = "python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60"

COMPOSE = f"""\
services:
  main:
    networks: [front, back]
    depends_on:
      web:
        condition: service_healthy
  web:
    image: {IMAGE}
    command: [python, -m, http.server, "8000", --directory, /srv]
    volumes: [./web:/srv:ro]
    networks: [front]
    healthcheck:
      test: [CMD, python, -c, "import urllib.request; urllib.request.urlopen('http://localhost:8000/answer', timeout=2)"]
      interval: 2s
      retries: 30
  vault:
    image: {IMAGE}
    command: [sh, -c, "mkdir -p /vault && echo s3cret > /vault/secret && exec python -m http.server 8000 --directory /vault"]
    networks: [back]
networks:
  front: {{}}
  back:
    internal: true
"""


def fetch(url: str, timeout: int = 15) -> list[str]:
    return [
        "python",
        "-c",
        f"import urllib.request; print(urllib.request.urlopen({url!r}, timeout={timeout}).read().decode().strip())",
    ]


class ComposeHarness(vf.Harness):
    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        return await runtime.run(
            [
                "python",
                "-c",
                (
                    "import os, urllib.request\n"
                    "os.makedirs('/app', exist_ok=True)\n"
                    "read = lambda url: urllib.request.urlopen(url, timeout=15).read().decode().strip()\n"
                    "open('/app/answer', 'w').write(read('http://web:8000/answer') + ' ' + read('http://vault:8000/secret'))\n"
                    "open('/app/literal', 'w').write(os.environ['LITERAL'])\n"
                ),
            ],
            {},
        )


class ComposeTask(HarborTask):
    async def setup(self, trace, runtime, deployment=None):
        await super().setup(runtime, deployment)
        if deployment is None:
            return
        backends = deployment.target_backends
        assert sorted(backends) == ["main", "vault", "web"]
        assert len({backend.info.id for backend in backends.values()}) == 3
        web, vault = deployment.target("web"), deployment.target("vault")
        # The bound task files are read-only, as declared.
        assert (await web.run(["sh", "-c", "echo x > /srv/answer"], {})).exit_code
        # Names resolve only across a shared network, and LAN rules follow the same lines.
        assert (await web.run(fetch("http://vault:8000/secret", 5), {})).exit_code
        lan = f"http://{deployment.hosts['vault']}:8000/secret"
        assert (await web.run(fetch(lan, 5), {})).exit_code
        # A service only on an internal network has no internet, from the start.
        assert (await vault.run(fetch("https://example.com", 5), {})).exit_code
        trace.info["isolation_checked"] = True


@pytest.mark.parametrize("mode", ["shared", "isolated"])
@pytest.mark.prime
@pytest.mark.skipif(not os.environ.get("PRIME_API_KEY"), reason="needs PRIME_API_KEY")
async def test_native_compose(tmp_path, mode):
    environment = tmp_path / "environment"
    (environment / "web").mkdir(parents=True)
    (environment / "web" / "answer").write_text("42\n")
    (environment / "docker-compose.yaml").write_text(COMPOSE)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "set -eu\n"
        'test "$(cat /app/answer)" = "42 s3cret"\n'
        "test \"$(cat /app/literal)\" = 'a$b'\n"
        "echo 1 > /logs/verifier/reward.txt\n"
    )
    task = ComposeTask(
        HarborData(
            prompt="Combine the web answer and the vault secret",
            task_dir=str(tmp_path),
            image=IMAGE,
            # Compose interpolates `$`; task values must arrive verbatim.
            env={"LITERAL": "a$b"},
            artifacts=[
                HarborArtifact(source="/app/answer"),
                HarborArtifact(source="/app/literal"),
            ],
        )
    )
    config = vf.AgentConfig(
        model="unused",
        runtime=vf.PrimeConfig(labels=["vf-ci"]),
        scoring=vf.ScoringConfig(mode=mode, retries=vf.RetryConfig(max_retries=0)),
        # A cold run builds and imports the VM image during setup.
        timeout=TimeoutConfig(setup=1200, finalize=60, scoring=300),
    )
    async with vf.Agent(config) as agent:
        agent.harness = ComposeHarness(vf.HarnessConfig())
        trace = await agent.run(task)
    assert trace.ok and trace.reward == 1, trace.errors
    assert trace.info.get("isolation_checked")
