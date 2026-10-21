"""Compose scoring uses live shared services or a released, collected submission, on
a Prime VM."""

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

# Solver service backends by trace id, to check their VM is gone before grading.
SOLVERS: dict[str, list] = {}

IMAGE = "python:3.11-slim"
COMPOSE = f"""
services:
  main:
    image: {IMAGE}
    working_dir: /app
    command: [sleep, infinity]
    depends_on:
      sidecar:
        condition: service_healthy
  sidecar:
    environment:
      COLLECTION_SCOPE: sidecar
    image: {IMAGE}
    command: [sh, -c, "mkdir -p /srv /evidence; echo 42 > /srv/answer; cd /srv; exec python -m http.server 8000"]
    healthcheck:
      test: [CMD, python, -c, "import urllib.request; urllib.request.urlopen('http://localhost:8000/answer')"]
      interval: 1s
      timeout: 2s
      retries: 30
"""


class ComposeHarness(vf.Harness):
    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        if "build-main" in data.tags:
            result = await runtime.run(["test", "-f", "/built-from-dockerfile"], {})
            assert result.exit_code == 0, "workspace did not use its Dockerfile"
        return await runtime.run(
            [
                "python",
                "-c",
                (
                    "import urllib.request; "
                    "open('/app/answer', 'wb').write(urllib.request.urlopen('http://sidecar:8000/answer').read())"
                ),
            ],
            {},
        )


class ComposeTask(HarborTask):
    async def setup(self, trace, runtime, deployment=None):
        await super().setup(runtime)
        if deployment is not None:
            SOLVERS[trace.id] = list(deployment.target_backends.values())
        else:
            # Fresh verifier setup happens only after the solver VM is removed.
            assert all(
                backend._host_backend.stopped for backend in SOLVERS[trace.id]
            ), "solver services survived into grading"
            if "build-main" in self.data.tags:
                result = await runtime.run(
                    [
                        "sh",
                        "-c",
                        "test -f /built-from-dockerfile && test ! -e /app/answer",
                    ],
                    {},
                )
                assert result.exit_code == 0, (
                    "verifier was not rebuilt from clean inputs"
                )
                trace.info["verifier_rebuilt_from_dockerfile"] = True
            trace.info["solver_released_before_grading"] = True

    async def finalize_services(self, trace, targets):
        main = targets["main"].backend
        result = await main._run_host(
            "docker", "inspect", "--format", "{{.State.Running}}", main._container
        )
        assert result.stdout.strip() == "false", (
            "workspace must stop before sidecar collection"
        )
        trace.info["collected_sidecars"] = True
        await super().finalize_services(trace, targets)


@pytest.mark.parametrize(
    ("mode", "collect_fails", "scoring_timeout", "build_main"),
    [
        ("shared", True, False, False),
        ("isolated", False, False, False),
        ("isolated", True, False, False),
        ("isolated", False, True, False),
        ("shared", True, False, True),
        ("isolated", False, False, True),
    ],
    ids=[
        "shared-skips-sidecars",
        "isolated-restores-sidecars",
        "isolated-collect-failure",
        "isolated-scoring-timeout",
        "shared-builds-main",
        "isolated-builds-main",
    ],
)
@pytest.mark.prime
@pytest.mark.skipif(not os.environ.get("PRIME_API_KEY"), reason="needs PRIME_API_KEY")
async def test_compose_scoring(
    tmp_path, mode, collect_fails, scoring_timeout, build_main
):
    task = compose_task(tmp_path, collect_fails, scoring_timeout, mode, build_main)
    config = vf.AgentConfig(
        model="unused",
        runtime=vf.PrimeConfig(labels=["vf-ci"]),
        scoring=vf.ScoringConfig(
            mode=mode,
            env={"EXPECT_SIDECAR": "1" if mode == "isolated" else "0"},
            retries=vf.RetryConfig(max_retries=0),
        ),
        timeout=TimeoutConfig(
            setup=600, finalize=60, scoring=1 if scoring_timeout else 120
        ),
    )
    async with vf.Agent(config) as agent:
        agent.harness = ComposeHarness(vf.HarnessConfig())
        trace = await agent.run(task)
    assert all(backend._host_backend.stopped for backend in SOLVERS.pop(trace.id, []))
    if mode == "isolated" and collect_fails:
        assert not trace.ok
        assert "collect hook failed" in str(trace.errors)
        assert "solver_released_before_grading" not in trace.info
    elif scoring_timeout:
        assert trace.ok and trace.is_timeout and not trace.errors
        assert trace.stop_condition == "scoring_timeout"
        assert all(reward is None for reward in trace.rewards.values())
    else:
        assert trace.ok and trace.reward == 1, trace.errors
        assert bool(trace.info.get("collected_sidecars")) == (mode == "isolated")
        assert bool(trace.info.get("solver_released_before_grading")) == (
            mode == "isolated"
        )
        assert bool(trace.info.get("verifier_rebuilt_from_dockerfile")) == (
            mode == "isolated" and build_main
        )


def compose_task(
    tmp_path, collect_fails: bool, scoring_timeout: bool, mode: str, build_main: bool
):
    environment = tmp_path / "environment"
    environment.mkdir()
    compose = COMPOSE
    if build_main:
        compose = compose.replace(f"    image: {IMAGE}\n", "", 1)
        (environment / "Dockerfile").write_text(
            f"FROM {IMAGE}\nRUN touch /built-from-dockerfile\n"
        )
    (environment / "docker-compose.yaml").write_text(compose)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "set -eu\n"
        'test "$(cat /app/answer)" = 42\n'
        'if [ "$EXPECT_SIDECAR" = 1 ]; then test "$(cat /evidence/result)" = 42; fi\n'
        + ("sleep 30\n" if scoring_timeout else "")
        + "echo 1 > /logs/verifier/reward.txt\n"
    )
    return ComposeTask(
        HarborData(
            prompt="Copy the sidecar answer",
            tags=["build-main"] if build_main else [],
            task_dir=str(tmp_path),
            env={"COLLECTION_SCOPE": "workspace"},
            # Exercise agent overrides in both directions, not just task defaults.
            scoring=vf.ScoringConfig(mode="isolated" if mode == "shared" else "shared"),
            artifacts=[
                HarborArtifact(source="/app/answer"),
                HarborArtifact(source="/evidence/result", service="sidecar"),
            ],
            collect=[
                CollectHook(
                    service="sidecar",
                    command="exit 17"
                    if collect_fails
                    else 'test "$COLLECTION_SCOPE" = sidecar && cp /srv/answer /evidence/result',
                )
            ],
        )
    )
