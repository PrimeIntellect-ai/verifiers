"""Native Prime services exchange data over a LAN, then grade against live services
or a released, collected submission."""

import asyncio
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

# Solver service backends by trace id, to check their VMs are gone before grading.
SOLVERS: dict[str, list] = {}

IMAGE = "python:3.11-slim"


class ServiceHarness(vf.Harness):
    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        return await runtime.run(
            [
                "python",
                "-c",
                (
                    "import urllib.request; "
                    "url = open('/app/service-url').read(); "
                    "open('/app/answer', 'wb').write(urllib.request.urlopen(url, timeout=15).read())"
                ),
            ],
            {},
        )


class ServiceTask(HarborTask):
    def deployment_spec(self):
        return vf.ServicesSpec(
            {"sidecar": vf.PrimeConfig(image=IMAGE, allow=[], labels=["vf-ci"])}
        )

    async def setup(self, trace, runtime, deployment=None):
        await super().setup(runtime)
        if deployment is not None:
            SOLVERS[trace.id] = list(deployment.target_backends.values())
            assert len({backend.info.id for backend in SOLVERS[trace.id]}) == 2
            for name, target in (
                ("main", runtime),
                ("sidecar", deployment.target("sidecar")),
            ):
                await target.write("/srv/answer", b"42")
                await target.run_background(
                    [
                        "python",
                        "-m",
                        "http.server",
                        "8000",
                        "--bind",
                        "0.0.0.0",
                        "--directory",
                        "/srv",
                    ],
                    {"COLLECTION_SCOPE": name},
                    log="/tmp/service.log",
                )
                async with asyncio.timeout(60):
                    while (
                        await target.run(
                            [
                                "python",
                                "-c",
                                "import urllib.request; urllib.request.urlopen('http://localhost:8000/answer', timeout=2)",
                            ],
                            {},
                        )
                    ).exit_code:
                        await asyncio.sleep(1)
            await runtime.write(
                "/app/service-url",
                f"http://{deployment.hosts['sidecar']}:8000/answer".encode(),
            )
            # Finalize probes the reverse direction after both egress policies apply.
            await deployment.target("sidecar").write(
                "/app/service-url",
                f"http://{deployment.hosts['main']}:8000/answer".encode(),
            )
        else:
            # Fresh verifier setup happens only after all solver VMs are removed.
            assert all(backend.stopped for backend in SOLVERS[trace.id]), (
                "solver services survived into grading"
            )
            trace.info["solver_released_before_grading"] = True

    async def finalize(self, trace, runtime, sidecars=None, deployment=None):
        if deployment is not None:
            result = await deployment.target("sidecar").run(
                [
                    "python",
                    "-c",
                    "import urllib.request; assert urllib.request.urlopen(open('/app/service-url').read(), timeout=15).read() == b'42'",
                ],
                {},
            )
            assert result.exit_code == 0, result.stderr
        await super().finalize(trace, runtime, sidecars)

    async def finalize_services(self, trace, targets):
        assert targets["main"].backend.stopped, (
            "workspace must stop before sidecar collection"
        )
        assert not targets["sidecar"].backend.stopped
        trace.info["collected_sidecars"] = True
        await super().finalize_services(trace, targets)


@pytest.mark.parametrize(
    ("mode", "collect_fails", "scoring_timeout"),
    [
        ("shared", True, False),
        ("isolated", False, False),
        ("isolated", True, False),
        ("isolated", False, True),
    ],
    ids=[
        "shared-skips-sidecars",
        "isolated-restores-sidecars",
        "isolated-collect-failure",
        "isolated-scoring-timeout",
    ],
)
@pytest.mark.prime
@pytest.mark.skipif(not os.environ.get("PRIME_API_KEY"), reason="needs PRIME_API_KEY")
async def test_service_scoring(tmp_path, mode, collect_fails, scoring_timeout):
    task = service_task(tmp_path, collect_fails, scoring_timeout, mode)
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
        agent.harness = ServiceHarness(vf.HarnessConfig())
        trace = await agent.run(task)
    assert all(backend.stopped for backend in SOLVERS.pop(trace.id, []))
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


def service_task(tmp_path, collect_fails: bool, scoring_timeout: bool, mode: str):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "set -eu\n"
        'test "$(cat /app/answer)" = 42\n'
        'if [ "$EXPECT_SIDECAR" = 1 ]; then test "$(cat /evidence/result)" = 42; fi\n'
        + ("sleep 30\n" if scoring_timeout else "")
        + "echo 1 > /logs/verifier/reward.txt\n"
    )
    return ServiceTask(
        HarborData(
            prompt="Copy the sidecar answer",
            task_dir=str(tmp_path),
            image=IMAGE,
            network=vf.NetworkPolicyConfig(allow=[]),
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
                    else 'test "${COLLECTION_SCOPE-unset}" = unset && mkdir -p /evidence && cp /srv/answer /evidence/result',
                )
            ],
        )
    )
