"""Dockerfile graders rebuild inputs and retain image defaults under isolated policy."""

import os
import shutil

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.tasksets.harbor.taskset import (
    HarborArtifact,
    HarborData,
    HarborTask,
    VerifierConfig,
)

IMAGE = "python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60"


class SubmitHarness(vf.Harness):
    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        return await runtime.run(["sh", "-c", "echo 42 > answer"], {})


@pytest.mark.parametrize(
    "runtime",
    [
        pytest.param(vf.DockerConfig(), marks=pytest.mark.docker, id="docker"),
        pytest.param(
            vf.PrimeConfig(labels=["vf-ci"]), marks=pytest.mark.prime, id="prime"
        ),
    ],
)
async def test_verifier_build(tmp_path, runtime):
    if not shutil.which("docker"):
        pytest.skip("needs Docker for image preparation")
    if runtime.type == "prime" and not os.environ.get("PRIME_API_KEY"):
        pytest.skip("needs PRIME_API_KEY")
    tests = tmp_path / "tests"
    tests.mkdir()
    # Only the image's bundled suite should run, not the task's staging copy.
    (tests / "test.sh").write_text("exit 41\n")
    (tests / "baked.sh").write_text(
        "set -eu\n"
        'test "$(id -u)" = 10001\n'
        'test "$(id -g)" = "$EXPECTED_GID"\n'
        'test "$(pwd)" = /app\n'
        "test \"$BUILD_TOKEN\" = 'a$b'\n"
        'test "$(cat /tests/marker)" = "$EXPECTED"\n'
        'test "$(cat answer)" = 42\n'
        "python - <<'PY'\n"
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('https://example.com', timeout=5)\n"
        "except Exception:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('grader bypassed network policy')\n"
        "PY\n"
        "echo 1 > /logs/verifier/reward.txt\n"
    )
    dockerfile = (
        f"FROM {IMAGE}\n"
        "RUN groupadd -g 10001 grader && groupadd -g 10002 verifier "
        "&& useradd -u 10001 -g grader -m grader "
        "&& mkdir /app /tests && chown grader /tests "
        "&& echo stale > /app/answer\n"
        "COPY baked.sh /tests/test.sh\n"
        "COPY marker /tests/marker\n"
        "ENV BUILD_TOKEN=a\\$b\n"
        'ENTRYPOINT ["sh", "-c", "exit 42"]\n'
    )
    config = vf.AgentConfig(
        model="unused",
        runtime=runtime,
        scoring=vf.ScoringConfig(
            mode="isolated",
            network=vf.NetworkPolicyConfig(allow=[]),
            retries=vf.RetryConfig(max_retries=0),
        ),
        timeout=TimeoutConfig(setup=1200, finalize=60, scoring=1200),
    )
    for marker in ("first", "changed"):
        (tests / "marker").write_text(marker)
        user = "grader:verifier" if marker == "changed" else "10001"
        gid = "10002" if marker == "changed" else "10001"
        # Missing WORKDIR uses the standalone /app default; an image declaration
        # retains it on the rebuilt grader, with the same relative artifact.
        (tests / "Dockerfile").write_text(
            dockerfile
            + f"USER {user}\n"
            + ("WORKDIR /app\n" if marker == "changed" else "")
        )
        task = HarborTask(
            HarborData(
                prompt="Submit the answer",
                task_dir=str(tmp_path),
                image=IMAGE,
                verifier=VerifierConfig(),
                verifier_build=str(tests),
                verifier_env={"EXPECTED": marker, "EXPECTED_GID": gid},
                artifacts=[HarborArtifact(source="answer")],
            )
        )
        async with vf.Agent(config) as agent:
            agent.harness = SubmitHarness(vf.HarnessConfig())
            trace = await agent.run(task)
        assert trace.ok and trace.reward == 1, trace.errors
