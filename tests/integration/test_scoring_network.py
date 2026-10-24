"""Scoring runs under the network its policy declares: a task with framework-only egress
probes two public hosts from its scoring hook, in the workspace (shared) and in a fresh
verifier (isolated), in rollouts and in vf-validate, and a Harbor task's trusted grading
lifts the policy unless scoring declares its network."""

import json
import shutil

import pytest

import verifiers.v1 as vf
from verifiers.v1.cli.validate import _run_check
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.configs.cli.validate import ValidateConfig
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not shutil.which("docker"), reason="needs Docker"),
]

IMAGE = "python:3.11-slim"
HOSTS = ["example.com", "pypi.org"]
# Prints which hosts answer 200. Requests go through the runtime's egress proxy, which
# refuses a host the policy denies.
PROBE = f"""
import json, urllib.request
reachable = []
for host in {HOSTS!r}:
    try:
        with urllib.request.urlopen("https://" + host, timeout=15) as response:
            if response.status == 200:
                reachable.append(host)
    except Exception:
        pass
print(json.dumps(reachable))
"""


class IdleHarness(vf.Harness):
    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        return await runtime.run(["true"], {})


class ProbeTask(vf.Task):
    @vf.reward(weight=1.0)
    async def probe(self, trace, runtime) -> float:
        result = await runtime.run(["python", "-c", PROBE], {})
        trace.info["reachable"] = json.loads(result.stdout)
        return 1.0


async def run(task) -> vf.Trace:
    config = vf.AgentConfig(
        model="unused",
        runtime=vf.DockerConfig(),
        timeout=TimeoutConfig(setup=180, finalize=30, scoring=120),
    )
    async with vf.Agent(config) as agent:
        agent.harness = IdleHarness(vf.HarnessConfig())
        return await agent.run(task)


@pytest.mark.parametrize("mode", ["shared", "isolated"])
@pytest.mark.parametrize(
    ("network", "reachable"),
    [
        (None, []),
        (vf.NetworkPolicyConfig(allow=["example.com"]), ["example.com"]),
        (vf.NetworkPolicyConfig(), HOSTS),
    ],
    ids=["undeclared", "one-host", "open"],
)
async def test_scoring_network(mode, network, reachable):
    task = ProbeTask(
        vf.TaskData(
            prompt="Do nothing",
            image=IMAGE,
            network=vf.NetworkPolicyConfig(allow=[]),
            scoring=vf.ScoringConfig(
                mode=mode, network=network, retries=vf.RetryConfig(max_retries=0)
            ),
        )
    )
    trace = await run(task)
    assert trace.ok, trace.errors
    assert trace.info["reachable"] == reachable


class ReachTask(vf.Task):
    @vf.reward(weight=1.0)
    async def reach(self, trace, runtime) -> float:
        result = await runtime.run(["python", "-c", PROBE], {})
        return float(json.loads(result.stdout) == ["example.com"])


@pytest.mark.parametrize("mode", ["shared", "isolated"])
async def test_validate_scores_under_the_declared_network(mode):
    task = ReachTask(
        vf.TaskData(
            prompt="Do nothing",
            image=IMAGE,
            network=vf.NetworkPolicyConfig(allow=[]),
            scoring=vf.ScoringConfig(
                mode=mode,
                network=vf.NetworkPolicyConfig(allow=["example.com"]),
                retries=vf.RetryConfig(max_retries=0),
            ),
        )
    )
    row = await _run_check(task, ValidateConfig(runtime=vf.DockerConfig()), "noop")
    # vf-validate never applies the agent phase's policy, so the shared workspace is open
    # until scoring declares a network. Scoring that reaches only example.com passes the
    # untouched task, which the noop check reports as invalid.
    assert row["error"] is None
    assert row["valid"] is False


class GoldReachTask(ReachTask):
    async def validate(self, trace, runtime) -> bool:
        await vf.grade_submission(self, trace, runtime)
        return trace.reward == 1.0


async def test_validate_grades_a_submission_under_the_declared_network():
    task = GoldReachTask(
        vf.TaskData(
            prompt="Do nothing",
            image=IMAGE,
            network=vf.NetworkPolicyConfig(allow=[]),
            scoring=vf.ScoringConfig(
                mode="isolated",
                network=vf.NetworkPolicyConfig(allow=["example.com"]),
                retries=vf.RetryConfig(max_retries=0),
            ),
        )
    )
    # vf-validate's trace records its own runtime config, so the verifier can reach
    # example.com although the task's policy is framework-only.
    row = await _run_check(task, ValidateConfig(runtime=vf.DockerConfig()), "gold")
    assert row["error"] is None
    assert row["valid"] is True


@pytest.mark.parametrize(
    ("network", "reward"),
    [(None, 1.0), (vf.NetworkPolicyConfig(allow=["pypi.org"]), 0.0)],
    ids=["trusted-default", "declared"],
)
async def test_harbor_grading_network(tmp_path, network, reward):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "python - <<'EOF'\n"
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('https://example.com', timeout=15)\n"
        "    reward = 1\n"
        "except Exception:\n"
        "    reward = 0\n"
        "open('/logs/verifier/reward.txt', 'w').write(str(reward))\n"
        "EOF\n"
    )
    task = HarborTask(
        HarborData(
            prompt="Do nothing",
            task_dir=str(tmp_path),
            image=IMAGE,
            network=vf.NetworkPolicyConfig(allow=[]),
            scoring=vf.ScoringConfig(network=network),
        )
    )
    trace = await run(task)
    assert trace.ok, trace.errors
    assert trace.reward == reward
