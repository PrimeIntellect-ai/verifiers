"""Scoring runs under the egress its policy declares: a task with framework-only egress
probes two public hosts from its scoring hook, in the workspace (shared) and in a fresh
verifier (isolated), and a Harbor task's trusted grading lifts the policy unless scoring
declares its egress."""

import json
import shutil

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig
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
        return await runtime.run_program(["true"], {})


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
    ("egress", "reachable"),
    [(None, []), (["https://example.com"], ["example.com"]), (["*"], HOSTS)],
    ids=["undeclared", "one-url", "lifted"],
)
async def test_scoring_egress(mode, egress, reachable):
    task = ProbeTask(
        vf.TaskData(
            prompt="Do nothing",
            image=IMAGE,
            network_allow=[],
            scoring=vf.ScoringConfig(
                mode=mode, egress=egress, retries=vf.RetryConfig(max_retries=0)
            ),
        )
    )
    trace = await run(task)
    assert trace.ok, trace.errors
    assert trace.info["reachable"] == reachable


@pytest.mark.parametrize(
    ("egress", "reward"),
    [(None, 1.0), (["https://pypi.org"], 0.0)],
    ids=["trusted-default", "declared"],
)
async def test_harbor_grading_egress(tmp_path, egress, reward):
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
            network_allow=[],
            scoring=vf.ScoringConfig(egress=egress),
        )
    )
    trace = await run(task)
    assert trace.ok, trace.errors
    assert trace.reward == reward
