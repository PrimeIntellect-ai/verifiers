"""Scoring's declared network: how agent and task policies combine, how it combines with
the runtime's own rules (in `grade_submission`, the agent's), and what the runtime asks
its provider for."""

from typing import cast

import pytest

import verifiers.v1 as vf
from verifiers.v1 import grading
from verifiers.v1.configs.scoring import resolve_scoring
from verifiers.v1.errors import SandboxError, TaskError
from verifiers.v1.grading import grade_submission, scoring_network
from verifiers.v1.runtimes import deployment
from verifiers.v1.runtimes.base import RuntimeBackend
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.compile import resolve_runtime_config

PYPI = vf.NetworkPolicyConfig(allow=["pypi.org"])


def test_agent_network_replaces_the_tasks():
    task = vf.ScoringConfig(network=PYPI)
    assert resolve_scoring(task, vf.ScoringConfig()).network == PYPI
    agent = vf.ScoringConfig(network=vf.NetworkPolicyConfig())
    assert resolve_scoring(task, agent).network == vf.NetworkPolicyConfig()


def test_scoring_network_combines_with_the_runtimes_rules():
    assert scoring_network(vf.DockerConfig(), PYPI).allow == ["pypi.org"]
    narrowed = vf.DockerConfig(allow=["pypi.org", "example.com"])
    opened = scoring_network(narrowed, vf.NetworkPolicyConfig())
    assert opened.allow == ["pypi.org", "example.com"]
    # As for `TaskData.network`, a concrete allowlist can't sit beside a blocklist.
    with pytest.raises(ValueError):
        scoring_network(vf.DockerConfig(block=["example.com"]), PYPI)


def test_scoring_network_needs_a_runtime_with_policies():
    with pytest.raises(TaskError):
        scoring_network(vf.SubprocessConfig(), PYPI)
    open_network = vf.NetworkPolicyConfig()
    assert scoring_network(vf.SubprocessConfig(), open_network) is open_network


class Backend:
    SUPPORTS_USER_SWITCHING = False
    name = "fake"

    def __init__(self, config, restricted=True):
        self.config = config
        self.network_restricted = restricted
        self.calls = []

    async def prepare_execution(self, routes, network=None):
        self.calls.append((routes, None if network is None else network.allow))


async def test_set_network_enforces_a_policy_or_lifts_it():
    backend = Backend(vf.DockerConfig(allow=[]))
    runtime = Runtime(cast(RuntimeBackend, backend))
    await runtime.set_network(PYPI)
    await runtime.set_network(vf.NetworkPolicyConfig())
    assert backend.calls == [([], ["pypi.org"]), (None, None)]


async def test_set_network_cannot_restrict_a_runtime_that_started_open():
    backend = Backend(vf.DockerConfig(), restricted=False)
    runtime = Runtime(cast(RuntimeBackend, backend))
    with pytest.raises(SandboxError):
        await runtime.set_network(PYPI)
    await runtime.set_network(vf.NetworkPolicyConfig())
    assert backend.calls == [(None, None)]


async def test_grade_submission_widens_from_the_agents_runtime(monkeypatch):
    graded = []

    async def grade_task(task, trace, config, *args, **kwargs):
        graded.append(config)

    class Deployment:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(grading, "grade_task", grade_task)
    monkeypatch.setattr(deployment, "Deployment", Deployment)
    task = vf.Task(
        vf.TaskData(
            prompt="",
            image="python:3.11-slim",
            network=vf.NetworkPolicyConfig(allow=[]),
            scoring=vf.ScoringConfig(mode="isolated", network=PYPI),
        )
    )
    agent_runtime = vf.DockerConfig()
    # The solver's resolved config carries the task's framework-only policy.
    solver = Runtime(
        cast(RuntimeBackend, Backend(resolve_runtime_config(agent_runtime, task)))
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig(runtime=agent_runtime)),
        task=vf.TraceTask(type="Task", data=task.data),
    )
    await grade_submission(task, trace, solver)
    [config] = graded
    assert config.allow == ["pypi.org"]
