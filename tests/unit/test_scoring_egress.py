"""Scoring's declared egress: what it accepts, how agent and task policies combine, and
what the runtime asks its provider for."""

from typing import cast

import pytest
from pydantic import ValidationError

import verifiers.v1 as vf
from verifiers.v1.configs.scoring import resolve_scoring
from verifiers.v1.runtimes.base import RuntimeBackend
from verifiers.v1.runtimes.runtime import Runtime


@pytest.mark.parametrize(
    "egress", [None, [], ["*"], ["https://pypi.org", "http://10.0.0.5:8080/path"]]
)
def test_accepts_urls_or_a_lone_wildcard(egress):
    assert vf.ScoringConfig(egress=egress).egress == egress


@pytest.mark.parametrize(
    "egress", [["pypi.org"], ["https://"], ["*", "https://pypi.org"]]
)
def test_rejects_hosts_without_a_scheme_and_mixed_wildcards(egress):
    with pytest.raises(ValidationError):
        vf.ScoringConfig(egress=egress)


def test_agent_egress_replaces_the_tasks():
    task = vf.ScoringConfig(egress=["https://pypi.org"])
    assert resolve_scoring(task, vf.ScoringConfig()).egress == ["https://pypi.org"]
    agent = vf.ScoringConfig(egress=["*"])
    assert resolve_scoring(task, agent).egress == ["*"]


class Backend:
    supports_user = False

    def __init__(self):
        self.routes = []

    async def prepare_execution(self, routes):
        self.routes.append(routes)


async def test_set_egress_enforces_the_policy_plus_urls_or_lifts_it():
    backend = Backend()
    runtime = Runtime(cast(RuntimeBackend, backend))
    await runtime.set_egress(["https://pypi.org"])
    await runtime.set_egress([])
    await runtime.set_egress(["*"])
    assert backend.routes == [["https://pypi.org"], [], None]
    with pytest.raises(ValueError):
        await runtime.set_egress(["pypi.org"])
