"""A task's network policy holds during execution, and the agent is told about it: a
scripted agent probes a blocked host, a permitted host, and the framework's own route
from inside the runtime, and records the system prompt it was handed
(network-policy-v1). Each task declares one policy in its data; the agent's
`network_notice` is on or off. Runs across the runtimes with a network policy; a
runtime that cannot enforce a task's policy refuses that task instead."""

import pytest
from grid import runtimes_grid
from pydantic import TypeAdapter, ValidationError

from verifiers.v1.runtimes import RuntimeConfig

NOTICES = {
    "open": None,
    "off": "External network access is disabled.",
    "block": (
        "External network access is allowed except for these blocked destinations: "
        "example.com. Do not circumvent this selection of domains."
    ),
    "allow": (
        "External network access is limited to these destinations: pypi.org. "
        "Do not circumvent this selection of domains."
    ),
}

POLICIES = {
    "open": {},
    "off": {"allow": []},
    "block": {"block": ["example.com"]},
    "allow": {"allow": ["pypi.org"]},
}


def enforces(runtime: dict, policy: dict) -> bool:
    """Whether the runtime config accepts the policy: modal has no deny lists, and e2b
    denies only IP addresses and CIDR blocks."""
    try:
        TypeAdapter(RuntimeConfig).validate_python({**runtime, **policy})
    except ValidationError:
        return False
    return True


@pytest.mark.parametrize("notice", [True, False], ids=["notice-on", "notice-off"])
@pytest.mark.parametrize(
    "harness_runtime", runtimes_grid(needs=("network",)), indirect=True
)
async def test_network_policy(run, harness_runtime, notice):
    episodes = await run(
        "network-policy-v1",
        harness="network-policy-v1",
        runtime=harness_runtime,
        num_tasks=len(NOTICES),
        env={"agent": {"network_notice": notice}},
    )
    by_name = {episode.task.data.name: episode for episode in episodes}
    assert set(by_name) == set(NOTICES)
    for name, episode in by_name.items():
        if not enforces(harness_runtime, POLICIES[name]):
            assert not episode.ok and not episode.traces, name
            assert "ValidationError" in episode.errors[0].message, name
            continue
        [trace] = episode.traces
        assert trace.ok, [error.message for error in trace.errors]
        assert trace.reward == 1.0, (name, trace.info["network_report"])
        assert trace.info["system_prompt"] == (NOTICES[name] if notice else None), name
