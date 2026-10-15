"""A task's network policy holds during execution, and the model is told about it: the
agent runs an egress probe under each policy the task data declares
(network-policy-v1), with the agent's `network_notice` on or off, across every harness
on prime, and on the chat loop across every runtime. Runtimes without network policy
support refuse the restricted tasks."""

import pytest
from grid import grid, runtimes_grid

from verifiers.v1.utils.loaders import harness_class

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


def prompt_text(trace) -> str:
    """Every prompt-supplied message the model saw, as one text."""
    texts = []
    for node in trace.nodes:
        if node.sampled:
            continue
        content = node.message.content
        if isinstance(content, str):
            texts.append(content)
        elif content:
            texts.extend(getattr(part, "text", "") or "" for part in content)
    return "\n".join(texts)


def check_policies(episodes, harness: str, notice: bool) -> None:
    by_name = {episode.task.data.name: episode for episode in episodes}
    assert set(by_name) == set(NOTICES)
    for name, episode in by_name.items():
        [trace] = episode.traces
        # mini-swe-agent only acts through tool calls: it drops a tool-less reply
        # (the probe output as a final answer) from its history, so the trace forks
        # by design. Recheck on version bumps:
        # https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/agents/default.py#L114
        if harness != "mini-swe-agent":
            assert trace.num_branches == 1
        assert trace.ok, [error.message for error in trace.errors]
        prompt = prompt_text(trace)
        if notice and NOTICES[name]:
            assert NOTICES[name] in prompt, name
        else:
            assert "External network access" not in prompt, name
        if harness_class(harness).EXECUTES_CODE:
            assert trace.reward == 1.0, (
                name,
                trace.info["network_report"],
                trace.last_reply,
            )


@pytest.mark.parametrize("notice", [True, False], ids=["notice-on", "notice-off"])
@pytest.mark.parametrize(
    "harness,harness_runtime", grid(runtimes=["prime"]), indirect=True
)
async def test_network_policy(run, harness, harness_runtime, notice):
    episodes = await run(
        "network-policy-v1",
        harness=harness,
        runtime=harness_runtime,
        num_tasks=len(NOTICES),
        max_turns=6,
        max_tokens=4096,
        env={"agent": {"network_notice": notice}},
    )
    check_policies(episodes, harness.id, notice)


@pytest.mark.parametrize(
    "harness_runtime", runtimes_grid(needs=("network",)), indirect=True
)
async def test_network_policy_runtimes(run, harness_runtime):
    """Each runtime either applies the task policies or refuses the restricted tasks."""
    episodes = await run(
        "network-policy-v1",
        runtime=harness_runtime,
        num_tasks=len(NOTICES),
        max_turns=2,
    )
    check_policies(episodes, "null", True)
