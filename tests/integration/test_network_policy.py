"""A task's network policy holds during execution, and the model is told about it: the
agent runs an egress probe under each policy the task data declares
(network-policy), with the agent's `network_notice` on or off, across every harness
on prime."""

import pytest
from grid import assert_one_branch, grid

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


@pytest.mark.parametrize("notice", [True, False], ids=["notice-on", "notice-off"])
@pytest.mark.parametrize("harness,harness_runtime", grid(), indirect=True)
async def test_network_policy(run, harness, harness_runtime, notice):
    episodes = await run(
        "network-policy",
        harness=harness,
        runtime=harness_runtime,
        num_tasks=len(NOTICES),
        max_turns=6,
        max_tokens=4096,
        env={"agent": {"network_notice": notice}},
    )
    by_name = {episode.task.data.name: episode for episode in episodes}
    assert set(by_name) == set(NOTICES)
    for name, episode in by_name.items():
        [trace] = episode.traces
        assert_one_branch(trace, harness, forks=("mini-swe-agent", "openclaw"))
        assert trace.ok, [error.message for error in trace.errors]
        prompt = prompt_text(trace)
        if notice and NOTICES[name]:
            assert NOTICES[name] in prompt, name
        else:
            assert "External network access" not in prompt, name
        if harness_class(harness.id).EXECUTES_CODE:
            assert trace.reward == 1.0, (
                name,
                trace.info["network_report"],
                trace.last_reply,
            )
