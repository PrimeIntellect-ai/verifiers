"""A request interceptor denies a tool call before it runs (tool-gate), across the
harness x runtime grid. The tool never executes: the harness's gate or the tool server
hands the denial to the model, or the hook stops a program that has no gate."""

import pytest
from grid import grid

from verifiers.v1.utils.loaders import harness_class


@pytest.mark.parametrize(
    "harness,harness_runtime",
    grid(needs=("mcp",)),
    indirect=True,
)
async def test_tool_gate(run, harness, harness_runtime):
    (episode,) = await run(
        "tool-gate",
        harness=harness,
        runtime=harness_runtime,
        max_turns=6,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert trace.state.revealed == 0  # the tool never executed
    assert trace.request_rewrites, "the model never called the tool"
    assert {r.handler for r in trace.request_rewrites} == {"deny_reveal"}
    denied = any("DENIED-7c1" in str(m.content) for m in trace.tool_messages)
    if harness_class(harness.id).SUPPORTS_TOOL_INTERCEPTION:
        assert trace.ok, [error.message for error in trace.errors]
        assert denied, "the denial never reached the model"
    else:
        # The tool server enforces the verdict however the program reached the tool; a
        # direct call to a program without a gate instead ends the rollout at the hook.
        assert denied or trace.stop_condition == "deny_reveal", trace.stop_condition
