"""A request interceptor denies a tool call before it runs (tool-gate-v1), across the
harness x runtime grid. A harness with a tool gate hands the denial to the model; any
other harness has its rollout stopped by the hook. The tool never executes."""

import pytest
from grid import grid, harness_class


@pytest.mark.parametrize(
    "harness,harness_runtime",
    grid(needs=("mcp",)),
    indirect=True,
)
async def test_tool_gate(run_v1, harness, harness_runtime, tmp_path):
    (trace,) = await run_v1(
        "tool-gate-v1",
        harness=harness,
        runtime=harness_runtime,
        output_dir=tmp_path,
        max_turns=6,
        max_tokens=8192,
    )
    assert trace.state.revealed == 0  # the tool never executed
    assert [r.handler for r in trace.request_rewrites] == ["deny_reveal"] * len(
        trace.request_rewrites
    )
    assert trace.request_rewrites, "the model never called the tool"
    if harness_class(harness.id).SUPPORTS_TOOL_INTERCEPTION:
        assert trace.ok, [error.message for error in trace.errors]
        assert trace.stop_condition != "deny_reveal"
        assert any("DENIED-7c1" in str(m.content) for m in trace.tool_messages)
    else:
        assert trace.stop_condition == "deny_reveal"
