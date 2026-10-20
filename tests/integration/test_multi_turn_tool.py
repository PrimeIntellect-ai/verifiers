"""One task, one MCP tool: the model calls a tool whose output carries a stamp the prompt
never reveals (echo-tool), so reward 1.0 proves the tool ran. Every MCP harness reaches
it colocated across the harness x runtime grid; the chat loop reaches it across its
placements (colocated in the harness's box, in a Compose service beside it, or its own
runtime) on the runtime x runtime grid."""

import pytest
from grid import assert_one_branch, grid, placements

import verifiers.v1 as vf


@pytest.mark.parametrize("harness,harness_runtime", grid(needs=("mcp",)), indirect=True)
async def test_tool(run, harness, harness_runtime):
    """Every MCP-capable harness reaches a tool colocated in its runtime."""
    (episode,) = await run(
        "echo-tool",
        harness=harness,
        runtime=harness_runtime,
        max_turns=6,
        max_tokens=8192,
        taskset_overrides={"task": {"tools": {"colocated": True}}},
    )
    (trace,) = episode.traces
    assert_one_branch(trace, harness, forks=("openclaw",))
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 1
    # Reward 1.0 is the proof the tool ran. How a program surfaces an MCP tool to the
    # model is its own business (rlm calls it from code; chat loops advertise it), so
    # the advertised schema is checked on the chat loop in `test_tool_placement`.
    assert trace.reward == 1.0, "\n".join(
        message.model_dump_json(exclude_none=True)
        for message in trace.messages
        if message.role in {"assistant", "tool"}
    )


@pytest.mark.parametrize(
    "harness_runtime,tool_runtime",
    placements(tool_runtimes=("colocated", "service", "subprocess", "prime")),
    indirect=True,
)
async def test_tool_placement(run, harness_runtime, tool_runtime):
    """A `vf.Toolset` across its placement (`tool_runtime`: colocated in the harness's
    runtime, in a Compose service beside it, or its own runtime) x the harness `runtime`,
    on the chat loop. Eval-wide SHARED servers are a different scope (`Taskset.toolsets`),
    covered by `test_shared_tool`."""
    (episode,) = await run(
        "echo-tool",
        harness="null",
        runtime=harness_runtime,
        max_turns=6,
        taskset_overrides={"task": {"tools": tool_runtime}},
    )
    (trace,) = episode.traces
    assert trace.num_branches == 1
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0
    # The interception server captured the harness's advertised name and schema. Harnesses
    # may qualify the same raw MCP tool differently, so the test checks its stable suffix.
    assert trace.tools
    (echo_tool,) = [tool for tool in trace.tools if tool.name.endswith("back")]
    assert "message" in echo_tool.parameters.get("properties", {})


@pytest.mark.parametrize(
    "placement",
    [
        {"colocated": True},
        {"url": "http://example.com/mcp"},
        {"runtime": {"type": "prime"}},
    ],
)
def test_service_placement_is_exclusive(placement):
    """A server in a Compose service runs there: not colocated, remote or in a runtime of
    its own."""
    with pytest.raises(ValueError):
        vf.ToolsetConfig(service="tools", **placement)
