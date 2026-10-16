"""One task, one MCP tool: the model calls a tool whose output carries a stamp the prompt
never reveals (echo-tool-v1), so reward 1.0 proves the tool ran. Every MCP harness reaches
it colocated across the harness x runtime grid; the chat loop reaches it across its
placements (colocated in the harness's box, or its own runtime) on the runtime x runtime
grid."""

import pytest
from grid import grid, placements


@pytest.mark.parametrize("harness,harness_runtime", grid(needs=("mcp",)), indirect=True)
async def test_tool(run, harness, harness_runtime):
    """Every MCP-capable harness reaches a tool colocated in its runtime."""
    (episode,) = await run(
        "echo-tool-v1",
        harness=harness,
        runtime=harness_runtime,
        max_turns=6,
        max_tokens=8192,
        taskset_overrides={"task": {"tools": {"colocated": True}}},
    )
    (trace,) = episode.traces
    # OpenClaw strips the runtime context it injects before each turn from the
    # replayed history, so the trace forks by design. Recheck on version bumps:
    # https://github.com/openclaw/openclaw/blob/v2026.9.5/src/agents/embedded-agent-runner/run/attempt-llm-boundary.ts#L145
    if harness.id != "openclaw":
        assert trace.num_branches == 1
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 1
    # Reward 1.0 is the proof the tool ran. How a program surfaces an MCP tool to the
    # model is its own business (rlm calls it from code; chat loops advertise it), so
    # the advertised schema is checked on the chat loop in `test_tool_placement`.
    assert trace.reward == 1.0


@pytest.mark.parametrize("harness_runtime,tool_runtime", placements(), indirect=True)
async def test_tool_placement(run, harness_runtime, tool_runtime):
    """A `vf.Toolset` across its placement (`tool_runtime`: colocated in the harness's
    runtime, or its own runtime) x the harness `runtime`, on the chat loop. Eval-wide SHARED servers
    are a different scope (`Taskset.toolsets`), covered by `test_shared_tool`."""
    (episode,) = await run(
        "echo-tool-v1",
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
