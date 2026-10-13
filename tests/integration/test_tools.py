"""Multi-turn tool use: the model calls an MCP tool (echo-tool-v1, whose output carries a
stamp the prompt never reveals, so reward 1.0 proves the tool ran) across the harness x
runtime grid, and the tool server's placement (colocated in the harness's box, or its own
runtime) across the runtime x runtime grid."""

import pytest
from grid import RUNTIMES, grid, placements


@pytest.mark.parametrize("harness,harness_runtime", grid(needs=("mcp",)), indirect=True)
async def test_tool(run_v1, harness, harness_runtime, tmp_path):
    """Every MCP-capable harness reaches a tool colocated in its runtime."""
    (trace,) = await run_v1(
        "echo-tool-v1",
        harness=harness,
        runtime=harness_runtime,
        output_dir=tmp_path,
        max_turns=6,
        max_tokens=8192,
        taskset_overrides={"task": {"tools": {"colocated": True}}},
    )
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 1
    # Reward 1.0 is the proof the tool ran. How a program surfaces an MCP tool to the
    # model is its own business (rlm calls it from code; chat loops advertise it), so
    # the advertised schema is checked on the chat loop in `test_tool_placement`.
    assert trace.reward == 1.0


@pytest.mark.parametrize("harness_runtime,tool_runtime", placements(), indirect=True)
async def test_tool_placement(run_v1, harness_runtime, tool_runtime, tmp_path):
    """A `vf.Toolset` across its placement (`tool_runtime`: colocated in the harness's
    runtime, or its own runtime) x the harness `runtime`, on the chat loop. Eval-wide
    SHARED servers are a different scope (`Taskset.toolsets`) with their own
    env-server-path coverage: `test_shared_tool_isolation`."""
    (trace,) = await run_v1(
        "echo-tool-v1",
        harness="null",
        runtime=harness_runtime,
        output_dir=tmp_path,
        max_turns=6,
        taskset_overrides={"task": {"tools": tool_runtime}},
    )
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0
    # The interception server captured the harness's advertised name and schema. Harnesses
    # may qualify the same raw MCP tool differently, so the test checks its stable suffix.
    assert trace.tools
    (echo_tool,) = [tool for tool in trace.tools if tool.name.endswith("back")]
    assert "message" in echo_tool.parameters.get("properties", {})


@pytest.mark.parametrize("harness_runtime,tool_runtime", placements(), indirect=True)
async def test_tool_state(run_v1, harness_runtime, tool_runtime, tmp_path):
    """The shared-state round-trip: a `@vf.tool` increments the typed `trace.state` each
    call (synced over the interception server) and the `@reward` reads it back — reward
    1.0 proves tool writes reach the host's `trace.state`, colocated and own-runtime (a
    SHARED server's per-rollout state channel is covered by `test_shared_tool_isolation`)."""
    (trace,) = await run_v1(
        "counter-tool-v1",
        harness="null",
        runtime=harness_runtime,
        output_dir=tmp_path,
        max_turns=8,
        taskset_overrides={"task": {"tools": tool_runtime}},
    )
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 2  # at least two tool calls accumulated
    assert trace.reward == 1.0


@pytest.mark.parametrize(
    "harness_runtime,tool_runtime",
    placements(tool_runtimes=tuple(RUNTIMES)),
    indirect=True,
)
async def test_shared_tool_isolation(
    run_v1_server, harness_runtime, tool_runtime, tmp_path
):
    """A shared writable tool isolates state across concurrent rollouts and runtimes.
    Shared servers always run in their own runtime (colocation is per-rollout, shared is
    eval-level)."""
    traces = await run_v1_server(
        "scratchpad",
        harness="null",
        runtime=harness_runtime,
        output_dir=tmp_path,
        num_tasks=2,
        n=1,
        max_turns=4,
        taskset_overrides={"tools": tool_runtime},
    )
    assert len(traces) == 2
    for trace in traces:
        assert trace.ok, [error.message for error in trace.errors]
        assert trace.num_turns >= 2  # tool call + answer
        assert trace.reward == 1.0


async def test_tool_response_image(run_v1, tmp_path):
    """MCP image content from a tool result survives into the v1 trace (needs a vision model)."""
    (trace,) = await run_v1(
        "tool-response-image-v1",
        harness="null",
        runtime={"type": "subprocess"},
        reasoning_effort="none",
        output_dir=tmp_path,
        max_turns=4,
    )
    assert trace.ok
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0
