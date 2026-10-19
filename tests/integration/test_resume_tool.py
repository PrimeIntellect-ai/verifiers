"""One task, two segments: the harness preserves context and MCP access across a resumed
exchange (echo-acp-resume), across the harness x runtime grid."""

import pytest
from grid import assert_one_branch, grid


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("resume", "mcp")), indirect=True
)
async def test_resume_tool(run, harness, harness_runtime):
    """Each harness preserves context and MCP access across two segments."""
    (episode,) = await run(
        "echo-acp-resume",
        harness=harness,
        runtime=harness_runtime,
        max_turns=8,
        max_tokens=8192,
        rollout_timeout=600,
    )
    (trace,) = episode.traces
    assert_one_branch(trace, harness, forks=("openclaw",))
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.stop_condition == "user_closed"
    assert trace.rewards["resumed"].score == 1.0
    segments = trace.info["acp_segments"]
    assert len(segments) == 2
    assert segments[0]["terminated"] is False
    assert segments[1]["terminated"] is False
    assert trace.last_reply == segments[1]["last_reply"]
    assert "tool" in segments[1]["roles"]
    assert segments[1]["tool_outputs"]
    if harness.id == "pi":
        recall = next(tool for tool in trace.tools if tool.name == "resume_recall")
        assert recall.parameters["properties"]["codeword"]["type"] == "string"
        assert "codeword" in recall.parameters["required"]
    if harness.id == "codex":
        recall = next(tool for tool in trace.tools if tool.name.endswith("recall"))
        assert recall.namespace
        assert "codeword" in recall.parameters["properties"]
        assert any(
            call.namespace == recall.namespace and call.name == recall.name
            for message in trace.assistant_messages
            for call in message.tool_calls or []
        )
    if harness.id == "rlm":
        assert "turns_since_last_compaction" in trace.metrics
        assert all(call.acp is not None for call in trace.calls)
        assert all(
            trace.nodes[parent.node].sampled and node.sampled
            for node in trace.nodes
            for parent in node.semantic_parents
        )
    if harness.id == "prime-agent":
        lifecycle = trace.info["acp_lifecycle"]["ai.primeintellect.prime-agent"]
        assert len(lifecycle) == 2
        for status in lifecycle:
            assert status["infrastructure_status"] == "ok"
            assert status["terminal_quiescence_observed"] is True
            assert (
                status["terminal_quiescence"]["quiescence"]["outstandingSubagents"] == 0
            )
