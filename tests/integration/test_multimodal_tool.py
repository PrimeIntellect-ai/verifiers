"""A tool that returns an image: MCP image content from a tool result survives into the
trace (tool-response-image-v1; needs a vision model)."""

from conftest import CI_VISION_MODEL


async def test_multimodal_tool(run):
    """MCP image content from a tool result survives into the v1 trace (needs a vision model)."""
    (episode,) = await run(
        "tool-response-image-v1",
        harness="null",
        runtime={"type": "subprocess"},
        model=CI_VISION_MODEL,
        reasoning_effort="none",
        max_turns=4,
    )
    (trace,) = episode.traces
    assert trace.ok
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0
