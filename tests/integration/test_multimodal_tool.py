"""A tool that returns an image: MCP image content from a tool result survives into the
trace (tool-response-image-v1; needs a vision model)."""


async def test_multimodal_tool(run_v1, tmp_path):
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
