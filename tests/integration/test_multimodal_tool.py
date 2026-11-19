"""A tool that returns an image: MCP image content from a tool result survives into the
trace (tool-response-image; needs a vision model)."""


async def test_multimodal_tool(run):
    """MCP image content from a tool result survives into the v1 trace (needs a vision model)."""
    (episode,) = await run(
        "tool-response-image",
        harness="null",
        runtime={"type": "prime"},
        reasoning_effort="none",
        max_turns=4,
    )
    (trace,) = episode.traces
    assert trace.num_branches == 1
    assert trace.ok
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0


async def test_rlm_reads_workspace_image(run):
    from verifiers.v1.harnesses.rlm.harness import RLMHarnessConfig

    (episode,) = await run(
        "read-image",
        harness=RLMHarnessConfig(
            id="rlm",
            version="bd5b007fcd863fa3262daec2de71ecc794d45346",
            vision=True,
        ),
        runtime={"type": "prime"},
        max_turns=6,
    )
    (trace,) = episode.traces
    assert trace.ok
    assert trace.reward == 1.0
