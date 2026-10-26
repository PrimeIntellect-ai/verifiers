"""A tool that raises: its exception text reaches the model as the tool result
(tool-error). The server side is the same for every harness, so the chat loop covers it."""


async def test_tool_error(run):
    """A tool's exception message comes back to the model, not just that the call failed."""
    (episode,) = await run("tool-error", max_turns=4)
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0, [
        message.model_dump_json(exclude_none=True) for message in trace.tool_messages
    ]
