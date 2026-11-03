"""Sampling overrides reach the provider: the eval's temperature, top_p, and reasoning
effort replace whatever the harness sends, across every harness. codex speaks Responses,
claude-code Anthropic messages, and the rest chat completions."""

import pytest
from grid import grid

SAMPLING = {"temperature": 0.3, "top_p": 0.9, "reasoning_effort": "high"}


@pytest.mark.parametrize("harness,harness_runtime", grid(), indirect=True)
async def test_sampling_overrides(run, harness, harness_runtime):
    (episode,) = await run(
        "echo",
        harness=harness,
        runtime=harness_runtime,
        env={"agent": {"sampling": SAMPLING}},
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.calls
    # Each call records the sampling parsed from the request that went upstream.
    for call in trace.calls:
        assert call.sampling is not None
        assert call.sampling.temperature == SAMPLING["temperature"]
        assert call.sampling.top_p == SAMPLING["top_p"]
        assert call.sampling.reasoning_effort == SAMPLING["reasoning_effort"]
