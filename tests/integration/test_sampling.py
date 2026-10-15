"""Sampling overrides reach the provider: the eval's temperature, top_p, and reasoning
effort replace whatever the harness sends, across every harness. pi and kimi-code pick
their transport, so they also cover the Responses and Anthropic dialects."""

import pytest
from grid import HARNESSES, grid

TRANSPORTS = [
    {"id": harness, "transport": transport}
    for harness in ("pi", "kimi-code")
    for transport in ("responses", "anthropic_messages")
]


@pytest.mark.parametrize(
    "harness,harness_runtime",
    grid([*HARNESSES, *TRANSPORTS], runtimes=("subprocess", "prime")),
    indirect=True,
)
async def test_sampling_overrides(run, harness, harness_runtime):
    # The effort comes from the run's sampling, temperature and top_p from the seat's,
    # so the merge of the two layers is on the path too.
    (episode,) = await run(
        "echo-v1",
        harness=harness,
        runtime=harness_runtime,
        reasoning_effort="low",
        env={"agent": {"sampling": {"temperature": 0.3, "top_p": 0.9}}},
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.calls
    # Each call records the sampling parsed from the request that went upstream.
    for call in trace.calls:
        assert call.sampling is not None
        assert call.sampling.temperature == 0.3
        assert call.sampling.top_p == 0.9
        assert call.sampling.reasoning_effort == "low"
