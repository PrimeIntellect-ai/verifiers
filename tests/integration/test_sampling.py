"""Sampling overrides reach the provider: the eval's temperature and top_p replace whatever
the harness sends, across every harness. codex speaks Responses, claude-code Anthropic
messages, and the rest chat completions. Unset, they stay off the request, so the
provider's default applies."""

import pytest
from grid import grid


@pytest.mark.parametrize(
    "sampling", [{"temperature": 0.3, "top_p": 0.9}, {}], ids=["set", "unset"]
)
@pytest.mark.parametrize(
    "harness,harness_runtime", grid(runtimes=("prime",)), indirect=True
)
async def test_sampling_overrides(run, harness, harness_runtime, sampling):
    # The effort comes from the run's sampling, temperature and top_p from the seat's,
    # so the merge of the two layers is on the path too.
    (episode,) = await run(
        "echo-v1",
        harness=harness,
        runtime=harness_runtime,
        reasoning_effort="low",
        env={"agent": {"sampling": sampling}},
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.calls
    # Each call records the sampling parsed from the request that went upstream.
    for call in trace.calls:
        assert call.sampling is not None
        assert call.sampling.temperature == sampling.get("temperature")
        assert call.sampling.top_p == sampling.get("top_p")
        assert call.sampling.reasoning_effort == "low"
