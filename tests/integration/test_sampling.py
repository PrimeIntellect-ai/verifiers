"""Sampling overrides reach the provider: the eval's temperature, top_p, and reasoning
effort replace whatever the harness sends, in each dialect's native shape. Every harness
runs one turn against a scripted upstream that records the requests it receives; pi and
kimi-code pick their transport, so they also cover the Responses and Anthropic dialects."""

import pytest
from grid import HARNESSES, grid

TRANSPORTS = [
    {"id": harness, "transport": transport}
    for harness in ("pi", "kimi-code")
    for transport in ("responses", "anthropic_messages")
]

# Where each dialect carries the reasoning effort.
EFFORT = {
    "/chat/completions": lambda body: body.get("reasoning_effort"),
    "/responses": lambda body: (body.get("reasoning") or {}).get("effort"),
    "/messages": lambda body: (body.get("output_config") or {}).get("effort"),
}


@pytest.mark.parametrize(
    "harness,harness_runtime",
    grid([*HARNESSES, *TRANSPORTS], runtimes=("subprocess", "prime")),
    indirect=True,
)
async def test_sampling_overrides(
    run, harness, harness_runtime, scripted_model, upstream_requests
):
    # The effort comes from the run's sampling, temperature and top_p from the seat's,
    # so the merge of the two layers is on the path too.
    (episode,) = await run(
        "echo-v1",
        harness=harness,
        runtime=harness_runtime,
        client=scripted_model,
        reasoning_effort="low",
        env={"agent": {"sampling": {"temperature": 0.3, "top_p": 0.9}}},
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert upstream_requests
    for endpoint, body in upstream_requests:
        sent = {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "reasoning_effort": EFFORT[endpoint](body),
        }
        assert sent == {"temperature": 0.3, "top_p": 0.9, "reasoning_effort": "low"}, (
            endpoint
        )
    for call in trace.calls:
        assert call.sampling is not None
        assert call.sampling.temperature == 0.3
        assert call.sampling.top_p == 0.9
        assert call.sampling.reasoning_effort == "low"
