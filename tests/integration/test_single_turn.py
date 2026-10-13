"""Single-turn rollouts: the model echoes a phrase back (echo-v1), across the harness x
runtime grid."""

import pytest
from grid import grid


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("reply",)), indirect=True
)
async def test_single_turn(run_v1, harness, harness_runtime, tmp_path):
    (trace,) = await run_v1(
        "echo-v1",
        harness=harness,
        runtime=harness_runtime,
        env={"agent": {"sampling": {"temperature": 0.0}}},
        output_dir=tmp_path,
        max_turns=2,
    )
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns == 1
    assert trace.stop_condition == "agent_completed"
    assert trace.reward == 1.0
    assert trace.task.key == f"echo:{trace.task.data.answer}"
    assert trace.task.hash is not None
    # The seat's resolved identity rides the trace (policy metadata for trainers).
    assert trace.agent is not None
    assert trace.agent.config.sampling.max_tokens == 2048
    assert trace.agent.config.sampling.temperature == 0.0
    # Every sampled turn has one per-call record, linked to its assistant node.
    sampled = [i for i, n in enumerate(trace.nodes) if n.sampled]
    assert [c.node for c in trace.calls if c.error is None] == sampled
    for call in trace.calls:
        assert call.model and call.sampling is not None
        assert call.time.duration > 0


@pytest.mark.browser_use
@pytest.mark.docker
async def test_browser_use(run_v1, tmp_path):
    """The browser_use harness runs in an explicitly browser-capable runtime."""
    image = (
        "mcr.microsoft.com/playwright/python:v1.61.0-noble@"
        "sha256:a9731514f24121d1dcd25d58d0a38146646d290a5998fd80d3e533e7b5e21c69"
    )
    (trace,) = await run_v1(
        "echo-v1",
        harness="browser_use",
        runtime={"type": "docker", "image": image},
        output_dir=tmp_path,
        max_turns=2,
    )
    assert trace.ok
    assert trace.reward == 1.0
