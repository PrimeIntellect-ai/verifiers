"""The seat's caps stop a rollout that would run on: the counter task wants three tool
turns (counter-tool), and each cap ends it after the first with its own stop
condition. The turn cap also runs across the harness x runtime grid, since each program
handles a refused request in its own way."""

import pytest
from grid import grid

CAPS = [
    pytest.param({"max_turns": 1}, "max_turns", id="max-turns"),
    pytest.param({"max_input_tokens": 1}, "max_input_tokens", id="max-input-tokens"),
    pytest.param({"max_output_tokens": 1}, "max_output_tokens", id="max-output-tokens"),
    pytest.param({"max_total_tokens": 1}, "max_total_tokens", id="max-total-tokens"),
]


@pytest.mark.parametrize("cap,stop_condition", CAPS)
async def test_cap(run, cap, stop_condition):
    (episode,) = await run(
        "counter-tool",
        harness="null",
        runtime={"type": "subprocess"},
        env={"agent": cap},
    )
    (trace,) = episode.traces
    assert trace.num_branches == 1
    assert trace.stop_condition == stop_condition
    assert trace.is_truncated
    assert trace.num_turns == 1


async def test_rollout_timeout(run):
    (episode,) = await run(
        "counter-tool",
        harness="null",
        runtime={"type": "subprocess"},
        rollout_timeout=1,
    )
    (trace,) = episode.traces
    assert trace.is_timeout
    assert trace.stop_condition == "agent_timeout"


@pytest.mark.parametrize("harness,harness_runtime", grid(needs=("mcp",)), indirect=True)
async def test_max_turns(run, harness, harness_runtime):
    (episode,) = await run(
        "counter-tool",
        harness=harness,
        runtime=harness_runtime,
        max_turns=1,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert trace.num_branches == 1
    assert trace.stop_condition == "max_turns", [e.message for e in trace.errors]
    assert trace.num_turns == 1
