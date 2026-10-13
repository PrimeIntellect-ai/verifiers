"""The seat's caps stop a rollout that would run on: the counter task wants three tool
turns (counter-tool-v1), and each cap ends it after the first with its own stop
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
async def test_cap(run_v1, cap, stop_condition, tmp_path):
    (trace,) = await run_v1(
        "counter-tool-v1",
        harness="null",
        runtime={"type": "subprocess"},
        output_dir=tmp_path,
        env={"agent": cap},
    )
    assert trace.stop_condition == stop_condition
    assert trace.is_truncated
    assert trace.num_turns == 1


async def test_rollout_timeout(run_v1, tmp_path):
    (trace,) = await run_v1(
        "counter-tool-v1",
        harness="null",
        runtime={"type": "subprocess"},
        output_dir=tmp_path,
        rollout_timeout=1,
    )
    assert trace.is_timeout
    assert trace.stop_condition == "agent_timeout"


@pytest.mark.parametrize("harness,harness_runtime", grid(needs=("mcp",)), indirect=True)
async def test_max_turns(run_v1, harness, harness_runtime, tmp_path):
    (trace,) = await run_v1(
        "counter-tool-v1",
        harness=harness,
        runtime=harness_runtime,
        output_dir=tmp_path,
        max_turns=1,
        max_tokens=8192,
    )
    assert trace.stop_condition == "max_turns", [e.message for e in trace.errors]
    assert trace.num_turns == 1
