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
        runtime={"type": "prime"},
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
        runtime={"type": "prime"},
        rollout_timeout=1,
    )
    (trace,) = episode.traces
    assert trace.is_timeout
    assert trace.stop_condition == "agent_timeout"


@pytest.mark.parametrize("skip", [False, True], ids=["scored", "skipped"])
async def test_skip_scoring_on_truncated(run, skip):
    (episode,) = await run(
        "counter-tool",
        harness="null",
        runtime={"type": "prime"},
        taskset_overrides={"task": {"constant_reward": True}},
        env={"agent": {"max_turns": 1, "scoring": {"skip_on_truncated": skip}}},
    )
    (trace,) = episode.traces
    assert trace.is_truncated and trace.ok
    if skip:
        assert trace.reward == 0 and trace.rewards == {}
        assert trace.info["scoring_skipped"] == "truncated"
    else:
        assert trace.reward == 1
        assert "scoring_skipped" not in trace.info


@pytest.mark.parametrize("skip", [False, True], ids=["scored", "skipped"])
async def test_skip_scoring_on_timeout(run, skip):
    (episode,) = await run(
        "counter-tool",
        harness="null",
        runtime={"type": "prime"},
        rollout_timeout=1,
        taskset_overrides={"task": {"constant_reward": True}},
        env={"agent": {"scoring": {"skip_on_timeout": skip}}},
    )
    (trace,) = episode.traces
    assert trace.is_timeout and trace.ok
    if skip:
        assert trace.reward == 0 and trace.rewards == {}
        assert trace.info["scoring_skipped"] == "timeout"
    else:
        assert trace.reward == 1


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


@pytest.mark.parametrize("skip", [False, True], ids=["scored", "skipped"])
async def test_skip_scoring_on_stop(run, scripted_model, skip):
    stop = {"fn": "counter_tool:any_response", "skip_scoring": skip}
    (episode,) = await run(
        "counter-tool",
        harness="null",
        runtime={"type": "subprocess"},
        client=scripted_model,
        taskset_overrides={
            "task": {"constant_reward": True, "stops": {"any_response": stop}}
        },
    )
    (trace,) = episode.traces
    assert trace.stop_condition == "any_response" and trace.ok
    if skip:
        assert trace.reward == 0 and trace.rewards == {}
        assert trace.info["scoring_skipped"] == "any_response"
    else:
        assert trace.reward == 1
        assert "scoring_skipped" not in trace.info
