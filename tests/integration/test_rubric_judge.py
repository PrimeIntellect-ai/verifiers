"""Config-plugged judges score the rollout on top of the task's own reward."""

import pytest
from conftest import CI_DECISION_JUDGE, CI_JUDGE


@pytest.mark.parametrize("judge_id", ["rubric", "decision"])
async def test_rubric_judge(run, tmp_path, judge_id):
    """Keep rubric coverage and exercise Clef through the existing decision judge.

    Both variants verify config loading, grading, trace records and usage; the
    rubric variant also checks its per-criterion metric.
    """
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(
        "[[criteria]]\n"
        'name = "always_yes"\n'
        'text = "Always satisfied — answer yes regardless of the response."\n'
    )
    judge = {
        "id": judge_id,
        **(CI_DECISION_JUDGE if judge_id == "decision" else CI_JUDGE),
    }
    if judge_id == "rubric":
        judge["path"] = str(rubric)
    (episode,) = await run(
        "echo",
        harness="null",
        runtime={"type": "prime"},
        taskset_overrides={"task": {"judges": [judge]}},
        max_turns=2,
    )
    (trace,) = episode.traces
    assert trace.num_branches == 1
    assert trace.ok
    assert trace.rewards[judge_id].score > 0  # the judge's verdict landed in the reward
    if judge_id == "rubric":
        assert trace.metrics["rubric/always_yes"] == 1.0
    assert trace.info["judge_calls"]  # the call was recorded onto the trace
    assert trace.info["judge_calls"][0]["request"]["model"] == judge["model"]
    assert trace.extra_usage[-1].prompt_tokens > 0
