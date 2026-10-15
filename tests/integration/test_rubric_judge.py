"""Config-plugged judges score the rollout on top of the task's own reward."""

import pytest
from conftest import CI_JUDGE


@pytest.mark.parametrize("judge_id", ["reference", "rubric", "decision"])
async def test_plugged_judge(run, tmp_path, judge_id):
    """Judge config, calls, rewards, rubric metrics and usage reach the trace.

    The reference answer is echoed exactly and the rubric's criterion is always
    satisfied, so this checks the integration rather than grading quality."""
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(
        "[[criteria]]\n"
        'name = "always_yes"\n'
        'text = "Always satisfied — answer yes regardless of the response."\n'
    )
    judge = {"id": judge_id, **CI_JUDGE}
    if judge_id == "rubric":
        judge["path"] = str(rubric)
    (episode,) = await run(
        "echo-v1",
        harness="null",
        runtime={"type": "subprocess"},
        taskset_overrides={"task": {"judges": [judge]}},
        max_turns=2,
    )
    (trace,) = episode.traces
    assert trace.ok
    assert trace.rewards[judge_id].score > 0  # the judge's verdict landed in the reward
    if judge_id == "rubric":
        assert trace.metrics["rubric/always_yes"] == 1.0
    assert trace.info["judge_calls"]  # the call was recorded onto the trace
    assert trace.info["judge_calls"][0]["request"]["model"] == "cloudflare/clef"
    assert trace.extra_usage[-1].prompt_tokens > 0
