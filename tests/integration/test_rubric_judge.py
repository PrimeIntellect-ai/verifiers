"""A config-plugged rubric judge scores the rollout on top of the task's own reward."""

from conftest import CI_MODEL


async def test_rubric_judge(run, tmp_path):
    """A config-plugged rubric judge scores the rollout on top of the taskset's own reward.

    The single criterion is trivially satisfiable ("answer yes"), so any live judge model
    scores it 1.0 — the test asserts the plumbing (config narrowing -> judge call -> reward +
    per-criterion metric on the trace), not judge quality."""
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(
        "[[criteria]]\n"
        'name = "always_yes"\n'
        'text = "Always satisfied — answer yes regardless of the response."\n'
    )
    (episode,) = await run(
        "echo-v1",
        harness="null",
        runtime={"type": "subprocess"},
        taskset_overrides={
            "task": {
                "judges": [{"id": "rubric", "path": str(rubric), "model": CI_MODEL}]
            }
        },
        max_turns=2,
    )
    (trace,) = episode.traces
    assert trace.ok
    assert trace.rewards["rubric"].score > 0  # the judge's verdict landed in the reward
    assert trace.metrics["rubric/always_yes"] == 1.0
    assert trace.info["judge_calls"]  # the call was recorded onto the trace
