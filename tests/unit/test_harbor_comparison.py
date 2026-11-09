"""Diagnostic comparison preserves the original verdict's training eligibility."""

from verifiers.v1.tasksets.harbor.comparison import VerifierComparison


def test_dropped_original_scores_remain_available_for_diagnosis(make_trace):
    original = make_trace()
    original.ok = True
    original.stop("agent_completed")
    original.record_reward("reward", 0.2)
    original.record_metric("criterion", 0.3)
    original.drop("excluded from training")

    current = original.model_copy(deep=True)
    current.dropped = None
    current.record_reward("reward", 0.75)
    dropped = current.model_copy(deep=True)
    dropped.drop("no usable verdict")
    failed = current.model_copy(deep=True)
    failed.ok = False
    timed_out = current.model_copy(deep=True)
    timed_out.record_timeout("scoring")
    comparison = VerifierComparison(
        original=original,
        traces={
            "current": current,
            "dropped": dropped,
            "failed": failed,
            "timeout": timed_out,
        },
    )

    assert comparison.scores == {
        "original": {"reward/reward": 0.2, "metric/criterion": 0.3},
        "current": {"reward/reward": 0.75, "metric/criterion": 0.3},
        "dropped": {},
        "failed": {},
        "timeout": {},
    }
    assert comparison.original.is_dropped
    assert comparison.original.dropped == "excluded from training"
    assert comparison.traces["dropped"].is_dropped
