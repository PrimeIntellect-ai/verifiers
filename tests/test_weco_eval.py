from types import SimpleNamespace

from verifiers.v1.cli.weco_eval import summary


def test_summary_excludes_errored_rollouts_from_optimizer_reward():
    traces = [
        SimpleNamespace(
            has_error=False, rewards={"score": 1.0}, metrics={"ok": 1.0}, reward=1.0
        ),
        SimpleNamespace(has_error=True, rewards={}, metrics={}, reward=0.0),
    ]

    assert summary(traces).splitlines() == [
        "rollouts: 2",
        "errors: 1",
        "reward/score: 1.0000",
        "metric/ok: 1.0000",
        "reward: 1.0000",
    ]
