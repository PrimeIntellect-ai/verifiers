"""Consumers of sibling rewards leave a dropped trace's reward out."""

from types import SimpleNamespace

from verifiers.v1.envs.best_of_n.env import BestOfNEnv
from verifiers.v1.gepa.adapter import _episode_score


def _traces(make_trace, *rewards, dropped=()):
    traces = []
    for n, reward in enumerate(rewards):
        trace = make_trace()
        trace.record_reward("r", reward)
        if n in dropped:
            trace.drop("judge: outcome=task_defect")
        traces.append(trace)
    return traces


async def test_best_of_n_ignores_a_dropped_sibling(make_trace):
    traces = _traces(make_trace, 1.0, 0.5, dropped=(0,))
    env = SimpleNamespace(config=SimpleNamespace(threshold=1.0))
    await BestOfNEnv.finalize(env, None, SimpleNamespace(traces=traces))
    assert [t.metrics["best"] for t in traces] == [0.0, 1.0]
    assert {t.metrics["pass_at_n"] for t in traces} == {0.0}


async def test_best_of_n_with_every_sibling_dropped_marks_none(make_trace):
    traces = _traces(make_trace, 1.0, dropped=(0,))
    env = SimpleNamespace(config=SimpleNamespace(threshold=1.0))
    await BestOfNEnv.finalize(env, None, SimpleNamespace(traces=traces))
    assert "best" not in traces[0].metrics


def test_gepa_scores_without_dropped_traces(make_trace):
    traces = _traces(make_trace, 1.0, 0.0, dropped=(1,))
    assert _episode_score(SimpleNamespace(traces=traces)) == 1.0
    assert _episode_score(SimpleNamespace(traces=traces[1:])) == 0.0
