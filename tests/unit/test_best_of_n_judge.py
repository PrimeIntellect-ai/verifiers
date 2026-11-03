"""best-of-n-judge: n attempts ranked side by side by an ensemble of agentic judges.
Fake agents drive the env's own `run`, `RankTask.finalize` and `finalize`, so the
ranking contract and the reward composition are checked without a model or a
sandbox; tests/integration/test_agentic_judge.py runs the real thing."""

import json
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.envs.agentic_judge.env import VERDICT_FILE
from verifiers.v1.envs.best_of_n_judge.env import (
    BestOfNJudgeEnv,
    Ranking,
    RankTask,
)
from verifiers.v1.utils.loaders import resolve_env_config


class VerdictBox:
    """A judge's box after it exited, serving back the verdict it wrote."""

    def __init__(self, verdict: dict) -> None:
        self.verdict = json.dumps(verdict).encode()

    async def read(self, path: str) -> bytes:
        assert path == VERDICT_FILE
        return self.verdict


def make_env(**overrides) -> BestOfNJudgeEnv:
    data = {
        "id": "best-of-n-judge",
        "n": 3,
        "taskset": {"id": "reverse-text"},
        "agent": {"harness": {"id": "null"}},
        "judge": {"harness": {"id": "bash"}},
        **overrides,
    }
    return BestOfNJudgeEnv(resolve_env_config(data))


def attempt(make_trace, reward: float, *, ok: bool = True) -> vf.Trace:
    trace = make_trace()
    trace.ok = ok
    trace.record_reward("correct", reward)
    return trace


class OracleJudge:
    """Ranks attempts by their task reward (ties shared), through the real
    `RankTask.finalize`, and remembers each task it was handed."""

    def __init__(self) -> None:
        self.tasks: list[RankTask] = []

    async def run(self, task: RankTask) -> vf.Trace:
        self.tasks.append(task)
        by_reward: dict[float, list[str]] = {}
        for label, trace in task.labels.items():
            by_reward.setdefault(trace.reward, []).append(label)
        verdict = {
            "ranking": [by_reward[r] for r in sorted(by_reward, reverse=True)],
            "reasons": {label: "checked it" for label in task.labels},
        }
        trace = vf.Trace(
            agent=vf.AgentInfo(config=vf.AgentConfig(), name="judge"),
            task=vf.TraceTask(type="RankTask", data=task.data),
        )
        await task.finalize(trace, VerdictBox(verdict))
        trace.ok = True
        return trace


class ReplayAgent:
    def __init__(self, traces: list[vf.Trace]) -> None:
        self.traces = iter(traces)

    async def run(self, task, collect_artifacts: bool = False) -> vf.Trace:
        return next(self.traces)


def test_rank_scores_count_wins_and_half_ties() -> None:
    assert Ranking(ranking=[["A"], ["B"], ["C"]], reasons={}).scores() == {
        "A": 1.0,
        "B": 0.5,
        "C": 0.0,
    }
    tied = Ranking(ranking=[["A", "B"], ["C"], ["D"]], reasons={}).scores()
    assert tied == {"A": 5 / 6, "B": 5 / 6, "C": 1 / 3, "D": 0.0}
    assert set(Ranking(ranking=[["A", "B", "C"]], reasons={}).scores().values()) == {
        0.5
    }


@pytest.mark.parametrize(
    "ranking,reasons",
    [
        ([["A"], ["B"]], {"A": "x", "B": "y"}),  # C missing
        ([["A", "A"], ["B", "C"]], {"A": "x", "B": "y", "C": "z"}),  # A twice
        ([["A"], [], ["B", "C"]], {"A": "x", "B": "y", "C": "z"}),  # empty tier
        ([["A"], ["B"], ["C"]], {"A": "x", "B": "y"}),  # C has no reason
    ],
)
def test_ranking_must_cover_every_label_once(ranking, reasons) -> None:
    with pytest.raises(ValueError):
        Ranking(ranking=ranking, reasons=reasons).check(["A", "B", "C"])


def test_construction_refuses_impossible_judges(tmp_path) -> None:
    with pytest.raises(ValueError, match="execute code"):
        make_env(judge={"harness": {"id": "null"}})
    with pytest.raises(ValueError, match="container judge runtime"):
        make_env(judge={"harness": {"id": "bash"}, "runtime": {"type": "subprocess"}})
    with pytest.raises(FileNotFoundError):
        make_env(task={"prompt": {"path": str(tmp_path / "missing.md")}})
    with pytest.raises(ValueError):
        make_env(n=1)


async def test_run_averages_the_ensemble_and_weighs_the_task(make_trace):
    env = make_env(judges=2, score={"task_weight": 0.5, "judge_weight": 2.0})
    attempts = [attempt(make_trace, r) for r in (1.0, 0.0, 1.0)]
    judge = OracleJudge()
    agents = SimpleNamespace(agent=ReplayAgent(attempts), judge=judge)
    await env.run(vf.Task(vf.TaskData(prompt="go")), agents)

    assert len(judge.tasks) == 2
    for task in judge.tasks:
        assert {t.id for t in task.labels.values()} == {t.id for t in attempts}
    for trace in attempts:
        assert trace.metrics["task_reward"] in (0.0, 1.0)
        assert trace.rewards["correct"].weight == 0.5
    assert [t.rewards["judge"] for t in attempts] == [
        vf.Reward(score=0.75, weight=2.0),
        vf.Reward(score=0.0, weight=2.0),
        vf.Reward(score=0.75, weight=2.0),
    ]

    episode = vf.Episode(
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="go")),
        traces=attempts,
    )
    await env.finalize(vf.Task(vf.TaskData(prompt="go")), episode)
    assert [t.metrics["best"] for t in attempts] == [1.0, 0.0, 1.0]
    assert all(t.metrics["pass_at_n"] == 1.0 for t in attempts)


async def test_a_failed_attempt_skips_judging(make_trace):
    env = make_env()
    attempts = [
        attempt(make_trace, 1.0),
        attempt(make_trace, 0.0, ok=False),
        attempt(make_trace, 0.0),
    ]
    judge = OracleJudge()
    agents = SimpleNamespace(agent=ReplayAgent(attempts), judge=judge)
    await env.run(vf.Task(vf.TaskData(prompt="go")), agents)
    assert judge.tasks == []
    assert all("judge" not in t.rewards for t in attempts)
