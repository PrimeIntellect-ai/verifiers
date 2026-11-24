"""`Task` hooks: config-plugged functions merge with decorated ones, and scoring can be deferred."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage

PLUGGED_FNS_PY = """
from verifiers.v1 import Trace


async def reply_length(trace) -> float:
    return float(len(trace.last_reply or ""))


async def exact_match(task, trace) -> float:
    return float(task.answer in (trace.last_reply or ""))


async def marker(trace) -> float:
    return 0.125


async def two_turns(trace: Trace) -> bool:
    return trace.num_turns >= 2
"""


class HookData(vf.TaskData):
    answer: str = ""


class HookTask(vf.Task[HookData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.stop(skip_scoring=True)
    async def no_reply(self, trace: vf.Trace) -> bool:
        return trace.last_reply is None

    @vf.reward
    async def lcs(self, trace: vf.Trace) -> float:
        return 0.5

    @vf.reward(weight=0.0)
    async def fmt(self, trace: vf.Trace) -> float:
        return 1.0


async def test_config_plugged_fns_merge_and_override(tmp_path) -> None:
    fns = tmp_path / "fns.py"
    fns.write_text(PLUGGED_FNS_PY)
    config = vf.TaskConfig(
        stops={"single_turn": vf.StopFunctionConfig(fn=f"{fns}:two_turns")},
        metrics={"reply_length": vf.DecoratedFunctionConfig(fn=f"{fns}:reply_length")},
        rewards={
            "exact_match": vf.RewardFunctionConfig(fn=f"{fns}:exact_match", weight=0.5),
            "lcs": vf.RewardFunctionConfig(fn=f"{fns}:marker", weight=2.0),
            "fmt": vf.RewardFunctionConfig(weight=1.0),
        },
    )
    task = HookTask(HookData(idx=0, prompt="abc", answer="cba"), config)
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="HookTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="abc"), sampled=False),
            MessageNode(
                parent=0, message=AssistantMessage(content="cba"), sampled=True
            ),
        ],
    )

    # the plugged `single_turn` replaces the decorated one: no stop after one turn
    no_reply, stop = task.hooks("stop")
    assert stop.__name__ == "single_turn"
    assert not await stop(trace)
    assert no_reply.stop_skip_scoring and not stop.stop_skip_scoring

    await task.score(trace)
    # `lcs` scored by the plugged marker (config wins the name clash), config weight
    assert trace.rewards["lcs"].score == 0.125
    assert trace.rewards["lcs"].weight == 2.0
    assert trace.rewards["exact_match"].score == 1.0
    assert trace.rewards["exact_match"].weight == 0.5
    assert trace.metrics["reply_length"] == 3.0
    # `fmt` keeps the decorated body, only its weight is overridden (0.0 -> 1.0)
    assert trace.rewards["fmt"].score == 1.0
    assert trace.rewards["fmt"].weight == 1.0

    # a fn-less entry must name an existing decorated method
    config = vf.TaskConfig(rewards={"nope": vf.RewardFunctionConfig(weight=1.0)})
    with pytest.raises(ValueError, match="no @vf.reward method named 'nope'"):
        HookTask(HookData(idx=0, prompt="abc"), config).hooks("reward")


async def test_defer_scoring_defers_only_task_scoring() -> None:
    task = HookTask(HookData(idx=0, prompt="abc"))
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="HookTask", data=task.data),
    )

    await task.defer_scoring().score(trace)
    assert trace.rewards == {}
    invalid = trace.model_copy(deep=True)
    invalid.stop("format_violation")
    await task.score(invalid)
    assert invalid.rewards == {}
    assert invalid.reward == 0.0
    await task.score(trace)
    assert set(trace.rewards) == {"fmt", "lcs"}

    trace.stop("format_violation")
    trace.ok = True
    await task.rescore(trace)
    assert trace.rewards == {}
    assert trace.reward == 0.0


def test_plugged_stops_set_skip_scoring(tmp_path) -> None:
    fns = tmp_path / "fns.py"
    fns.write_text(PLUGGED_FNS_PY)
    data = HookData(idx=0, prompt="abc")

    def skips(stops: dict) -> dict[str, bool]:
        config = vf.TaskConfig(stops=stops)
        return {
            fn.__name__: fn.stop_skip_scoring
            for fn in HookTask(data, config).hooks("stop")
        }

    # a fresh plugged stop scores by default; a fn-less entry keeps or overrides the decorator's
    assert skips({"two": vf.StopFunctionConfig(fn=f"{fns}:two_turns")})["two"] is False
    assert skips(
        {"two": vf.StopFunctionConfig(fn=f"{fns}:two_turns", skip_scoring=True)}
    )["two"]
    assert skips({"no_reply": vf.StopFunctionConfig()})["no_reply"] is True
    assert (
        skips({"no_reply": vf.StopFunctionConfig(skip_scoring=False)})["no_reply"]
        is False
    )


def test_a_nan_reward_is_an_error() -> None:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="HookTask", data=HookData(idx=0, prompt="abc")),
    )
    with pytest.raises(ValueError, match="must be finite"):
        trace.record_reward("judge", float("nan"))


def test_failed_scoring_rolls_back_its_drop() -> None:
    from verifiers.v1.grading import scoring_transaction

    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="HookTask", data=HookData(idx=0, prompt="abc")),
    )
    with pytest.raises(RuntimeError), scoring_transaction(trace):
        trace.drop("judge: outcome=task_defect")
        raise RuntimeError("the judge failed")
    assert not trace.is_dropped


async def test_rollout_close_marks_trace_ok_before_finalize_and_reward_hooks() -> None:
    from types import SimpleNamespace

    from verifiers.v1.configs.scoring import ScoringConfig
    from verifiers.v1.rollout import Rollout, RolloutTimeouts

    observed: list[tuple[str, bool, bool]] = []

    class ErrorCheckingTask(vf.Task[HookData]):
        async def finalize(self, trace: vf.Trace) -> None:
            observed.append(("finalize", trace.ok, trace.has_error))

        @vf.reward
        async def correct(self, trace: vf.Trace) -> float:
            observed.append(("reward", trace.ok, trace.has_error))
            return 0.0 if trace.has_error else 1.0

    def build_rollout(task: vf.Task[HookData]) -> Rollout:
        run = object.__new__(Rollout)
        run.task = task
        run._scoring_task = task
        run._grader = None
        run._grading_config = vf.SubprocessConfig()
        run._scoring_policy = ScoringConfig()
        run._timeouts = RolloutTimeouts()
        run._on_score = None
        run._collect_artifacts = False
        run._skip_scoring_stops = set()
        run._opened = True
        run._closed = False
        run._failed = False
        run._failure = None
        run._borrowed_runtime = None
        run._harness_cleaned = True
        run._harness_session = None
        run._agent_runtime = None
        run._stack = SimpleNamespace(aclose=lambda: _noop())
        run.harness = SimpleNamespace(score=lambda trace, rt: _noop())
        run.runtime = SimpleNamespace(
            name="local",
            env={},
            with_env=lambda env: run.runtime,
            backend=SimpleNamespace(stopped=True),
        )
        run.deployment = SimpleNamespace(close=lambda: _noop())
        run.trace = vf.Trace(
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            task=vf.TraceTask(type=type(task).__name__, data=task.data),
            nodes=[
                MessageNode(
                    parent=None, message=UserMessage(content="q"), sampled=False
                ),
                MessageNode(
                    parent=0,
                    message=AssistantMessage(content="answer"),
                    sampled=True,
                ),
            ],
        )
        return run

    async def _noop() -> None:
        return None

    healthy = build_rollout(ErrorCheckingTask(HookData(idx=0, prompt="q")))
    trace = await healthy.close()
    assert observed == [("finalize", True, False), ("reward", True, False)]
    assert trace.ok and not trace.has_error and trace.is_completed
    assert trace.reward == 1.0

    class RaisingRewardTask(vf.Task[HookData]):
        @vf.reward
        async def broken(self, trace: vf.Trace) -> float:
            assert not trace.has_error
            raise RuntimeError("reward boom")

    failing_in_score = build_rollout(RaisingRewardTask(HookData(idx=1, prompt="q")))
    failed_trace = await failing_in_score.close()
    assert not failed_trace.ok and failed_trace.has_error and failed_trace.is_completed
    assert [e.type for e in failed_trace.errors] == ["TaskError"]

    pre_failed = build_rollout(ErrorCheckingTask(HookData(idx=2, prompt="q")))
    pre_failed.fail(RuntimeError("agent boom"))
    observed.clear()
    errored_trace = await pre_failed.close()
    assert observed == []
    assert (
        not errored_trace.ok and errored_trace.has_error and errored_trace.is_completed
    )
