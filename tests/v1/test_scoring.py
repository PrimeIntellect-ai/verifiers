import asyncio

import pytest

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.session import RolloutLimits
from verifiers.v1.types import AssistantMessage, UserMessage
from verifiers.v1.utils.loaders import load_harness

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
        stops={"single_turn": vf.DecoratedFunctionConfig(fn=f"{fns}:two_turns")},
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
    (stop,) = task.hooks("stop")
    assert stop.__name__ == "single_turn"
    assert not await stop(trace)

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
    await task.score(trace)
    assert set(trace.rewards) == {"fmt", "lcs"}


async def test_rollout_close_waits_for_both_scorers_before_teardown() -> None:
    class FakeRuntime:
        stopped = False

        async def stop(self) -> None:
            self.stopped = True

    task = vf.Task(vf.TaskData(idx=0, prompt="hi"))
    task_started = asyncio.Event()
    release_task = asyncio.Event()
    harness_started = asyncio.Event()
    harness_failed = asyncio.Event()
    allow_harness_failure = asyncio.Event()
    observation: dict[str, bool] = {}

    async def slow_score(trace, runtime) -> None:
        task_started.set()
        await release_task.wait()
        observation.update(
            runtime_stopped=runtime.stopped,
            trace_completed=trace.is_completed,
        )
        trace.record_reward("task", 1.0)

    task.score = slow_score
    harness = load_harness(HarnessConfig(id="null"))

    async def failing_score(trace, runtime) -> None:
        harness_started.set()
        await allow_harness_failure.wait()
        harness_failed.set()
        raise RuntimeError("judge call failed")

    harness.score = failing_score
    run = Rollout(
        task=task,
        agent_config=vf.AgentConfig(model="m"),
        harness=harness,
        ctx=vf.ModelContext(model="m", client=vf.EvalClientConfig()),
        runtime_config=vf.SubprocessConfig(),
        timeouts=RolloutTimeouts(),
        limits=RolloutLimits(),
    )
    run._opened = True
    runtime = FakeRuntime()
    run.runtime = runtime

    close_task = asyncio.create_task(run.close())
    await task_started.wait()
    await harness_started.wait()
    allow_harness_failure.set()
    await harness_failed.wait()
    await asyncio.sleep(0)

    assert not close_task.done()
    assert not runtime.stopped

    release_task.set()
    trace = await close_task

    assert observation == {"runtime_stopped": False, "trace_completed": False}
    assert trace.rewards["task"].score == 1.0
    assert trace.is_completed
    assert not trace.ok
    assert trace.last_error is not None
    assert "judge call failed" in trace.last_error.message
    assert runtime.stopped


def test_compare_stdout_results_accepts_token_equal_text() -> None:
    assert vf.compare_stdout_results("hello   world\n", "hello world\n")


def test_compare_stdout_results_keeps_numeric_tolerance() -> None:
    assert vf.compare_stdout_results("1.0001 2.0\n", "1.0002 2.0009\n")


def test_parse_pytest_outcomes_strips_xfail_xpass_reasons() -> None:
    output = (
        "XFAIL tests/test_mod.py::test_xfail - known bug - still tracked\n"
        "XPASS tests/test_mod.py::test_xpass - always xfail - unexpectedly passed\n"
        "FAILED tests/test_mod.py::test_param[a - b] - assert left - right\n"
        "PASSED tests/test_mod.py::test_ok"
    )

    assert vf.parse_pytest_outcomes(output) == {
        "tests/test_mod.py::test_xfail": "XFAIL",
        "tests/test_mod.py::test_xpass": "XPASS",
        "tests/test_mod.py::test_param[a - b]": "FAILED",
        "tests/test_mod.py::test_ok": "PASSED",
    }


def test_parse_judge_choice_prefers_final_marker_then_boxed() -> None:
    assert (
        vf.parse_judge_choice(
            "Draft: \\boxed{A}\nFinal Judgment: B", choices=("A", "B")
        )
        == "B"
    )
    assert (
        vf.parse_judge_choice(
            "Draft: \\boxed{A}\nFinal Judgment:\nReasoning mentions B",
            choices=("A", "B"),
        )
        == "A"
    )
    assert (
        vf.parse_judge_choice(
            "Draft: \\boxed{B}\nFinal Judgment: This is a \\boxed{B}",
            choices=("A", "B"),
        )
        == "B"
    )
    assert (
        vf.parse_judge_choice(
            "Final Judgment: A better choice is \\boxed{B}", choices=("A", "B")
        )
        == "A"
    )
    assert (
        vf.parse_judge_choice(
            "Final Judgment: \\boxed{A}\nFinal Judgment: B", choices=("A", "B")
        )
        == "B"
    )
    assert (
        vf.parse_judge_choice("Draft: \\boxed{A}\nAnswer: B", choices=("A", "B")) == "A"
    )
