"""--env.id: the env as its own plugin axis, and the bundled envs' pure logic."""

import asyncio

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.envs.best_of_n import BestOfNEnv, BestOfNParams
from verifiers.v1.envs.judge.env import JudgeParams
from verifiers.v1.judges.rubric import RubricJudgeConfig
from verifiers.v1.judges.score import ScoreJudge
from verifiers.v1.trace import Trace, TraceTask

from kuhn_poker_v1.taskset import LEGAL, TO_ACT, parse_action, payoff


def test_env_id_resolves_bundled():
    assert vf.environment_class("", "best-of-n") is BestOfNEnv
    assert vf.env_params_type("", "best-of-n") is BestOfNParams


def test_env_id_wins_over_taskset_export():
    """An explicit `--env.id` pairs its env with any taskset — including one that
    ships its own (the id is the escape hatch, the bundled env the default)."""
    assert vf.environment_class("duet-v1").__name__ == "DuetEnv"
    assert vf.environment_class("duet-v1", "best-of-n") is BestOfNEnv


def test_unknown_env_id_raises():
    """An explicit pairing must not silently fall back to the base env."""
    with pytest.raises(ModuleNotFoundError, match="environment"):
        vf.environment_class("echo-v1", "no-such-env")


def test_env_field_narrows_by_env_id():
    config = EvalConfig(taskset={"id": "echo-v1"}, env={"id": "best-of-n", "n": 2})
    assert isinstance(config.env, BestOfNParams) and config.env.n == 2
    assert config.env_id == "best-of-n+echo-v1"  # runs stay distinguishable
    # Round-trip (config.toml, resume): the id re-narrows the field.
    rebuilt = EvalConfig.model_validate(config.model_dump(mode="json"))
    assert isinstance(rebuilt.env, BestOfNParams) and rebuilt.env.n == 2


def test_load_environment_honors_env_id():
    config = EvalConfig(taskset={"id": "echo-v1"}, env={"id": "best-of-n", "n": 3})
    env = vf.load_environment(config)
    assert isinstance(env, BestOfNEnv)
    assert set(env.roles()) == {"solver"}


def test_bundled_env_solver_runs_the_runs_harness():
    """The axes stay orthogonal when paired: `--env.id best-of-n --harness.id X`
    seats the solver on X (an unpinned role late-binds to the run's harness), while
    a pinned role (the judge env's judge) keeps its own."""
    config = EvalConfig(
        taskset={"id": "echo-v1"}, harness={"id": "null"}, env={"id": "best-of-n"}
    )
    env = vf.load_environment(config)
    assert env._harnesses["solver"] is env.harness
    assert env.harness.config.id == "null"
    judged = vf.load_environment(
        EvalConfig(
            taskset={"id": "echo-v1"}, harness={"id": "null"}, env={"id": "judge"}
        )
    )
    assert judged._harnesses["solver"] is judged.harness
    assert judged._harnesses["judge"].config.id == "direct"  # the pin survives


def _scored_trace(reward: float) -> Trace:
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")))
    trace.record_reward("task", reward)
    return trace


def test_best_of_n_sibling_scoring():
    config = EvalConfig(taskset={"id": "echo-v1"}, env={"id": "best-of-n", "n": 2})
    env = vf.load_environment(config)
    traces = [_scored_trace(0.4), _scored_trace(1.0)]
    task = vf.Task(vf.TaskData(idx=0, prompt="hi"))
    asyncio.run(env.score(task, traces))
    assert [t.metrics["best"] for t in traces] == [0.0, 1.0]
    assert all(t.metrics["pass_at_n"] == 1.0 for t in traces)

    misses = [_scored_trace(0.2), _scored_trace(0.2)]
    asyncio.run(env.score(task, misses))
    # Ties share `best`; nothing reached the threshold.
    assert [t.metrics["best"] for t in misses] == [1.0, 1.0]
    assert all(t.metrics["pass_at_n"] == 0.0 for t in misses)


def _judged(reply: str) -> float:
    judge = ScoreJudge()
    task = vf.TaskData(idx=0, prompt="q")
    return judge.verdict(task, _scored_trace(0.0), reply)


def test_score_judge_verdict():
    assert _judged("SCORE: 7") == 0.7
    assert _judged("The answer is right.\nSCORE: 10") == 1.0
    assert _judged("SCORE: 3\n...revised...\nSCORE: 8") == 0.8  # last wins
    assert _judged("SCORE: 42") == 1.0  # clamped
    with pytest.raises(ValueError, match="no 'SCORE:"):
        _judged("I refuse to grade this.")  # a judge failure, never a 0


def test_judge_spec_resolves_like_a_judges_entry():
    """The env's verdict spec rides the judge-plugin registry: an explicit id swaps
    (and narrows) it; a partial override tunes the default without resetting it."""
    swapped = JudgeParams.model_validate(
        {"spec": {"id": "rubric", "path": "grading.toml"}}
    )
    assert isinstance(swapped.spec, RubricJudgeConfig)
    assert swapped.spec.view == "full_trace"  # the rubric's own default
    tuned = JudgeParams.model_validate({"spec": {"view": "full_trace"}})
    assert type(tuned.spec).__name__ == "ScoreJudgeConfig"
    assert tuned.spec.view == "full_trace"
    assert tuned.spec.name == "judge"  # the pinned reward key survives


def test_rubric_spec_agent_execution_round_trip(tmp_path):
    """The unification: the same rubric file drives an agent-executed judge —
    `render` carries the criteria + JSON contract, `verdict` parses the agent's
    reply into the identical per-criterion metrics and weighted total the plugged
    tier records."""
    import json

    from verifiers.v1.judges.rubric import RubricJudge

    rubric = tmp_path / "grading.json"
    rubric.write_text(
        json.dumps(
            {
                "criteria": [
                    {"name": "correct", "text": "Is it right?", "weight": 3.0},
                    {"name": "concise", "text": "Is it short?"},
                ]
            }
        )
    )
    judge = RubricJudge(RubricJudgeConfig(path=rubric))
    task = vf.TaskData(idx=0, prompt="What is 2+2?")
    trace = _scored_trace(0.0)
    prompt = judge.render(task, trace)
    assert "correct" in prompt and "concise" in prompt and "verdicts" in prompt
    reply = json.dumps(
        {
            "verdicts": [
                {"name": "correct", "reason": "it is", "verdict": "yes"},
                {"name": "concise", "reason": "it is not", "verdict": "no"},
            ]
        }
    )
    total = judge.verdict(task, trace, f"Here is my grading:\n{reply}")
    assert total == 0.75  # (3*1 + 1*0) / 4
    assert trace.metrics["rubric/correct"] == 1.0
    assert trace.metrics["rubric/concise"] == 0.0


def test_kuhn_payoffs_are_zero_sum_and_correct():
    assert payoff("check-check", ["K", "J"]) == 1
    assert payoff("check-check", ["J", "K"]) == -1
    assert payoff("bet-fold", ["J", "K"]) == 1  # folding surrenders the antes
    assert payoff("check-bet-fold", ["K", "J"]) == -1
    assert payoff("bet-call", ["K", "Q"]) == 2
    assert payoff("bet-call", ["Q", "K"]) == -2
    assert payoff("check-bet-call", ["J", "Q"]) == -2


def test_kuhn_state_machine_is_closed():
    """Every non-terminal history offers legal actions; every extension is either
    another decision point or a payoff-defined terminal."""
    for history, legal in LEGAL.items():
        assert history in TO_ACT
        for action in legal:
            extended = f"{history}-{action}" if history else action
            if extended not in TO_ACT:
                payoff(extended, ["K", "J"])  # raises KeyError if unmapped


def test_kuhn_parse_action():
    assert parse_action("I will [bet]!", ("check", "bet")) == "bet"
    assert parse_action("[check] no wait, [bet]", ("check", "bet")) == "bet"
    assert parse_action("[BET]", ("check", "bet")) == "bet"
    assert parse_action("[raise]", ("check", "bet")) is None
    assert parse_action("bet", ("check", "bet")) is None  # brackets required
