"""--env.id: the env as its own plugin axis, and the bundled envs' pure logic."""

import asyncio

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.envs.best_of_n import BestOfNEnv, BestOfNParams
from verifiers.v1.envs.judge.env import parse_score
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


def test_judge_parse_score():
    assert parse_score("SCORE: 7") == 0.7
    assert parse_score("The answer is right.\nSCORE: 10") == 1.0
    assert parse_score("SCORE: 3\n...revised...\nSCORE: 8") == 0.8  # last wins
    assert parse_score("SCORE: 42") == 1.0  # clamped
    assert parse_score("I refuse to grade this.") == 0.0
    assert parse_score(None) == 0.0


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
