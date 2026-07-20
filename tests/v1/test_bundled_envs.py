"""--env.id: the env as its own plugin axis, and the bundled envs' pure logic."""

import asyncio

import pytest

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.envs.best_of_n import BestOfNEnv, BestOfNEnvConfig
from verifiers.v1.envs.agentic_judge import AgenticJudgeEnvConfig
from verifiers.v1.judges.rubric import RubricJudgeConfig
from verifiers.v1.judges.score import ScoreJudge
from verifiers.v1.trace import Trace, TraceTask


def test_env_id_resolves_bundled():
    assert vf.environment_class("", "best-of-n") is BestOfNEnv
    assert vf.env_config_type("", "best-of-n") is BestOfNEnvConfig


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
    config = EvalConfig(env={"id": "best-of-n", "taskset": {"id": "echo-v1"}, "n": 2})
    assert isinstance(config.env, BestOfNEnvConfig) and config.env.n == 2
    assert config.env_id == "best-of-n+echo-v1"  # runs stay distinguishable
    # Round-trip (config.toml, resume): the id re-narrows the field.
    rebuilt = EvalConfig.model_validate(config.model_dump(mode="json"))
    assert isinstance(rebuilt.env, BestOfNEnvConfig) and rebuilt.env.n == 2


def test_load_environment_honors_env_id():
    config = EvalConfig(env={"id": "best-of-n", "taskset": {"id": "echo-v1"}, "n": 3})
    env = vf.load_environment(config.env)
    assert isinstance(env, BestOfNEnv)
    assert set(env.roles()) == {"solver"}


def test_minted_task_roles_pair_with_tool_tasksets():
    """A role whose tasks the env mints itself declares its own needs: the agentic
    judge env loads over a tool-declaring taskset — the judge's `mcp=False` waives
    the taskset's MCP need — while the SOLVER role still validates against the
    dataset."""
    env = vf.load_environment(
        vf.resolve_env_config(
            {
                "id": "agentic-judge",
                "taskset": {"id": "echo-tool-v1"},
                "judge": {"harness": {"runtime": {"type": "docker"}}},
            }
        )
    )
    assert env._role_needs_mcp["judge"] is False
    assert env._role_needs_mcp["solver"]
    # The dataset-playing role keeps failing loudly: a tool taskset on a harness
    # that can't mount MCP is still an impossible pairing.
    with pytest.raises(ValueError, match="role 'solver' plays tasks with MCP"):
        vf.load_environment(
            vf.resolve_env_config(
                {
                    "id": "best-of-n",
                    "taskset": {"id": "echo-tool-v1"},
                    "solver": {"harness": {"id": "terminus-2"}},
                }
            )
        )


def test_agentic_judge_is_sandboxed():
    """The agentic judge is never played on the host: its role's container need is
    STATIC (no mode-switching) — a judge resolving to the subprocess runtime refuses
    at construction, and the fix is the judge seat's own runtime pin. A tool-less
    judge harness belongs on `judge`."""
    with pytest.raises(ValueError, match="role 'judge' plays tasks that need a"):
        vf.load_environment(
            vf.resolve_env_config({"id": "agentic-judge", "taskset": {"id": "echo-v1"}})
        )
    env = vf.load_environment(
        vf.resolve_env_config(
            {
                "id": "agentic-judge",
                "taskset": {"id": "echo-v1"},
                "judge": {"harness": {"runtime": {"type": "docker"}}},
            }
        )
    )
    judge = env._roles["judge"]
    assert judge.container is True
    assert env._harnesses["judge"].config.runtime.type == "docker"
    assert env._harnesses["solver"].config.runtime.type == "subprocess"  # unpinned
    # The env-server config round-trip resolves to the same shape.
    rebuilt = vf.load_environment(
        vf.resolve_env_config(env.config.model_dump(mode="json"))
    )
    assert rebuilt._roles["judge"].container is True
    # A tool-less judge harness is refused: a verdict that needs no execution is
    # a plugged judge (env.taskset.task.judges), not an agent.
    with pytest.raises(ValueError, match="plugged judge"):
        vf.load_environment(
            vf.resolve_env_config(
                {
                    "id": "agentic-judge",
                    "taskset": {"id": "echo-v1"},
                    "judge": {"harness": {"id": "null", "runtime": {"type": "docker"}}},
                }
            )
        )


def test_roles_are_always_roles():
    """`SingleAgentEnv`'s seat is one dataset-playing `agent` role; a roles()
    override handing back a bare AgentConfig gets a wrap-it error."""
    env = vf.SingleAgentEnv(_bundled_config())
    (name,), (role,) = env._roles.keys(), env._roles.values()
    assert name == "agent" and role.mcp is None and role.container is None

    class Bare(vf.Environment):
        def roles(self):
            return {"solo": vf.AgentConfig()}

        async def rollout(self, task, agents):
            return {}

    with pytest.raises(TypeError, match="wrap it: vf.Role"):
        Bare(_bundled_config())


def _bundled_config() -> vf.SingleAgentEnvConfig:
    return vf.SingleAgentEnvConfig(taskset={"id": "echo-v1"})


def test_paired_env_seats_pin_their_own_harness():
    """There is no run-level harness to inherit: a seat pin runs that seat on its
    own harness, and an unpinned seat runs the taskset's default."""
    env = vf.load_environment(
        vf.resolve_env_config(
            {
                "id": "best-of-n",
                "taskset": {"id": "echo-v1"},
                "solver": {"harness": {"id": "null"}},
            }
        )
    )
    assert env._harnesses["solver"].config.id == "null"
    judged = vf.load_environment(
        vf.resolve_env_config(
            {
                "id": "agentic-judge",
                "taskset": {"id": "echo-v1"},
                "judge": {"harness": {"runtime": {"type": "docker"}}},
            }
        )
    )
    assert judged._harnesses["solver"].config.id == "bash"  # taskset's default
    assert judged._harnesses["judge"].config.runtime.type == "docker"  # the pin


def _scored_trace(reward: float) -> Trace:
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")))
    trace.record_reward("task", reward)
    return trace


def test_best_of_n_sibling_scoring():
    env = vf.load_environment(
        vf.resolve_env_config({"id": "best-of-n", "taskset": {"id": "echo-v1"}, "n": 2})
    )
    traces = [_scored_trace(0.4), _scored_trace(1.0)]
    task = vf.Task(vf.TaskData(idx=0, prompt="hi"))
    asyncio.run(env.score(task, {"solver": traces}))
    assert [t.metrics["best"] for t in traces] == [0.0, 1.0]
    assert all(t.metrics["pass_at_n"] == 1.0 for t in traces)

    misses = [_scored_trace(0.2), _scored_trace(0.2)]
    asyncio.run(env.score(task, {"solver": misses}))
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
    assert _judged("SCORE: 7.5") == 0.75  # a decimal is a verdict, not a 0.7
    with pytest.raises(ValueError, match="no 'SCORE:"):
        _judged("I refuse to grade this.")  # a judge failure, never a 0
    with pytest.raises(ValueError, match="off the 0-10 scale"):
        _judged("SCORE: 95")  # a judge on its own scale must not clamp to full marks


def test_judge_spec_resolves_like_a_judges_entry():
    """The env's verdict spec rides the judge-plugin registry: an explicit id swaps
    (and narrows) it; a partial override tunes the default without resetting it."""
    swapped = AgenticJudgeEnvConfig.model_validate(
        {"spec": {"id": "rubric", "path": "grading.toml"}}
    )
    assert isinstance(swapped.spec, RubricJudgeConfig)
    assert swapped.spec.view == "full_trace"  # the rubric's own default
    tuned = AgenticJudgeEnvConfig.model_validate({"spec": {"view": "full_trace"}})
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


def _reply_trace(reply: str, role: str | None = None) -> Trace:
    trace = Trace(
        task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")), role=role
    )
    graph.prepare_turn(trace, [vf.UserMessage(content="q")]).commit(
        vf.Response(
            id="",
            created=0,
            model="test",
            message=vf.AssistantMessage(content=reply),
            finish_reason="stop",
        )
    )
    return trace


def _golf_trace(reply: str, passed: float, latency: float) -> Trace:
    trace = _reply_trace(reply)
    trace.record_metrics({"passed": passed, "latency": latency})
    return trace


def test_code_golf_only_passing_attempts_compete():
    """The sibling comparisons rank passing attempts only: a short-but-wrong
    program must not beat a long-but-right sibling, and an all-failed group
    pays nobody."""
    env = vf.load_environment(
        vf.resolve_env_config({"taskset": {"id": "code-golf-v1"}})
    )
    task = vf.Task(vf.TaskData(idx=0, prompt="q"))
    wrong = _golf_trace("```python\nx\n```", passed=0.0, latency=0.01)
    right = _golf_trace(
        "```python\nprint(sum(range(101)))\n```", passed=1.0, latency=0.9
    )
    asyncio.run(env.score(task, {"golfer": [wrong, right]}))
    assert wrong.rewards["most_concise"] == 0.0
    assert wrong.rewards["fastest"] == 0.0
    assert right.rewards["most_concise"] == 0.5  # the sole passing attempt (weighted)
    assert right.rewards["fastest"] == 0.5

    misses = [
        _golf_trace("```python\nx\n```", passed=0.0, latency=0.01),
        _golf_trace("no code at all", passed=0.0, latency=0.02),
    ]
    asyncio.run(env.score(task, {"golfer": misses}))
    assert all(t.rewards["most_concise"] == t.rewards["fastest"] == 0.0 for t in misses)


def test_solve_task_minted_from_the_proposer_contract():
    from proposer_solver_v1.taskset import SolveTask

    minted = SolveTask.from_trace(
        _reply_trace('Verified.\n{"problem": "How many cats?", "answer": 12}')
    )
    assert minted.data.answer == "12"
    assert minted.data.prompt.startswith("How many cats?")


def test_solve_task_refuses_off_contract_answers():
    """Off-contract output raises (the env-rollout fails, retryable): a float or
    bool answer must not be coerced into a ground truth."""
    from proposer_solver_v1.taskset import SolveTask

    with pytest.raises(ValueError, match="JSON integer"):
        SolveTask.from_trace(_reply_trace('{"problem": "p", "answer": 4.9}'))
    with pytest.raises(ValueError, match="JSON integer"):
        SolveTask.from_trace(_reply_trace('{"problem": "p", "answer": true}'))
    with pytest.raises(ValueError, match="JSON contract"):
        SolveTask.from_trace(_reply_trace("no contract here"))


def test_proposer_learnability_peaks_at_half():
    """The curriculum signal is 4p(1-p) over the solver seat: 0 when the problem
    is impossible or trivial for the solvers, 1 when half crack it."""
    env = vf.load_environment(
        vf.resolve_env_config({"taskset": {"id": "proposer-solver-v1"}})
    )
    task = vf.Task(vf.TaskData(idx=0, prompt="q"))

    def score(hits: int, n: int = 4) -> Trace:
        proposer = _reply_trace("proposed", role="proposer")
        solvers = [_reply_trace("42", role="solver") for _ in range(n)]
        for solver in solvers[:hits]:
            solver.record_reward("correct", 1.0)
        asyncio.run(env.score(task, {"proposer": proposer, "solver": solvers}))
        return proposer

    assert score(0).rewards["learnability"] == 0.0
    assert score(2).rewards["learnability"] == 1.0
    assert score(4).rewards["learnability"] == 0.0
    assert score(1).metrics["solve_rate"] == 0.25


def test_solve_task_contract_tolerates_latex_escapes():
    """Math proposers write LaTeX inside the JSON contract; the parse doubles the
    off-spec backslashes instead of failing the episode."""
    from proposer_solver_v1.taskset import SolveTask

    trace = _reply_trace(
        'Here it is:\n{"problem": "Let \\( S \\) be a set with \\\\leq 5 elements.", "answer": 3}'
    )
    task = SolveTask.from_trace(trace)
    assert task.data.answer == "3"
    assert "\\( S \\)" in task.data.prompt
