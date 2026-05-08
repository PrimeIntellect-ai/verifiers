"""Tests for the math_group environment rubric behaviour."""
import os
import sys

import pytest
import verifiers as vf

# math_group is a standalone package under environments/; add it to the path
# so the import works without installing it.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "environments", "math_group"),
)
from math_group import load_environment  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
async def env_group():
    """Load the math_group EnvGroup once per module; teardown rubric executors after."""
    eg = load_environment()
    yield eg
    for env in eg.envs:
        await env.rubric.teardown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(make_input, completion: str, answer: str) -> vf.State:
    state = vf.State(
        input=make_input(
            prompt=[{"role": "user", "content": "Solve it."}],
            answer=answer,
        )
    )
    state["completion"] = completion
    state["trajectory"] = []
    state["timing"] = {
        "generation_ms": 0.0,
        "scoring_ms": 0.0,
        "total_ms": 0.0,
        "start_time": 0.0,
    }
    return state


# ---------------------------------------------------------------------------
# math sub-env: equivalent LaTeX forms must score 1.0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "completion,answer",
    [
        # display-fraction vs regular fraction — same value, different macro
        (r"\boxed{\frac{3}{4}}", r"\dfrac{3}{4}"),
        # fraction vs decimal
        (r"\boxed{\frac{1}{2}}", "0.5"),
        # commutativity
        (r"\boxed{1 + x}", "x + 1"),
        # decimal vs fraction (reversed direction)
        (r"\boxed{0.75}", r"\frac{3}{4}"),
    ],
    ids=[
        r"\frac{3}{4}==\dfrac{3}{4}",
        r"\frac{1}{2}==0.5",
        "1+x==x+1",
        r"0.75==\frac{3}{4}",
    ],
)
async def test_math_env_equivalent_forms_score_1(env_group, make_input, completion, answer):
    """Equivalent answers must receive full credit from the math sub-env rubric.

    MathRubric uses math_verify under the hood, so symbolic equivalence is
    respected rather than raw string equality.
    """
    state = _make_state(make_input, completion, answer)
    await env_group.env_map["math"].rubric.score_rollout(state)

    assert state["metrics"]["correct_answer"] == 1.0


# ---------------------------------------------------------------------------
# gsm8k sub-env: same behaviour required
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "completion,answer",
    [
        (r"\boxed{\frac{3}{4}}", r"\dfrac{3}{4}"),
        (r"\boxed{\frac{1}{2}}", "0.5"),
        (r"\boxed{1 + x}", "x + 1"),
        (r"\boxed{0.75}", r"\frac{3}{4}"),
    ],
    ids=[
        r"\frac{3}{4}==\dfrac{3}{4}",
        r"\frac{1}{2}==0.5",
        "1+x==x+1",
        r"0.75==\frac{3}{4}",
    ],
)
async def test_gsm8k_env_equivalent_forms_score_1(env_group, make_input, completion, answer):
    """gsm8k sub-env rubric must also accept equivalent LaTeX representations."""
    state = _make_state(make_input, completion, answer)
    await env_group.env_map["gsm8k"].rubric.score_rollout(state)

    assert state["metrics"]["correct_answer"] == 1.0


# ---------------------------------------------------------------------------
# Format reward contributes to state["reward"] for the math sub-env
# ---------------------------------------------------------------------------

async def test_math_format_reward_adds_to_total(env_group, make_input):
    """A correctly boxed answer must earn the 0.2 format bonus on top of the 1.0 answer score.

    The math rubric is: weight 1.0 (correct_answer) + weight 0.2 (format_reward).
    A \boxed{} completion that is correct should therefore yield reward > 1.0.
    """
    state = _make_state(make_input, r"\boxed{\frac{1}{2}}", "0.5")
    await env_group.env_map["math"].rubric.score_rollout(state)

    assert state["metrics"]["correct_answer"] == 1.0
    assert state["reward"] > 1.0  # format bonus applied


# ---------------------------------------------------------------------------
# Regression: wrong answers must still score 0
# ---------------------------------------------------------------------------

async def test_wrong_answer_scores_0(env_group, make_input):
    """A clearly wrong answer must score 0 — math-awareness must not over-accept."""
    state = _make_state(make_input, r"\boxed{2}", "3")
    await env_group.env_map["math"].rubric.score_rollout(state)

    assert state["metrics"]["correct_answer"] == 0.0


async def test_gsm8k_wrong_answer_scores_0(env_group, make_input):
    """Same regression check for the gsm8k sub-env."""
    state = _make_state(make_input, r"\boxed{42}", "7")
    await env_group.env_map["gsm8k"].rubric.score_rollout(state)

    assert state["metrics"]["correct_answer"] == 0.0


# ---------------------------------------------------------------------------
# EnvGroupRubric teardown must propagate to child rubrics
# ---------------------------------------------------------------------------

async def test_env_group_teardown_propagates_to_child_rubrics():
    """EnvGroup.rubric.teardown() must shut down child MathRubric executors.

    MathRubric spawns a ProcessPoolExecutor per instance. If EnvGroupRubric
    does not propagate teardown to the child RubricGroups, those workers leak
    when the EnvGroup is torn down.

    Chain: EnvGroupRubric → RubricGroup.teardown() → MathRubric.teardown()
    """
    eg = load_environment()
    await eg.rubric.teardown()
    for env in eg.envs:
        math_rubric = next(r for r in env.rubric.rubrics if hasattr(r, "executor"))
        with pytest.raises(RuntimeError, match="shutdown"):
            math_rubric.executor.submit(lambda: None)
