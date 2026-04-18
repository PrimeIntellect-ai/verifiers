"""Seam tests for the real JudgeRubric class.

These tests live in the fork's own test suite (not the orchestrator's
test_debate_env.py) so they import and exercise the production
JudgeRubric class — the orchestrator test file stubs
``verifiers.rubrics.judge_rubric`` with a minimal ``_StubJudgeRubric``
mirror to avoid pulling httpx/openai transitively, which means
composition-level tests there can only assert on the stub contract.
This file closes that gap by loading the real module, which is
importable in the fork's own venv.

Contract under test:
  - ``judge()`` renders the ``judge_prompt`` format string with
    ``{question}``, ``{answer}``, ``{response}`` substitutions drawn
    from the caller-supplied arguments, and routes the rendered text
    through ``vf.Client.get_response`` with the configured ``model``
    and normalized ``sampling_args``. ``state`` is threaded through as
    a kwarg.
  - ``state["judge_response"]`` acts as a per-rendered-prompt verdict
    cache so a single rollout's repeated grader calls coalesce.
"""

from __future__ import annotations

import pytest

from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.types import State


@pytest.mark.asyncio
async def test_judge_rubric_routes_rendered_prompt_model_and_args(mock_client):
    """The composition seam. Asserts (a) the rendered prompt actually
    carries the substituted {question}/{answer}/{response} fragments, not
    the raw format string, (b) the configured ``judge_model`` reaches the
    client, (c) ``judge_sampling_args`` are forwarded with the
    ``max_tokens -> max_completion_tokens`` normalization, (d) ``state``
    is threaded through as a kwarg.

    A bug in the template renderer, the sampling-args normalization, or
    the state passthrough would slip past a verdict-only assertion. This
    test catches all four at the seam.
    """
    mock_client.set_default_response("CORRECT")
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt=(
            "Q: {question}\n"
            "GT: {answer}\n"
            "Resp: {response}\n"
            "Reply CORRECT or INCORRECT."
        ),
        # Use the legacy max_tokens form so the normalization path runs.
        judge_sampling_args={"temperature": 0.0, "max_tokens": 64},
    )
    state: State = State(_seed=True)
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "four"}]

    verdict = await rubric.judge(
        prompt=prompt, completion=completion, answer="4", state=state
    )

    assert verdict == "CORRECT"
    assert mock_client.call_count == 1

    # --- (a) the rendered prompt carries the substituted fragments ---
    routed = mock_client.last_call_kwargs
    routed_prompt = routed["prompt"]
    assert isinstance(routed_prompt, list) and len(routed_prompt) == 1
    rendered = routed_prompt[0]
    # Real client path wraps in UserMessage, but MockClient receives it
    # verbatim. Pull content via dict OR attr to stay stable across
    # Pydantic/dict forms.
    rendered_content = (
        rendered.get("content") if isinstance(rendered, dict)
        else getattr(rendered, "content", "")
    )
    assert "Q: What is 2+2?" in rendered_content
    assert "GT: 4" in rendered_content
    assert "Resp: four" in rendered_content
    # Confirm renderer actually substituted rather than leaving braces.
    assert "{question}" not in rendered_content
    assert "{answer}" not in rendered_content
    assert "{response}" not in rendered_content

    # --- (b) model passthrough ---
    assert routed["model"] == "test-model"

    # --- (c) sampling args normalized: max_tokens -> max_completion_tokens,
    #         None-valued keys stripped ---
    sampling_args = routed["sampling_args"]
    assert sampling_args == {"temperature": 0.0, "max_completion_tokens": 64}
    assert "max_tokens" not in sampling_args

    # --- (d) state threaded through as a kwarg ---
    assert routed.get("state") is state


@pytest.mark.asyncio
async def test_judge_rubric_strips_none_sampling_arg_values(mock_client):
    """None-valued ``max_completion_tokens`` / ``max_tokens`` must be
    dropped before routing — passing ``None`` through triggers a
    ``pydantic_core.ValidationError`` on real Chat Completions backends."""
    mock_client.set_default_response("YES")
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question} A: {answer} R: {response}",
        judge_sampling_args={
            "temperature": 0.0,
            "max_completion_tokens": None,
            "max_tokens": None,
            "top_p": None,
        },
    )
    state: State = State(_seed=True)
    await rubric.judge(
        prompt=[{"role": "user", "content": "q"}],
        completion=[{"role": "assistant", "content": "a"}],
        answer="a",
        state=state,
    )
    sampling_args = mock_client.last_call_kwargs["sampling_args"]
    assert sampling_args == {"temperature": 0.0}
    assert all(v is not None for v in sampling_args.values())


@pytest.mark.asyncio
async def test_judge_rubric_caches_via_state_response(mock_client):
    """Cache seam: repeat calls with the same rendered prompt on the
    same state must be served from ``state['judge_response']`` without
    hitting the backend. This is the coalescing path that lets
    DebateRubric's ``_grade`` and ``_match`` both fire with the same
    target without double-billing the judge."""
    mock_client.set_default_response("INCORRECT")
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
    )
    state: State = State(_seed=True)
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "five"}]

    first = await rubric.judge(
        prompt=prompt, completion=completion, answer="4", state=state
    )
    assert first == "INCORRECT"
    assert mock_client.call_count == 1
    cache = state.get("judge_response")
    assert isinstance(cache, dict) and len(cache) == 1

    second = await rubric.judge(
        prompt=prompt, completion=completion, answer="4", state=state
    )
    assert second == "INCORRECT"
    assert mock_client.call_count == 1  # served from cache
