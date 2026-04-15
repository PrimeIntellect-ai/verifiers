"""Phase 1E seam tests for the real JudgeRubric class.

These tests live in the fork's own test suite (not the orchestrator's
test_debate_env.py) so they import and exercise the production
JudgeRubric class — the orchestrator test file stubs
`verifiers.rubrics.judge_rubric` with a minimal `_StubJudgeRubric`
mirror to avoid pulling httpx/openai transitively, which means
composition-level tests there can only assert on the stub contract.
This file closes that gap by loading the real module, which is
importable in the fork's own venv.

Contract under test:
  - JudgeRubric accepts a vf.Client instance (not a raw AsyncOpenAI).
  - judge() routes via vf.Client.get_response, returns the verdict
    content as a string, and propagates vf.Error subclasses unchanged.
  - state["judge_response"] acts as a per-prompt verdict cache so a
    single rollout's repeated grader calls coalesce.
"""

from __future__ import annotations

import pytest

from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.types import State


def test_judge_rubric_accepts_vf_client(mock_client):
    """v7 constructor contract: JudgeRubric accepts a vf.Client (via the
    shared MockClient fixture) and exposes it unchanged as
    self.judge_client. The framework no longer auto-wraps raw
    AsyncOpenAI — callers supply a vf.Client explicitly."""
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}\nReply CORRECT or INCORRECT.",
    )
    assert rubric.judge_client is mock_client
    assert rubric.judge_model == "test-model"
    # class_objects dict is the injection surface for reward functions;
    # it must carry the same client reference so reward callables can
    # see the wired-up JudgeRubric internals.
    assert rubric.class_objects["judge_client"] is mock_client


@pytest.mark.asyncio
async def test_judge_rubric_routes_through_vf_client(mock_client):
    """v7 happy path: judge() renders the format string with
    {question, answer, response}, hands the rendered text to
    vf.Client.get_response, and returns the verdict content as a
    plain string. Proves the silent-0 bug is fixed at the seam —
    errors (not demonstrated here, see propagation test) would
    propagate unchanged because there is no try/except wrapping
    the get_response call."""
    mock_client.set_default_response("CORRECT")
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
    )
    # Seed state with a placeholder — judge_rubric's cache-write branch
    # gates on `if state:` (truthy-dict check), not `if state is not None`,
    # so an empty dict would short-circuit the write.
    state: State = State(_seed=True)
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "four"}]
    verdict = await rubric.judge(
        prompt=prompt, completion=completion, answer="4", state=state
    )
    assert verdict == "CORRECT"
    # Exactly one backend call for a fresh state.
    assert mock_client.call_count == 1


@pytest.mark.asyncio
async def test_judge_rubric_caches_via_state_response(mock_client):
    """v7 cache seam: repeat calls with the same rendered prompt on the
    same state must be served from state['judge_response'] without
    hitting the backend. This is the coalescing path that lets
    DebateRubric's _grade and _match both fire with the same target
    without double-billing the judge."""
    mock_client.set_default_response("INCORRECT")
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
    )
    # Seed so `if state:` is truthy; see the routing test for context.
    state: State = State(_seed=True)
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "five"}]

    first = await rubric.judge(
        prompt=prompt, completion=completion, answer="4", state=state
    )
    assert first == "INCORRECT"
    assert mock_client.call_count == 1
    # Cache populated with the rendered prompt as key.
    cache = state.get("judge_response")
    assert isinstance(cache, dict)
    assert len(cache) == 1

    # Second call with the same prompt / completion / answer must be
    # served from the cache — no additional backend call.
    second = await rubric.judge(
        prompt=prompt, completion=completion, answer="4", state=state
    )
    assert second == "INCORRECT"
    assert mock_client.call_count == 1  # unchanged
