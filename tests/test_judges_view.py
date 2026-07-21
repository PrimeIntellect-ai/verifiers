"""Tests for the Judges View capture path.

JudgeRubric should record each judge call into state["judges"], and
state_to_output should propagate that list into the RolloutOutput so it
reaches the platform without an opt-in state_columns declaration.
"""

import json

import pytest

from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.utils.save_utils import (
    make_serializable,
    states_to_outputs,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._response_text)


class _FakeChat:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions


class _FakeJudgeClient:
    def __init__(self, response_text: str = "yes") -> None:
        self.chat = _FakeChat(_FakeChatCompletions(response_text))


@pytest.mark.asyncio
async def test_judge_records_appended_to_state(make_state):
    client = _FakeJudgeClient(response_text="yes")
    rubric = JudgeRubric(judge_client=client, judge_model="fake-judge")
    state = make_state(
        prompt=[{"role": "user", "content": "What is 2+2?"}],
        completion=[{"role": "assistant", "content": "4"}],
        answer="4",
    )

    out = await rubric.judge(
        prompt=state["prompt"],
        completion=state["completion"],
        answer=state["answer"],
        state=state,
    )
    assert out == "yes"

    judges = state.get("judges")
    assert isinstance(judges, list) and len(judges) == 1
    record = judges[0]
    assert record["judge_output"] == "yes"
    assert record["model"] == "fake-judge"
    assert record["rubric"] == "JudgeRubric"
    assert isinstance(record["judge_input"], list)
    assert record["judge_input"][0]["role"] == "user"
    assert "What is 2+2?" in record["judge_input"][0]["content"]


@pytest.mark.asyncio
async def test_judge_records_distinguish_named_rubrics(make_state):
    correctness = JudgeRubric(
        judge_client=_FakeJudgeClient("yes"),
        judge_model="judge-a",
        name="correctness_judge",
    )
    style = JudgeRubric(
        judge_client=_FakeJudgeClient("no"),
        judge_model="judge-b",
        name="style_judge",
    )
    state = make_state(
        prompt=[{"role": "user", "content": "Q"}],
        completion=[{"role": "assistant", "content": "A"}],
        answer="A",
    )

    await correctness.judge(state["prompt"], state["completion"], state["answer"], state)
    await style.judge(state["prompt"], state["completion"], state["answer"], state)

    judges = state["judges"]
    assert [r["rubric"] for r in judges] == ["correctness_judge", "style_judge"]
    assert [r["model"] for r in judges] == ["judge-a", "judge-b"]


@pytest.mark.asyncio
async def test_state_to_output_propagates_judges(make_state):
    rubric = JudgeRubric(judge_client=_FakeJudgeClient("yes"), judge_model="fake-judge")
    state = make_state(
        prompt=[{"role": "user", "content": "Q"}],
        completion=[{"role": "assistant", "content": "A"}],
        answer="A",
    )
    await rubric.judge(state["prompt"], state["completion"], state["answer"], state)

    output = states_to_outputs([state], state_columns=[])[0]
    assert "judges" in output
    serialized = json.loads(json.dumps(output, default=make_serializable))
    assert serialized["judges"][0]["judge_output"] == "yes"


def test_state_to_output_omits_judges_when_absent(make_state):
    state = make_state()
    output = states_to_outputs([state], state_columns=[])[0]
    assert "judges" not in output
