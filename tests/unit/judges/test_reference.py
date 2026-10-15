"""The built-in `reference` judge: verdict parsing, prompt inputs, and configurable labels."""

import pytest
from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.judge import JudgeResponse
from verifiers.v1.types import AssistantMessage, UserMessage


def test_reference_parse():
    judge = vf.ReferenceJudge()
    assert judge.parse(JudgeResponse(text="yes")) == 1.0
    assert judge.parse(JudgeResponse(text="Final answer: NO")) == 0.0
    # an unparseable verdict is a judge failure: raise (-> rollout error), don't score 0
    with pytest.raises(ValueError, match="no yes/no verdict"):
        judge.parse(JudgeResponse(text="gibberish"))


async def test_reference_score(fake_judge_model, make_trace):
    trace = make_trace()
    verdict = await vf.ReferenceJudge(vf.ReferenceJudgeConfig(id="reference")).score(
        trace.task.data, trace
    )
    assert verdict == 1.0
    assert (
        "Capital of France?" in fake_judge_model[0]
    )  # the task prompt is in the judge prompt
    assert len(trace.info["judge_calls"]) == 1  # the call is recorded onto the trace
    judge_record = trace.info["judge_calls"][0]
    assert judge_record["name"] == "reference"
    assert judge_record["request"]["model"] == "openai/gpt-6-luna"
    assert "Capital of France?" in judge_record["request"]["messages"][0]["content"]
    assert judge_record["response"]["message"] == {
        "role": "assistant",
        "content": "yes",
        "reasoning_content": None,
        "tool_calls": None,
        "provider_state": None,
    }
    assert judge_record["response"]["parsed"] == 1.0

    request_messages = [{"role": "user", "content": "original"}]
    request = {
        "model": "openai/gpt-5.4-nano",
        "messages": request_messages,
    }
    trace.record_judge_call(
        name="snapshot",
        request=request,
        response=JudgeResponse(text="yes"),
    )
    request_messages[0]["content"] = "mutated"
    assert (
        trace.info["judge_calls"][-1]["request"]["messages"][0]["content"] == "original"
    )

    override_trace = make_trace()
    await vf.ReferenceJudge().complete(
        "Judge this response.", trace=override_trace, model="openai/gpt-5.4-mini"
    )
    assert (
        override_trace.info["judge_calls"][0]["request"]["model"]
        == "openai/gpt-5.4-mini"
    )

    trace = make_trace(reply="It is Rome.")
    assert await vf.ReferenceJudge().score(trace.task.data, trace) == 0.0

    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(answer_field="gold"))
    with pytest.raises(ValueError, match="no 'gold' field"):  # misconfig raises, not 0
        await judge.score(trace.task.data, trace)


async def test_reference_score_messages_prompt(fake_judge_model, qa_data):
    # A Messages-form prompt still reaches the judge as text (via TaskData.prompt_text).
    from verifiers.v1.types import TextContentPart
    from verifiers.v1.types import UserMessage as UM

    task = qa_data(
        idx=0,
        prompt=[UM(content=[TextContentPart(text="Capital of France?")])],
        answer="Paris",
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=task),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(
                parent=0, message=AssistantMessage(content="It is Paris."), sampled=True
            ),
        ],
    )
    assert await vf.ReferenceJudge().score(task, trace) == 1.0
    assert "Capital of France?" in fake_judge_model[0]


async def test_reference_question_field(fake_judge_model):
    # question_field points {question} at a dedicated task field instead of the full prompt.
    class FieldTask(vf.TaskData):
        question: str = ""
        answer: str = ""

    task = FieldTask(
        idx=0,
        prompt="SYSTEM INSTRUCTIONS\n\nQuestion: Capital of France?",
        question="Capital of France?",
        answer="Paris",
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=task),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(
                parent=0, message=AssistantMessage(content="It is Paris."), sampled=True
            ),
        ],
    )
    await vf.ReferenceJudge(vf.ReferenceJudgeConfig(question_field="question")).score(
        task, trace
    )
    assert "Capital of France?" in fake_judge_model[0]
    assert "SYSTEM INSTRUCTIONS" not in fake_judge_model[0]

    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(question_field="typo"))
    with pytest.raises(ValueError, match="no 'typo' field"):
        await judge.score(task, trace)


async def test_reference_list_answer(fake_judge_model, make_trace):
    # A list-valued answer field is judged as multiple acceptable answers, one per line.
    class MultiTask(vf.TaskData):
        aliases: list[str] = Field(default_factory=list)

    task = MultiTask(idx=0, prompt="q?", aliases=["Paris", "Lutetia"])
    trace = make_trace()
    await vf.ReferenceJudge(vf.ReferenceJudgeConfig(answer_field="aliases")).score(
        task, trace
    )
    assert "Paris\nLutetia" in fake_judge_model[0]


async def test_reference_choices(fake_judge_model, make_trace):
    # Verdict labels are configurable; the positive (first) label scores 1.0 and the
    # default prompt asks for the configured labels (not a hardcoded yes/no).
    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(choices=("A", "B")))
    assert judge.parse(JudgeResponse(text="A")) == 1.0
    assert judge.parse(JudgeResponse(text="Final verdict: B")) == 0.0
    with pytest.raises(ValueError, match="no A/B verdict"):
        judge.parse(JudgeResponse(text="gibberish"))
    trace = make_trace()
    with pytest.raises(
        ValueError
    ):  # the yes-replying fake is now an unparseable verdict
        await judge.score(trace.task.data, trace)
    assert 'Respond either "A" or "B"' in fake_judge_model[0]
    # degenerate labels are a config error (duplicates would score every verdict 1.0)
    for choices in (("yes", "yes"), ("A", "a"), ("", "no")):
        with pytest.raises(ValueError, match="two distinct, non-empty"):
            vf.ReferenceJudgeConfig(choices=choices)
