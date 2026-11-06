"""The built-in `decision` judge: its graded score, and any question the decision model answers."""

import json

import httpx
import pytest
from openai import AsyncOpenAI

import verifiers.v1 as vf
from verifiers.v1.judges import decision


def serve(monkeypatch, reply: dict) -> list[dict]:
    """Answer the judge's request with `reply`, recording each request body."""
    sent: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=reply)

    monkeypatch.setattr(
        decision,
        "build_async_openai",
        lambda config: AsyncOpenAI(
            base_url="https://judge.test/v1",
            api_key="k",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ),
    )
    return sent


async def test_evaluate_grades_on_the_levels(monkeypatch, make_trace):
    sent = serve(
        monkeypatch,
        {
            "answers": {"grade": {"score": 0.75}},
            "usage": {"inputTokens": 9, "outputTokens": 1},
        },
    )
    trace = make_trace()
    judge = vf.DecisionJudge()
    graded = await judge.evaluate(trace=trace, question="q", response="r")
    assert graded.parsed == 0.75
    assert sent[0] == {
        "model": "cloudflare/clef",
        "state": {"question": "q", "response": "r"},
        "questions": {
            "grade": {
                "type": "score",
                "instructions": judge.prompt,
                "criteria": ["Incorrect", "Correct"],
            }
        },
    }
    (call,) = trace.info["judge_calls"]
    assert call["name"] == "decision" and call["response"]["parsed"] == 0.75
    assert trace.extra_usage[0].prompt_tokens == 9


async def test_ask_asks_any_questions_and_records_the_answers(monkeypatch, make_trace):
    answers = {
        "chance": {"type": "choice", "choice": "80"},
        "right": {"type": "noul", "noul": 0.9},
    }
    sent = serve(monkeypatch, {"result": {"answers": answers}})
    questions = {
        "chance": {
            "type": "choice",
            "instructions": "What chance does the reply give?",
            "criteria": {"80": "80%", "none": "No number."},
        },
        "right": {"type": "noul", "instructions": "Is the reply right?"},
    }
    trace = make_trace()
    asked = await vf.DecisionJudge().ask("A reply: about 80%.", questions, trace=trace)
    assert asked.parsed == answers
    assert sent[0] == {
        "model": "cloudflare/clef",
        "state": "A reply: about 80%.",
        "questions": questions,
    }
    assert trace.info["judge_calls"][0]["response"]["parsed"] == answers


async def test_a_question_left_unanswered_raises_and_is_still_recorded(
    monkeypatch, make_trace
):
    serve(monkeypatch, {"answers": {}})
    trace = make_trace()
    with pytest.raises(ValueError, match="answered no"):
        await vf.DecisionJudge().ask(
            "state", {"right": {"type": "noul", "instructions": "?"}}, trace=trace
        )
    assert len(trace.info["judge_calls"]) == 1


@pytest.mark.parametrize(
    "answers, error",
    [({"grade": {"score": 5}}, ValueError), ({}, KeyError)],
)
async def test_an_ungradable_answer_raises_and_is_still_recorded(
    monkeypatch, make_trace, answers, error
):
    serve(monkeypatch, {"answers": answers})
    trace = make_trace()
    with pytest.raises(error):
        await vf.DecisionJudge().evaluate(trace=trace, question="q", response="r")
    assert len(trace.info["judge_calls"]) == 1
