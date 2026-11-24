import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from verifiers.v1.judges.decision import (
    DecisionJudge,
    DecisionJudgeConfig,
    DecisionScoreError,
)


async def test_task_images_are_sent_in_order_and_omitted_from_trace(monkeypatch):
    images = ["data:image/png;base64,AQ==", "data:image/jpeg;base64,Ag=="]
    client = AsyncMock()
    monkeypatch.setattr(
        "verifiers.v1.judges.decision.build_async_openai", lambda _: client
    )
    post = client.__aenter__.return_value.post
    post.return_value = {
        "result": {
            "answers": {"grade": {"score": 1}},
            "images": images,
            "usage": {"inputTokens": 7},
        }
    }
    judge = DecisionJudge(DecisionJudgeConfig(images_field="plots"))
    trace = Mock(last_reply="The second figure has the larger slope.")
    assert (
        await judge.score(
            SimpleNamespace(prompt_text="Compare the slopes.", plots=images), trace
        )
        == 1
    )
    body = post.await_args.kwargs["body"]
    assert body["images"] == images
    assert "images" not in body["state"]
    logged = trace.record_judge_call.call_args.kwargs
    assert logged["request"]["image_count"] == 2
    assert "base64" not in json.dumps(logged["request"]) + logged["response"].text
    assert logged["response"].usage.prompt_tokens == 7


async def test_malformed_grade_keeps_usage_and_compact_trace(monkeypatch):
    client = AsyncMock()
    monkeypatch.setattr(
        "verifiers.v1.judges.decision.build_async_openai", lambda _: client
    )
    post = client.__aenter__.return_value.post
    post.return_value = {
        "answers": {"grade": {"score": True}},
        "state": {"artifact": "large evidence"},
        "usage": {"inputTokens": 5},
    }
    trace = Mock()
    with pytest.raises(DecisionScoreError):
        await DecisionJudge().evaluate(
            trace=trace, trace_state={}, artifact="large evidence"
        )
    assert "images" not in post.await_args.kwargs["body"]
    logged = trace.record_judge_call.call_args.kwargs
    assert logged["request"]["state"] == {}
    assert "large evidence" not in logged["response"].text
    assert logged["response"].usage.prompt_tokens == 5


async def test_batch_shares_evidence_and_matches_scores_by_question(monkeypatch):
    client = AsyncMock()
    monkeypatch.setattr(
        "verifiers.v1.judges.decision.build_async_openai", lambda _: client
    )
    post = client.__aenter__.return_value.post
    images = ["data:image/png;base64,AQ=="]
    criteria = [str(index) for index in range(65)]
    judge = DecisionJudge()
    trace = Mock()

    async def evaluate(url, *, body, cast_to):
        assert body["state"] == {"artifact": "figure"} and body["images"] == images
        return {
            "answers": {
                key: {"score": int(question["instructions"]) / 64}
                for key, question in reversed(body["questions"].items())
            }
        }

    post.side_effect = evaluate
    scores = await judge.evaluate_batch(
        criteria, images=images, trace=trace, artifact="figure"
    )
    assert scores == [i / 64 for i in range(65)]
    assert [len(call.kwargs["body"]["questions"]) for call in post.await_args_list] == [
        64,
        1,
    ]
    assert trace.record_judge_call.call_count == 2

    post.side_effect = None
    post.return_value = {"answers": {"criterion_0": {"score": 1}}}
    with pytest.raises(DecisionScoreError, match="criterion_1"):
        await judge.evaluate_batch(criteria[:2], trace=trace)
    assert trace.record_judge_call.call_count == 3
