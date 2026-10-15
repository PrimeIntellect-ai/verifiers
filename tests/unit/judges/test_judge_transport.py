"""Judge transport: transient-failure retries under the config's `retry` budget and the
input-size checks before and after a call. The provider is an in-process
`httpx.MockTransport`."""

import json
from typing import ClassVar

import httpx
import openai
import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import base as client_base
from verifiers.v1.judge import BYTES_PER_TOKEN, input_token_floor

FAST_RETRY = vf.ProviderRetryConfig(budget=5.0, max_backoff=0.01)


@pytest.fixture
def provider(monkeypatch):
    """Route every judge HTTP request to `provider.replies` (one per request, in order),
    recording each request body in `provider.requests`."""

    class Provider:
        replies: ClassVar[list[httpx.Response | Exception]] = []
        requests: ClassVar[list[dict]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        Provider.requests.append(json.loads(request.content))
        reply = Provider.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def mock_httpx(config):
        return httpx.AsyncClient(transport=httpx.MockTransport(handle))

    monkeypatch.setattr(client_base, "build_async_httpx", mock_httpx)
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    Provider.replies = []
    Provider.requests = []
    return Provider


def chat(text: str, prompt_tokens: int = 1) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "test/fake-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": prompt_tokens + 1,
            },
        },
    )


def evaluation(score: float, input_tokens: int | None = None) -> httpx.Response:
    body: dict = {"answers": {"grade": {"score": score}}}
    if input_tokens is not None:
        body["usage"] = {"inputTokens": input_tokens, "outputTokens": 0}
    return httpx.Response(200, json=body)


def judge(**config) -> vf.Judge:
    return vf.Judge(
        vf.JudgeConfig(
            **{
                "model": "test/fake-model",
                "base_url": "https://judge.test/v1",
                "retry": FAST_RETRY,
                **config,
            }
        )
    )


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(503, json={"error": "busy"}),
        httpx.Response(429, json={"error": "slow down"}),
        httpx.ConnectError("reset"),
        httpx.ReadTimeout("slow"),
    ],
)
async def test_a_transient_failure_is_retried_then_succeeds(provider, failure):
    provider.replies = [failure, chat("yes")]

    response = await judge().complete("Is it done?")

    assert response.text == "yes"
    assert len(provider.requests) == 2


async def test_a_spent_retry_budget_raises_instead_of_scoring(provider):
    provider.replies = [httpx.Response(503, json={"error": "down"})] * 50
    with pytest.raises(openai.InternalServerError):
        await judge(
            retry=vf.ProviderRetryConfig(budget=0.05, max_backoff=0.01)
        ).complete("Is it done?")
    assert 1 < len(provider.requests) < 50


async def test_no_retry_budget_sends_once(provider):
    provider.replies = [httpx.Response(503, json={"error": "down"})]
    with pytest.raises(openai.InternalServerError):
        await judge(retry=vf.ProviderRetryConfig(budget=0)).complete("Is it done?")
    assert len(provider.requests) == 1


async def test_a_request_error_is_not_retried(provider):
    provider.replies = [httpx.Response(400, json={"error": "bad request"})]
    with pytest.raises(openai.BadRequestError):
        await judge().complete("Is it done?")
    assert len(provider.requests) == 1


async def test_a_far_oversized_input_raises_before_any_call(provider):
    with pytest.raises(vf.JudgeInputTooLarge) as raised:
        await judge(max_input_tokens=100).complete("x" * 1000)
    assert raised.value.limit == 100 and raised.value.tokens > 100
    assert not raised.value.reported
    assert provider.requests == []


async def test_an_input_over_the_limit_in_bytes_but_not_tokens_is_sent(provider):
    provider.replies = [chat("yes", prompt_tokens=60)]
    response = await judge(max_input_tokens=100).complete("x" * 300)
    assert response.text == "yes"
    assert len(provider.requests) == 1


async def test_a_reported_input_count_at_the_limit_raises(provider):
    provider.replies = [chat("yes", prompt_tokens=100)]
    with pytest.raises(vf.JudgeInputTooLarge) as raised:
        await judge(max_input_tokens=100).complete("Is it done?")
    assert raised.value.reported and raised.value.tokens == 100
    assert len(provider.requests) == 1


def test_the_token_floor_divides_utf8_bytes():
    assert input_token_floor("é") >= input_token_floor("e")
    assert input_token_floor("a" * 900) * BYTES_PER_TOKEN >= 900
    assert input_token_floor("a" * 900) < 900


def decision(**config) -> vf.DecisionJudge:
    return vf.DecisionJudge(
        vf.DecisionJudgeConfig(
            model="test/fake-model",
            base_url="https://judge.test/v1/evaluations",
            levels=["No", "Yes"],
            retry=FAST_RETRY,
            **config,
        )
    )


async def test_a_decision_judge_sends_one_request_per_question(provider):
    provider.replies = [evaluation(1.0)]

    result = await decision().evaluate(question="Is it done?", response="done")

    assert result.parsed == 1.0
    (request,) = provider.requests
    assert list(request["questions"]) == ["grade"]
    assert request["state"] == {"question": "Is it done?", "response": "done"}


async def test_a_decision_judge_retries_a_transient_failure(provider):
    provider.replies = [httpx.Response(502, json={"error": "gateway"}), evaluation(0.0)]

    result = await decision().evaluate(question="Is it done?", response="no")

    assert result.parsed == 0.0
    assert len(provider.requests) == 2


async def test_a_decision_judge_refuses_oversized_state_before_the_call(provider):
    with pytest.raises(vf.JudgeInputTooLarge):
        await decision(max_input_tokens=64).evaluate(
            question="Is it done?", response="y" * 500
        )
    assert provider.requests == []


async def test_a_decision_judge_grades_state_over_the_limit_in_bytes(provider):
    provider.replies = [evaluation(1.0, input_tokens=150)]

    result = await decision(max_input_tokens=200).evaluate(
        question="Is it done?", response="y" * 500
    )

    assert result.parsed == 1.0
    assert len(provider.requests) == 1


async def test_a_decision_judge_refuses_a_cut_input_after_the_call(
    provider, make_trace
):
    # A truncating endpoint reports exactly its cap and grades the head it kept.
    provider.replies = [evaluation(1.0, input_tokens=64)]
    trace = make_trace()

    with pytest.raises(vf.JudgeInputTooLarge) as raised:
        await decision(max_input_tokens=64).evaluate(
            trace=trace, question="Is it done?", response="done"
        )

    assert raised.value.reported and raised.value.tokens == 64
    assert len(trace.info["judge_calls"]) == 1
