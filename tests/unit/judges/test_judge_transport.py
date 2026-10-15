"""Judge transport: transient-failure retries under the config's `retry` budget and the
input-size check before a call. The provider is an in-process `httpx.MockTransport`."""

import json

import httpx
import openai
import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import base as client_base
from verifiers.v1.judge import input_token_bound

FAST_RETRY = vf.ProviderRetryConfig(budget=5.0, max_backoff=0.01)


@pytest.fixture
def provider(monkeypatch):
    """Route every judge HTTP request to `provider.replies` (one per request, in order),
    recording each request body in `provider.requests`."""

    class Provider:
        replies: list[httpx.Response | Exception] = []
        requests: list[dict] = []

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


def chat(text: str) -> httpx.Response:
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
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def evaluation(score: float) -> httpx.Response:
    return httpx.Response(200, json={"answers": {"grade": {"score": score}}})


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


async def test_an_oversized_input_raises_before_any_call(provider):
    with pytest.raises(vf.JudgeInputTooLarge) as raised:
        await judge(max_input_tokens=100).complete("x" * 200)
    assert raised.value.limit == 100 and raised.value.tokens > 100
    assert provider.requests == []


async def test_an_input_within_the_limit_is_sent(provider):
    provider.replies = [chat("yes")]
    await judge(max_input_tokens=200).complete("x" * 50)
    assert len(provider.requests) == 1


def test_the_token_bound_counts_utf8_bytes():
    assert input_token_bound("é") > input_token_bound("e")
    assert input_token_bound("a" * 1000) >= 1000


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
