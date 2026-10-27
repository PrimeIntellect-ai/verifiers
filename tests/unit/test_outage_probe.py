"""The outage probe: read-only Prime calls and one-token completions say whether an
outage of a kind has ended (cleared, still out, or unknown), with fake SDK clients."""

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest
from prime_sandboxes.core import APIError, PaymentRequiredError

import verifiers.v1 as vf
from verifiers.v1 import outage_probe as probe_module
from verifiers.v1.runtimes import SubprocessConfig


class FakePrime:
    """A Prime API client: `answers` maps an endpoint to a response or an error."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.calls: list[tuple[str, dict | None]] = []
        self.config = SimpleNamespace(team_id="team-1")

    async def get(self, endpoint: str, params: dict | None = None) -> dict:
        self.calls.append((endpoint, params))
        answer = self.answers[endpoint]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    async def aclose(self) -> None:
        pass


class FakeOpenAI:
    """An OpenAI client whose completion `answer(request)` may raise."""

    def __init__(self, calls: list, answer):
        async def create(**request):
            calls.append(request)
            answer(request)
            return SimpleNamespace(choices=[])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    async def close(self) -> None:
        pass


def status_error(cls: type[openai.APIStatusError], status: int, message="refused"):
    request = httpx.Request("POST", "https://models.test/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=None)


def raising(error):
    def answer(request):
        raise error

    return answer


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    state = {"prime": FakePrime({}), "model": lambda request: None, "completions": []}
    monkeypatch.setattr(probe_module, "_prime_api", lambda: state["prime"])
    monkeypatch.setattr(
        probe_module,
        "build_async_openai",
        lambda endpoint: FakeOpenAI(
            state["completions"], lambda request: state["model"](request)
        ),
    )
    return state


SEAT = vf.AgentConfig(model="test/model", runtime=vf.PrimeConfig())


async def test_an_account_outage_clears_when_the_wallet_has_funds(fakes):
    fakes["prime"] = FakePrime(
        {"/sandbox": {"sandboxes": []}, "/billing/wallet": {"balance_usd": 0.0}}
    )
    report = await vf.probe("account", [SEAT, SEAT])
    assert report.status == "out"
    (failure,) = report.failures
    assert failure.what == "prime wallet" and failure.kind == "account"
    # Reads only: a team-scoped page of one sandbox, then the team's wallet. The two
    # seats' one model gets one completion of one token.
    assert fakes["prime"].calls == [
        ("/sandbox", {"page": 1, "per_page": 1, "team_id": "team-1"}),
        ("/billing/wallet", {"limit": 1, "teamId": "team-1"}),
    ]
    assert [call["max_tokens"] for call in fakes["completions"]] == [1]

    fakes["prime"].answers["/billing/wallet"] = {"balance_usd": 12.5}
    assert (await vf.probe("account", [SEAT])).cleared

    fakes["prime"].answers["/billing/wallet"] = PaymentRequiredError("Payment.")
    (failure,) = (await vf.probe("account", [SEAT])).failures
    assert failure.what == "prime wallet" and failure.kind == "account"


async def test_a_wallet_that_cannot_be_read_does_not_block(fakes):
    fakes["prime"] = FakePrime(
        {"/sandbox": {"sandboxes": []}, "/billing/wallet": APIError("HTTP 403: no")}
    )
    report = await vf.probe("account", [SEAT])
    assert report.cleared and len(report.notes) == 1
    assert "prime wallet" in report.notes[0]

    fakes["prime"].answers["/billing/wallet"] = {"wallet_id": "w"}  # no balance
    report = await vf.probe("account", [SEAT])
    assert report.cleared and len(report.notes) == 1


async def test_a_quota_outage_is_out_while_any_check_meets_an_outage(fakes):
    fakes["prime"] = FakePrime({"/sandbox": APIError("HTTP 429: burst limit")})
    report = await vf.probe("quota", [SEAT])
    assert report.status == "out" and [f.kind for f in report.failures] == ["quota"]
    assert [call[0] for call in fakes["prime"].calls] == ["/sandbox"]  # no wallet

    # An outage blocks even when another check cannot tell.
    fakes["model"] = raising(status_error(openai.NotFoundError, 404))
    report = await vf.probe("quota", [SEAT])
    assert report.status == "out"
    assert {f.status for f in report.failures} == {"out", "unknown"}

    fakes["prime"].answers["/sandbox"] = {"sandboxes": []}
    fakes["model"] = raising(status_error(openai.PermissionDeniedError, 403))
    (failure,) = (await vf.probe("quota", [SEAT])).failures
    assert failure.what.startswith("model test/model") and failure.kind == "account"

    fakes["model"] = lambda request: None
    assert (await vf.probe("quota", [SEAT])).status == "cleared"


async def test_a_reasoning_model_is_asked_again_with_max_completion_tokens(fakes):
    def reasoning(request):
        if "max_tokens" in request:
            raise status_error(
                openai.BadRequestError,
                400,
                "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'.",
            )

    fakes["model"] = reasoning
    seat = vf.AgentConfig(model="o-model", runtime=SubprocessConfig())
    assert (await vf.probe("quota", [seat])).cleared
    assert [call.get("max_completion_tokens") for call in fakes["completions"]] == [
        None,
        1,
    ]

    fakes["completions"].clear()
    fakes["model"] = raising(status_error(openai.BadRequestError, 400, "bad model"))
    assert (await vf.probe("quota", [seat])).status == "unknown"
    assert len(fakes["completions"]) == 1  # another 400 is not retried


async def test_the_probe_is_unknown_when_no_check_can_tell(fakes):
    local = vf.AgentConfig(runtime=SubprocessConfig())
    assert (await vf.probe("io", [local])).status == "unknown"  # nothing to check
    assert fakes["prime"].calls == []

    fakes["prime"] = FakePrime({"/sandbox": APIError("No API key configured.")})
    fakes["model"] = raising(status_error(openai.NotFoundError, 404))
    report = await vf.probe("io", [SEAT])
    assert report.status == "unknown"
    assert {f.status for f in report.failures} == {"unknown"}


async def test_a_check_that_outlasts_its_timeout_is_an_io_outage(fakes):
    class Hanging(FakePrime):
        async def get(self, endpoint, params=None):
            await asyncio.sleep(60)

    fakes["prime"] = Hanging({})
    report = await vf.probe("io", [vf.AgentConfig()], timeout=0.01)
    (failure,) = report.failures
    assert failure.what == "prime sandboxes" and failure.kind == "io"


async def test_a_model_refused_without_a_key_is_unknown_not_an_account_outage(
    fakes, monkeypatch
):
    """An unset key variable sends the request with no key: its 401 is the run's
    configuration, not an account outage, so the probe cannot tell (`unknown`). With
    the key set, the same 401 is `out`. A keyless endpoint that answers clears."""
    monkeypatch.delenv("VF_PROBE_TEST_KEY", raising=False)
    endpoint = vf.EvalClientConfig(
        base_url="https://models.test/v1", api_key_var="VF_PROBE_TEST_KEY"
    )
    seat = vf.AgentConfig(
        model="test/model", client=endpoint, runtime=SubprocessConfig()
    )
    fakes["model"] = raising(status_error(openai.AuthenticationError, 401))
    report = await vf.probe("account", [seat])
    assert report.status == "unknown"
    (failure,) = report.failures
    assert failure.kind is None and "$VF_PROBE_TEST_KEY is unset" in failure.error

    monkeypatch.setenv("VF_PROBE_TEST_KEY", "sk-test")
    (failure,) = (await vf.probe("account", [seat])).failures
    assert (failure.status, failure.kind) == ("out", "account")

    monkeypatch.delenv("VF_PROBE_TEST_KEY")
    fakes["model"] = lambda request: None
    assert (await vf.probe("account", [seat])).cleared
