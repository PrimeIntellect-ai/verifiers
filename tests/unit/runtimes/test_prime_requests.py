import asyncio
import json
import logging
from datetime import UTC, datetime
from email.utils import format_datetime
from itertools import pairwise
from multiprocessing import get_context
from types import MethodType, SimpleNamespace

import httpx
import pytest
from prime_sandboxes import AsyncSandboxClient
from prime_sandboxes.core import APIError, AsyncAPIClient
from pydantic import ValidationError
from tenacity import wait_none

from verifiers.v1.configs.runtime import PrimeConfig
from verifiers.v1.runtimes import prime_requests as pacing


def _attempt_from_process(limiter, results):
    results.put(limiter._update())


def test_processes_share_admission_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(pacing, "LIMITER_DIR", tmp_path)
    limiter = pacing.PrimeRequestLimiter(2, "shared-process-run")
    context = get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(target=_attempt_from_process, args=(limiter, results))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    try:
        waits = sorted(results.get(timeout=30) for _ in workers)
        assert waits[0] == 0
        assert 0 < waits[1] <= 30
    finally:
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.kill()
                worker.join()
        results.close()
    assert all(worker.exitcode == 0 for worker in workers)


@pytest.fixture
def clock(monkeypatch, tmp_path):
    class Clock:
        def __init__(self):
            self.now = 1000.0
            self.waits = []

        async def sleep(self, delay):
            self.waits.append(delay)
            self.now += delay
            await asyncio.sleep(0)

    clock = Clock()
    monkeypatch.setattr(pacing, "LIMITER_DIR", tmp_path)
    monkeypatch.setattr(pacing, "time", SimpleNamespace(time=lambda: clock.now))
    monkeypatch.setattr(
        pacing,
        "asyncio",
        SimpleNamespace(to_thread=asyncio.to_thread, sleep=clock.sleep),
    )
    return clock


async def test_instances_share_spacing_without_reserving_waiters(clock):
    first = pacing.PrimeRequestLimiter(120, "run")
    second = pacing.PrimeRequestLimiter(120, "run")
    await first.acquire()
    # A waiting check does not advance the cursor or reserve future slots.
    assert second._update() == 0.5
    assert second._update() == 0.5
    assert json.loads(first._path.read_text())["next"] == 1000.5
    await second.acquire()
    assert clock.waits == [0.5]
    assert json.loads(first._path.read_text())["next"] == 1001.0
    independent = pacing.PrimeRequestLimiter(120, "other/run")
    assert independent._update() == 0
    with pytest.raises(ValueError, match="Conflicting"):
        pacing.PrimeRequestLimiter(60, "run")._update()


async def test_pending_waiter_rechecks_new_shared_cooldown(clock, monkeypatch):
    first = pacing.PrimeRequestLimiter(120, "run")
    second = pacing.PrimeRequestLimiter(120, "run")
    await first.acquire()
    entered = asyncio.Event()
    resume = asyncio.Event()

    async def sleep(delay):
        clock.waits.append(delay)
        if len(clock.waits) == 1:
            entered.set()
            await resume.wait()
        clock.now += delay

    monkeypatch.setattr(pacing.asyncio, "sleep", sleep)
    pending = asyncio.create_task(second.acquire())
    await entered.wait()
    await first.cooldown(20)
    resume.set()
    await pending
    assert clock.waits == [0.5, 19.5]
    assert clock.now == 1020


async def test_cancelled_waiter_leaves_no_future_reservation(clock, monkeypatch):
    limiter = pacing.PrimeRequestLimiter(120, "run")
    await limiter.acquire()
    entered = asyncio.Event()

    async def sleep(delay):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(pacing.asyncio, "sleep", sleep)
    pending = asyncio.create_task(limiter.acquire())
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert json.loads(limiter._path.read_text())["next"] == 1000.5
    clock.now += 0.5
    assert limiter._update() == 0


@pytest.mark.parametrize("value", [None, "invalid", "-1", "nan", "inf"])
def test_invalid_retry_after_has_bounded_fallback(clock, value):
    delay = pacing._retry_after_delay(value)
    assert 10 <= delay <= 60


def test_retry_after_numeric_and_date_are_minimums_with_positive_jitter(clock):
    for value in ("25", format_datetime(datetime.fromtimestamp(1025, UTC))):
        delay = pacing._retry_after_delay(value)
        assert 25 < delay <= 26
    past = format_datetime(datetime.fromtimestamp(900, UTC))
    assert 0 < pacing._retry_after_delay(past) <= 1


def sandbox_client(http_client):
    # Real SDK request/retry methods without credentials, config files, or networking.
    sandbox = object.__new__(AsyncSandboxClient)
    api = object.__new__(AsyncAPIClient)
    api.client = http_client
    api.require_auth = False
    api.api_key = None
    api.base_url = "https://prime.invalid"
    sandbox.client = api
    return sandbox


async def test_hooks_once_preserve_existing_and_log_only_templates(clock, caplog):
    seen = []

    async def request_hook(request):
        seen.append("request")

    async def response_hook(response):
        seen.append("response")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})),
        event_hooks={"request": [request_hook], "response": [response_hook]},
    ) as client:
        sandbox = sandbox_client(client)
        pacing.install_prime_request_hooks(sandbox, 120, "run")
        pacing.install_prime_request_hooks(sandbox, 120, "run")
        assert len(client.event_hooks["request"]) == 2
        assert len(client.event_hooks["response"]) == 2
        with pytest.raises(ValueError, match="Conflicting"):
            pacing.install_prime_request_hooks(sandbox, 60, "run")
        with pytest.raises(ValueError, match="Conflicting"):
            pacing.install_prime_request_hooks(sandbox, None, "run")
        caplog.set_level(logging.INFO, logger=pacing.__name__)
        await client.get(
            "https://URL_SECRET/api/v1/sandbox/ID_SECRET/egress-policy?token=QUERY_SECRET",
            headers={"Authorization": "TOKEN_SECRET"},
        )
        await client.post("https://URL_SECRET/UNKNOWN_SECRET", content="BODY_SECRET")
    assert seen == ["request", "response"] * 2
    assert "path_template=/api/v1/sandbox/{id}/egress-policy" in caplog.text
    assert "path_template=/api/v1/{endpoint}" in caplog.text
    for secret in (
        "URL_SECRET",
        "ID_SECRET",
        "QUERY_SECRET",
        "TOKEN_SECRET",
        "UNKNOWN_SECRET",
        "BODY_SECRET",
    ):
        assert secret not in caplog.text


async def test_sdk_physical_retries_all_paced_and_budget_unchanged(clock):
    attempts = []

    def respond(request):
        attempts.append(clock.now)
        return httpx.Response(429, headers={"Retry-After": "2"}, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        sandbox = sandbox_client(client)
        # Remove only SDK test sleeps; keep its native three-attempt stop condition.
        retry = sandbox.client._non_idempotent_request_with_retry.retry_with(
            wait=wait_none()
        )
        sandbox.client._non_idempotent_request_with_retry = MethodType(
            retry, sandbox.client
        )
        pacing.install_prime_request_hooks(sandbox, 120, "run")
        with pytest.raises(APIError):
            await sandbox.client.request("POST", "/sandbox", json={"body": "SECRET"})
    assert len(attempts) == 3
    assert all(b - a > 2 for a, b in pairwise(attempts))


async def test_no_added_post_or_gateway_exec_replay(clock):
    attempts = []

    def fail(request):
        attempts.append(request.url.path)
        raise httpx.ReadError("ambiguous committed request", request=request)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(fail)) as api_client,
        httpx.AsyncClient(transport=httpx.MockTransport(fail)) as gateway,
    ):
        sandbox = sandbox_client(api_client)
        sandbox._gateway_client = gateway
        pacing.install_prime_request_hooks(sandbox, 120, "run")
        with pytest.raises(APIError):
            await sandbox.client.request("POST", "/sandbox", json={})
        with pytest.raises(httpx.ReadError):
            await sandbox._gateway_post("https://gateway.invalid/exec", {}, 1, json={})
        assert gateway.event_hooks == {"request": [], "response": []}
    assert attempts == ["/api/v1/sandbox", "/exec"]


def test_config_is_opt_in_and_positive():
    assert PrimeConfig().requests_per_min is None
    assert PrimeConfig(requests_per_min=120).requests_per_min == 120
    for invalid in (0, -1):
        with pytest.raises(ValidationError):
            PrimeConfig(requests_per_min=invalid)
