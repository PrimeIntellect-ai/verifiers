"""Outages: infrastructure failures typed from evidence (never message text), the seat
on an error record, and the stock probe of sandbox creation and model endpoints."""

import aiohttp
import httpx
import pytest
from aiohttp import web

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError, model_error, outage_kind
from verifiers.v1.runtimes import SubprocessConfig


def test_provider_and_transport_failures_are_outages_by_kind():
    assert (
        outage_kind(vf.ProviderError("<html>bad gateway</html>", status_code=502))
        == "io"
    )
    assert outage_kind(vf.ProviderError("no", status_code=402)) == "account"
    assert outage_kind(vf.ProviderError("slow down", status_code=429)) == "quota"
    assert outage_kind(vf.ProviderError("context overflow", status_code=400)) is None
    assert outage_kind(model_error("ReadTimeout", status_code=504)) == "io"
    assert isinstance(httpx.ReadTimeout("read timed out"), vf.Outage)
    assert isinstance(aiohttp.ClientConnectionResetError("reset by peer"), vf.Outage)
    assert outage_kind(SandboxError("quota", kind="quota")) == "quota"
    assert not isinstance(SandboxError("ENOSPC", kind="disk_full"), vf.Outage)


def test_message_text_never_makes_an_outage():
    assert not isinstance(RuntimeError("401 unauthorized: invalid api key"), vf.Outage)
    assert not isinstance(ValueError("upstream unavailable"), vf.Outage)


def test_a_failed_runs_record_types_the_error_that_carries_it():
    record = vf.Error(type="ProviderError", message="upstream 503", status_code=503)
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task", data=vf.TaskData(idx=0, prompt=None), key="t", hash="h"
        ),
    )
    trace.errors.append(record)
    assert isinstance(vf.RunFailed(trace), vf.Outage)
    assert vf.RunFailed(trace, seat="author").error.seat == "author"
    assert outage_kind(record) == "io"
    assert (
        outage_kind(vf.Error(type="SandboxError", message="x", kind="account"))
        == "account"
    )
    assert (
        outage_kind(vf.Error(type="TaskError", message="503 in the task's own output"))
        is None
    )


def test_an_error_record_carries_the_seat_it_was_raised_for():
    error = vf.Outage("billing", kind="account", seat="author")
    assert isinstance(error, vf.Outage) and outage_kind(error) == "account"
    assert vf.Error.of(error).seat == "author"
    assert "seat" not in vf.Error.of(RuntimeError("x")).model_dump()


@pytest.fixture
async def endpoint():
    """A local chat endpoint: `status` says what it answers."""
    state = {"status": 200}

    async def complete(request: web.Request) -> web.Response:
        if state["status"] != 200:
            return web.Response(status=state["status"], text="<html>unavailable</html>")
        body = await request.json()
        return web.json_response(
            {
                "id": "probe",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "p"},
                        "finish_reason": "length",
                    }
                ],
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/v1", state
    await runner.cleanup()


@pytest.mark.asyncio
async def test_the_probe_checks_each_runtime_and_model_once(endpoint, monkeypatch):
    url, state = endpoint
    monkeypatch.setenv("PROBE_TEST_KEY", "k")
    client = vf.EvalClientConfig(base_url=url, api_key_var="PROBE_TEST_KEY")
    seat = vf.AgentConfig(model="test/fake-model", runtime=SubprocessConfig())
    report = await vf.probe([seat, seat], client=client)
    assert report.ok and len(report.checks) == 2
    state["status"] = 503
    report = await vf.probe([seat], client=client)
    (failure,) = report.failures
    assert failure.what.startswith("model") and failure.kind == "io"
