"""A marked compaction may use an authorized model without becoming a solver turn."""

import asyncio
from contextlib import asynccontextmanager

import aiohttp

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.interception.server import (
    AUXILIARY_PURPOSE_HEADER,
    InterceptionServer,
)
from verifiers.v1.session import RolloutSession
from verifiers.v1.types import SamplingConfig, Usage


class FakeClient:
    def __init__(self, *, empty_auxiliary: bool = False):
        self.requests = []
        self.empty_auxiliary = empty_auxiliary

    async def get_response(self, dialect, body, sampling, **kwargs):
        self.requests.append((body, kwargs["headers"]))
        auxiliary = body["model"] == "openai/gpt-5.6-sol"
        text = (
            ""
            if auxiliary and self.empty_auxiliary
            else ("<summary>continuity</summary>" if auxiliary else "solver answer")
        )
        raw = {"id": "r", "model": body["model"], "output": []}
        return vf.Response(
            id="r",
            created=0,
            model=body["model"],
            message=vf.AssistantMessage(content=text),
            finish_reason="stop",
            usage=Usage(prompt_tokens=20, completion_tokens=3),
            raw=raw,
        )


@asynccontextmanager
async def interception():
    server = InterceptionServer()
    await server.start()
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="question")),
    )
    session = RolloutSession(
        ctx=ModelContext(
            model="z-ai/glm-5.3-flash",
            client=EvalClientConfig(),
            sampling=SamplingConfig(),
        ),
        trace=trace,
        auxiliary_models=frozenset({"openai/gpt-5.6-sol"}),
    )
    model_secret, state_secret = server.register(session)
    fake = FakeClient()
    session.client = fake
    try:
        yield server, session, fake, model_secret
    finally:
        server.unregister(model_secret, state_secret)
        await server.stop()


def test_compaction_model_is_authorized_recorded_and_unscored():
    async def check():
        async with interception() as (server, session, fake, secret):
            url = f"{server.base_url}/v1/responses"
            request = {
                "model": "openai/gpt-5.6-sol",
                "input": "Summarize the trajectory",
                "stream": False,
                "tool_choice": "none",
            }
            auth = {"Authorization": f"Bearer {secret}"}
            async with aiohttp.ClientSession() as client:
                async with client.post(url, json=request, headers=auth) as response:
                    assert response.status == 200
                assert fake.requests[0][0]["model"] == "z-ai/glm-5.3-flash"
                assert session.trace.last_reply == "solver answer"

                async with client.post(
                    url,
                    json=request,
                    headers={**auth, AUXILIARY_PURPOSE_HEADER: "compaction"},
                ) as response:
                    assert response.status == 200
                assert fake.requests[1][0]["model"] == "openai/gpt-5.6-sol"
                assert AUXILIARY_PURPOSE_HEADER not in fake.requests[1][1]

            assert session.trace.num_turns == 1
            assert session.trace.last_reply == "solver answer"
            assert [
                (call.purpose, call.model, call.node) for call in session.trace.calls
            ] == [
                ("agent", "z-ai/glm-5.3-flash", 1),
                ("compaction", "openai/gpt-5.6-sol", None),
            ]
            assert session.trace.usage.completion_tokens == 6
            assert session.trace.model_dump()["calls"][1]["purpose"] == "compaction"

    asyncio.run(check())


def test_unapproved_compaction_model_is_rejected():
    async def check():
        async with interception() as (server, session, fake, secret):
            async with (
                aiohttp.ClientSession() as client,
                client.post(
                    f"{server.base_url}/v1/responses",
                    json={
                        "model": "other/model",
                        "input": "Summarize",
                        "stream": False,
                        "tool_choice": "none",
                    },
                    headers={
                        "Authorization": f"Bearer {secret}",
                        AUXILIARY_PURPOSE_HEADER: "compaction",
                    },
                ) as response,
            ):
                assert response.status == 403
            assert not fake.requests
            assert not session.trace.calls

    asyncio.run(check())


def test_empty_compaction_is_retryable_provider_error():
    async def check():
        async with interception() as (server, session, fake, secret):
            fake.empty_auxiliary = True
            async with (
                aiohttp.ClientSession() as client,
                client.post(
                    f"{server.base_url}/v1/responses",
                    json={
                        "model": "openai/gpt-5.6-sol",
                        "input": "Summarize",
                        "stream": False,
                        "tool_choice": "none",
                    },
                    headers={
                        "Authorization": f"Bearer {secret}",
                        AUXILIARY_PURPOSE_HEADER: "compaction",
                    },
                ) as response,
            ):
                assert response.status == 502
            assert session.trace.num_turns == 0
            assert session.trace.calls[0].purpose == "compaction"
            assert session.trace.calls[0].error.type == "ProviderError"

    asyncio.run(check())
