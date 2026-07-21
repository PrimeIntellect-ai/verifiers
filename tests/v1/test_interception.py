"""The v1 interception server's session seal: a rollout that concludes while an
exchange is still upstream must not have its trace mutated by the straggler."""

import asyncio
from types import SimpleNamespace

import httpx

import verifiers.v1 as vf
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import Trace, TraceTask


class HangingClient:
    """A model client whose call parks until cancelled — the slow-provider tail."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.cancelled = False

    async def get_response(self, *args, **kwargs):
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def test_unregister_seals_the_trace_and_cancels_stragglers():
    """The deadline path: the harness window closes (rollout unregisters) while the
    model call is still upstream. The handler must be cancelled, and the sealed
    trace must keep zero turns and zero call records — what was persisted is what
    the in-memory trace still says."""
    client = HangingClient()
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")))
    session = RolloutSession(
        ctx=SimpleNamespace(client=client, model="m", sampling=vf.SamplingConfig()),
        trace=trace,
    )
    server = InterceptionServer()
    await server.start()
    try:
        secret = server.register(session)
        async with httpx.AsyncClient() as http:
            request = asyncio.ensure_future(
                http.post(
                    f"{server.base_url}/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    headers={"Authorization": f"Bearer {secret}"},
                    timeout=30,
                )
            )
            await asyncio.wait_for(client.entered.wait(), 10)
            assert session.tasks  # the handler adopted itself
            server.unregister(secret)
            # The straggler dies instead of finishing the exchange: either the fence
            # answered 409 or the cancellation dropped the connection.
            try:
                response = await asyncio.wait_for(request, 10)
                assert response.status_code == 409
            except httpx.HTTPError:
                pass
        await asyncio.sleep(0)  # let cancellation unwind
        assert client.cancelled
        assert session.released
        assert trace.nodes == [] and trace.calls == []  # the seal held
        assert trace.stop_condition is None  # refused() never ran post-seal
    finally:
        await server.stack.aclose()


async def test_released_session_refuses_new_exchanges():
    """A request arriving for a concluded rollout is refused outright (409), it
    never reaches the model client."""
    client = HangingClient()
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")))
    session = RolloutSession(
        ctx=SimpleNamespace(client=client, model="m", sampling=vf.SamplingConfig()),
        trace=trace,
    )
    session.released = True
    server = InterceptionServer()
    await server.start()
    try:
        server.sessions["still-routed"] = session  # released but not yet unregistered
        async with httpx.AsyncClient() as http:
            response = await http.post(
                f"{server.base_url}/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer still-routed"},
                timeout=10,
            )
        assert response.status_code == 409
        assert not client.entered.is_set()
        assert trace.nodes == [] and trace.calls == []
    finally:
        await server.stack.aclose()
