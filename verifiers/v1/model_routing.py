"""Binding a session to one deployment of its model pool.

At a session's first model request, `request_binding` picks a deployment: the pool's
deployments are tried in random order, each with one non-queuing admission at its request
gate (`ModelGate.try_acquire`), and the first that grants a slot wins. With none free it
tries again after a wait that grows from 30 ms to about 1 s: a session waiting to bind
yields to the requests already queued at a gate (those of bound sessions). The slot it took is handed to that first request
(`reserved_permit`), so the choice and the admission are one step. The session then keeps
the deployment, model and client for every later request (provider prefix caches stay
warm); each of them passes that deployment's gate like any gated request.

The binding is recorded on the trace: `trace.info["model_binding"]` (deployment, model,
base URL), the agent config on the trace (`trace.agent.config.model` / `.client`), and
each call's `ModelCall.deployment`. A resumed session keeps its saved deployment while the
pool still has it. In a pool of different models, a resume whose deployment has left the
pool binds to another and so may change model mid-conversation (each call still records
its own); a `same_model` pool cannot.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

import httpx

from verifiers.v1.configs.client import (
    BaseClientConfig,
    resolve_api_key,
    resolve_headers,
)
from verifiers.v1.configs.model_pool import ModelDeployment, ModelPoolConfig
from verifiers.v1.request_gate import (
    _GATES,
    ModelGate,
    RequestPermit,
    gate_dir,
    gate_key,
    model_gate,
    reserved_permit,
)

if TYPE_CHECKING:
    from verifiers.v1.clients import Client
    from verifiers.v1.dialects import Dialect
    from verifiers.v1.session import RolloutSession

MODEL_BINDING = "model_binding"
"""`trace.info` key of the session's deployment."""
POLL_SECONDS = 0.03
"""First wait before trying the pool again when every deployment is at its limit; it
doubles each round (jittered), up to `MAX_POLL_SECONDS`."""
MAX_POLL_SECONDS = 1.0


async def _gate(deployment: ModelDeployment, dialect: Dialect) -> ModelGate:
    """The deployment's gate, keyed as its requests will be (`GatedAsyncClient`)."""
    client = deployment.client
    config = client.request_gate
    assert config is not None  # ModelDeployment requires one
    headers = httpx.Headers(resolve_headers(client))
    headers.update(dialect.auth_headers(resolve_api_key(client)))
    credential = headers.get("authorization") or headers.get("x-api-key") or ""
    key = gate_key(
        httpx.URL(client.base_url), credential, deployment.model, config.scope
    )
    directory = gate_dir(config)
    gate = _GATES.get((directory, key))
    if gate is None:  # its first use may start the daemon: off the event loop
        gate = await asyncio.to_thread(model_gate, directory, key, deployment.model)
    return gate


async def choose(
    pool: ModelPoolConfig, dialect: Dialect
) -> tuple[ModelDeployment, RequestPermit]:
    """The first deployment, in random order, whose gate admits a request now, and the
    slot it admitted. Waits (polling) while every deployment is at its limit."""
    candidates = [(item, await _gate(item, dialect)) for item in pool.deployments]
    random.shuffle(candidates)
    delay = POLL_SECONDS
    while True:
        for deployment, gate in candidates:
            config = deployment.client.request_gate
            assert config is not None
            slot = await gate.try_acquire(config)
            if slot is not None:
                return deployment, RequestPermit(gate, slot)
        # Admission, not a snapshot of load, makes the choice; each gate still serves
        # its own queued requests first (a newcomer yields to them) and holds its cap
        # host-wide. Back off, so a crowd of newcomers cannot flood the daemon.
        await asyncio.sleep(random.uniform(delay / 2, delay))
        delay = min(delay * 2, MAX_POLL_SECONDS)


def bind(
    session: RolloutSession,
    deployment: ModelDeployment,
    make_client: Callable[[BaseClientConfig], Client],
) -> None:
    """Point the session at `deployment` for the rest of its requests, and record it."""
    session.ctx = replace(session.ctx, model=deployment.model, client=deployment.client)
    session.client = make_client(deployment.client)
    session.deployment = deployment.id
    trace = session.trace
    trace.info[MODEL_BINDING] = {
        "deployment": deployment.id,
        "model": deployment.model,
        "base_url": deployment.client.base_url,
    }
    trace.agent.config = trace.agent.config.model_copy(
        update={"model": deployment.model, "client": deployment.client}
    )
    trace.notify()


@asynccontextmanager
async def request_binding(
    session: RolloutSession,
    dialect: Dialect,
    make_client: Callable[[BaseClientConfig], Client],
) -> AsyncIterator[None]:
    """Around one model request of a pooled session: bind the session first if it is not
    yet, handing the chosen slot to this request."""
    pool = session.ctx.model_pool
    if pool is None or session.deployment is not None:
        yield
        return
    permit: RequestPermit | None = None
    token = None
    try:
        async with session.model_lock:
            if session.deployment is None:
                saved = session.trace.info.get(MODEL_BINDING) or {}
                chosen = next(
                    (
                        item
                        for item in pool.deployments
                        if item.id == saved.get("deployment")
                        and item.model == saved.get("model")
                    ),
                    None,
                )
                if chosen is None:
                    chosen, permit = await choose(pool, dialect)
                bind(session, chosen, make_client)
        token = reserved_permit.set(permit)
        yield
    finally:
        if token is not None:
            reserved_permit.reset(token)
        if permit is not None:  # the request never reached the gate: give it back
            permit.close()
