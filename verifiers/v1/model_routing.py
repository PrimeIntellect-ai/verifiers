"""Session-pinned deployment choice using the native request gates."""

import asyncio
import random
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx

from verifiers.v1.clients.gate import (
    RequestPermit,
    gate_dir,
    gate_key,
    model_gate,
    reserved_permit,
)
from verifiers.v1.configs.client import resolve_api_key, resolve_headers

MODEL_BINDING = "model_binding"


async def _choose(pool, dialect):
    eligible = list(pool.deployments)
    if pool.order == "shuffle":
        random.shuffle(eligible)
    gates = []
    for deployment in eligible:
        client = deployment.client
        config = client.request_gate
        headers = httpx.Headers(resolve_headers(client))
        headers.update(dialect.auth_headers(resolve_api_key(client)))
        credential = headers.get("authorization") or headers.get("x-api-key") or ""
        gate = model_gate(
            gate_dir(config),
            gate_key(
                httpx.URL(client.base_url), credential, deployment.model, config.scope
            ),
            deployment.model,
        )
        gates.append((deployment, gate, config))
    while True:
        for deployment, gate, config in gates:
            slot = gate.try_admit(config)
            if slot is not None:
                return deployment, RequestPermit(gate, config, slot)
        # Admission, not a telemetry snapshot, makes the choice. Each gate still
        # gives its older queued requests priority and enforces capacity host-wide.
        await asyncio.sleep(random.uniform(0.03, 0.08))


@asynccontextmanager
async def request_binding(session, dialect, make_client):
    permit = None
    token = None
    try:
        async with session.model_lock:
            pending, session.model_change = session.model_change, None
            pool = session.ctx.model_pool
            saved = session.trace.info.get(MODEL_BINDING) or {}
            chosen = None
            if pending is not None and pending.get("pool") is not None:
                session.ctx = replace(session.ctx, model_pool=pending["pool"])
                pool = pending["pool"]
                session.model_bound = False
                saved = {}
            if pending is not None and pending.get("pool") is None:
                session.ctx = replace(
                    session.ctx,
                    model=pending["model"],
                    client=pending.get("client") or session.ctx.client,
                    model_pool=None,
                )
                session.model_bound = True
            elif pool is not None and not session.model_bound:
                chosen = next(
                    (
                        item
                        for item in pool.deployments
                        if item.id == saved.get("deployment")
                        and item.model == saved.get("model")
                        and item.client.model_dump(mode="json") == saved.get("client")
                    ),
                    None,
                )
                if chosen is None:
                    chosen, permit = await _choose(pool, dialect)
                session.ctx = replace(
                    session.ctx, model=chosen.model, client=chosen.client
                )
                session.model_bound = True
            direct_changed = (
                pool is None
                and saved
                and (
                    saved.get("model") != session.ctx.model
                    or saved.get("client") != session.ctx.client.model_dump(mode="json")
                )
            )
            if pending is not None or chosen is not None or direct_changed:
                affinity = (
                    saved.get("session_id")
                    or session.ctx.session_id
                    or session.trace.id
                )
                session.ctx = replace(session.ctx, session_id=affinity)
                session.client = make_client(session.ctx.client)
                binding = {
                    "deployment": chosen.id if chosen is not None else None,
                    "model": session.ctx.model,
                    "client": session.ctx.client.model_dump(mode="json"),
                    "session_id": affinity,
                }
                if binding != saved:
                    session.trace.info.setdefault("model_changes", []).append(
                        {**binding, "from_call": len(session.trace.calls)}
                    )
                session.trace.info[MODEL_BINDING] = binding
                session.trace.agent.config = session.trace.agent.config.model_copy(
                    update={
                        "model": session.ctx.model,
                        "client": session.ctx.client,
                        "session_id": affinity,
                        "model_pool": session.ctx.model_pool,
                    }
                )
                session.trace.notify()
        token = reserved_permit.set(permit)
        yield
    finally:
        if token is not None:
            reserved_permit.reset(token)
        if permit is not None:
            permit.close()
