"""Preflight: run-blocking facts, checked before any spend.

`preflight(config)` answers "can this run start here?" with typed refusals: sandbox
credentials, and — under `contact=True` — whether each distinct model endpoint serves
every model a seat will ask it for. The Engine never calls this itself: the producer
decides when to check (a refusal names what to fix; nothing is killed, no settings
change). Ported from the proven upstream data-flywheel run.py `prerequisites`/
`credentials`/`served` (upstream-comparison-871ae9d8), reading credentials exactly as
verifiers' own auth helper does (`ensure_prime_auth`: `$PRIME_API_KEY`, then the
Prime CLI config) and keys exactly as the client will send them (`resolve_api_key`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from verifiers.v1.agent import agent_config_fields
from verifiers.v1.configs.client import (
    BaseClientConfig,
    EvalClientConfig,
    resolve_api_key,
)
from verifiers.v1.utils.prime import load_prime_config

if TYPE_CHECKING:
    from verifiers.v1.flow.flow import FlowConfig


@dataclass(frozen=True)
class Check:
    """One prerequisite of a run on this host: the `subject`, the `fact`; `ok` False
    refuses the start."""

    subject: str
    fact: str
    ok: bool = True

    def __str__(self) -> str:
        return f"{self.subject}: {self.fact}"


SANDBOXES = "sandboxes"
NO_KEY = "EMPTY"  # what verifiers' `resolve_api_key` answers for a key nobody set
MODELS_TIMEOUT_S = 15.0  # seconds one `GET <base_url>/models` gets
SHOWN = 8  # model ids named when the endpoint does not serve the one asked


async def preflight(config: FlowConfig, contact: bool = True) -> list[Check]:
    """The sandbox credentials; per distinct model endpoint its key and, under
    `contact`, whether it serves the seats' models. One `GET /models` per distinct
    endpoint (base_url + key var + headers), the seats' models checked against its
    list — the seats resolve exactly as the engine's `Engine.seat` resolves them."""
    checks = [credentials()]
    endpoints: dict[tuple, tuple[BaseClientConfig, set[str | None]]] = {}
    for name, seat in agent_config_fields(config).items():
        model = seat.model if seat.model is not None else config.model
        client = seat.client if seat.client is not None else config.client
        client = client if client is not None else EvalClientConfig()
        if model is None:
            checks.append(
                Check(f"seat {name}", "no model; set the seat's or config.model", ok=False)
            )
            continue
        identity = (
            client.base_url,
            client.api_key_var,
            tuple(sorted(client.headers.items())),
        )
        endpoints.setdefault(identity, (client, set()))[1].add(model)
    for _identity, (client, models) in sorted(
        endpoints.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        checks.append(keyed(client))
        if contact:
            checks.extend(await served(client, sorted(models)))
    return checks


def credentials() -> Check:
    """The sandbox platform's credentials, as verifiers' runtime reads them
    (`ensure_prime_auth`)."""
    if os.environ.get("PRIME_API_KEY"):
        return Check(SANDBOXES, "Prime credentials from PRIME_API_KEY")
    if load_prime_config().get("api_key"):
        return Check(SANDBOXES, "Prime credentials from `prime login`")
    return Check(
        SANDBOXES, "no Prime credentials; run `prime login` or set PRIME_API_KEY", ok=False
    )


def endpoint(client: BaseClientConfig) -> str:
    return f"model endpoint {client.base_url}"


def keyed(client: BaseClientConfig) -> Check:
    """Where the endpoint's key comes from: `$<var>`, `prime login`, or nowhere (a
    fact: a server of yours may take none)."""
    if os.environ.get(client.api_key_var):
        return Check(endpoint(client), f"key from ${client.api_key_var}")
    if resolve_api_key(client) != NO_KEY:
        return Check(
            endpoint(client), f"key from `prime login` (${client.api_key_var} is not set)"
        )
    return Check(endpoint(client), f"no key: ${client.api_key_var} is not set; the requests carry none")


async def served(client: BaseClientConfig, models: list[str]) -> list[Check]:
    """`GET <base_url>/models` as the client's requests go: a model absent from the
    list refuses; anything else the endpoint says is one line, a fact (the first turn
    tells whether it serves)."""
    headers = {**client.headers, "Authorization": f"Bearer {resolve_api_key(client)}"}
    try:
        async with httpx.AsyncClient(timeout=MODELS_TIMEOUT_S) as http:
            response = await http.get(
                client.base_url.rstrip("/") + "/models", headers=headers
            )
        response.raise_for_status()
        listed = response.json()
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        return [
            Check(endpoint(client), f"/models answered {status}; the first turn tells whether it serves")
        ]
    except (httpx.HTTPError, ValueError) as error:
        return [
            Check(
                endpoint(client),
                f"/models not read ({type(error).__name__}); the first turn tells whether it serves",
            )
        ]
    data = listed.get("data") if isinstance(listed, dict) else None
    entries = data if isinstance(data, list) else []
    ids = sorted(
        {m["id"] for m in entries if isinstance(m, dict) and isinstance(m.get("id"), str)}
    )
    if not ids:
        return [Check(endpoint(client), "/models lists nothing; the first turn tells whether it serves")]
    checks = []
    for model in models:
        if model in ids:
            checks.append(Check(endpoint(client), f"serves {model}"))
            continue
        name = model.rsplit("/", 1)[-1].lower()
        alike = [served_id for served_id in ids if name in served_id.lower()] or ids
        named = ", ".join(alike[:SHOWN])
        checks.append(
            Check(endpoint(client), f"does not serve {model}; it serves {named} ({len(ids)} in all)", ok=False)
        )
    return checks
