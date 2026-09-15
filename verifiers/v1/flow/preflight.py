"""Preflight: can this run start here? Sandbox credentials, each model endpoint's
key, and (with `contact`) whether the endpoint serves every model the seats ask
for. Nothing is spent; a refusal names what to fix."""

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
    subject: str
    fact: str
    ok: bool = True

    def __str__(self) -> str:
        return f"{self.subject}: {self.fact}"


NO_KEY = "EMPTY"  # what `resolve_api_key` answers when nobody set a key
MODELS_TIMEOUT_S = 15.0
SHOWN = 8  # model ids named when the asked-for one is missing


async def preflight(config: FlowConfig, contact: bool = True) -> list[Check]:
    checks = [credentials()]
    endpoints: dict[str, tuple[BaseClientConfig, set[str]]] = {}
    for name, seat in agent_config_fields(config).items():
        model = seat.model if seat.model is not None else config.model
        client = seat.client if seat.client is not None else config.client
        client = client if client is not None else EvalClientConfig()
        if model is None:
            checks.append(
                Check(
                    f"seat {name}", "no model; set the seat's or config.model", ok=False
                )
            )
            continue
        endpoints.setdefault(client.model_dump_json(), (client, set()))[1].add(model)
    for client, models in endpoints.values():
        checks.append(keyed(client))
        if contact:
            checks.extend(await served(client, sorted(models)))
    return checks


def credentials() -> Check:
    """The sandbox platform's credentials, read as verifiers' runtime reads them."""
    if os.environ.get("PRIME_API_KEY"):
        return Check("sandboxes", "Prime credentials from PRIME_API_KEY")
    if load_prime_config().get("api_key"):
        return Check("sandboxes", "Prime credentials from `prime login`")
    return Check(
        "sandboxes",
        "no Prime credentials; run `prime login` or set PRIME_API_KEY",
        ok=False,
    )


def endpoint(client: BaseClientConfig) -> str:
    return f"model endpoint {client.base_url}"


def keyed(client: BaseClientConfig) -> Check:
    """Where the endpoint's key comes from; no key is a fact, not a refusal."""
    if os.environ.get(client.api_key_var):
        return Check(endpoint(client), f"key from ${client.api_key_var}")
    if resolve_api_key(client) != NO_KEY:
        return Check(
            endpoint(client), f"key from `prime login` (${client.api_key_var} unset)"
        )
    return Check(endpoint(client), f"no key: ${client.api_key_var} is not set")


async def served(client: BaseClientConfig, models: list[str]) -> list[Check]:
    """`GET <base_url>/models`: a model absent from the list refuses; an endpoint
    that will not answer is one line, and the first turn tells."""
    headers = {**client.headers, "Authorization": f"Bearer {resolve_api_key(client)}"}
    try:
        async with httpx.AsyncClient(timeout=MODELS_TIMEOUT_S) as http:
            response = await http.get(
                client.base_url.rstrip("/") + "/models", headers=headers
            )
        response.raise_for_status()
        listed = response.json()
    except httpx.HTTPStatusError as error:
        return [
            Check(endpoint(client), f"/models answered {error.response.status_code}")
        ]
    except (httpx.HTTPError, ValueError) as error:
        return [Check(endpoint(client), f"/models not read ({type(error).__name__})")]
    data = listed.get("data") if isinstance(listed, dict) else None
    ids = sorted(
        {
            m["id"]
            for m in data or []
            if isinstance(m, dict) and isinstance(m.get("id"), str)
        }
    )
    if not ids:
        return [Check(endpoint(client), "/models lists nothing")]
    checks = []
    for model in models:
        if model in ids:
            checks.append(Check(endpoint(client), f"serves {model}"))
            continue
        name = model.rsplit("/", 1)[-1].lower()
        alike = [i for i in ids if name in i.lower()] or ids
        checks.append(
            Check(
                endpoint(client),
                f"does not serve {model}; it serves {', '.join(alike[:SHOWN])} ({len(ids)} in all)",
                ok=False,
            )
        )
    return checks
