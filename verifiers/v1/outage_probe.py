"""The outage probe: has an outage of a given kind (`account`, `quota`, `io`) ended?

    report = await vf.probe("account", seats, client=run_client)
    report.status       # "cleared" | "out" | "unknown"
    report.failures     # the checks that did not clear, with what they met
    report.notes        # what was read but does not decide, such as an unread wallet

A caller paused on an outage resumes only on `cleared`. Each check is read-only or
stateless, and none creates a sandbox:

- the Prime platform, when a seat runs on a Prime box: one `GET /sandbox` (a page of
  one, scoped to the team) for every kind; for `account`, also `GET /billing/wallet`.
- each distinct endpoint, key, and model the seats call (eval clients only; a
  training endpoint is a local engine with no billing): one chat completion of one
  token, the smallest request that meets a provider's billing, auth, or rate limit.
  A model that refuses `max_tokens` (a reasoning model) is asked once more with
  `max_completion_tokens`. A refusal of a request sent with no key (the endpoint's
  key variable is unset) is `unknown`: a configuration problem, not an outage. An endpoint without `/chat/completions` (Anthropic-native,
  Responses-only) always reports `unknown`.

A check that fails with an outage is `out`, with the kind it met, whatever kind was
probed (an account outage still blocks a run paused on quota). One that fails with no
outage (no API key configured, an unknown model, an unexpected response) is `unknown`;
so is a probe with no checks. `out` wins over `unknown`. Failures are recorded, never
raised.

The wallet blocks only when it says the account is unpaid: a balance of zero or less,
or a 402. A wallet the key may not read (401, 403, 404), a reply without a balance,
or one that fails in transit is a note: the sandbox listing already proved auth and
the platform. A postpaid or invoiced team whose balance is zero or less stays paused
until an operator clears it.

`GET /sandbox` meets neither the create rate limit nor a sandbox cap, which only a
created box meets. So after a Prime `quota` pause the probe clears as soon as the
platform answers, and acts as a retry every probe interval: a cap or limit that still
binds pauses the run again on its next box.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from verifiers.v1.clients.base import build_async_openai
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.client import (
    ClientConfig,
    EvalClientConfig,
    resolve_api_key,
)
from verifiers.v1.errors import OutageKind, outage_kind, sandbox_error_kind

ProbeStatus = Literal["cleared", "out", "unknown"]
"""`cleared`: the evidence says the outage is over. `out`: a check met an outage.
`unknown`: the probe could not tell (a check failed with no outage, or nothing was
checked)."""

PROBE_TIMEOUT = 60.0
"""Seconds one check may take before it counts as an `io` outage."""


@dataclass(frozen=True)
class ProbeCheck:
    """One check: `what` it reached, its status, and when it did not clear, the
    outage `kind` it met (None when it met no outage) and the error."""

    what: str
    status: ProbeStatus
    kind: OutageKind | None = None
    error: str | None = None


@dataclass(frozen=True)
class ProbeReport:
    """The checks one probe of `kind` ran. `out` when any check met an outage, else
    `unknown` when any could not tell or none ran, else `cleared`."""

    kind: OutageKind
    checks: tuple[ProbeCheck, ...] = ()
    notes: tuple[str, ...] = ()
    """What a check read that does not decide the status, such as an unread wallet."""

    @property
    def status(self) -> ProbeStatus:
        statuses = {check.status for check in self.checks}
        if "out" in statuses:
            return "out"
        if "unknown" in statuses or not statuses:
            return "unknown"
        return "cleared"

    @property
    def cleared(self) -> bool:
        return self.status == "cleared"

    @property
    def failures(self) -> list[ProbeCheck]:
        return [check for check in self.checks if check.status != "cleared"]


async def probe(
    kind: OutageKind,
    seats: Iterable[AgentConfig] = (),
    *,
    client: ClientConfig | None = None,
    prime: bool | None = None,
    timeout: float = PROBE_TIMEOUT,
) -> ProbeReport:
    """Check whether an outage of `kind` has ended for `seats`, all checks at once.
    `prime` probes the Prime platform (None: when a seat runs on a Prime box). A seat
    without its own client calls `client` (the run's), else Prime Inference."""
    seats = list(seats)
    endpoints: dict[tuple[str, str, str], EvalClientConfig] = {}
    for seat in seats:
        endpoint = seat.client or client or EvalClientConfig()
        if seat.model and isinstance(endpoint, EvalClientConfig):
            key = (endpoint.base_url, seat.model, endpoint.api_key_var)
            endpoints.setdefault(key, endpoint)
    if prime is None:
        prime = any(seat.runtime.type == "prime" for seat in seats)
    batches = [
        _model(endpoint, model, timeout)
        for (_, model, _), endpoint in endpoints.items()
    ]
    if prime:
        batches.append(_prime(kind, timeout))
    results = [item for batch in await asyncio.gather(*batches) for item in batch]
    return ProbeReport(
        kind=kind,
        checks=tuple(item for item in results if isinstance(item, ProbeCheck)),
        notes=tuple(item for item in results if isinstance(item, str)),
    )


async def _model(
    endpoint: EvalClientConfig, model: str, timeout: float
) -> list[ProbeCheck | str]:
    what = f"model {model} at {endpoint.base_url}"
    check = await _check(what, _complete(endpoint, model), timeout)
    if check.kind == "account" and resolve_api_key(endpoint) == "EMPTY":
        # Sent without a key (its variable is unset): the refusal is the run's
        # configuration, not an account outage, and no probe will see it clear. A
        # keyless endpoint that answers still clears.
        return [
            ProbeCheck(
                what,
                "unknown",
                error=f"${endpoint.api_key_var} is unset: {check.error}",
            )
        ]
    return [check]


async def _complete(endpoint: EvalClientConfig, model: str) -> None:
    """One chat completion of one token, as `max_completion_tokens` when the model
    refuses `max_tokens` (OpenAI's reasoning models answer that with a 400)."""
    import openai

    client = build_async_openai(endpoint)
    messages: Any = [{"role": "user", "content": "ping"}]
    try:
        try:
            await client.chat.completions.create(
                model=model, messages=messages, max_tokens=1
            )
        except openai.BadRequestError as error:
            if "max_tokens" not in f"{getattr(error, 'param', None)} {error}":
                raise
            await client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=1
            )
    finally:
        await client.close()


def _prime_api() -> Any:
    """A Prime API client from the user's Prime config (`prime login` or
    `$PRIME_API_KEY`), as the sandbox SDK builds one."""
    from prime_sandboxes.core import AsyncAPIClient

    return AsyncAPIClient()


async def _prime(kind: OutageKind, timeout: float) -> list[ProbeCheck | str]:
    what = "prime sandboxes"
    try:
        api = _prime_api()
    except Exception as error:  # noqa: BLE001 - a failed check is the report's
        return [_failed(what, error)]
    try:
        checks: list[Coroutine[Any, Any, ProbeCheck | str]] = [
            _check(what, _sandboxes(api), timeout)
        ]
        if kind == "account":
            checks.append(_wallet(api, timeout))
        return list(await asyncio.gather(*checks))
    finally:
        with contextlib.suppress(Exception):
            await api.aclose()


async def _sandboxes(api: Any) -> None:
    """List one of the team's sandboxes: the platform answers, auth holds, and it does
    not rate-limit reads."""
    params: dict[str, Any] = {"page": 1, "per_page": 1}
    if team := api.config.team_id:
        params["team_id"] = team
    await api.get("/sandbox", params=params)


async def _wallet(api: Any, timeout: float) -> ProbeCheck | str:
    """Read the team's wallet: `out` when it is unpaid (a balance of zero or less, or a
    402), else a note when it cannot be read."""
    from prime_sandboxes.core import PaymentRequiredError

    what = "prime wallet"
    params: dict[str, Any] = {"limit": 1}
    if team := api.config.team_id:
        params["teamId"] = team
    try:
        reply = await asyncio.wait_for(
            api.get("/billing/wallet", params=params), timeout
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - an unread wallet is a note
        if isinstance(error, PaymentRequiredError):
            return ProbeCheck(what, "out", "account", _describe(error))
        return f"{what} not read, so it does not block: {_describe(error)}"
    balance = reply.get("balance_usd") if isinstance(reply, dict) else None
    if isinstance(balance, bool) or not isinstance(balance, int | float):
        return f"{what} has no balance, so it does not block: {balance!r}"
    if balance <= 0:
        return ProbeCheck(what, "out", "account", f"wallet balance {balance} USD")
    return ProbeCheck(what, "cleared")


async def _check(
    what: str, work: Coroutine[Any, Any, None], timeout: float
) -> ProbeCheck:
    """Run one check: it clears when `work` returns, else is `out` on the outage it
    raised (a timeout is `io`) or `unknown`."""
    try:
        await asyncio.wait_for(work, timeout)
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - a failed check is the report's
        return _failed(what, error)
    return ProbeCheck(what=what, status="cleared")


def _failed(what: str, error: BaseException) -> ProbeCheck:
    kind = _outage(error)
    return ProbeCheck(
        what=what,
        status="out" if kind is not None else "unknown",
        kind=kind,
        error=_describe(error),
    )


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:500]


def _outage(error: BaseException) -> OutageKind | None:
    """The outage a failed check met, from typed evidence (`outage_kind`; a timeout is
    `io`), or for a Prime API error, the sandbox SDK's typed errors and its
    `HTTP <status>:` prefix, read as for any sandbox call."""
    if (kind := outage_kind(error)) is not None:
        return kind
    if type(error).__module__.startswith("prime_sandboxes"):
        kind = sandbox_error_kind(error)
        return kind if kind in ("account", "quota", "io") else None
    return None


__all__ = ["PROBE_TIMEOUT", "ProbeCheck", "ProbeReport", "ProbeStatus", "probe"]
