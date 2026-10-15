"""The stock outage probe: can the run create a sandbox, and does each model answer?

    report = await vf.probe(seats, client=run_client)
    if report.ok: ...                       # every check passed: the outage is over
    report.failures                         # the checks that did not, with their kind

`probe` creates (and deletes) one box per distinct runtime the seats use, and sends one
tiny completion (one token) per distinct endpoint and model, all at once. A failure is
recorded with its `OutageKind` (None when it is not infrastructure, e.g. an unknown
model), never raised. A caller paused on an outage clears it when the probe passes.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass, field

from verifiers.v1.clients import ClientConfig, EvalClientConfig
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.errors import OutageKind, outage_kind
from verifiers.v1.runtimes import RuntimeConfig

PROBE_TIMEOUT = 120.0
"""Seconds one check may take before it counts as an `io` outage."""


@dataclass(frozen=True)
class ProbeCheck:
    """One check: `what` it reached (`sandbox <type>` or `model <name> at <url>`),
    whether it passed, and the outage it met when it did not."""

    what: str
    ok: bool
    error: str | None = None
    kind: OutageKind | None = None


@dataclass(frozen=True)
class ProbeReport:
    checks: list[ProbeCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> list[ProbeCheck]:
        return [check for check in self.checks if not check.ok]


async def probe(
    seats: Iterable[AgentConfig],
    *,
    client: ClientConfig | None = None,
    sandboxes: bool = True,
    models: bool = True,
    timeout: float = PROBE_TIMEOUT,
) -> ProbeReport:
    """Check each distinct runtime of `seats` (a box created and torn down) and each
    distinct endpoint and model (one completion of one token), concurrently. A seat
    without its own client uses `client` (the run's), else Prime Inference."""
    runtimes: dict[str, RuntimeConfig] = {}
    endpoints: dict[tuple[str, str], tuple[ClientConfig, str]] = {}
    for seat in seats:
        if sandboxes:
            runtimes.setdefault(
                json.dumps(seat.runtime.model_dump(mode="json"), sort_keys=True),
                seat.runtime,
            )
        endpoint = seat.client or client or EvalClientConfig()
        if models and seat.model:
            endpoints.setdefault(
                (endpoint.base_url, seat.model), (endpoint, seat.model)
            )
    checks = [
        *(_sandbox(runtime, timeout) for runtime in runtimes.values()),
        *(_model(endpoint, model, timeout) for endpoint, model in endpoints.values()),
    ]
    return ProbeReport(checks=list(await asyncio.gather(*checks)))


async def _sandbox(config: RuntimeConfig, timeout: float) -> ProbeCheck:
    from verifiers.v1.runtimes import provision_backend

    what = f"sandbox {config.type}"

    async def create() -> None:
        async with provision_backend(config, env={}):
            pass

    return await _check(what, create(), timeout)


async def _model(endpoint: ClientConfig, model: str, timeout: float) -> ProbeCheck:
    from verifiers.v1.clients.base import build_async_openai

    what = f"model {model} at {endpoint.base_url}"

    async def ask() -> None:
        openai = build_async_openai(endpoint)
        try:
            await openai.with_options(max_retries=0).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        finally:
            await openai.close()

    return await _check(what, ask(), timeout)


async def _check(what: str, work, timeout: float) -> ProbeCheck:
    try:
        await asyncio.wait_for(work, timeout)
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - a failed check is the report's
        kind = "io" if isinstance(error, TimeoutError) else outage_kind(error)
        return ProbeCheck(
            what=what,
            ok=False,
            error=f"{type(error).__name__}: {error}"[:500],
            kind=kind,
        )
    return ProbeCheck(what=what, ok=True)


__all__ = ["PROBE_TIMEOUT", "ProbeCheck", "ProbeReport", "probe"]
