"""The fault taxonomy: what kind of thing failed a row.

`fault_kind` classifies a trace error (`Trace.errors` entries, typed by their `type`
name) or a live exception into one of four kinds:

- `permanent` — the endpoint refused the credentials for good (a provider 401/403):
  no retry re-enables it; the producer decides what a permanent fault means.
- `outage` — the endpoint's failure, not the turn's (a provider 5xx/429, or a request
  that never got a status): the engine holds new admissions while it lasts.
- `platform` — the sandbox/runtime side failed (provisioning, exec, the
  interception/tunnel path).
- `turn` — everything else: the attempt's own fault.

Ported from the proven upstream data-flywheel seat.py `provider_outage`/
`permanent_provider_error` (prime-envs-private origin/main; its evidence: a
12-minute endpoint restart cost 45 turns / 2,705 calls / 24 attempt ids before the
fix — the outage must never count as the turn's fault).
"""

from __future__ import annotations

from typing import Literal

import verifiers.v1.errors as vf_errors
from verifiers.v1.errors import InterceptionError, ProviderError, SandboxError

FaultKind = Literal["permanent", "outage", "platform", "turn"]


def _class_of(error) -> type | None:
    """A live exception's class, or the class a trace error's `type` names."""
    if isinstance(error, BaseException):
        return type(error)
    return getattr(vf_errors, getattr(error, "type", None) or "", None)


def _is(error, base: type) -> bool:
    cls = _class_of(error)
    return isinstance(cls, type) and issubclass(cls, base)


def _status(error) -> int | None:
    """The typed HTTP status a provider error carries, None when it carries none."""
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def fault_kind(error) -> FaultKind:
    """Classify one trace error or exception; a None or untyped failure is `turn`."""
    if error is not None and _is(error, ProviderError):
        status = _status(error)
        if status in (401, 403):
            return "permanent"
        if status is None or status == 429 or status >= 500:
            return "outage"
    elif _is(error, (SandboxError, InterceptionError)):
        return "platform"
    return "turn"
