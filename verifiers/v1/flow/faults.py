"""What kind of thing failed a row: `permanent` (the endpoint refused the
credentials), `outage` (the endpoint's failure, a 5xx/429 or no status), `platform`
(the runtime or interception side), or `turn` (the attempt's own). The engine
reports it on `RowResult.fault`; what to do about it is the producer's call."""

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


def _is(error, base: type | tuple[type, ...]) -> bool:
    cls = _class_of(error)
    return isinstance(cls, type) and issubclass(cls, base)


def fault_kind(error) -> FaultKind:
    if error is not None and _is(error, ProviderError):
        status = getattr(error, "status_code", None)
        if status in (401, 403):
            return "permanent"
        if not isinstance(status, int) or status == 429 or status >= 500:
            return "outage"
    elif _is(error, (SandboxError, InterceptionError)):
        return "platform"
    return "turn"
