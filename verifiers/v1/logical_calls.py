"""Harness-owned request identity at the model interception boundary."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LogicalCall:
    """One stable model call that may cross several transport attempts."""

    key: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ResolvedCallHeaders:
    """Logical identity plus the request headers safe to forward upstream."""

    call: LogicalCall | None
    forward_headers: dict[str, str]


class LogicalCallResolver(Protocol):
    def resolve(
        self, headers: Mapping[str, str], *, session_id: str
    ) -> ResolvedCallHeaders: ...
