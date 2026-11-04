"""Run-wide pacing of Prime control-plane HTTP attempts, including SDK retries.

Only the API client's HTTPX hooks are changed, never the separate gateway/exec
transport. Workers in a run must share LIMITER_DIR and a comparable wall clock.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import math
import random
import re
import time
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from verifiers.v1.runtimes.limiters import LIMITER_DIR

logger = logging.getLogger(__name__)


class PrimeRequestLimiter:
    """Admit only a request due *now*, then sleep and recheck without reservations.

    A response can extend the shared cooldown while other workers are waiting.
    Cancellation can consume at most the current slot, not a future backlog.
    File locking and IO always run outside the event loop.
    """

    def __init__(self, requests_per_min: int, scope: str) -> None:
        if requests_per_min <= 0:
            raise ValueError("Prime requests_per_min must be positive")
        self._rate = requests_per_min
        self._interval = 60 / requests_per_min
        key = hashlib.sha256(scope.encode()).hexdigest()
        self._path = LIMITER_DIR / f"prime-requests-{key}.json"

    def _update(self, cooldown: float | None = None) -> float:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a+") as state_file:
            fcntl.flock(state_file.fileno(), fcntl.LOCK_EX)
            try:
                state_file.seek(0)
                data = state_file.read()
                state = (
                    json.loads(data)
                    if data
                    else {"rate": self._rate, "next": 0.0, "cooldown": 0.0}
                )
                if state["rate"] != self._rate:
                    raise ValueError(
                        "Conflicting Prime requests_per_min for the same run scope"
                    )
                now = time.time()
                if cooldown is not None:
                    state["cooldown"] = max(state["cooldown"], now + cooldown)
                    wait = state["cooldown"] - now
                else:
                    wait = max(state["next"], state["cooldown"]) - now
                    if wait > 0:
                        return wait
                    state["next"] = now + self._interval
                    wait = 0.0
                state_file.seek(0)
                state_file.truncate()
                json.dump(state, state_file)
                state_file.flush()
                return wait
            finally:
                fcntl.flock(state_file.fileno(), fcntl.LOCK_UN)

    async def acquire(self) -> None:
        while True:
            wait = await asyncio.to_thread(self._update)
            if wait <= 0:
                return
            await asyncio.sleep(wait)

    async def cooldown(self, seconds: float) -> float:
        return await asyncio.to_thread(self._update, seconds)


def _retry_after_delay(value: str | None) -> float:
    """Honor server seconds/date as a minimum; invalid/missing uses 10–60s jitter."""
    seconds = None
    if value is not None:
        try:
            seconds = float(value)
            if not math.isfinite(seconds) or seconds < 0:
                seconds = None
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=UTC)
                seconds = max(0.0, date.timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                pass
    if seconds is None:
        return random.uniform(10.0, 60.0)
    return seconds + random.uniform(0.1, 1.0)


# Only literal templates are logged. Unknown routes never reproduce URL segments.
_PATH_TEMPLATES = (
    "/api/v1/sandbox",
    "/api/v1/sandbox/batch",
    "/api/v1/sandbox/{id}",
    "/api/v1/sandbox/{id}/egress-policy",
    "/api/v1/sandbox/{id}/logs",
    "/api/v1/sandbox/{id}/checkpoints",
    "/api/v1/sandbox/{id}/ssh-session",
    "/api/v1/sandbox/{id}/ssh-session/{id}",
    "/api/v1/sandbox/checkpoints/batch",
    "/api/v1/sandbox/checkpoints/{id}",
    "/api/v1/sandbox/networks",
    "/api/v1/sandbox/networks/{id}",
    "/api/v1/sandbox/networks/{id}/rules",
    "/api/v1/sandbox/networks/{id}/members/{id}",
)
_PATH_PATTERNS = [
    (re.compile(re.escape(template).replace(r"\{id\}", "[^/]+")), template)
    for template in _PATH_TEMPLATES
]


def _path_template(request: httpx.Request) -> str:
    # Prefer literal paths over the generic sandbox/{id} route.
    for pattern, template in sorted(_PATH_PATTERNS, key=lambda item: "{id}" in item[1]):
        if pattern.fullmatch(request.url.path):
            return template
    return "/api/v1/{endpoint}"


def install_prime_request_hooks(
    sandbox_client: Any, requests_per_min: int | None, scope: str
) -> None:
    """Install once on AsyncSandboxClient.client.client, preserving existing hooks.

    A loop's shared SDK client must use one rate and scope, including when disabled;
    conflicting leases fail rather than silently bypassing or multiplying pacing.
    None disables pacing. This function adds no retries or retry-budget changes.
    """
    client = sandbox_client.client.client
    config = (requests_per_min, scope)
    existing = getattr(client, "_verifiers_prime_request_config", None)
    if existing is not None:
        if existing != config:
            raise ValueError("Conflicting Prime request pacing on a shared SDK client")
        return
    if requests_per_min is not None:
        limiter = PrimeRequestLimiter(requests_per_min, scope)

        async def before_request(request: httpx.Request) -> None:
            await limiter.acquire()

        async def after_response(response: httpx.Response) -> None:
            cooldown = 0.0
            if response.status_code == 429:
                cooldown = await limiter.cooldown(
                    _retry_after_delay(response.headers.get("Retry-After"))
                )
            method = response.request.method
            if method not in {
                "GET",
                "POST",
                "PUT",
                "PATCH",
                "DELETE",
                "HEAD",
                "OPTIONS",
            }:
                method = "OTHER"
            logger.info(
                "prime_request method=%s path_template=%s status=%s cooldown_s=%.3f",
                method,
                _path_template(response.request),
                response.status_code,
                cooldown,
            )

        client.event_hooks["request"].append(before_request)
        client.event_hooks["response"].append(after_response)
    client._verifiers_prime_request_config = config
