"""Datadog telemetry for verifiers.

Sends discrete structured log events to Datadog's HTTP Logs API using
``httpx`` (already a runtime dependency).  Each instrumentation point
emits one event per occurrence so Datadog can aggregate, filter, and
alert on raw data.

Configuration (env vars):

- ``DD_API_KEY``:  Datadog API key.  When set, HTTP POSTs are enabled.
- ``DD_SITE``:     Datadog site (default ``"datadoghq.com"``).
                   Examples: ``"us5.datadoghq.com"``, ``"datadoghq.eu"``.
- ``VF_TELEMETRY``: Set to ``"1"`` to enable stderr logging even without
                    a Datadog key.  When ``DD_API_KEY`` is set, telemetry
                    is always enabled regardless of this flag.
- ``VF_TELEMETRY_SERVICE``: service tag (default ``"verifiers"``).

When ``DD_API_KEY`` is absent *and* ``VF_TELEMETRY`` is not ``"1"``,
every public function is a no-op with near-zero overhead (a single
boolean check).

All errors inside the telemetry layer are swallowed and logged to
stderr so they **never** propagate to callers or interfere with
evaluations / training.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import platform
import sys
import threading
from datetime import datetime, timezone
from typing import Any

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_INSTANCE: _Telemetry | None = None
_LOCK = threading.Lock()


def _get() -> _Telemetry:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = _Telemetry()
    return _INSTANCE


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def enabled() -> bool:
    """Return True when at least one telemetry sink is active."""
    return _get().enabled


def emit(
    event_name: str,
    data: dict[str, Any] | None = None,
    *,
    level: str = "info",
) -> None:
    """Emit a single discrete event.  Safe to call from sync or async code."""
    tel = _get()
    if not tel.enabled:
        return
    try:
        tel.emit(event_name, data or {}, level=level)
    except Exception as exc:
        _safe_stderr(f"VF_TELEMETRY_EMIT_ERROR event={event_name} err={exc!r}")


def flush() -> None:
    """Flush pending log entries to Datadog.  Best-effort, non-blocking."""
    tel = _get()
    if not tel.enabled:
        return
    try:
        tel.flush()
    except Exception as exc:
        _safe_stderr(f"VF_TELEMETRY_FLUSH_ERROR err={exc!r}")


# ---------------------------------------------------------------------------
# Convenience wrappers used by instrumented code
# ---------------------------------------------------------------------------


def rollout_started(
    *,
    env_id: str = "",
    model: str = "",
    example_id: int | str = "",
    trajectory_id: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    emit(
        "rollout.started",
        {
            "env_id": env_id,
            "model": model,
            "example_id": example_id,
            "trajectory_id": trajectory_id,
            **(extra or {}),
        },
    )


def rollout_completed(
    *,
    env_id: str = "",
    model: str = "",
    example_id: int | str = "",
    trajectory_id: str = "",
    reward: float | None = None,
    num_turns: int | None = None,
    duration_s: float | None = None,
    stop_condition: str = "",
    error: str = "",
    input_tokens: float = 0,
    output_tokens: float = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    emit(
        "rollout.completed",
        {
            "env_id": env_id,
            "model": model,
            "example_id": example_id,
            "trajectory_id": trajectory_id,
            "reward": reward,
            "num_turns": num_turns,
            "duration_s": duration_s,
            "stop_condition": stop_condition,
            "error": error,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **(extra or {}),
        },
        level="error" if error else "info",
    )


def group_started(
    *,
    env_id: str = "",
    model: str = "",
    example_id: int | str = "",
    group_size: int = 0,
) -> None:
    emit(
        "group.started",
        {
            "env_id": env_id,
            "model": model,
            "example_id": example_id,
            "group_size": group_size,
        },
    )


def group_completed(
    *,
    env_id: str = "",
    model: str = "",
    example_id: int | str = "",
    group_size: int = 0,
    duration_s: float | None = None,
    avg_reward: float | None = None,
) -> None:
    emit(
        "group.completed",
        {
            "env_id": env_id,
            "model": model,
            "example_id": example_id,
            "group_size": group_size,
            "duration_s": duration_s,
            "avg_reward": avg_reward,
        },
    )


def setup_started(*, env_id: str = "", trajectory_id: str = "") -> None:
    emit("setup.started", {"env_id": env_id, "trajectory_id": trajectory_id})


def setup_completed(
    *,
    env_id: str = "",
    trajectory_id: str = "",
    duration_s: float | None = None,
    error: str = "",
) -> None:
    emit(
        "setup.completed",
        {
            "env_id": env_id,
            "trajectory_id": trajectory_id,
            "duration_s": duration_s,
            "error": error,
        },
        level="error" if error else "info",
    )


def turn_started(
    *,
    env_id: str = "",
    trajectory_id: str = "",
    turn_index: int = 0,
) -> None:
    emit(
        "turn.started",
        {
            "env_id": env_id,
            "trajectory_id": trajectory_id,
            "turn_index": turn_index,
        },
    )


def turn_completed(
    *,
    env_id: str = "",
    trajectory_id: str = "",
    turn_index: int = 0,
    duration_s: float | None = None,
    model_duration_s: float | None = None,
    env_duration_s: float | None = None,
    is_truncated: bool = False,
    error: str = "",
) -> None:
    emit(
        "turn.completed",
        {
            "env_id": env_id,
            "trajectory_id": trajectory_id,
            "turn_index": turn_index,
            "duration_s": duration_s,
            "model_duration_s": model_duration_s,
            "env_duration_s": env_duration_s,
            "is_truncated": is_truncated,
            "error": error,
        },
        level="error" if error else "info",
    )


def model_request(
    *,
    env_id: str = "",
    model: str = "",
    trajectory_id: str = "",
    turn_index: int = 0,
    duration_s: float | None = None,
    input_tokens: float = 0,
    output_tokens: float = 0,
    is_truncated: bool = False,
    error: str = "",
) -> None:
    emit(
        "model.response",
        {
            "env_id": env_id,
            "model": model,
            "trajectory_id": trajectory_id,
            "turn_index": turn_index,
            "duration_s": duration_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "is_truncated": is_truncated,
            "error": error,
        },
        level="error" if error else "info",
    )


def tool_call_started(
    *,
    env_id: str = "",
    trajectory_id: str = "",
    tool_name: str = "",
    tool_call_id: str = "",
) -> None:
    emit(
        "tool_call.started",
        {
            "env_id": env_id,
            "trajectory_id": trajectory_id,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
        },
    )


def tool_call_completed(
    *,
    env_id: str = "",
    trajectory_id: str = "",
    tool_name: str = "",
    tool_call_id: str = "",
    duration_s: float | None = None,
    error: str = "",
) -> None:
    emit(
        "tool_call.completed",
        {
            "env_id": env_id,
            "trajectory_id": trajectory_id,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "duration_s": duration_s,
            "error": error,
        },
        level="error" if error else "info",
    )


def scoring_completed(
    *,
    env_id: str = "",
    trajectory_id: str = "",
    duration_s: float | None = None,
    reward: float | None = None,
) -> None:
    emit(
        "scoring.completed",
        {
            "env_id": env_id,
            "trajectory_id": trajectory_id,
            "duration_s": duration_s,
            "reward": reward,
        },
    )


def stop_condition_triggered(
    *,
    env_id: str = "",
    trajectory_id: str = "",
    condition: str = "",
    error: str = "",
) -> None:
    emit(
        "stop_condition.triggered",
        {
            "env_id": env_id,
            "trajectory_id": trajectory_id,
            "condition": condition,
            "error": error,
        },
        level="warn" if error else "info",
    )


def timeout_triggered(
    *,
    env_id: str = "",
    trajectory_id: str = "",
    timeout_seconds: float | None = None,
) -> None:
    emit(
        "rollout.timeout",
        {
            "env_id": env_id,
            "trajectory_id": trajectory_id,
            "timeout_seconds": timeout_seconds,
        },
        level="warn",
    )


def generate_started(
    *,
    env_id: str = "",
    model: str = "",
    total_rollouts: int = 0,
    num_examples: int = 0,
    rollouts_per_example: int = 0,
    max_concurrent: int = -1,
) -> None:
    emit(
        "generate.started",
        {
            "env_id": env_id,
            "model": model,
            "total_rollouts": total_rollouts,
            "num_examples": num_examples,
            "rollouts_per_example": rollouts_per_example,
            "max_concurrent": max_concurrent,
        },
    )


def generate_completed(
    *,
    env_id: str = "",
    model: str = "",
    total_rollouts: int = 0,
    duration_s: float | None = None,
    avg_reward: float | None = None,
) -> None:
    emit(
        "generate.completed",
        {
            "env_id": env_id,
            "model": model,
            "total_rollouts": total_rollouts,
            "duration_s": duration_s,
            "avg_reward": avg_reward,
        },
    )


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _safe_stderr(msg: str) -> None:
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:
        pass


def _json_default(obj: object) -> str:
    """Fallback serializer for json.dumps — converts unknown types to repr."""
    return repr(obj)[:1000]


class _Telemetry:
    """Singleton telemetry backend.  Thread-safe, sync-only (uses httpx)."""

    def __init__(self) -> None:
        self.api_key = (os.environ.get("DD_API_KEY") or "").strip()
        self.site = (os.environ.get("DD_SITE") or "datadoghq.com").strip()
        self.service_name = os.environ.get("VF_TELEMETRY_SERVICE", "verifiers")
        self.stderr_enabled = (
            bool(self.api_key) or os.environ.get("VF_TELEMETRY", "") == "1"
        )
        self.dd_endpoint = f"https://http-intake.logs.{self.site}/api/v2/logs"
        self._hostname = platform.node() or "unknown"

        self._queue: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._max_batch = 20
        self._flush_timer: threading.Timer | None = None
        self._flush_interval = 2.0

        status = "on" if self.api_key else "off"
        stderr = "on" if self.stderr_enabled else "off"
        _safe_stderr(
            f"VF_TELEMETRY_INIT datadog={status} stderr={stderr} "
            f"service={self.service_name!r}"
        )

        atexit.register(self.flush)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) or self.stderr_enabled

    def emit(
        self, event_name: str, data: dict[str, Any], *, level: str = "info"
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        full_event = {**data, "event": event_name, "ts": ts}

        if self.stderr_enabled:
            self._write_stderr(event_name, full_event, level)

        if self.api_key:
            entry = self._build_entry(event_name, full_event, level)
            with self._lock:
                self._queue.append(entry)
                should_flush = len(self._queue) >= self._max_batch
            if should_flush:
                self.flush()
            else:
                self._schedule_flush()

    def flush(self) -> None:
        with self._lock:
            if not self._queue:
                return
            batch = list(self._queue)
            self._queue.clear()
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None
        if not self.api_key:
            return
        self._post(batch)

    # -- private helpers --

    def _schedule_flush(self) -> None:
        with self._lock:
            if self._flush_timer is not None and self._flush_timer.is_alive():
                return
            self._flush_timer = threading.Timer(self._flush_interval, self.flush)
            self._flush_timer.daemon = True
            self._flush_timer.start()

    def _post(self, batch: list[dict[str, Any]]) -> None:
        try:
            import httpx

            payload = json.dumps(batch, default=_json_default).encode()
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(
                    self.dd_endpoint,
                    content=payload,
                    headers={
                        "Content-Type": "application/json",
                        "DD-API-KEY": self.api_key,
                    },
                )
                if resp.status_code >= 400:
                    _safe_stderr(
                        f"VF_TELEMETRY_DD_ERROR status={resp.status_code} "
                        f"body={resp.text[:500]!r}"
                    )
        except Exception as exc:
            _safe_stderr(f"VF_TELEMETRY_DD_ERROR err={exc!r}")

    def _build_entry(
        self, event_name: str, event: dict[str, Any], level: str
    ) -> dict[str, Any]:
        ddtags = f"event:{event_name}"
        env_id = event.get("env_id")
        if env_id:
            ddtags += f",env_id:{env_id}"
        return {
            "ddsource": "verifiers",
            "ddtags": ddtags,
            "hostname": self._hostname,
            "message": json.dumps(event, default=_json_default),
            "service": self.service_name,
            "status": level,
        }

    def _write_stderr(self, event_name: str, event: dict[str, Any], level: str) -> None:
        parts = [f"VF_TELEMETRY [{level.upper()}] {event_name}"]
        for key in (
            "env_id",
            "model",
            "trajectory_id",
            "example_id",
            "tool_name",
            "duration_s",
            "reward",
            "error",
            "condition",
            "turn_index",
            "num_turns",
            "stop_condition",
            "input_tokens",
            "output_tokens",
            "group_size",
        ):
            val = event.get(key)
            if val is not None and val != "" and val != 0:
                parts.append(f"{key}={val!r}")
        _safe_stderr(" ".join(parts))
