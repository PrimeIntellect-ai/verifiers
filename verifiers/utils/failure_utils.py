from __future__ import annotations

from typing import Any, TypedDict

from verifiers.errors import ModelError, SandboxError, TunnelError


class RolloutFailure(TypedDict):
    reason: str
    origin: str
    error_type: str | None
    root_error_type: str | None
    message: str
    logs: dict[str, str]


FAILURE_REASON_AGENT_NONZERO_EXIT = "agent_nonzero_exit"
FAILURE_REASON_AGENT_POLL_FAILED = "agent_poll_failed"
FAILURE_REASON_AGENT_EMPTY_TRAJECTORY = "agent_empty_trajectory"
FAILURE_REASON_ROLLOUT_TIMEOUT = "rollout_timeout"
FAILURE_REASON_SANDBOX_OOM = "sandbox_oom"
FAILURE_REASON_SANDBOX_TIMEOUT = "sandbox_timeout"
FAILURE_REASON_SANDBOX_COMMAND_FAILED = "sandbox_command_failed"
FAILURE_REASON_SANDBOX_SETUP_FAILED = "sandbox_setup_failed"
FAILURE_REASON_TUNNEL_ERROR = "tunnel_error"
FAILURE_REASON_STREAM_INTERRUPTED = "stream_interrupted"
FAILURE_REASON_MODEL_ERROR = "model_error"
FAILURE_REASON_ENV_SERVER_ERROR = "env_server_error"
FAILURE_REASON_UNKNOWN = "unknown"

FAILURE_ORIGIN_AGENT = "agent"
FAILURE_ORIGIN_SANDBOX = "sandbox"
FAILURE_ORIGIN_TUNNEL = "tunnel"
FAILURE_ORIGIN_MODEL = "model"
FAILURE_ORIGIN_ENV_SERVER = "env_server"
FAILURE_ORIGIN_ROLLOUT = "rollout"
FAILURE_ORIGIN_UNKNOWN = "unknown"

DEFAULT_FAILURE_MESSAGE_CHARS = 2000
DEFAULT_FAILURE_LOG_CHARS = 12000


def tail_text(value: Any, max_chars: int = DEFAULT_FAILURE_LOG_CHARS) -> str:
    """Return a bounded string tail for logs and diagnostic messages."""
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return f"...<truncated {len(text) - max_chars} chars>\n{text[-max_chars:]}"


def _error_chain(error: BaseException | None) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = error.__cause__
    return chain


def _chain_type_names(error: BaseException | None) -> list[str]:
    return [type(item).__name__ for item in _error_chain(error)]


def _has_chain_type(error: BaseException | None, type_name: str) -> bool:
    return type_name in _chain_type_names(error)


def classify_rollout_failure(
    state: dict[str, Any],
    error: BaseException | None = None,
    *,
    detect_empty_trajectory: bool = True,
) -> tuple[str, str]:
    """Classify a rollout failure into a dashboard-oriented reason/origin pair."""
    error = state.get("error") if error is None else error

    if state.get("timed_out"):
        return FAILURE_REASON_ROLLOUT_TIMEOUT, FAILURE_ORIGIN_ROLLOUT
    if state.get("sandbox_oom"):
        return FAILURE_REASON_SANDBOX_OOM, FAILURE_ORIGIN_SANDBOX
    if state.get("sandbox_timeout"):
        return FAILURE_REASON_SANDBOX_TIMEOUT, FAILURE_ORIGIN_SANDBOX

    if error is not None:
        if _has_chain_type(error, "StreamInterrupted"):
            return FAILURE_REASON_STREAM_INTERRUPTED, FAILURE_ORIGIN_TUNNEL
        if isinstance(error, TunnelError) or _has_chain_type(error, "TunnelError"):
            return FAILURE_REASON_TUNNEL_ERROR, FAILURE_ORIGIN_TUNNEL
        if isinstance(error, ModelError) or _has_chain_type(error, "ModelError"):
            return FAILURE_REASON_MODEL_ERROR, FAILURE_ORIGIN_MODEL
        if state.get("agent_poll_failed") or _has_chain_type(error, "AgentPollError"):
            return FAILURE_REASON_AGENT_POLL_FAILED, FAILURE_ORIGIN_AGENT
        if _has_chain_type(error, "AgentError"):
            agent_exit_code = state.get("agent_exit_code")
            if agent_exit_code is not None and agent_exit_code != 0:
                return FAILURE_REASON_AGENT_NONZERO_EXIT, FAILURE_ORIGIN_AGENT
            return FAILURE_REASON_AGENT_POLL_FAILED, FAILURE_ORIGIN_AGENT
        if _has_chain_type(error, "SandboxSetupError"):
            return FAILURE_REASON_SANDBOX_SETUP_FAILED, FAILURE_ORIGIN_SANDBOX
        if isinstance(error, SandboxError) or _has_chain_type(error, "SandboxError"):
            return FAILURE_REASON_SANDBOX_COMMAND_FAILED, FAILURE_ORIGIN_SANDBOX

    agent_exit_code = state.get("agent_exit_code")
    if agent_exit_code is not None and agent_exit_code != 0:
        return FAILURE_REASON_AGENT_NONZERO_EXIT, FAILURE_ORIGIN_AGENT

    trajectory = state.get("trajectory")
    if (
        detect_empty_trajectory
        and isinstance(trajectory, list)
        and len(trajectory) == 0
    ):
        return FAILURE_REASON_AGENT_EMPTY_TRAJECTORY, FAILURE_ORIGIN_AGENT

    return FAILURE_REASON_UNKNOWN, FAILURE_ORIGIN_UNKNOWN


def make_rollout_failure(
    state: dict[str, Any],
    *,
    reason: str | None = None,
    origin: str | None = None,
    error: BaseException | None = None,
    message: str | None = None,
    logs: dict[str, str] | None = None,
    detect_empty_trajectory: bool = True,
) -> RolloutFailure:
    """Build a JSON-serializable rollout failure payload."""
    error = state.get("error") if error is None else error
    classified_reason, classified_origin = classify_rollout_failure(
        state, error, detect_empty_trajectory=detect_empty_trajectory
    )
    reason = reason or classified_reason
    origin = origin or classified_origin
    chain = _error_chain(error)
    error_type = type(error).__name__ if error is not None else None
    root_error_type = type(chain[-1]).__name__ if chain else None
    if message is None:
        message = str(error) if error is not None else reason
    return {
        "reason": reason,
        "origin": origin,
        "error_type": error_type,
        "root_error_type": root_error_type,
        "message": tail_text(message, DEFAULT_FAILURE_MESSAGE_CHARS),
        "logs": {str(k): tail_text(v) for k, v in (logs or {}).items()},
    }


def normalize_rollout_failure(value: Any) -> RolloutFailure | None:
    """Normalize an existing failure-like mapping into the public payload shape."""
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    logs = value.get("logs")
    return {
        "reason": str(value.get("reason") or FAILURE_REASON_UNKNOWN),
        "origin": str(value.get("origin") or FAILURE_ORIGIN_UNKNOWN),
        "error_type": value.get("error_type"),
        "root_error_type": value.get("root_error_type"),
        "message": str(value.get("message") or ""),
        "logs": {
            str(k): tail_text(v)
            for k, v in (logs if isinstance(logs, dict) else {}).items()
        },
    }


def ensure_rollout_failure(
    state: dict[str, Any],
    *,
    reason: str | None = None,
    origin: str | None = None,
    error: BaseException | None = None,
    message: str | None = None,
    logs: dict[str, str] | None = None,
    detect_empty_trajectory: bool = True,
    overwrite: bool = False,
) -> RolloutFailure | None:
    """Set and return state['failure'] when the state has a classified failure."""
    existing = normalize_rollout_failure(state.get("failure"))
    if existing is not None and not overwrite:
        if logs:
            existing["logs"].update({str(k): tail_text(v) for k, v in logs.items()})
            state["failure"] = existing
        return existing

    error = state.get("error") if error is None else error
    should_create = (
        error is not None
        or state.get("timed_out")
        or state.get("sandbox_oom")
        or state.get("sandbox_timeout")
    )
    trajectory = state.get("trajectory")
    should_create = should_create or (
        detect_empty_trajectory
        and isinstance(trajectory, list)
        and len(trajectory) == 0
    )
    should_create = should_create or reason is not None or origin is not None
    if not should_create:
        return None

    failure = make_rollout_failure(
        state,
        reason=reason,
        origin=origin,
        error=error,
        message=message,
        logs=logs,
        detect_empty_trajectory=detect_empty_trajectory,
    )
    state["failure"] = failure
    return failure


def add_failure_logs(
    state: dict[str, Any],
    logs: dict[str, str],
    *,
    detect_empty_trajectory: bool = True,
) -> RolloutFailure | None:
    """Merge bounded diagnostic logs into state['failure'] if a failure exists."""
    if not logs:
        return normalize_rollout_failure(state.get("failure"))
    failure = ensure_rollout_failure(
        state, logs=logs, detect_empty_trajectory=detect_empty_trajectory
    )
    if failure is None:
        return None
    failure["logs"].update({str(k): tail_text(v) for k, v in logs.items()})
    state["failure"] = failure
    return failure
