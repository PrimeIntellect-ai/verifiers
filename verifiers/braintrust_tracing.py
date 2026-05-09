"""
Braintrust tracing for verifiers.

Provides nested span traces for rollouts, model calls, tool calls, and scoring.
Each rollout becomes a trace in Braintrust with child spans showing the full
execution timeline.

Activation:
    Set BRAINTRUST_API_KEY to enable.  No-op when unset.
    Optionally set VF_BRAINTRUST_PROJECT to override the default project name
    (default: "verifiers").

Span hierarchy produced per rollout:

    rollout (type=task)          ← root span / trace
    ├── setup_state (type=task)
    ├── turn_0 (type=task)
    │   ├── model_request (type=llm)
    │   └── env_response (type=task)   ← includes tool_call children
    │       ├── tool_call:navigate (type=tool)
    │       └── tool_call:computer (type=tool)
    ├── turn_1 (type=task)
    │   └── model_request (type=llm)
    └── scoring (type=score)

All public helpers are safe to call even when Braintrust is not configured;
they degrade to no-ops with near-zero overhead.  Errors inside telemetry
code are swallowed so they never interfere with evaluation runs.
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
import threading
import time
import uuid
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_INSTANCE: _Tracing | None = None
_LOCK = threading.Lock()

# Coroutine-local storage for passing the rollout span from
# _run_rollout_state → rollout() across the await boundary without
# storing mutable state on the shared Environment instance.
_pending_rollout_span: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_pending_rollout_span", default=None
)

# Run-level tags: coroutine-local storage so concurrent generate() calls
# each get their own tag set without overwriting each other.
_run_tags: contextvars.ContextVar[list[str]] = contextvars.ContextVar(
    "_run_tags", default=[]
)


def _get() -> _Tracing:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = _Tracing()
    return _INSTANCE


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def enabled() -> bool:
    """Return True when Braintrust tracing is active."""
    return _get().enabled


def flush() -> None:
    """Flush any buffered spans to Braintrust."""
    try:
        inst = _get()
        if inst.enabled and inst._logger is not None:
            inst._logger.flush()
    except Exception:
        pass


def set_run_tags(tags: list[str] | None = None) -> list[str]:
    """Set tags for the current eval run.

    If *tags* is ``None`` a unique tag is auto-generated from the current
    timestamp (``run-<epoch>-<short_uuid>``).  Returns the active tag list
    so callers can inspect or log it.

    Uses a ``ContextVar`` so concurrent ``generate()`` calls each get their
    own isolated tag set.
    """
    if tags is not None:
        new_tags = list(tags)
    else:
        short_id = uuid.uuid4().hex[:8]
        new_tags = [f"run-{int(time.time())}-{short_id}"]
    _run_tags.set(new_tags)
    return new_tags


def get_run_tags() -> list[str]:
    """Return the currently active run tags (empty list when unset)."""
    return list(_run_tags.get())


def clear_run_tags() -> None:
    """Clear run tags (called after generate completes)."""
    _run_tags.set([])


# -- Span lifecycle helpers ------------------------------------------------
# These return an opaque span object (or None when disabled).  Callers store
# the span and later call end_span() when the phase completes.


def start_rollout_span(
    *,
    env_id: str = "",
    model: str = "",
    example_id: Any = "",
    trajectory_id: str = "",
) -> Any:
    """Start a root span for one rollout.  Returns span or None."""
    try:
        inst = _get()
        if not inst.enabled or inst._logger is None:
            return None
        kwargs: dict[str, Any] = {
            "name": "rollout",
            "span_attributes": {"type": "task"},
            "input": {"example_id": _safe(example_id)},
            "metadata": {
                "env_id": env_id,
                "model": model,
                "trajectory_id": trajectory_id,
            },
        }
        tags = _run_tags.get()
        if tags:
            kwargs["tags"] = list(tags)
        span = inst._logger.start_span(**kwargs)
        return span
    except Exception:
        return None


def start_child_span(
    parent: Any,
    *,
    name: str,
    span_type: str = "task",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Start a child span under *parent*.  Returns span or None."""
    try:
        if parent is None:
            return None
        kwargs: dict[str, Any] = {
            "name": name,
            "span_attributes": {"type": span_type},
        }
        if input is not None:
            kwargs["input"] = _safe(input)
        if metadata:
            kwargs["metadata"] = _safe(metadata)
        return parent.start_span(**kwargs)
    except Exception:
        return None


def log_to_span(
    span: Any,
    *,
    input: Any = None,
    output: Any = None,
    metadata: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
    scores: dict[str, float] | None = None,
) -> None:
    """Log data to an existing span."""
    try:
        if span is None:
            return
        kwargs: dict[str, Any] = {}
        if input is not None:
            kwargs["input"] = _safe(input)
        if output is not None:
            kwargs["output"] = _safe(output)
        if metadata:
            kwargs["metadata"] = _safe(metadata)
        if metrics:
            kwargs["metrics"] = _safe(metrics)
        if error:
            kwargs["error"] = error
        if scores:
            kwargs["scores"] = scores
        if kwargs:
            span.log(**kwargs)
    except Exception:
        pass


def end_span(span: Any) -> None:
    """End (close) a span.  Safe to call with None."""
    try:
        if span is not None:
            span.end()
    except Exception:
        pass


# -- Convenience: rollout lifecycle ----------------------------------------


def rollout_started(
    *,
    env_id: str = "",
    model: str = "",
    example_id: Any = "",
    trajectory_id: str = "",
) -> Any:
    """Start a rollout root span.  Returns the span."""
    return start_rollout_span(
        env_id=env_id,
        model=model,
        example_id=example_id,
        trajectory_id=trajectory_id,
    )


def rollout_completed(
    span: Any,
    *,
    reward: Any = None,
    num_turns: int = 0,
    duration_s: float = 0.0,
    stop_condition: str = "",
    error: str = "",
    input_tokens: float = 0.0,
    output_tokens: float = 0.0,
) -> None:
    """Finalize and close a rollout span."""
    try:
        if span is None:
            return
        metrics: dict[str, Any] = {
            "duration_s": duration_s,
            "num_turns": num_turns,
            "tokens": input_tokens + output_tokens,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        }
        meta: dict[str, Any] = {"stop_condition": stop_condition}
        scores: dict[str, float] | None = None
        if reward is not None:
            try:
                scores = {"reward": float(reward)}
            except (TypeError, ValueError):
                meta["reward_raw"] = repr(reward)
        log_to_span(
            span,
            output={"stop_condition": stop_condition, "num_turns": num_turns},
            metadata=meta,
            metrics=metrics,
            error=error or None,
            scores=scores,
        )
        end_span(span)
    except Exception:
        pass


# -- Convenience: setup ----------------------------------------------------


def setup_started(parent: Any, *, env_id: str = "", trajectory_id: str = "") -> Any:
    """Start a setup_state child span."""
    return start_child_span(
        parent,
        name="setup_state",
        span_type="task",
        metadata={"env_id": env_id, "trajectory_id": trajectory_id},
    )


def setup_completed(span: Any, *, duration_s: float = 0.0, error: str = "") -> None:
    """Finalize and close a setup span."""
    log_to_span(
        span,
        metrics={"duration_s": duration_s},
        error=error or None,
    )
    end_span(span)


# -- Convenience: turns ----------------------------------------------------


def turn_started(
    parent: Any,
    *,
    turn_index: int = 0,
    trajectory_id: str = "",
) -> Any:
    """Start a turn child span."""
    return start_child_span(
        parent,
        name=f"turn_{turn_index}",
        span_type="task",
        metadata={"turn_index": turn_index, "trajectory_id": trajectory_id},
    )


def turn_completed(
    span: Any,
    *,
    duration_s: float = 0.0,
    model_duration_s: float | None = None,
    env_duration_s: float | None = None,
    is_truncated: bool = False,
    error: str = "",
) -> None:
    """Finalize and close a turn span."""
    metrics: dict[str, Any] = {"duration_s": duration_s}
    if model_duration_s is not None:
        metrics["model_duration_s"] = model_duration_s
    if env_duration_s is not None:
        metrics["env_duration_s"] = env_duration_s
    log_to_span(
        span,
        metrics=metrics,
        metadata={"is_truncated": is_truncated},
        error=error or None,
    )
    end_span(span)


# -- Convenience: model requests -------------------------------------------


def model_request_span(
    parent: Any,
    *,
    model: str = "",
    turn_index: int = 0,
    messages: Any = None,
) -> Any:
    """Start a model_request child span (type=llm)."""
    input_val = None
    if messages is not None:
        input_val = _safe(messages)
    return start_child_span(
        parent,
        name="model_request",
        span_type="llm",
        input=input_val,
        metadata={"model": model, "turn_index": turn_index},
    )


def model_request_completed(
    span: Any,
    *,
    duration_s: float = 0.0,
    input_tokens: float = 0.0,
    output_tokens: float = 0.0,
    response: Any = None,
    error: str = "",
) -> None:
    """Finalize and close a model_request span."""
    metrics: dict[str, Any] = {
        "duration_s": duration_s,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "tokens": input_tokens + output_tokens,
    }
    output_val = None
    if response is not None:
        output_val = _safe_response(response)
    log_to_span(
        span,
        output=output_val,
        metrics=metrics,
        error=error or None,
    )
    end_span(span)


# -- Convenience: tool calls -----------------------------------------------


def tool_call_started(
    parent: Any,
    *,
    tool_name: str = "",
    tool_call_id: str = "",
    tool_args: Any = None,
) -> Any:
    """Start a tool_call child span (type=tool)."""
    return start_child_span(
        parent,
        name=f"tool_call:{tool_name}",
        span_type="tool",
        input=_safe(tool_args) if tool_args is not None else {"tool_name": tool_name},
        metadata={"tool_name": tool_name, "tool_call_id": tool_call_id},
    )


def tool_call_completed(
    span: Any,
    *,
    duration_s: float = 0.0,
    result: Any = None,
    error: str = "",
) -> None:
    """Finalize and close a tool_call span."""
    log_to_span(
        span,
        output=_safe(result) if result is not None else None,
        metrics={"duration_s": duration_s},
        error=error or None,
    )
    end_span(span)


# -- Convenience: scoring --------------------------------------------------


def scoring_started(parent: Any, *, trajectory_id: str = "") -> Any:
    """Start a scoring child span (type=score)."""
    return start_child_span(
        parent,
        name="scoring",
        span_type="score",
        metadata={"trajectory_id": trajectory_id},
    )


def scoring_completed(
    span: Any,
    *,
    duration_s: float = 0.0,
    reward: Any = None,
) -> None:
    """Finalize and close a scoring span."""
    scores: dict[str, float] | None = None
    if reward is not None:
        try:
            scores = {"reward": float(reward)}
        except (TypeError, ValueError):
            pass
    log_to_span(
        span,
        metrics={"duration_s": duration_s},
        scores=scores,
    )
    end_span(span)


# -- Convenience: groups ---------------------------------------------------


def group_started(
    *,
    env_id: str = "",
    model: str = "",
    example_id: Any = "",
    group_size: int = 0,
) -> Any:
    """Start a root span for a group of rollouts."""
    try:
        inst = _get()
        if not inst.enabled or inst._logger is None:
            return None
        kwargs: dict[str, Any] = {
            "name": "group",
            "span_attributes": {"type": "task"},
            "input": {"example_id": _safe(example_id), "group_size": group_size},
            "metadata": {"env_id": env_id, "model": model},
        }
        tags = _run_tags.get()
        if tags:
            kwargs["tags"] = list(tags)
        return inst._logger.start_span(**kwargs)
    except Exception:
        return None


def group_completed(
    span: Any,
    *,
    duration_s: float = 0.0,
    avg_reward: float | None = None,
    group_size: int = 0,
) -> None:
    """Finalize and close a group span."""
    scores: dict[str, float] | None = None
    if avg_reward is not None:
        scores = {"avg_reward": avg_reward}
    log_to_span(
        span,
        output={"group_size": group_size},
        metrics={"duration_s": duration_s},
        scores=scores,
    )
    end_span(span)


# -- Convenience: generate -------------------------------------------------


def generate_started(
    *,
    env_id: str = "",
    model: str = "",
    num_inputs: int = 0,
) -> Any:
    """Start a root span for a generate() call."""
    try:
        inst = _get()
        if not inst.enabled or inst._logger is None:
            return None
        kwargs: dict[str, Any] = {
            "name": "generate",
            "span_attributes": {"type": "eval"},
            "input": {"num_inputs": num_inputs},
            "metadata": {"env_id": env_id, "model": model},
        }
        tags = _run_tags.get()
        if tags:
            kwargs["tags"] = list(tags)
        return inst._logger.start_span(**kwargs)
    except Exception:
        return None


def generate_completed(
    span: Any,
    *,
    duration_s: float = 0.0,
    num_outputs: int = 0,
    avg_reward: float | None = None,
) -> None:
    """Finalize and close a generate span."""
    scores: dict[str, float] | None = None
    if avg_reward is not None:
        scores = {"avg_reward": avg_reward}
    log_to_span(
        span,
        output={"num_outputs": num_outputs},
        metrics={"duration_s": duration_s},
        scores=scores,
    )
    end_span(span)
    flush()


# -- Convenience: stop condition / timeout ---------------------------------


def stop_condition_triggered(
    parent: Any,
    *,
    condition: str = "",
    error: str = "",
) -> None:
    """Log a stop condition event on the rollout span."""
    log_to_span(
        parent,
        metadata={"stop_condition": condition},
        error=error or None,
    )


def timeout_triggered(
    parent: Any,
    *,
    timeout_seconds: float | None = None,
) -> None:
    """Log a timeout event on the rollout span."""
    log_to_span(
        parent,
        metadata={"timed_out": True, "timeout_seconds": timeout_seconds},
        error="timeout",
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _safe(obj: Any) -> Any:
    """Best-effort JSON-safe conversion of arbitrary objects."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    # Pydantic models
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Fallback to repr
    try:
        return repr(obj)[:2000]
    except Exception:
        return "<unserializable>"


def _safe_response(response: Any) -> Any:
    """Extract serializable data from a model response object."""
    try:
        if hasattr(response, "model_dump"):
            d = response.model_dump()
            # Trim large fields to keep span data manageable
            if isinstance(d, dict) and "choices" in d:
                for choice in d.get("choices", []):
                    if isinstance(choice, dict) and "message" in choice:
                        msg = choice["message"]
                        if isinstance(msg, dict) and "content" in msg:
                            content = msg["content"]
                            if isinstance(content, str) and len(content) > 5000:
                                msg["content"] = content[:5000] + "...(truncated)"
            return d
        if hasattr(response, "message"):
            return _safe(response.message)
        return repr(response)[:2000]
    except Exception:
        return "<response>"


# ---------------------------------------------------------------------------
# Singleton backend
# ---------------------------------------------------------------------------


class _Tracing:
    """Lazy singleton that manages the Braintrust logger."""

    def __init__(self) -> None:
        self._logger: Any = None
        self._enabled = False
        self._api_key = os.environ.get("BRAINTRUST_API_KEY", "")
        self._project = os.environ.get("VF_BRAINTRUST_PROJECT", "verifiers")

        if not self._api_key:
            return

        try:
            import braintrust

            self._logger = braintrust.init_logger(
                project=self._project,
                api_key=self._api_key,
            )
            self._enabled = True
            _log.info("Braintrust tracing enabled for project=%s", self._project)
        except ImportError:
            print(
                "WARNING: BRAINTRUST_API_KEY is set but 'braintrust' package "
                "is not installed. Run: pip install braintrust",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"WARNING: Failed to initialize Braintrust tracing: {exc}",
                file=sys.stderr,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled
