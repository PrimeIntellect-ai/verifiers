# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "aiohttp>=3.11",
#   "openai>=2.0",
#   "orjson>=3.10",
#   "pillow>=11.0",
#   "pydantic>=2.0",
#   "pypdf>=5.4",
#   "pypdfium2>=4.30",
#   "python-pptx>=1.0",
# ]
# ///
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

ERROR_CATEGORY_CODES = {
    "none": 0,
    "harness_disconnect": 1,
    "request_too_large_bytes": 2,
    "request_too_large_tokens": 3,
    "model_rate_limit": 4,
    "model_auth": 5,
    "model_bad_request": 6,
    "model_internal_error": 7,
    "max_steps_exceeded": 8,
    "browser_or_sandbox": 9,
    "agent_logic_error": 10,
    "unknown": 11,
    "model_endpoint_gone": 12,
    "model_connection_failure": 13,
}

TOOL_ERROR_PREFIXES = (
    "ValidationError",
    "KeyError",
    "ValueError",
    "RuntimeError",
    "AttributeError",
    "TypeError",
    "Unknown tool",
)

TOOL_ERROR_BREAKDOWN_NAMES = ("computer", "read_page", "find", "get_page_text")
HTTP_STATUS_RE = re.compile(
    r"(?:Error code:|status(?: code)?[=:]?)\s*(\d{3})", re.IGNORECASE
)
DEFAULT_PROGRESS_PATH = "/logs/mini_browse/progress.jsonl"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    return _read_json(path)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _write_progress(progress_path: Path, event: str, **fields: Any) -> None:
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event": event,
            "timestamp": time.time(),
            **{key: _json_safe(value) for key, value in fields.items()},
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    except Exception:
        return


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _http_status_from_exception(exc: BaseException | None) -> int | None:
    if exc is None:
        return None
    status = getattr(exc, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _http_status_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = HTTP_STATUS_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _classify_exception(exc: BaseException) -> str:
    if isinstance(exc, asyncio.CancelledError):
        return "harness_disconnect"

    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return "model_rate_limit"
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return "model_auth"
        if isinstance(exc, openai.BadRequestError):
            text = str(exc).lower()
            bytes_markers = (
                "request entity too large",
                "payload too large",
                "413 request",
                "413 payload",
            )
            if any(marker in text for marker in bytes_markers):
                return "request_too_large_bytes"
            token_markers = (
                "context length",
                "maximum context",
                "too many tokens",
                "context_length_exceeded",
                "context window",
                "input is too long",
            )
            if any(marker in text for marker in token_markers):
                return "request_too_large_tokens"
            return "model_bad_request"
        status = _http_status_from_exception(exc)
        if isinstance(exc, openai.NotFoundError) or status == 404:
            return "model_endpoint_gone"
        if isinstance(exc, openai.InternalServerError) or (
            status is not None and 500 <= status < 600
        ):
            return "model_internal_error"
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return "model_connection_failure"
        if isinstance(exc, openai.APIError):
            if status == 404:
                return "model_endpoint_gone"
            if status is not None and 500 <= status < 600:
                return "model_internal_error"
            return "model_connection_failure"
    except ImportError:
        pass

    try:
        import aiohttp

        if isinstance(exc, aiohttp.ClientError):
            return "model_connection_failure"
    except ImportError:
        pass

    if isinstance(exc, TimeoutError):
        return "model_connection_failure"
    if isinstance(exc, ConnectionError):
        return "model_connection_failure"
    if isinstance(exc, OSError):
        return "browser_or_sandbox"
    if isinstance(
        exc, (KeyError, TypeError, AttributeError, ValueError, RuntimeError, IndexError)
    ):
        return "agent_logic_error"
    return "unknown"


def _diagnose(exc: BaseException | None, error_text: str | None) -> dict[str, Any]:
    if exc is not None:
        category = _classify_exception(exc)
        error_type = type(exc).__name__
        excerpt = str(exc)[:1200]
        http_status = _http_status_from_exception(exc) or _http_status_from_text(
            excerpt
        )
    elif error_text:
        text = str(error_text)
        error_type = text.split(":", 1)[0][:120] if ":" in text else text[:120]
        excerpt = text[:1200]
        http_status = _http_status_from_text(text)
        if http_status == 404:
            category = "model_endpoint_gone"
        elif http_status is not None and 500 <= http_status < 600:
            category = "model_internal_error"
        elif "maximum steps exceeded" in text.lower():
            category = "max_steps_exceeded"
        else:
            category = "unknown"
    else:
        return {
            "error_type": None,
            "error_category": "none",
            "error_category_code": ERROR_CATEGORY_CODES["none"],
            "error_excerpt": None,
            "error_http_status": None,
        }
    return {
        "error_type": error_type,
        "error_category": category,
        "error_category_code": ERROR_CATEGORY_CODES[category],
        "error_excerpt": excerpt,
        "error_http_status": http_status,
    }


def _count_image_parts(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    count += 1
    return count


def _json_size_bytes(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def _summarize_tool_errors(messages: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    validation = 0
    streak = 0
    max_streak = 0
    by_tool: dict[str, int] = {}
    unique_kinds: set[str] = set()
    id_to_tool: dict[str, str] = {}

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if isinstance(tool_call, str):
                    try:
                        tool_call = json.loads(tool_call)
                    except json.JSONDecodeError:
                        continue
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = tool_call.get("id")
                function = tool_call.get("function")
                if isinstance(function, dict):
                    name = function.get("name")
                else:
                    name = tool_call.get("name")
                if isinstance(tool_call_id, str) and isinstance(name, str):
                    id_to_tool[tool_call_id] = name
        elif role == "tool":
            content = message.get("content")
            if not isinstance(content, str):
                streak = 0
                continue
            stripped = content.lstrip()
            if not any(stripped.startswith(prefix) for prefix in TOOL_ERROR_PREFIXES):
                streak = 0
                continue

            total += 1
            if stripped.startswith("ValidationError"):
                validation += 1
            tool_name = id_to_tool.get(message.get("tool_call_id") or "", "unknown")
            by_tool[tool_name] = by_tool.get(tool_name, 0) + 1
            unique_kinds.add(stripped.split("\n", 1)[0][:200])
            streak += 1
            max_streak = max(max_streak, streak)

    return {
        "tool_error_count": total,
        "tool_error_validation": validation,
        "tool_error_max_streak": max_streak,
        "tool_error_unique_kinds": len(unique_kinds),
        "tool_error_by_tool": by_tool,
    }


def _load_task_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Mini Browse task payload must be an object: {path}")
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Mini Browse task payload requires non-empty instruction")
    output_schema = payload.get("output_schema")
    if not isinstance(output_schema, dict):
        raise ValueError("Mini Browse task payload requires object output_schema")
    return payload


async def _run(args: argparse.Namespace) -> int:
    from mini_browse import run_bcu_task

    task_path = Path(args.task)
    result_path = Path(args.result)
    transcript_path = Path(args.transcript)
    metrics_path = Path(args.metrics)
    progress_path = Path(args.progress)
    workspace_root = Path(args.workspace_root)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    os.environ["MINI_BROWSE_PROGRESS_PATH"] = str(progress_path)
    _write_progress(
        progress_path,
        "harness_program_start",
        task_path=str(task_path),
        result_path=str(result_path),
        transcript_path=str(transcript_path),
        metrics_path=str(metrics_path),
        workspace_root=str(workspace_root),
    )

    task_payload = _load_task_payload(task_path)
    _write_progress(
        progress_path,
        "task_payload_loaded",
        source=task_payload.get("source"),
        start_url=task_payload.get("start_url"),
        instruction_chars=len(task_payload.get("instruction") or ""),
        output_schema_keys=sorted((task_payload.get("output_schema") or {}).keys()),
        has_browser_api_url=bool(task_payload.get("browser_api_url")),
        has_http_proxy=bool(task_payload.get("http_proxy")),
    )
    instruction = task_payload["instruction"].strip()
    output_schema = task_payload["output_schema"]
    start_url = str(task_payload.get("start_url") or "about:blank")
    browser_api_url = str(task_payload.get("browser_api_url") or "").strip()
    if browser_api_url:
        os.environ["MINI_BROWSE_BROWSER_API_URL"] = browser_api_url
    http_proxy = str(task_payload.get("http_proxy") or "").strip()
    if http_proxy:
        os.environ["PERPLEXITY_TAILSCALE_HTTP_PROXY"] = http_proxy
    source = str(task_payload.get("source") or "verifiers-mini-browse")
    task_preamble = str(
        task_payload.get("task_preamble")
        or os.environ.get("MINI_BROWSE_TASK_PREAMBLE")
        or ""
    )
    conversation = (
        _read_optional_json(Path(args.conversation)) if args.conversation else None
    )
    if conversation is not None and not isinstance(conversation, list):
        raise ValueError("Mini Browse conversation payload must be a list")
    from openai import AsyncOpenAI

    model_client = _read_json(Path(args.model_client))
    model = model_client["model"]
    client = AsyncOpenAI(
        base_url=model_client["base_url"], api_key=model_client["api_key"]
    )
    coordinate_mode = os.environ.get("MINI_BROWSE_COORDINATE_MODE", "relative_1000")

    payload: dict[str, Any]
    messages: list[dict[str, Any]] = []
    exc_caught: BaseException | None = None
    try:
        _write_progress(
            progress_path,
            "run_bcu_task_start",
            model=model,
            coordinate_mode=coordinate_mode,
            max_steps=int(args.max_steps),
        )
        run_result = await run_bcu_task(
            task=instruction,
            url=start_url,
            output_schema=output_schema,
            model=model,
            client=client,
            max_steps=int(args.max_steps),
            workspace_root=workspace_root,
            include_builtin_tools=_env_bool("MINI_BROWSE_INCLUDE_BUILTIN_TOOLS"),
            source=source,
            task_preamble=task_preamble,
            coordinate_mode=coordinate_mode,
            conversation=conversation,
            browser_start_min_interval_seconds=_env_float(
                "MINI_BROWSE_BROWSER_START_MIN_INTERVAL_SECONDS"
            ),
            browser_start_jitter_seconds=_env_float(
                "MINI_BROWSE_BROWSER_START_JITTER_SECONDS"
            ),
            browser_start_max_in_flight=_env_int(
                "MINI_BROWSE_BROWSER_START_MAX_IN_FLIGHT"
            ),
        )
        _write_progress(
            progress_path,
            "run_bcu_task_done",
            is_error=run_result.is_error,
            submitted_result_present=bool(run_result.submitted_result),
            message_count=len(run_result.messages),
            browser_session_id=run_result.browser_session_id,
        )
        messages = run_result.messages
        payload = {
            "response": run_result.response,
            "is_error": run_result.is_error,
            "error": run_result.error,
            "is_cancelled": run_result.is_cancelled,
            "browser_session_id": run_result.browser_session_id,
            "tab_group_id": run_result.tab_group_id,
            "submitted_result": _json_safe(run_result.submitted_result),
            "workspace_root": run_result.workspace_root,
            "message_count": len(messages),
            "coordinate_mode": coordinate_mode,
        }
    except BaseException as exc:
        exc_caught = exc
        _write_progress(
            progress_path,
            "run_bcu_task_exception",
            error_type=type(exc).__name__,
            error_excerpt=str(exc)[:500],
            is_cancelled=isinstance(exc, asyncio.CancelledError),
        )
        payload = {
            "response": "",
            "is_error": True,
            "error": f"{type(exc).__name__}: {exc}",
            "is_cancelled": isinstance(exc, asyncio.CancelledError),
            "browser_session_id": None,
            "tab_group_id": None,
            "submitted_result": None,
            "workspace_root": str(workspace_root),
            "message_count": len(messages),
            "coordinate_mode": coordinate_mode,
        }

    diagnostics = _diagnose(exc_caught, payload.get("error"))
    payload.update(diagnostics)
    payload["transcript_image_count"] = _count_image_parts(messages)
    payload["transcript_json_bytes"] = _json_size_bytes(messages)
    payload.update(_summarize_tool_errors(messages))

    submitted = payload.get("submitted_result")
    response = payload.get("response")
    answered = bool(submitted) or bool(isinstance(response, str) and response.strip())
    category = payload.get("error_category")
    metrics = {
        "answered": float(answered and not payload.get("is_error")),
        "is_error": float(bool(payload.get("is_error"))),
        "message_count": float(payload.get("message_count") or 0),
        "submitted_result_present": float(bool(submitted)),
        "has_browser_session": float(bool(payload.get("browser_session_id"))),
        "error_category_code": float(payload.get("error_category_code") or 0),
        "error_http_status": float(payload.get("error_http_status") or 0),
        "transcript_image_count": float(payload.get("transcript_image_count") or 0),
        "transcript_json_bytes": float(payload.get("transcript_json_bytes") or 0),
        "tool_error_count": float(payload.get("tool_error_count") or 0),
        "tool_error_validation": float(payload.get("tool_error_validation") or 0),
        "tool_error_max_streak": float(payload.get("tool_error_max_streak") or 0),
        "tool_error_unique_kinds": float(payload.get("tool_error_unique_kinds") or 0),
    }
    for category_name in ERROR_CATEGORY_CODES:
        if category_name == "none":
            continue
        metrics[f"error_{category_name}"] = float(category == category_name)
    by_tool = payload.get("tool_error_by_tool") or {}
    for tool_name in TOOL_ERROR_BREAKDOWN_NAMES:
        metrics[f"tool_error_{tool_name}"] = float(by_tool.get(tool_name, 0))

    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    transcript_path.write_text(json.dumps(messages, ensure_ascii=False, indent=2))
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    _write_progress(
        progress_path,
        "harness_program_artifacts_written",
        error_category=category,
        is_error=payload.get("is_error"),
        result_path=str(result_path),
        transcript_path=str(transcript_path),
        metrics_path=str(metrics_path),
    )

    print(
        json.dumps(
            {
                "result_path": str(result_path),
                "metrics_path": str(metrics_path),
                "transcript_path": str(transcript_path),
                "progress_path": str(progress_path),
                "error_category": category,
                "error_type": payload.get("error_type"),
                "error_excerpt": payload.get("error_excerpt"),
            }
        )
    )
    if exc_caught is not None and not isinstance(exc_caught, Exception):
        raise exc_caught
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Mini Browse harness.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-client", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--progress", default=DEFAULT_PROGRESS_PATH)
    parser.add_argument("--conversation")
    parser.add_argument("--max-steps", type=int, default=75)
    parser.add_argument("--workspace-root", default="/workspace/mini-browse")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
