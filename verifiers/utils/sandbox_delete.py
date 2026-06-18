import asyncio
from collections.abc import MutableMapping
from typing import Any

from verifiers.errors import SandboxDeleteError
from verifiers.utils.threaded_sandbox_client import ThreadedAsyncSandboxClient


def record_sandbox_delete_error(
    state: MutableMapping[str, Any],
    error: SandboxDeleteError,
    *,
    scope: str,
    sandbox_id: str,
) -> None:
    cleanup_errors = state.setdefault("cleanup_errors", [])
    if not isinstance(cleanup_errors, list):
        cleanup_errors = []
        state["cleanup_errors"] = cleanup_errors
    cleanup_errors.append(
        {
            "type": type(error).__name__,
            "message": str(error),
            "scope": scope,
            "sandbox_id": sandbox_id,
        }
    )
    if state.get("error") is None:
        set_error = getattr(state, "_set_error", None)
        if callable(set_error):
            set_error(error)
        else:
            state["error"] = error


async def delete_sandbox_for_rollout(
    sandbox_client: ThreadedAsyncSandboxClient,
    sandbox_id: str,
) -> None:
    try:
        # DELETE already has client-level timeout/retry; keep persistent outages visible.
        await asyncio.shield(sandbox_client.delete(sandbox_id))
    except Exception as exc:
        # Catches APIError/APITimeoutError/HTTP/request failures; BaseException signals propagate.
        raise SandboxDeleteError(
            f"Failed to delete sandbox {sandbox_id}: {exc}"
        ) from exc


async def cleanup_sandbox_for_rollout(
    sandbox_client: ThreadedAsyncSandboxClient,
    sandbox_id: str,
    state: MutableMapping[str, Any],
    *,
    scope: str,
) -> bool:
    try:
        await delete_sandbox_for_rollout(sandbox_client, sandbox_id)
    except SandboxDeleteError as exc:
        record_sandbox_delete_error(state, exc, scope=scope, sandbox_id=sandbox_id)
        return False
    deregister = state.get("_sandbox_deregister")
    if callable(deregister):
        deregister(sandbox_id)
    return True
