import asyncio
from collections.abc import Callable

from verifiers.errors import SandboxDeleteError
from verifiers.types import State
from verifiers.utils.threaded_sandbox_client import ThreadedAsyncSandboxClient


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
    state: State,
    *,
    scope: str,
    on_deleted: Callable[[str], None] | None = None,
) -> bool:
    """Delete during cleanup, recording ordinary DELETE failures on rollout state."""
    try:
        await delete_sandbox_for_rollout(sandbox_client, sandbox_id)
    except SandboxDeleteError as exc:
        cleanup_errors = state.setdefault("cleanup_errors", [])
        if not isinstance(cleanup_errors, list):
            cleanup_errors = []
            state["cleanup_errors"] = cleanup_errors
        cleanup_errors.append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "scope": scope,
                "sandbox_id": sandbox_id,
            }
        )
        if state.get("error") is None:
            state["error"] = exc
        return False
    deregister = on_deleted or state.get("_sandbox_deregister")
    if callable(deregister):
        deregister(sandbox_id)
    return True
