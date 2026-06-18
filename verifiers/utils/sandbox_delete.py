import asyncio

from verifiers.errors import SandboxDeleteError
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
