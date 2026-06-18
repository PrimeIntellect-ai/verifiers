import asyncio
from collections.abc import Awaitable, Callable
from typing import cast

from verifiers.errors import SandboxDeleteError


async def delete_sandbox_for_rollout(
    sandbox_client: object,
    sandbox_id: str,
) -> None:
    try:
        delete = cast(
            Callable[[str], Awaitable[object]], getattr(sandbox_client, "delete")
        )
        # DELETE already has client-level timeout/retry; keep persistent outages visible.
        await asyncio.shield(delete(sandbox_id))
    except Exception as exc:
        # Catches APIError/APITimeoutError/HTTP/request failures; BaseException signals propagate.
        raise SandboxDeleteError(
            f"Failed to delete sandbox {sandbox_id}: {exc}"
        ) from exc
