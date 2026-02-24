"""Elastic endpoint pool — background polling loop that reloads endpoints.toml."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from verifiers.types import EndpointClientConfig
from verifiers.utils.async_utils import EndpointSlot, LeastLoadedDispatcher
from verifiers.utils.client_utils import resolve_client_configs
from verifiers.utils.eval_utils import load_endpoints

if TYPE_CHECKING:
    from verifiers.types import ClientConfig

logger = logging.getLogger(__name__)


class ElasticEndpointPool:
    """Periodically re-reads an endpoints file and updates a dispatcher.

    The pool runs a background ``asyncio.Task`` that polls the endpoints
    file at a fixed interval.  On each tick it rebuilds the
    :class:`EndpointSlot` list and calls
    :meth:`LeastLoadedDispatcher.update_variants` so that new endpoints
    are picked up and removed endpoints are drained.
    """

    def __init__(
        self,
        dispatcher: LeastLoadedDispatcher,
        endpoints_path: str,
        endpoint_id: str,
        poll_interval: float,
        base_client_config: ClientConfig,
    ) -> None:
        self._dispatcher = dispatcher
        self._endpoints_path = endpoints_path
        self._endpoint_id = endpoint_id
        self._poll_interval = poll_interval
        self._base_client_config = base_client_config
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Start the background polling loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Cancel the polling task and wait for it to finish."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _poll_loop(self) -> None:
        """Sleep → reload, repeat until cancelled."""
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._reload()
            except Exception:
                logger.warning(
                    "Elastic pool reload failed; keeping previous endpoints",
                    exc_info=True,
                )

    async def _reload(self) -> None:
        """Load endpoints file and push updated variants to the dispatcher."""
        endpoints = load_endpoints(self._endpoints_path)

        endpoint_group = endpoints.get(self._endpoint_id)
        if endpoint_group is None:
            logger.warning(
                "Elastic pool: endpoint_id %r not found in %s; skipping update",
                self._endpoint_id,
                self._endpoints_path,
            )
            return

        # Check that all variants have max_concurrent set
        missing = [
            i
            for i, ep in enumerate(endpoint_group)
            if ep.get("max_concurrent") is None
        ]
        if missing:
            logger.warning(
                "Elastic pool: endpoint_id %r has variants without max_concurrent "
                "(indices %s); skipping update",
                self._endpoint_id,
                missing,
            )
            return

        # Build EndpointClientConfig list (same pattern as eval.py)
        endpoint_configs = [
            EndpointClientConfig(
                api_key_var=ep["key"],
                api_base_url=ep["url"],
                max_concurrent=ep.get("max_concurrent"),
            )
            for ep in endpoint_group
        ]

        # Create a temporary ClientConfig with the new endpoint_configs
        # so resolve_client_configs can merge parent fields.
        temp_config = self._base_client_config.model_copy(
            update={"endpoint_configs": endpoint_configs}
        )
        resolved = resolve_client_configs(temp_config)

        slots = [
            EndpointSlot(
                config=cfg,
                max_concurrent=ep.max_concurrent,
            )
            for cfg, ep in zip(resolved, endpoint_configs)
        ]

        added, removed = await self._dispatcher.update_variants(slots)
        if added or removed:
            logger.info(
                "Elastic pool updated endpoint_id %r: +%d -%d endpoints",
                self._endpoint_id,
                added,
                removed,
            )
