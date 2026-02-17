import asyncio
import time
import uuid
from typing import cast

import msgpack
import zmq
import zmq.asyncio

from verifiers.utils.logging_utils import print_time
from verifiers.utils.worker_utils import msgpack_encoder
from verifiers.workers.client.env_client import EnvClient
from verifiers.workers.types import (
    BaseRequest,
    BaseResponseT,
    HealthRequest,
    HealthResponse,
    PendingRequest,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
    ServerState,
)


class ZMQEnvClient(EnvClient):
    """ZMQ-based environment client."""

    DEFAULT_REQUEST_TIMEOUT = 36_000  # 10h

    def __init__(self, address: str = "tcp://127.0.0.1:5000", **kwargs):
        super().__init__(address=address, **kwargs)

        # ZMQ context
        self.ctx = zmq.asyncio.Context()

        # DEALER socket for async request/response
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.socket.setsockopt(zmq.LINGER, 0)

        # TCP keepalive for faster dead server detection
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.socket.setsockopt(
            zmq.TCP_KEEPALIVE_IDLE, 10
        )  # Start probes after 10s idle
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 2)  # Probe every 2s
        self.socket.setsockopt(
            zmq.TCP_KEEPALIVE_CNT, 3
        )  # Give up after 3 failed probes

        self._receiver_lock = asyncio.Lock()
        self._receiver_task: asyncio.Task | None = None

        # Track pending requests
        self._pending_requests: dict[str, PendingRequest] = {}
        self._pending_lock = asyncio.Lock()

        # Run health check loop
        self._server_state = ServerState.STARTUP
        self._health_check_lock = asyncio.Lock()
        self._health_check_task: asyncio.Task | None = None
        self._failed_health_checks = 0

    async def handle_health_request(
        self, request: HealthRequest, timeout: float | None
    ) -> HealthResponse:
        return await self._send_request(request, HealthResponse, timeout=timeout)

    async def handle_run_rollout_request(
        self, request: RunRolloutRequest, timeout: float | None
    ) -> RunRolloutResponse:
        return await self._send_request(request, RunRolloutResponse, timeout=timeout)

    async def handle_run_group_request(
        self, request: RunGroupRequest, timeout: float | None
    ) -> RunGroupResponse:
        return await self._send_request(request, RunGroupResponse, timeout=timeout)

    async def _cancel_all_pending(
        self, reason: str = "Request cancelled"
    ) -> list[PendingRequest]:
        """Cancel all pending requests and return their metadata."""
        async with self._pending_lock:
            pending_count = len(self._pending_requests)
            if pending_count:
                self.logger.warning(
                    f"Cancelling {pending_count} pending request(s): {reason}"
                )

            # Collect metadata before clearing
            cancelled_requests = list(self._pending_requests.values())

            # Fail all futures with the provided reason
            for pending_req in cancelled_requests:
                if not pending_req.future.done():
                    pending_req.future.set_exception(RuntimeError(reason))

            # Clear tracking dict
            self._pending_requests.clear()

        return cancelled_requests

    async def _receive_loop(self):
        """Continuously receive responses from environment servers."""
        while True:
            try:
                # Receive multipart: [request_id, payload]
                msg = await self.socket.recv_multipart()

                if len(msg) < 2:
                    self.logger.error(
                        f"Invalid message format: expected 2 frames, got {len(msg)}"
                    )
                    continue

                request_id_bytes, response_data = msg[0], msg[1]
                request_id = request_id_bytes.decode()

                # Pop pending request atomically
                async with self._pending_lock:
                    pending_req = self._pending_requests.pop(request_id, None)

                if pending_req is not None and not pending_req.future.done():
                    try:
                        response = msgpack.unpackb(response_data, raw=False)
                        pending_req.future.set_result(response)
                    except Exception as unpack_error:
                        # Unpacking failed - fail the specific future
                        self.logger.error(
                            f"Failed to unpack response for request {request_id}: {unpack_error}"
                        )
                        pending_req.future.set_exception(
                            RuntimeError(
                                f"Failed to deserialize response: {unpack_error}"
                            )
                        )
                elif pending_req is None:
                    self.logger.warning(
                        f"Received response for unknown request_id={request_id} "
                        f"(pending={len(self._pending_requests)})"
                    )

            except asyncio.CancelledError:
                break
            except zmq.ZMQError as e:
                self.logger.error(f"ZMQ socket error in receive loop: {e}")
                await self._cancel_all_pending(f"ZMQ socket error: {e}")
                break
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in ZMQ receive loop: {e}", exc_info=True
                )
                # Don't break - log and continue for non-socket errors

    async def _start(self):
        self._receiver_task = asyncio.create_task(self._receive_loop())
        self.socket.connect(self.address)

    async def _send_request(
        self,
        request: BaseRequest,
        response_type: type[BaseResponseT],
        timeout: float | None = None,
        attempt: int = 0,
    ) -> BaseResponseT:
        """Send request to environment and await response with automatic retry."""
        # auto-start receiver if not already running (with lock to prevent race)
        if self._receiver_task is None:
            async with self._receiver_lock:
                if self._receiver_task is None:
                    await self._start()

        # Start health check task if enabled and not running
        if self.health_check_interval > 0 and self._health_check_task is None:
            async with self._health_check_lock:
                if self._health_check_task is None:
                    self._health_check_task = asyncio.create_task(
                        self._health_check_loop()
                    )

        effective_timeout = self.DEFAULT_REQUEST_TIMEOUT if timeout is None else timeout

        # Use request_id from Pydantic model, encode to bytes for ZMQ frame
        request_id = uuid.uuid4().hex

        # Serialize using Pydantic
        payload_bytes = cast(
            bytes,
            msgpack.packb(
                request.model_dump(mode="python", warnings=False),
                default=msgpack_encoder,
                use_bin_type=True,
            ),
        )

        # Create future and pending request atomically
        future: asyncio.Future[dict] = asyncio.Future()
        pending_req = PendingRequest(
            request_id=request_id,
            request=request,
            submitted_at=time.time(),
            timeout=effective_timeout,
            future=future,
        )

        async with self._pending_lock:
            self._pending_requests[request_id] = pending_req

        await self.socket.send_multipart([request_id.encode(), payload_bytes])

        try:
            raw_response = await asyncio.wait_for(future, timeout=effective_timeout)
        except asyncio.TimeoutError:
            # Clean up on timeout
            async with self._pending_lock:
                self._pending_requests.pop(request_id, None)
            self.logger.error(
                f"Timed out waiting for request_id={request_id} type={request.request_type} "
                f"after {effective_timeout:.1f}s (pending={len(self._pending_requests)})"
            )
            raise TimeoutError(
                f"Environment timeout for {request.request_type} request after {effective_timeout}s"
            )
        except RuntimeError as e:
            # Check if this is a server crash and we should retry
            error_msg = str(e)
            is_server_error = (
                "ZMQ socket error" in error_msg
                or "Request cancelled" in error_msg
                or "Server unhealthy" in error_msg
            )

            if is_server_error and attempt < self.max_auto_retries:
                self.logger.warning(
                    f"Request failed due to server error (attempt {attempt + 1}/{self.max_auto_retries}): {e}"
                )
                self.logger.info("Waiting for server recovery before retry...")

                # Wait for server to recover
                await self._wait_for_server_recovery()

                # Retry with incremented attempt counter
                self.logger.info(
                    f"Retrying request (attempt {attempt + 2}/{self.max_auto_retries + 1})"
                )
                return await self._send_request(
                    request, response_type, timeout, attempt + 1
                )

            # Either not a server error or exceeded max retries
            # Clean up metadata before re-raising
            async with self._pending_lock:
                self._pending_requests.pop(request_id, None)
            raise

        # Clean up metadata on success (receive loop already popped it, but do it here too for safety)
        async with self._pending_lock:
            self._pending_requests.pop(request_id, None)

        # validate response with Pydantic
        response = response_type.model_validate(raw_response)

        if not response.success:
            raise RuntimeError(response.error)

        return response

    async def close(self) -> None:
        """Close the client and clean up ZMQ resources."""
        # Cancel health check task
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        # Cancel receiver task
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None

        # Cancel all pending requests
        cancelled = await self._cancel_all_pending()
        if cancelled:
            self.logger.info(
                f"Cancelled {len(cancelled)} pending requests during close"
            )

        # Close socket and terminate context
        self.socket.close()
        self.ctx.term()

    async def _health_check_loop(self):
        """Background task that periodically checks server health."""
        self.logger.debug(
            f"Starting health check loop (interval={print_time(self.health_check_interval)})"
        )

        while True:
            try:
                # Skip health checks if server is recovering
                if self._server_state == ServerState.RECOVERING:
                    await asyncio.sleep(self.health_check_interval)
                    continue

                try:
                    await self.health(timeout=self.health_check_interval)
                    self._server_state = ServerState.HEALTHY
                    self._failed_health_checks = 0
                except Exception as e:
                    self._failed_health_checks += 1
                    self.logger.debug(
                        f"Health check failed ({self._failed_health_checks} consecutive): {e}"
                    )

                    # Transition from HEALTHY to UNHEALTHY after 3 consecutive failures
                    if (
                        self._server_state == ServerState.HEALTHY
                        and self._failed_health_checks >= 3
                    ):
                        self.logger.warning(
                            "Server is unhealthy after 3 consecutive health check failures - "
                            "cancelling pending requests to trigger retry"
                        )
                        self._server_state = ServerState.UNHEALTHY
                        await self._cancel_all_pending(
                            f"Server unhealthy: {self._failed_health_checks} consecutive health check failures"
                        )

                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                self.logger.debug("Health check loop cancelled")
                break
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in health check loop: {e}", exc_info=True
                )

    async def _wait_for_server_recovery(
        self,
        timeout: float | None = None,
        interval: float | None = None,
    ) -> None:
        """Wait for server to recover, managing state transitions."""
        self._server_state = ServerState.RECOVERING

        timeout = timeout if timeout is not None else self.recovery_timeout
        interval = interval if interval is not None else self.health_check_interval
        await self._wait_for_server_health(timeout=timeout, interval=interval)

        self._server_state = ServerState.HEALTHY
        self._failed_health_checks = 0
