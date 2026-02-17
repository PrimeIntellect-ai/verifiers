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
)


class ZMQEnvClient(EnvClient):
    """ZMQ-based environment client."""

    DEFAULT_REQUEST_TIMEOUT = 36_000  # 10h

    def __init__(self, address: str = "tcp://127.0.0.1:5000", **kwargs):
        super().__init__(address=address, **kwargs)

        # DEALER socket for async request/response
        self.ctx = zmq.asyncio.Context()
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

        # Single dict for all pending requests (includes futures)
        self.pending_requests: dict[str, PendingRequest] = {}
        self._receiver_task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()
        self._pending_lock = asyncio.Lock()
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

    async def _health_check_loop(self):
        """Background task that periodically checks server health."""
        self.logger.debug(
            f"Starting health check loop (interval={print_time(self.health_check_interval)})"
        )

        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Perform health check
                try:
                    is_healthy = await self.health(timeout=self.health_check_timeout)
                    if is_healthy:
                        if self._failed_health_checks > 0:
                            self.logger.info("Server health check passed")
                        self._failed_health_checks = 0
                    else:
                        # Health check returned False (shouldn't happen but handle it)
                        self._failed_health_checks += 1
                        self.logger.warning(
                            f"Health check failed ({self._failed_health_checks} consecutive)"
                        )
                except Exception as e:
                    self._failed_health_checks += 1
                    self.logger.warning(
                        f"Health check error ({self._failed_health_checks} consecutive): {e}"
                    )

                    # Log error after multiple consecutive failures
                    if self._failed_health_checks >= 2:
                        self.logger.error("Server health checks failing")

            except asyncio.CancelledError:
                self.logger.debug("Health check loop cancelled")
                break
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in health check loop: {e}", exc_info=True
                )

    async def cancel_all_pending(self) -> list[PendingRequest]:
        """Cancel all pending requests and return their metadata."""
        async with self._pending_lock:
            pending_count = len(self.pending_requests)
            if pending_count:
                self.logger.warning(f"Cancelling {pending_count} pending request(s)")

            # Collect metadata before clearing
            cancelled_requests = list(self.pending_requests.values())

            # Fail all futures
            for pending_req in cancelled_requests:
                if not pending_req.future.done():
                    pending_req.future.set_exception(RuntimeError("Request cancelled"))

            # Clear tracking dict
            self.pending_requests.clear()

        return cancelled_requests

    async def wait_for_server_health(
        self,
        timeout: float = 600.0,
        check_interval: float = 10.0,
    ) -> None:
        """Wait for server to become healthy.

        Universal method for both initial startup and recovery scenarios.
        """
        self.logger.info(
            f"Waiting for server to become healthy (timeout={print_time(timeout)})..."
        )
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                is_healthy = await self.health(timeout=check_interval)
                if is_healthy:
                    self._failed_health_checks = 0
                    self.logger.info(
                        f"Server is healthy after {print_time(time.time() - start_time)}"
                    )
                    return
            except Exception as e:
                self.logger.debug(f"Health check failed: {e}")

            await asyncio.sleep(check_interval)

        # Timeout reached
        raise TimeoutError(
            f"Server did not become healthy within {print_time(timeout)}"
        )

    def _fail_all_pending(self, reason: str):
        """Fail all pending requests synchronously."""
        pending_count = len(self.pending_requests)
        if pending_count:
            self.logger.warning(f"Failing {pending_count} pending request(s): {reason}")

        for pending_req in list(self.pending_requests.values()):
            if not pending_req.future.done():
                pending_req.future.set_exception(RuntimeError(reason))

        self.pending_requests.clear()

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
                    pending_req = self.pending_requests.pop(request_id, None)

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
                        f"(pending={len(self.pending_requests)})"
                    )

            except asyncio.CancelledError:
                break
            except zmq.ZMQError as e:
                # Socket-level error - fail all pending futures and exit
                self.logger.error(f"ZMQ socket error in receive loop: {e}")
                self._fail_all_pending(f"ZMQ socket error: {e}")
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
    ) -> BaseResponseT:
        """Send request to environment and await response."""
        # auto-start receiver if not already running (with lock to prevent race)
        if self._receiver_task is None:
            async with self._start_lock:
                if self._receiver_task is None:
                    await self._start()

        # Start health check task if enabled and not running
        if self.health_check_interval > 0 and self._health_check_task is None:
            async with self._start_lock:
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
        from verifiers.workers.types import PendingRequest

        pending_req = PendingRequest(
            request_id=request_id,
            request=request,
            submitted_at=time.time(),
            timeout=effective_timeout,
            future=future,
        )

        async with self._pending_lock:
            self.pending_requests[request_id] = pending_req

        await self.socket.send_multipart([request_id.encode(), payload_bytes])

        try:
            raw_response = await asyncio.wait_for(future, timeout=effective_timeout)
        except asyncio.TimeoutError:
            # Clean up on timeout
            async with self._pending_lock:
                self.pending_requests.pop(request_id, None)
            self.logger.error(
                f"Timed out waiting for request_id={request_id} type={request.request_type} "
                f"after {effective_timeout:.1f}s (pending={len(self.pending_requests)})"
            )
            raise TimeoutError(
                f"Environment timeout for {request.request_type} request after {effective_timeout}s"
            )

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
        cancelled = await self.cancel_all_pending()
        if cancelled:
            self.logger.info(
                f"Cancelled {len(cancelled)} pending requests during close"
            )

        # Close socket and terminate context
        self.socket.close()
        self.ctx.term()
