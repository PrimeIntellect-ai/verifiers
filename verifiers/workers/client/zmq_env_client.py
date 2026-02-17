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
    PendingTaskInfo,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
    ServerState,
)


class ZMQEnvClient(EnvClient):
    """ZMQ-based environment client."""

    DEFAULT_REQUEST_TIMEOUT = 36_000  # 10h

    def __init__(
        self,
        address: str = "tcp://127.0.0.1:5000",
        health_check_interval: float = 60.0,  # 1m, 0 to disable
        health_check_timeout: float = 1.0,  # 1s
        recovery_timeout: float = 600.0,  # 10m
    ):
        super().__init__(address=address)

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

        # Existing state
        self.pending: dict[str, asyncio.Future] = {}
        self._receiver_task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()

        # Task metadata cache for rescheduling
        self.pending_tasks: dict[str, PendingTaskInfo] = {}
        self._pending_lock = asyncio.Lock()

        # Health check configuration
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.recovery_timeout = recovery_timeout

        # Server state management
        self._server_state = ServerState.HEALTHY
        self._state_lock = asyncio.Lock()
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

                # Skip if we're already recovering
                async with self._state_lock:
                    if self._server_state == ServerState.RECOVERING:
                        continue

                # Perform health check
                try:
                    is_healthy = await self.health(timeout=self.health_check_timeout)
                    if is_healthy:
                        async with self._state_lock:
                            if self._server_state != ServerState.HEALTHY:
                                self.logger.info("Server recovered")
                                self._server_state = ServerState.HEALTHY
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

                    # Mark as unhealthy after 2 consecutive failures
                    if self._failed_health_checks >= 2:
                        async with self._state_lock:
                            if self._server_state == ServerState.HEALTHY:
                                self.logger.error("Server found to be unhealthy")
                                self._server_state = ServerState.UNHEALTHY

            except asyncio.CancelledError:
                self.logger.debug("Health check loop cancelled")
                break
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in health check loop: {e}", exc_info=True
                )

    async def cancel_all_pending(
        self, reason: str = "Cancelled"
    ) -> list[PendingTaskInfo]:
        """Cancel all pending requests and return their metadata."""
        async with self._pending_lock:
            pending_count = len(self.pending)
            if pending_count:
                self.logger.warning(
                    f"Cancelling {pending_count} pending request(s): {reason}"
                )

            # Collect metadata before clearing
            cancelled_tasks = list(self.pending_tasks.values())

            # Fail all futures
            for request_id, future in list(self.pending.items()):
                if not future.done():
                    future.set_exception(RuntimeError(reason))

            # Clear both tracking dicts
            self.pending.clear()
            self.pending_tasks.clear()

        return cancelled_tasks

    async def wait_for_recovery(
        self,
        timeout: float = 600.0,
        check_interval: float = 10.0,
    ) -> None:
        """Wait for server to recover after a failure."""
        async with self._state_lock:
            self._server_state = ServerState.RECOVERING

        self.logger.info(f"Waiting for server recovery (timeout={timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                is_healthy = await self.health(timeout=check_interval)
                if is_healthy:
                    async with self._state_lock:
                        self._server_state = ServerState.HEALTHY
                        self._failed_health_checks = 0
                    self.logger.info("Server has recovered")
                    return
            except Exception as e:
                self.logger.debug(f"Recovery check failed: {e}")

            await asyncio.sleep(check_interval)

        # Timeout reached
        async with self._state_lock:
            self._server_state = ServerState.UNHEALTHY
        raise TimeoutError(f"Server did not recover within {print_time(timeout)}")

    async def _cancel_and_maybe_recover(self, reason: str):
        """Cancel all pending and mark server as unhealthy."""
        await self.cancel_all_pending(reason)
        async with self._state_lock:
            self._server_state = ServerState.UNHEALTHY

    def _fail_all_pending(self, reason: str):
        """Synchronous wrapper for failing all pending requests."""
        # Schedule the async cancel method
        try:
            loop = asyncio.get_running_loop()
            # Fire and forget - don't await
            loop.create_task(self._cancel_and_maybe_recover(reason))
        except RuntimeError:
            # No event loop running, fail synchronously
            pending_count = len(self.pending)
            if pending_count:
                self.logger.warning(
                    f"Failing {pending_count} pending request(s): {reason}"
                )
            for request_id, future in list(self.pending.items()):
                if not future.done():
                    future.set_exception(RuntimeError(reason))
            self.pending.clear()
            self.pending_tasks.clear()

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

                # Pop both future and metadata atomically
                async with self._pending_lock:
                    future = self.pending.pop(request_id, None)
                    self.pending_tasks.pop(request_id, None)  # Clean up metadata

                if future is not None and not future.done():
                    try:
                        response = msgpack.unpackb(response_data, raw=False)
                        future.set_result(response)
                    except Exception as unpack_error:
                        # Unpacking failed - fail the specific future
                        self.logger.error(
                            f"Failed to unpack response for request {request_id}: {unpack_error}"
                        )
                        future.set_exception(
                            RuntimeError(
                                f"Failed to deserialize response: {unpack_error}"
                            )
                        )
                elif future is None:
                    self.logger.warning(
                        f"Received response for unknown request_id={request_id} (pending={len(self.pending)})"
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

        future: asyncio.Future[dict] = asyncio.Future()

        # Store future and metadata atomically
        async with self._pending_lock:
            self.pending[request_id] = future
            # Cache metadata for potential rescheduling
            self.pending_tasks[request_id] = PendingTaskInfo(
                request_id=request_id,
                request=request,
                submitted_at=time.time(),
                timeout=effective_timeout,
            )

        await self.socket.send_multipart([request_id.encode(), payload_bytes])

        try:
            raw_response = await asyncio.wait_for(future, timeout=effective_timeout)
        except asyncio.TimeoutError:
            # Clean up on timeout
            async with self._pending_lock:
                self.pending.pop(request_id, None)
                self.pending_tasks.pop(request_id, None)
            self.logger.error(
                f"Timed out waiting for request_id={request_id} type={request.request_type} "
                f"after {effective_timeout:.1f}s (pending={len(self.pending)})"
            )
            raise TimeoutError(
                f"Environment timeout for {request.request_type} request after {effective_timeout}s"
            )
        except RuntimeError as e:
            # Check if this is a server crash and we should retry
            if "ZMQ socket error" in str(e) or "Client closed" in str(e):
                self.logger.warning(
                    "Request failed due to server error. Will retry after recovery..."
                )
                # Wait for server to recover
                await self.wait_for_recovery(timeout=self.recovery_timeout)
                # Retry with incremented attempt counter
                return await self._send_request(request, response_type, timeout)
            raise

        # Clean up metadata on success
        async with self._pending_lock:
            self.pending_tasks.pop(request_id, None)

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
        cancelled = await self.cancel_all_pending("Client closed")
        if cancelled:
            self.logger.info(f"Cancelled {len(cancelled)} pending tasks during close")

        # Close socket and terminate context
        self.socket.close()
        self.ctx.term()
