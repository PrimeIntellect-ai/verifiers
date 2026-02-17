import asyncio
import time
import uuid
from asyncio import Future
from typing import cast

import msgpack
import zmq
import zmq.asyncio

from verifiers.utils.logging_utils import print_time
from verifiers.utils.worker_utils import msgpack_encoder
from verifiers.workers.client.env_client import EnvClient
from verifiers.workers.server.zmq_env_server import derive_health_address
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
    ServerError,
    ServerState,
)


class ZMQEnvClient(EnvClient):
    """ZMQ-based environment client."""

    DEFAULT_REQUEST_TIMEOUT = 36_000  # 10h

    def __init__(self, address: str = "tcp://127.0.0.1:5000", **kwargs):
        super().__init__(address=address, **kwargs)

        # ZMQ context
        self.ctx = zmq.asyncio.Context()

        # DEALER socket for async request/response (work only)
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.socket.setsockopt(zmq.LINGER, 0)

        # TCP keepalive for faster dead server detection
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 10)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 2)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)

        # Separate REQ socket for health checks — connects to the server's
        # dedicated health thread, completely independent of work traffic.
        self.health_address = derive_health_address(address)
        self.health_socket = self.ctx.socket(zmq.REQ)
        self.health_socket.setsockopt(zmq.LINGER, 0)
        self.health_socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.health_socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self.health_socket_connected = False

        self.receiver_lock = asyncio.Lock()
        self.receiver_task: asyncio.Task | None = None

        # Track pending requests
        self.pending_requests: dict[str, PendingRequest] = {}
        self.pending_lock = asyncio.Lock()

        # Health check state
        self.server_state = ServerState.STARTUP
        self.health_check_lock = asyncio.Lock()
        self.health_check_task: asyncio.Task | None = None
        self.failed_health_checks = 0
        self.healthy_event = asyncio.Event()

    async def handle_health_request(
        self, request: HealthRequest, timeout: float | None
    ) -> HealthResponse:
        """Send health check via the dedicated health socket."""
        try:
            if not self.health_socket_connected:
                self.health_socket.connect(self.health_address)
                self.health_socket_connected = True

            await self.health_socket.send(b"ping")
            raw = await asyncio.wait_for(
                self.health_socket.recv(),
                timeout=timeout,
            )
            response = msgpack.unpackb(raw, raw=False)
            return HealthResponse.model_validate(response)
        except asyncio.TimeoutError:
            return HealthResponse(success=False, error="Health check timed out")
        except Exception as e:
            return HealthResponse(success=False, error=str(e))

    async def handle_run_rollout_request(
        self, request: RunRolloutRequest, timeout: float | None
    ) -> RunRolloutResponse:
        return await self.send_request(request, RunRolloutResponse, timeout=timeout)

    async def handle_run_group_request(
        self, request: RunGroupRequest, timeout: float | None
    ) -> RunGroupResponse:
        return await self.send_request(request, RunGroupResponse, timeout=timeout)

    async def wait_for_server_startup(
        self,
        timeout: float | None = None,
    ) -> None:
        """Wait for server to become healthy on initial startup."""
        timeout = timeout if timeout is not None else self.startup_timeout
        self.logger.info(
            f"Waiting for env server {self.name} to become healthy "
            f"(timeout={print_time(timeout)})"
        )
        await self.ensure_started()
        try:
            await asyncio.wait_for(self.healthy_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Env server {self.name} did not become healthy "
                f"within {print_time(timeout)}"
            )
        self.logger.info(f"Env server {self.name} is healthy")

    async def close(self) -> None:
        """Close the client and clean up ZMQ resources."""
        # Cancel health check task
        if self.health_check_task is not None:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None

        # Cancel receiver task
        if self.receiver_task is not None:
            self.receiver_task.cancel()
            try:
                await self.receiver_task
            except asyncio.CancelledError:
                pass
            self.receiver_task = None

        # Cancel all pending requests — use CancelledError (not ServerError)
        # so in-flight send_request calls propagate immediately instead of
        # waiting for a recovery that will never come.
        cancelled = await self.cancel_all_pending(
            reason="Client closed", use_cancelled=True
        )
        if cancelled:
            self.logger.info(
                f"Cancelled {len(cancelled)} pending requests during close of env server {self.name}"
            )

        # Close sockets and terminate context
        self.health_socket.close()
        self.socket.close()
        self.ctx.term()

    async def cancel_all_pending(
        self,
        reason: str = "Request cancelled",
        use_cancelled: bool = False,
    ) -> list[PendingRequest]:
        """Cancel all pending requests and return their metadata.

        Args:
            reason: Human-readable reason for cancellation.
            use_cancelled: If True, fail futures with CancelledError (non-retryable).
                If False (default), fail with ServerError (triggers retry in send_request).
        """
        async with self.pending_lock:
            pending_count = len(self.pending_requests)
            if pending_count:
                self.logger.debug(
                    f"Cancelling {pending_count} pending request(s) on env server {self.name} ({reason})"
                )

            # Collect metadata before clearing
            cancelled_requests = list(self.pending_requests.values())

            for pending_req in cancelled_requests:
                if not pending_req.future.done():
                    if use_cancelled:
                        pending_req.future.cancel()
                    else:
                        pending_req.future.set_exception(ServerError(reason))

            # Clear tracking dict
            self.pending_requests.clear()

        return cancelled_requests

    async def receive_loop(self):
        """Continuously receive responses from environment servers."""
        while True:
            try:
                # Receive multipart: [request_id, payload]
                msg = await self.socket.recv_multipart()

                if len(msg) < 2:
                    self.logger.error(
                        f"Received invalid message from env server {self.name}, expected 2 frames but got {len(msg)}"
                    )
                    continue

                request_id_bytes, response_data = msg[0], msg[1]
                request_id = request_id_bytes.decode()

                # Pop pending request atomically
                async with self.pending_lock:
                    pending_req = self.pending_requests.pop(request_id, None)

                if pending_req is not None and not pending_req.future.done():
                    try:
                        response = msgpack.unpackb(response_data, raw=False)
                        pending_req.future.set_result(response)
                    except Exception as unpack_error:
                        # Unpacking failed - fail the specific future
                        self.logger.error(
                            f"Request {request_id[:7]} failed to unpack response from env server {self.name} ({unpack_error})"
                        )
                        pending_req.future.set_exception(
                            RuntimeError(
                                f"Failed to deserialize response: {unpack_error}"
                            )
                        )
                elif pending_req is None:
                    pass  # ignore responses for requests we already popped (e.g. timed out)

            except asyncio.CancelledError:
                break
            except zmq.ZMQError as e:
                self.logger.error(
                    f"ZMQ socket error in receive loop for env server {self.name} ({e})"
                )
                await self.cancel_all_pending(f"ZMQ socket error: {e}")
                break
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in receive loop for env server {self.name} ({e})",
                    exc_info=True,
                )
                # Don't break - log and continue for non-socket errors

    async def ensure_started(self) -> None:
        """Ensure receiver and health check loop are running."""
        if self.receiver_task is None:
            async with self.receiver_lock:
                if self.receiver_task is None:
                    self.receiver_task = asyncio.create_task(self.receive_loop())
                    self.socket.connect(self.address)

        if self.health_check_interval > 0 and self.health_check_task is None:
            async with self.health_check_lock:
                if self.health_check_task is None:
                    self.health_check_task = asyncio.create_task(
                        self.health_check_loop()
                    )

    async def send_request(
        self,
        request: BaseRequest,
        response_type: type[BaseResponseT],
        timeout: float | None = None,
    ) -> BaseResponseT:
        """Send request to environment and await response with automatic retry."""
        await self.ensure_started()

        effective_timeout = self.DEFAULT_REQUEST_TIMEOUT if timeout is None else timeout

        # Serialize once — the payload doesn't change across retries
        payload_bytes = cast(
            bytes,
            msgpack.packb(
                request.model_dump(mode="python", warnings=False),
                default=msgpack_encoder,
                use_bin_type=True,
            ),
        )

        while True:
            request_id = uuid.uuid4().hex

            # Create future and pending request atomically
            future: Future = asyncio.Future()
            pending_req = PendingRequest(
                request_id=request_id,
                request=request,
                submitted_at=time.time(),
                timeout=effective_timeout,
                future=future,
            )

            async with self.pending_lock:
                self.pending_requests[request_id] = pending_req

            await self.socket.send_multipart([request_id.encode(), payload_bytes])

            try:
                raw_response = await asyncio.wait_for(future, timeout=effective_timeout)
            except asyncio.TimeoutError:
                # Clean up on timeout
                async with self.pending_lock:
                    self.pending_requests.pop(request_id, None)
                log = (
                    self.logger.debug
                    if isinstance(request, HealthRequest)
                    else self.logger.error
                )
                log(
                    f"Request {request_id[:7]} timed out on env server {self.name} "
                    f"after {effective_timeout:.1f}s "
                    f"(type={request.request_type}, pending={len(self.pending_requests)})"
                )
                raise TimeoutError(
                    f"Environment timeout for {request.request_type} "
                    f"request after {effective_timeout}s"
                )
            except ServerError as e:
                self.logger.debug(
                    f"Request {request_id[:7]} waiting for env server {self.name} to recover ({e})"
                )

                try:
                    await asyncio.wait_for(
                        self.healthy_event.wait(),
                        timeout=self.recovery_timeout,
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"Env server {self.name} did not recover within {print_time(self.recovery_timeout)}"
                    )

                continue  # retry the request
            except RuntimeError:
                async with self.pending_lock:
                    self.pending_requests.pop(request_id, None)
                raise

            # validate response with Pydantic
            response = response_type.model_validate(raw_response)

            if not response.success:
                raise RuntimeError(response.error)

            return response

    async def health_check_loop(self):
        """Background task that periodically checks server health and handles state transitions."""
        self.logger.debug(
            f"Starting health check loop for env server {self.name} (interval={print_time(self.health_check_interval)})"
        )

        probe_timeout = self.health_check_interval / 2

        while True:
            try:
                cycle_start = asyncio.get_event_loop().time()

                is_healthy = await self.health(timeout=probe_timeout)

                if is_healthy:
                    if self.server_state != ServerState.HEALTHY:
                        self.logger.info(
                            f"Env server {self.name} is healthy again "
                            f"(was {self.server_state.value}), "
                            f"rescheduling requests"
                        )
                    self.server_state = ServerState.HEALTHY
                    self.failed_health_checks = 0
                    self.healthy_event.set()
                else:
                    self.failed_health_checks += 1

                    # Transition to UNHEALTHY after 3 consecutive failures
                    if (
                        self.server_state == ServerState.HEALTHY
                        and self.failed_health_checks >= 3
                    ):
                        self.server_state = ServerState.UNHEALTHY
                        self.healthy_event.clear()
                        cancelled = await self.cancel_all_pending(
                            f"Env server {self.name} unhealthy: {self.failed_health_checks} "
                            f"consecutive health check failures"
                        )
                        self.logger.warning(
                            f"Env server {self.name} detected unhealthy, "
                            f"cancelling {len(cancelled)} pending request(s)"
                        )

                elapsed = asyncio.get_event_loop().time() - cycle_start
                await asyncio.sleep(max(0, self.health_check_interval - elapsed))

            except asyncio.CancelledError:
                self.logger.debug(
                    f"Health check loop for env server {self.name} cancelled"
                )
                break
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in health check loop for env server {self.name}: {e}",
                    exc_info=True,
                )
