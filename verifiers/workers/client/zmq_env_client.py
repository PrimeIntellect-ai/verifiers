import asyncio
import threading
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
        self.socket.setsockopt(zmq.SNDHWM, 0)  # no limit
        self.socket.setsockopt(zmq.RCVHWM, 0)  # no limit
        self.socket.setsockopt(zmq.LINGER, 0)  # discard msgs on socket close

        # TCP keepalive for faster dead server detection
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 10)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 2)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)

        # Health address — probed by the dedicated health-check thread
        self.health_address = derive_health_address(address)

        self.receiver_lock = asyncio.Lock()
        self.receiver_task: asyncio.Task | None = None

        # Track pending requests
        self.pending_requests: dict[str, PendingRequest] = {}
        self.pending_lock = asyncio.Lock()

        # Health check state — the periodic health monitor runs on a
        # dedicated thread so it is completely immune to event loop lag.
        self.server_state = ServerState.STARTUP
        self.healthy_event = asyncio.Event()
        self.health_thread: threading.Thread | None = None
        self.stop_health_thread = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None

    async def handle_health_request(
        self, request: HealthRequest, timeout: float | None
    ) -> HealthResponse:
        """Return current server health based on the dedicated health-check thread.

        The actual probing is done by ``run_health_check_thread`` on its own
        thread with its own ZMQ context, immune to event loop lag.
        This method simply reports the latest known state.
        """
        if self.server_state == ServerState.HEALTHY:
            return HealthResponse(success=True)
        return HealthResponse(
            success=False, error=f"Server state: {self.server_state.value}"
        )

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
        # Stop health check thread
        self.stop_health_thread.set()
        if self.health_thread is not None:
            self.health_thread.join(timeout=5)
            self.health_thread = None

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

        # Close async sockets and terminate context
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

        if self.health_check_interval > 0 and self.health_thread is None:
            self.loop = asyncio.get_running_loop()
            self.health_thread = threading.Thread(
                target=self.run_health_check_thread,
                name="health-checker",
                daemon=True,
            )
            self.health_thread.start()

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

    def run_health_check_thread(self):
        """Dedicated health-check thread.

        Runs the full probe-and-state-machine loop on its own thread with its
        own synchronous ZMQ context so that it is completely immune to asyncio
        event loop lag from high-concurrency workloads.  State transitions are
        forwarded to the event loop via ``call_soon_threadsafe``.
        """
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.REQ_RELAXED, 1)
        sock.setsockopt(zmq.REQ_CORRELATE, 1)
        sock.connect(self.health_address)

        # Generous probe timeout — no cost since this is a dedicated thread.
        probe_timeout_ms = max(int(self.health_check_interval * 1000), 2000)
        sock.setsockopt(zmq.SNDTIMEO, probe_timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, probe_timeout_ms)

        failed = 0
        state = ServerState.STARTUP

        assert self.loop is not None

        while not self.stop_health_thread.is_set():
            is_healthy = False
            try:
                sock.send(b"ping")
                raw = sock.recv()
                resp = msgpack.unpackb(raw, raw=False)
                is_healthy = resp.get("success", False)
            except zmq.Again:
                pass
            except Exception:
                pass

            if is_healthy:
                if state != ServerState.HEALTHY:
                    old_state = state
                    state = ServerState.HEALTHY
                    failed = 0
                    self.loop.call_soon_threadsafe(self.on_became_healthy, old_state)
                else:
                    failed = 0
            else:
                failed += 1
                if state == ServerState.HEALTHY and failed >= 5:
                    state = ServerState.UNHEALTHY
                    self.loop.call_soon_threadsafe(self.on_became_unhealthy, failed)

            self.stop_health_thread.wait(self.health_check_interval)

        sock.close()
        ctx.term()

    def on_became_healthy(self, old_state: ServerState):
        self.server_state = ServerState.HEALTHY
        self.healthy_event.set()
        self.logger.info(
            f"Env server {self.name} became healthy (was {old_state.value})"
        )

    def on_became_unhealthy(self, failed_checks: int):
        self.server_state = ServerState.UNHEALTHY
        self.healthy_event.clear()
        msg = f"Env server {self.name} became unhealthy ({failed_checks} consecutive health check failures)"
        asyncio.ensure_future(self.cancel_all_pending(msg))
        self.logger.warning(msg)
