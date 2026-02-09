import asyncio
import uuid
from typing import cast

import msgpack
import zmq
import zmq.asyncio

from verifiers.utils.worker_utils import msgpack_encoder
from verifiers.workers.client.env_client import EnvClient
from verifiers.workers.types import (
    BaseRequest,
    BaseResponseT,
    HealthRequest,
    HealthResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)


class ZMQEnvClient(EnvClient):
    """ZMQ-based environment client."""

    DEFAULT_REQUEST_TIMEOUT = 36_000  # 10h

    def __init__(self, address: str = "tcp://127.0.0.1:5000"):
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

        self.pending: dict[str, asyncio.Future] = {}
        self._receiver_task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()

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

    def _fail_all_pending(self, reason: str):
        """Fail all pending futures with the given reason."""
        pending_count = len(self.pending)
        if pending_count:
            self.logger.warning(f"Failing {pending_count} pending request(s): {reason}")
        for _, future in list(self.pending.items()):
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self.pending.clear()

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

                if request_id in self.pending:
                    future = self.pending.pop(request_id)
                    if not future.done():
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
                else:
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
        """Send request to environment and await response"""
        # auto-start receiver if not already running (with lock to prevent race)
        if self._receiver_task is None:
            async with self._start_lock:
                if self._receiver_task is None:
                    await self._start()

        effective_timeout = self.DEFAULT_REQUEST_TIMEOUT if timeout is None else timeout

        # Use request_id from Pydantic model, encode to bytes for ZMQ frame
        request_id = uuid.uuid4().hex

        # Serialize using Pydantic
        payload_bytes = cast(
            bytes,
            msgpack.packb(
                request.model_dump(mode="python"),
                default=msgpack_encoder,
                use_bin_type=True,
            ),
        )

        future: asyncio.Future[dict] = asyncio.Future()
        self.pending[request_id] = future
        await self.socket.send_multipart([request_id.encode(), payload_bytes])

        try:
            raw_response = await asyncio.wait_for(future, timeout=effective_timeout)
        except asyncio.TimeoutError:
            self.pending.pop(request_id, None)
            self.logger.error(
                f"Timed out waiting for request_id={request_id} type={request.request_type} after {effective_timeout:.1f}s (pending={len(self.pending)})"
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
        # Cancel the receiver task
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None

        # Fail any pending futures
        self._fail_all_pending("Client closed")

        # Close socket and terminate context
        self.socket.close()
        self.ctx.term()
