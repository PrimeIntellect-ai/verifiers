"""ZMQ client for the env server.

A DEALER socket + msgpack, with a single receive loop matching responses to
per-request futures by `request_id`. Speaks the typed pydantic request/response
models (`serve/types.py`) end-to-end. Native replies load as `AgentGraph`s with wire-typed
tasks and traces, preserving environment-specific fields without importing their packages.
Health is just another request.
"""

import asyncio
import contextlib
import logging
import time
import uuid
from typing import TypeVar

import msgpack
import zmq
import zmq.asyncio

from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.serve.types import (
    BaseRequest,
    BaseResponse,
    HealthRequest,
    HealthResponse,
    InfoRequest,
    InfoResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRequest,
    RunResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)
from verifiers.v1.task import WireTaskData
from verifiers.v1.topology import AgentGraph
from verifiers.v1.trace import Trace
from verifiers.v1.types import SamplingConfig

logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseResponse)


class EnvClient:
    def __init__(self, address: str = "tcp://127.0.0.1:5000") -> None:
        self.address = address
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.connect(address)
        self._pending: dict[str, asyncio.Future[bytes]] = {}
        self._receiver: asyncio.Task | None = None
        self._decode_slots = asyncio.BoundedSemaphore(1)

    def _ensure_receiver(self) -> None:
        if self._receiver is None:
            self._receiver = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        while True:
            try:
                request_id_bytes, data = await self.socket.recv_multipart()
            except asyncio.CancelledError:
                break
            future = self._pending.pop(request_id_bytes.decode(), None)
            if future is not None and not future.done():
                future.set_result(data)

    async def _request(
        self,
        request: BaseRequest,
        response_type: type[ResponseT],
        timeout: float | None = None,
    ) -> ResponseT:
        """Send a typed request and validate the reply into `response_type`. A
        `timeout` is only used for health polling — graph requests run untimed."""
        self._ensure_receiver()
        request_id = uuid.uuid4().hex
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = msgpack.packb(request.model_dump(mode="json"), use_bin_type=True)
        await self.socket.send_multipart(
            [request_id.encode(), request.method.encode(), payload]
        )
        try:
            data = await asyncio.wait_for(future, timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._pending.pop(request_id, None)
            raise
        if response_type in (HealthResponse, InfoResponse):
            response = response_type.model_validate(msgpack.unpackb(data, raw=False))
        else:
            # Keep large trace replies compact on the loop and expand only one at a time.
            await self._decode_slots.acquire()
            decoding = asyncio.create_task(
                asyncio.to_thread(
                    lambda: response_type.model_validate(
                        msgpack.unpackb(data, raw=False)
                    )
                )
            )
            # Hold the slot until the worker finishes so cancellation cannot overlap decodes.
            decoding.add_done_callback(lambda _: self._decode_slots.release())
            try:
                response = await asyncio.shield(decoding)
            except asyncio.CancelledError:
                decoding.add_done_callback(
                    lambda task: None if task.cancelled() else task.exception()
                )
                raise
        if not response.success:
            raise RuntimeError(response.error or "env server request failed")
        return response

    async def health(self, timeout: float = 2.0) -> bool:
        try:
            return (
                await self._request(HealthRequest(), HealthResponse, timeout=timeout)
            ).success
        except asyncio.TimeoutError:
            return False

    async def wait_for_server_startup(
        self, timeout: float = 120.0, interval: float = 1.0
    ) -> None:
        """Poll `health` until the server answers or `timeout` elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.health(timeout=min(interval, 2.0)):
                return
            await asyncio.sleep(interval)
        raise TimeoutError(
            f"env server at {self.address} did not become healthy in {timeout}s"
        )

    async def info(self) -> InfoResponse:
        """Return the server's ordered seed-task ids."""
        return await self._request(InfoRequest(), InfoResponse)

    async def run(
        self, task_idx: int, client: ClientConfig, model: str, sampling: SamplingConfig
    ) -> AgentGraph:
        """Run one independent native v1 invocation and return its agent graph."""
        response = await self._request(
            RunRequest(
                task_idx=task_idx, client=client, model=model, sampling=sampling
            ),
            RunResponse,
        )
        assert response.graph is not None
        return response.graph

    async def run_rollout(
        self, task_idx: int, client: ClientConfig, model: str, sampling: SamplingConfig
    ) -> Trace[WireTaskData]:
        """Run one legacy v0 rollout."""
        response = await self._request(
            RunRolloutRequest(
                task_idx=task_idx, client=client, model=model, sampling=sampling
            ),
            RunRolloutResponse,
        )
        return response.trace

    async def run_group(
        self,
        task_idx: int,
        n: int,
        client: ClientConfig,
        model: str,
        sampling: SamplingConfig,
    ) -> list[Trace[WireTaskData]]:
        """Run one legacy v0 rollout group."""
        response = await self._request(
            RunGroupRequest(
                task_idx=task_idx, n=n, client=client, model=model, sampling=sampling
            ),
            RunGroupResponse,
        )
        return response.traces

    async def close(self) -> None:
        if self._receiver is not None:
            self._receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receiver
            self._receiver = None
        self.socket.close()
        self.ctx.term()
