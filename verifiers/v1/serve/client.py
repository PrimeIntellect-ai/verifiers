"""ZMQ client for the env server.

A DEALER socket + msgpack, with a single receive loop matching responses to
per-request futures by `request_id`. Speaks the typed pydantic request/response
models (`serve/types.py`) end-to-end: a request is `model_dump`ed onto the wire
and the reply is `model_validate`d back — `Trace`s come back typed as
`Trace[WireTask]` (non-strict task, so env fields survive without importing the
env). Health is just another request (no dedicated probe thread).
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
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.errors import ProgramError
from verifiers.v1.retries import RolloutRetryConfig
from verifiers.v1.serve.types import (
    BaseRequest,
    BaseResponse,
    HealthRequest,
    HealthResponse,
    InfoRequest,
    InfoResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)
from verifiers.v1.task import WireTask
from verifiers.v1.trace import Trace
from verifiers.v1.types import SamplingConfig

logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseResponse)


class EnvClient:
    def __init__(
        self,
        address: str = "tcp://127.0.0.1:5000",
        retry: RolloutRetryConfig | None = None,
    ) -> None:
        self.address = address
        self.retry = retry or RolloutRetryConfig()
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.connect(address)
        self._pending: dict[str, asyncio.Future] = {}
        self._receiver: asyncio.Task | None = None

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
                future.set_result(msgpack.unpackb(data, raw=False))

    async def _request(
        self,
        request: BaseRequest,
        response_type: type[ResponseT],
        timeout: float | None = None,
    ) -> ResponseT:
        """Send a typed request and validate the reply into `response_type`. A
        `timeout` is only used for health polling — rollouts run untimed."""
        self._ensure_receiver()
        max_retries = (
            self.retry.max_retries
            if request.method in {"run_rollout", "run_group"}
            and self.retry.allows("ProgramError")
            else 0
        )

        retrying = AsyncRetrying(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential_jitter(initial=0.5, max=30),
            retry=retry_if_exception_type(ProgramError),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                request_id = uuid.uuid4().hex
                future: asyncio.Future = asyncio.get_running_loop().create_future()
                self._pending[request_id] = future
                payload = msgpack.packb(
                    request.model_dump(mode="json"), use_bin_type=True
                )
                await self.socket.send_multipart(
                    [request_id.encode(), request.method.encode(), payload]
                )
                try:
                    raw = await asyncio.wait_for(future, timeout)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self._pending.pop(request_id, None)
                    raise
                response = response_type.model_validate(raw)
                if not response.success:
                    message = response.error or "env server request failed"
                    if response.error_type == "ProgramError":
                        raise ProgramError(message)
                    raise RuntimeError(message)
                return response
            outcome = attempt.retry_state.outcome
            error = outcome.exception() if outcome is not None else None
            if (
                isinstance(error, ProgramError)
                and attempt.retry_state.attempt_number <= max_retries
            ):
                logger.warning(
                    "retrying env-server %s for task %s (retry %d/%d) "
                    "after worker failure: %s",
                    request.method,
                    getattr(request, "task_idx", "?"),
                    attempt.retry_state.attempt_number,
                    max_retries,
                    error,
                )
        raise RuntimeError("env server retry loop exited without a response")

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
        """Return the taskset `num_tasks` + `requires_group_scoring`."""
        return await self._request(InfoRequest(), InfoResponse)

    async def run_rollout(
        self, task_idx: int, client: ClientConfig, model: str, sampling: SamplingConfig
    ) -> Trace[WireTask]:
        """Run one rollout for `task_idx`; return a typed `Trace[WireTask]`."""
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
    ) -> list[Trace[WireTask]]:
        """Run `n` rollouts for `task_idx` as a scored group; return typed `Trace[WireTask]`s."""
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
