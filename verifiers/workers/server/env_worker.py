"""Self-contained environment worker process.

Receives requests from the router via a PULL socket, runs rollouts against
a local environment instance, and pushes responses + stats back via PUSH
sockets.
"""

import asyncio
import gc
import logging
import signal
import sys
import time
from typing import Any, cast

import msgpack
import numpy as np
import zmq
import zmq.asyncio

import verifiers as vf
from verifiers.clients import Client, resolve_client
from verifiers.types import ClientConfig
from verifiers.utils.async_utils import EventLoopLagMonitor
from verifiers.utils.client_utils import resolve_client_config
from verifiers.utils.logging_utils import print_time
from verifiers.utils.worker_utils import msgpack_encoder
from verifiers.workers.types import (
    BaseResponse,
    LagStats,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
    WorkerStats,
)


def _request_parent_death_signal() -> None:
    """Ask Linux to SIGTERM us when the parent dies."""
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)
    except Exception:
        pass


class EnvWorker:
    """Worker process that receives requests from a router via IPC PUSH/PULL.

    Owns a single environment instance, a client cache, and three ZMQ
    sockets (PULL for requests, PUSH for responses, PUSH for stats).
    """

    def __init__(
        self,
        env_id: str,
        env_args: dict[str, Any] | None = None,
        extra_env_kwargs: dict[str, Any] | None = None,
        log_level: str | None = None,
        log_file: str | None = None,
        log_file_level: str | None = None,
        *,
        worker_id: int,
        worker_name: str,
        request_address: str,
        response_address: str,
        stats_address: str,
    ):
        self.env_id = env_id
        self.worker_id = worker_id
        self.worker_name = worker_name

        # setup logging
        logger_kwargs: dict[str, Any] = {}
        if log_level is not None:
            logger_kwargs["level"] = log_level
        if log_file is not None:
            from pathlib import Path

            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            logger_kwargs["log_file"] = log_file
            logger_kwargs["log_file_level"] = log_file_level
        vf.setup_logging(**logger_kwargs)

        self.logger = logging.LoggerAdapter(
            logging.getLogger(f"{__name__}.{self.__class__.__name__}"),
        )
        self.logger.process = lambda msg, kwargs: (
            f"[{self.worker_name}] {msg}",
            kwargs,
        )

        # setup env
        self.logger.info(
            f"Loading environment {env_id} ({env_args=}, {extra_env_kwargs=})"
        )
        self.env = vf.load_environment(env_id, **(env_args or {}))
        if extra_env_kwargs:
            self.env.set_kwargs(**extra_env_kwargs)

        # setup zmq sockets
        self.ctx = zmq.asyncio.Context()

        self.pull_socket = self.ctx.socket(zmq.PULL)
        self.pull_socket.setsockopt(zmq.RCVHWM, 0)
        self.pull_socket.setsockopt(zmq.LINGER, 0)
        self.pull_socket.bind(request_address)

        self.response_socket = self.ctx.socket(zmq.PUSH)
        self.response_socket.setsockopt(zmq.SNDHWM, 0)
        self.response_socket.setsockopt(zmq.LINGER, 5000)
        self.response_socket.connect(response_address)

        self.stats_socket = self.ctx.socket(zmq.PUSH)
        self.stats_socket.setsockopt(zmq.SNDHWM, 100)
        self.stats_socket.setsockopt(zmq.LINGER, 0)
        self.stats_socket.connect(stats_address)

        # state tracking
        self.clients: dict[str, Client] = {}
        self.active_tasks: dict[str, asyncio.Task] = {}

        # stats
        self.lag_monitor = EventLoopLagMonitor()

        self.logger.info(f"Initialized EnvWorker {env_id} on {request_address}")

    # ── client resolution ────────────────────────────────────────

    async def resolve_client(self, client_config: ClientConfig) -> Client:
        resolved = resolve_client_config(client_config)
        key = resolved.model_dump_json()
        if key not in self.clients:
            self.clients[key] = resolve_client(resolved)
        return self.clients[key]

    # ── request handling ─────────────────────────────────────────

    async def handle_run_rollout(
        self, request: RunRolloutRequest
    ) -> RunRolloutResponse:
        client = await self.resolve_client(request.client_config)
        output = await self.env.run_rollout(
            input=request.input,
            client=client,
            model=request.model,
            sampling_args=request.sampling_args,
            max_retries=request.max_retries,
            state_columns=request.state_columns,
        )
        return RunRolloutResponse(output=output)

    async def handle_run_group(self, request: RunGroupRequest) -> RunGroupResponse:
        client = await self.resolve_client(request.client_config)
        outputs = await self.env.run_group(
            group_inputs=request.group_inputs,
            client=client,
            model=request.model,
            sampling_args=request.sampling_args,
            max_retries=request.max_retries,
            state_columns=request.state_columns,
        )
        return RunGroupResponse(outputs=outputs)

    async def process_request(
        self,
        client_id: bytes,
        request_id_bytes: bytes,
        payload_bytes: bytes,
    ) -> None:
        request_id = request_id_bytes.decode()
        response: BaseResponse

        try:
            raw = msgpack.unpackb(payload_bytes, raw=False)
            request_type = raw.get("request_type")
            request_id = raw.get("request_id", request_id)

            if request_type == "run_rollout":
                request = RunRolloutRequest.model_validate(raw)
                response = await self.handle_run_rollout(request)
            elif request_type == "run_group":
                request = RunGroupRequest.model_validate(raw)
                response = await self.handle_run_group(request)
            else:
                self.logger.warning(f"Unknown request type: {request_type}")
                response = BaseResponse(
                    success=False, error=f"Unknown request type: {request_type}"
                )

        except asyncio.CancelledError:
            response = BaseResponse(success=False, error="Request was cancelled")

        except Exception as e:
            self.logger.error(
                f"Error processing request {request_id}: {e}", exc_info=True
            )
            response = BaseResponse(success=False, error=repr(e))

        try:
            await asyncio.shield(
                self._serialize_and_send(client_id, request_id, response)
            )
        except asyncio.CancelledError:
            pass

    async def _serialize_and_send(
        self,
        client_id: bytes,
        request_id: str,
        response: BaseResponse,
    ) -> None:
        def _serialize() -> bytes:
            return cast(
                bytes,
                msgpack.packb(
                    response.model_dump(mode="python", warnings=False),
                    default=msgpack_encoder,
                    use_bin_type=True,
                ),
            )

        try:
            response_bytes = await asyncio.to_thread(_serialize)
        except Exception as e:
            self.logger.error(
                f"Failed to serialize response for {request_id}: {e}",
                exc_info=True,
            )
            response_bytes = cast(
                bytes,
                msgpack.packb(
                    BaseResponse(
                        success=False,
                        error=f"Response serialization failed: {repr(e)}",
                    ).model_dump(mode="python", warnings=False),
                    default=msgpack_encoder,
                    use_bin_type=True,
                ),
            )

        try:
            await self.response_socket.send_multipart(
                [client_id, request_id.encode(), response_bytes]
            )
        except zmq.ZMQError as e:
            self.logger.warning(f"Failed to send response for {request_id[:7]}: {e}")

    # ── stats push ───────────────────────────────────────────────

    async def _push_stats_loop(self, interval: float = 1.0) -> None:
        while True:
            await asyncio.sleep(interval)
            active = len(self.active_tasks)

            lags = self.lag_monitor.lags
            n = len(lags)
            lag = LagStats(n=n)
            if n > 0:
                arr = np.array(lags)
                lag = LagStats(
                    min=float(arr.min()),
                    mean=float(arr.mean()),
                    median=float(np.median(arr)),
                    p90=float(np.percentile(arr, 90)),
                    p99=float(np.percentile(arr, 99)),
                    max=float(arr.max()),
                    n=n,
                )

            stats = WorkerStats(
                worker_id=self.worker_id,
                timestamp=time.time(),
                active_tasks=active,
                lag=lag,
            )

            lag_str = ""
            if lag.n > 0:
                lag_str = (
                    f", Lag: mean={print_time(lag.mean)} "
                    f"p99={print_time(lag.p99)} max={print_time(lag.max)} (n={lag.n})"
                )
            self.logger.info(f"Active tasks: {active}{lag_str}")

            try:
                data = msgpack.packb(
                    stats.model_dump(mode="python"),
                    default=msgpack_encoder,
                    use_bin_type=True,
                )
                await self.stats_socket.send(data, zmq.NOBLOCK)
            except zmq.Again:
                pass  # best-effort

    # ── task bookkeeping ─────────────────────────────────────────

    def _cleanup_task(self, frame_request_id: str, task: asyncio.Task) -> None:
        if self.active_tasks.get(frame_request_id) is task:
            self.active_tasks.pop(frame_request_id, None)

    # ── main serve loop ──────────────────────────────────────────

    async def serve(self, stop_event: asyncio.Event | None = None) -> None:
        self.logger.info(f"Worker-{self.worker_id} serving")

        gc.collect()
        gc.freeze()
        gc.set_threshold(150_000, 10, 10)

        lag_task = asyncio.create_task(self.lag_monitor.run())
        stats_task = asyncio.create_task(self._push_stats_loop())

        poller = zmq.asyncio.Poller()
        poller.register(self.pull_socket, zmq.POLLIN)

        try:
            while True:
                if stop_event and stop_event.is_set():
                    break

                try:
                    events = dict(await poller.poll(timeout=1000))
                    if self.pull_socket not in events:
                        continue

                    frames = await self.pull_socket.recv_multipart()
                    if len(frames) != 3:
                        self.logger.warning(
                            f"Invalid message: expected 3 frames, got {len(frames)}"
                        )
                        continue

                    client_id, request_id, payload_bytes = frames

                    if not payload_bytes:
                        # Cancel signal
                        task = self.active_tasks.get(request_id.decode())
                        if task is not None:
                            task.cancel()
                        continue

                    frame_request_id = request_id.decode()
                    task = asyncio.create_task(
                        self.process_request(client_id, request_id, payload_bytes)
                    )
                    self.active_tasks[frame_request_id] = task
                    task.add_done_callback(
                        lambda t, rid=frame_request_id: self._cleanup_task(rid, t)
                    )

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in serve loop: {e}", exc_info=True)
        finally:
            poller.unregister(self.pull_socket)
            for t in (stats_task, lag_task):
                t.cancel()
            await asyncio.gather(stats_task, lag_task, return_exceptions=True)

    async def close(self) -> None:
        if self.active_tasks:
            tasks = list(self.active_tasks.values())
            self.logger.info(f"Cancelling {len(tasks)} active tasks")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.active_tasks.clear()

        for client in self.clients.values():
            await client.close()
        self.clients.clear()

        await self.env._teardown()

        self.pull_socket.close()
        self.response_socket.close()
        self.stats_socket.close()
        self.ctx.term()

        self.logger.info(f"Worker-{self.worker_id} shut down")

    async def run(self) -> None:
        _request_parent_death_signal()

        from verifiers.utils.thread_utils import install_default_executor

        install_default_executor()

        stop_event = asyncio.Event()

        def signal_handler(sig, _frame):
            stop_event.set()
            if sig == signal.SIGTERM:
                raise SystemExit(143)
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            await self.serve(stop_event=stop_event)
        finally:
            await self.close()

    @classmethod
    def run_worker(cls, *args, **kwargs) -> None:
        """Entry point for spawning in a subprocess."""
        worker = cls(*args, **kwargs)
        asyncio.run(worker.run())
