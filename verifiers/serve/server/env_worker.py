"""Environment worker.

Owns a single environment instance, a client cache, and three ZMQ
sockets (PULL for requests, PUSH for responses, PUSH for stats).
Receives requests from the router, runs rollouts, and pushes
responses + stats back.
"""

import asyncio
import gc
import logging
import signal
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, cast

import msgpack
import zmq
import zmq.asyncio
from pydantic import BaseModel

import verifiers as vf
from verifiers.clients import Client, resolve_client
from verifiers.serve.types import (
    BaseResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)
from verifiers.types import ClientConfig
from verifiers.utils.async_utils import EventLoopLagMonitor, EventLoopLagStats
from verifiers.utils.client_utils import resolve_client_config
from verifiers.utils.process_utils import monitor_death_pipe, set_proc_title
from verifiers.utils.serve_utils import msgpack_encoder


class EnvWorkerStats(BaseModel):
    worker_id: int
    timestamp: float
    active_tasks: int
    lag: EventLoopLagStats = EventLoopLagStats()

    def __str__(self) -> str:
        parts = []
        if self.lag.n > 0:
            parts.append(f"Lag: {self.lag}")
        return " | ".join(parts) if parts else "no lag data"


class EnvWorker:
    """Executes environment logic."""

    def __init__(
        self,
        env_id: str,
        env_args: dict[str, Any] | None = None,
        extra_env_kwargs: dict[str, Any] | None = None,
        log_level: str | None = None,
        log_dir: str | None = None,
        console_logging: bool = True,
        *,
        worker_id: int,
        worker_name: str,
        request_address: str,
        response_address: str,
        stats_address: str,
        death_pipe: Connection | None = None,
    ):
        set_proc_title(f"EnvWorker{worker_id}")
        self.death_pipe = death_pipe
        self.env_id = env_id
        self.worker_id = worker_id
        self.worker_name = worker_name

        # setup logging — each worker gets its own log file
        logger_kwargs: dict[str, Any] = {
            "console_logging": console_logging,
            "file_logging": log_dir is not None,
        }
        if log_level is not None:
            logger_kwargs["level"] = log_level
        if log_dir is not None:
            worker_log_file = EnvWorker.get_log_file(log_dir, worker_id)
            worker_log_file.parent.mkdir(parents=True, exist_ok=True)
            logger_kwargs["log_file"] = str(worker_log_file)
        vf.setup_logging(**logger_kwargs)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # setup env
        self.logger.debug(
            f"Loading environment {env_id} for worker {worker_name} ({env_args=}, {extra_env_kwargs=})"
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
        self.shutting_down: bool = False

        # stats
        self.lag_monitor = EventLoopLagMonitor()

        self.logger.info(f"Initialized worker {worker_name} on {request_address}")

    async def resolve_client(self, client_config: ClientConfig) -> Client:
        """Resolve the client instance given the request client config."""
        resolved = resolve_client_config(client_config)
        key = resolved.model_dump_json()
        if key not in self.clients:
            self.clients[key] = resolve_client(resolved)
        return self.clients[key]

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

        async def send_error_response(error: str) -> None:
            """Serialize and send an error response. Best-effort."""
            try:
                response_bytes = cast(
                    bytes,
                    msgpack.packb(
                        BaseResponse(success=False, error=error).model_dump(
                            mode="python", warnings=False
                        ),
                        default=msgpack_encoder,
                        use_bin_type=True,
                    ),
                )
                await self.response_socket.send_multipart(
                    [client_id, request_id.encode(), response_bytes]
                )
            except Exception:
                pass

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
            if self.shutting_down:
                return
            # shield prevents the still-set cancellation flag from killing the await
            await asyncio.shield(send_error_response("Request was cancelled"))
            return

        except Exception as e:
            self.logger.error(
                f"Error processing request {request_id}: {e}", exc_info=True
            )
            await send_error_response(repr(e))
            return

        try:
            response_bytes = await asyncio.to_thread(
                lambda: cast(
                    bytes,
                    msgpack.packb(
                        response.model_dump(mode="python", warnings=False),
                        default=msgpack_encoder,
                        use_bin_type=True,
                    ),
                )
            )
        except Exception as e:
            self.logger.error(
                f"Failed to serialize response for {request_id}: {e}",
                exc_info=True,
            )
            await send_error_response(f"Response serialization failed: {repr(e)}")
            return

        try:
            await self.response_socket.send_multipart(
                [client_id, request_id.encode(), response_bytes]
            )
        except zmq.ZMQError as e:
            self.logger.warning(f"Failed to send response for {request_id[:7]}: {e}")

    async def stats_loop(self, interval: float = 10.0) -> None:
        """Loop to push worker stats to the router."""
        while True:
            await asyncio.sleep(interval)

            stats = EnvWorkerStats(
                worker_id=self.worker_id,
                timestamp=time.time(),
                active_tasks=len(self.active_tasks),
                lag=EventLoopLagStats.from_monitor(self.lag_monitor),
            )

            try:
                data = msgpack.packb(
                    stats.model_dump(mode="python"),
                    default=msgpack_encoder,
                    use_bin_type=True,
                )
                await self.stats_socket.send(data, zmq.NOBLOCK)
            except zmq.Again:
                pass  # best-effort

    async def serve(self, stop_event: asyncio.Event | None = None) -> None:
        """Main worker loop."""
        self.logger.info(f"Starting worker {self.worker_name}")

        gc.collect()
        gc.freeze()
        gc.set_threshold(150_000, 10, 10)

        lag_task = asyncio.create_task(self.lag_monitor.run())
        stats_task = asyncio.create_task(self.stats_loop())

        poller = zmq.asyncio.Poller()
        poller.register(self.pull_socket, zmq.POLLIN)

        try:
            while True:
                if stop_event and stop_event.is_set():
                    break

                try:
                    events = dict(await poller.poll(timeout=100))
                    if self.pull_socket not in events:
                        continue

                    frames = await self.pull_socket.recv_multipart()
                    if len(frames) != 3:
                        self.logger.warning(
                            f"Invalid message: expected 3 frames, got {len(frames)}"
                        )
                        continue

                    raw_client_id, raw_request_id, raw_payload = frames
                    request_id = raw_request_id.decode()

                    if not raw_payload:
                        # Cancel signal
                        task = self.active_tasks.get(request_id)
                        if task is not None:
                            task.cancel()
                        continue

                    task = asyncio.create_task(
                        self.process_request(raw_client_id, raw_request_id, raw_payload)
                    )
                    self.active_tasks[request_id] = task

                    def cleanup_task(task: asyncio.Task, request_id: str) -> None:
                        if self.active_tasks.get(request_id) is task:
                            self.active_tasks.pop(request_id, None)

                    task.add_done_callback(
                        lambda t, rid=request_id: cleanup_task(t, rid)
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
        self.shutting_down = True
        if self.active_tasks:
            tasks = list(self.active_tasks.values())
            self.logger.info(
                f"Cancelling {len(tasks)} active tasks on worker {self.worker_name}"
            )
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

        self.logger.info(f"Shut down worker {self.worker_name}")

    async def run(self) -> None:
        if self.death_pipe is not None:
            monitor_death_pipe(self.death_pipe)

        from verifiers.utils.thread_utils import install_default_executor

        install_default_executor()

        stop_event = asyncio.Event()

        def signal_handler(sig, _frame):
            stop_event.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            await self.serve(stop_event=stop_event)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            await self.close()

    @staticmethod
    def get_log_file(log_dir: str, worker_id: int) -> Path:
        """Return the log file path for a given worker."""
        return Path(log_dir) / f"env_worker_{worker_id}.log"

    @classmethod
    def run_worker(cls, *args, **kwargs) -> None:
        worker = cls(*args, **kwargs)
        asyncio.run(worker.run())
