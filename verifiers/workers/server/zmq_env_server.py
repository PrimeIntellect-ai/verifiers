import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import msgpack
import zmq
import zmq.asyncio

from verifiers.utils.async_utils import EventLoopLagMonitor
from verifiers.utils.thread_utils import get_or_create_thread_loop
from verifiers.utils.worker_utils import msgpack_encoder
from verifiers.workers.server.env_server import EnvServer
from verifiers.workers.types import (
    BaseResponse,
    HealthRequest,
    RunGroupRequest,
    RunRolloutRequest,
)


class ZMQEnvServer(EnvServer):
    """ZMQ-based environment server."""

    def __init__(
        self,
        *args,
        address: str = "tcp://127.0.0.1:5000",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.address = address

        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(self.address)

        # Dedicated thread for health probes so they always respond
        # even when the main event loop is saturated with rollout work
        self._health_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="env-health"
        )

    async def run(self, stop_event: asyncio.Event | None = None):
        self.logger.info(f"{self.__class__.__name__} started on {self.address}")

        # Start event loop lag monitor
        lag_monitor = EventLoopLagMonitor(logger=self.logger)
        lag_monitor_task = lag_monitor.run_in_background(log_interval=30.0)

        # Use a poller to check for incoming data instead of asyncio.wait_for.
        # asyncio.wait_for wraps recv_multipart in a Task and cancels it on
        # timeout. There is a race in CPython's Task.__step where the recv
        # completes (consuming data from the ZMQ buffer) but _must_cancel is
        # already set, so the result is silently discarded — the message is
        # gone forever. A poller is non-destructive: it only checks socket
        # readability without consuming any data.
        poller = zmq.asyncio.Poller()
        poller.register(self.socket, zmq.POLLIN)

        try:
            while True:
                if stop_event and stop_event.is_set():
                    self.logger.info("Stop event received, shutting down gracefully")
                    break

                try:
                    events = dict(await poller.poll(timeout=1000))
                    if self.socket not in events:
                        continue

                    frames = await self.socket.recv_multipart()

                    if len(frames) != 3:
                        self.logger.warning(
                            f"Invalid message: expected 3 frames, got {len(frames)}"
                        )
                        continue

                    client_id, request_id, payload_bytes = frames

                    # Process in background, tracking the task for cleanup
                    task = asyncio.create_task(
                        self._process_request(client_id, request_id, payload_bytes)
                    )
                    self.pending_tasks.add(task)
                    task.add_done_callback(self.pending_tasks.discard)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in server loop: {e}", exc_info=True)
        finally:
            poller.unregister(self.socket)
            lag_monitor_task.cancel()
            try:
                await lag_monitor_task
            except asyncio.CancelledError:
                pass

    async def close(self):
        # cancel and await all pending tasks
        if self.pending_tasks:
            self.logger.info(f"Cancelling {len(self.pending_tasks)} pending tasks")
            for task in self.pending_tasks:
                task.cancel()
            await asyncio.gather(*self.pending_tasks, return_exceptions=True)
            self.pending_tasks.clear()

        await self._close_cached_clients()

        self._health_executor.shutdown(wait=False)

        self.socket.close()
        self.ctx.term()
        self.logger.info("Environment server shut down")

    async def _run_in_executor(
        self, executor: ThreadPoolExecutor, coro
    ) -> BaseResponse:
        """Run an async handler in a thread pool thread with a thread-local event loop."""
        loop = asyncio.get_running_loop()

        def run():
            thread_loop = get_or_create_thread_loop()
            return thread_loop.run_until_complete(coro)

        return await loop.run_in_executor(executor, run)

    async def _process_request(
        self,
        client_id: bytes,
        request_id_bytes: bytes,
        payload_bytes: bytes,
    ):
        request_id = request_id_bytes.decode()
        response: BaseResponse

        try:
            # deserialize request
            raw = msgpack.unpackb(payload_bytes, raw=False)
            request_type = raw.get("request_type")
            request_id = raw.get("request_id", request_id)

            # validate and route to handler
            if request_type == "health":
                request = HealthRequest.model_validate(raw)
                response = await self._run_in_executor(
                    self._health_executor, self._handle_health(request)
                )
            elif request_type == "run_rollout":
                request = RunRolloutRequest.model_validate(raw)
                response = await self._handle_run_rollout(request)
            elif request_type == "run_group":
                request = RunGroupRequest.model_validate(raw)
                response = await self._handle_run_group(request)
            else:
                self.logger.warning(f"Got unknown request type: {request_type}")
                response = BaseResponse(
                    success=False, error=f"Unknown request type: {request_type}"
                )

        except asyncio.CancelledError:
            return

        except Exception as e:
            self.logger.error(
                f"Error processing request {request_id}: {e}", exc_info=True
            )
            response = BaseResponse(
                success=False,
                error=repr(e),
            )

        # serialize response using Pydantic
        response_bytes = cast(
            bytes,
            msgpack.packb(
                response.model_dump(mode="python", warnings=False),
                default=msgpack_encoder,
                use_bin_type=True,
            ),
        )

        # send response: [client_id, request_id, response]
        await self.socket.send_multipart(
            [client_id, request_id.encode(), response_bytes]
        )
