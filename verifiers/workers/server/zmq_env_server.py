import asyncio
import threading
from typing import cast

import msgpack
import zmq
import zmq.asyncio

from verifiers.utils.async_utils import EventLoopLagMonitor
from verifiers.utils.logging_utils import print_time
from verifiers.utils.worker_utils import msgpack_encoder
from verifiers.workers.server.env_server import EnvServer
from verifiers.workers.types import (
    BaseResponse,
    RunGroupRequest,
    RunRolloutRequest,
)


def derive_health_address(address: str) -> str:
    """Derive health check address from main address (port + 1)."""
    prefix, port_str = address.rsplit(":", 1)
    return f"{prefix}:{int(port_str) + 1}"


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
        self.health_address = derive_health_address(address)

        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(self.address)

        # Health check runs on a dedicated thread with its own ZMQ socket,
        # completely decoupled from the main event loop so it always responds
        # even when rollout work saturates the loop.
        self._stop_health = threading.Event()
        self._health_thread: threading.Thread | None = None

        # Start event loop lag monitor
        self.lag_monitor = EventLoopLagMonitor(logger=self.logger)

    def _run_health_thread(self):
        """Blocking health check responder on a dedicated thread."""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout for clean shutdown
        sock.bind(self.health_address)
        self.logger.info(f"Health check responder started on {self.health_address}")

        health_response = msgpack.packb(
            {"success": True, "error": None}, use_bin_type=True
        )

        while not self._stop_health.is_set():
            try:
                sock.recv()  # block until request (with 1s timeout)
                sock.send(health_response)
            except zmq.Again:
                continue  # recv timeout, check stop flag
            except zmq.ZMQError:
                break

        sock.close()
        ctx.term()
        self.logger.debug("Health check responder stopped")

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        self.logger.info(f"{self.__class__.__name__} started on {self.address}")

        # Start health responder thread
        self._health_thread = threading.Thread(
            target=self._run_health_thread,
            name="health-responder",
            daemon=True,
        )
        self._health_thread.start()

        # Start statistics logger
        self.lag_monitor.run_in_background()
        log_stats_task = asyncio.create_task(self._log_stats_loop())

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
            for t in (log_stats_task,):
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    async def close(self):
        # Stop health thread
        self._stop_health.set()
        if self._health_thread is not None:
            self._health_thread.join(timeout=5)
            self._health_thread = None

        # Cancel and await all pending tasks
        if self.pending_tasks:
            self.logger.info(f"Cancelling {len(self.pending_tasks)} pending tasks")
            for task in self.pending_tasks:
                task.cancel()
            await asyncio.gather(*self.pending_tasks, return_exceptions=True)
            self.pending_tasks.clear()

        await self._close_cached_clients()

        self.socket.close()
        self.ctx.term()
        self.logger.info("Environment server shut down")

    async def _log_stats_loop(self, interval: float = 30.0):
        """Periodically log statistics."""
        while True:
            await asyncio.sleep(interval)
            lags = sorted(self.lag_monitor.lags)
            self.lag_monitor.reset()
            mean_lag = sum(lags) / len(lags)
            max_lag = lags[-1]
            p99_lag = lags[int(len(lags) * 0.99)]
            pending = len(self.pending_tasks)

            self.logger.info(
                f"Pending tasks: {pending}, Event loop lag: mean={print_time(mean_lag)}, p99={print_time(p99_lag)}, max={print_time(max_lag)}"
            )

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

            # Health requests are handled by the dedicated health thread,
            # so they should not arrive here. Handle just in case.
            if request_type == "run_rollout":
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
