"""ZMQ-based environment server.

Adds a ZMQ ROUTER frontend socket and health-check responder on top of
:class:`EnvServer`.  The poll loop bridges client frames to/from the
:class:`EnvRouter` worker pool.
"""

import asyncio
import multiprocessing as mp
import time

import zmq
import zmq.asyncio

from verifiers.utils.process_utils import terminate_process
from verifiers.utils.worker_utils import derive_health_address, run_health_responder
from verifiers.workers.server.env_server import EnvServer


class ZMQEnvServer(EnvServer):
    """ZMQ ROUTER frontend + EnvRouter worker pool."""

    def __init__(self, *args, address: str = "tcp://127.0.0.1:5000", **kwargs):
        super().__init__(*args, **kwargs)
        self.address = address
        self.health_address = derive_health_address(address)

        # ── ZMQ frontend (client-facing) ─────────────────────────
        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(self.address)

        # ── health check (separate process) ──────────────────────
        self.stop_health = mp.Event()
        self.health_process: mp.Process | None = None

    async def serve(self, stop_event: asyncio.Event | None = None) -> None:
        self.logger.info(f"ZMQEnvServer started on {self.address}")

        # Health responder
        self.health_process = mp.Process(
            target=run_health_responder,
            args=(self.health_address, self.stop_health),
            name="health-responder",
            daemon=True,
        )
        self.health_process.start()
        self.logger.info(f"Health responder on {self.health_address}")

        # Start worker pool
        self.router.start_workers()

        # Register all sockets in a single poller
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        poller.register(self.router.response_pull, zmq.POLLIN)
        poller.register(self.router.stats_pull, zmq.POLLIN)

        last_stats_log = time.time()
        last_heartbeat_check = time.time()

        try:
            while True:
                if stop_event and stop_event.is_set():
                    self.logger.info("Stop event received, shutting down")
                    break

                events = dict(await poller.poll(timeout=100))

                # ── client requests ──────────────────────────────
                if self.frontend in events:
                    frames = await self.frontend.recv_multipart()
                    if len(frames) != 3:
                        self.logger.warning(
                            f"Invalid message: expected 3 frames, got {len(frames)}"
                        )
                    else:
                        client_id, request_id, payload = frames
                        if not payload:
                            await self.router.forward_cancel(request_id, client_id)
                        else:
                            try:
                                await self.router.dispatch(
                                    client_id, request_id, payload
                                )
                            except zmq.ZMQError as e:
                                self.logger.error(f"Failed to dispatch request: {e}")

                # ── worker responses → client ────────────────────
                if self.router.response_pull in events:
                    while True:
                        try:
                            frames = await self.router.response_pull.recv_multipart(
                                zmq.NOBLOCK
                            )
                        except zmq.Again:
                            break
                        if len(frames) != 3:
                            continue
                        client_id, request_id, response_bytes = frames
                        try:
                            await self.frontend.send_multipart(
                                [client_id, request_id, response_bytes]
                            )
                        except zmq.ZMQError as e:
                            self.logger.warning(f"Failed to forward response: {e}")
                        self.router.complete_request(request_id)

                # ── worker stats ─────────────────────────────────
                if self.router.stats_pull in events:
                    while True:
                        try:
                            data = await self.router.stats_pull.recv(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        self.router.handle_stats_message(data)

                # ── periodic checks ──────────────────────────────
                now = time.time()
                if now - last_stats_log >= self.router.stats_log_interval:
                    self.router.log_aggregate_stats()
                    last_stats_log = now
                if now - last_heartbeat_check >= 5.0:
                    await self.router.check_workers()
                    last_heartbeat_check = now

        except asyncio.CancelledError:
            pass
        finally:
            poller.unregister(self.frontend)
            poller.unregister(self.router.response_pull)
            poller.unregister(self.router.stats_pull)

    async def close(self) -> None:
        # Health process
        self.stop_health.set()
        terminate_process(self.health_process)
        self.health_process = None

        # Frontend socket
        self.frontend.close()
        self.ctx.term()

        self.logger.info("ZMQEnvServer shut down")
