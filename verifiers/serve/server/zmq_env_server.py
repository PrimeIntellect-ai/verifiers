"""ZMQ-based environment server.

Adds a ZMQ ROUTER frontend socket and health-check responder on top of
:class:`EnvServer`.  Client requests are forwarded to the :class:`EnvRouter`
worker pool; the router's ``run()`` loop handles responses, stats, and
periodic checks in the background.
"""

import asyncio

import msgpack
import zmq
import zmq.asyncio

from verifiers.serve.server.env_server import EnvServer
from verifiers.utils.worker_utils import derive_health_address

# Pre-serialized health response — avoids repeated packing on every ping.
_HEALTH_RESPONSE = msgpack.packb({"success": True, "error": None}, use_bin_type=True)


class ZMQEnvServer(EnvServer):
    """ZMQ ROUTER frontend + EnvRouter worker pool."""

    def __init__(self, *args, address: str = "tcp://127.0.0.1:5000", **kwargs):
        super().__init__(*args, **kwargs)
        self.address = address
        self.health_address = derive_health_address(address)

        # Client-facing ROUTER socket
        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(self.address)

        # Health check REP socket (in-process, no subprocess needed)
        self.health_socket = self.ctx.socket(zmq.REP)
        self.health_socket.setsockopt(zmq.LINGER, 0)
        self.health_socket.bind(self.health_address)

    async def send_response(
        self, client_id: bytes, request_id: bytes, response_bytes: bytes
    ) -> None:
        """Forward a worker response to the client via the ROUTER socket."""
        try:
            await self.frontend.send_multipart([client_id, request_id, response_bytes])
        except zmq.ZMQError as e:
            self.logger.warning(f"Failed to forward response: {e}")

    async def serve(self, stop_event: asyncio.Event | None = None) -> None:
        self.logger.info(f"ZMQEnvServer started on {self.address}")
        self.logger.info(f"Health responder on {self.health_address}")

        stop = stop_event or asyncio.Event()

        # Start router background loop (drains responses, stats, periodic checks)
        router_task = asyncio.create_task(
            self.router.run(on_response=self.send_response, stop_event=stop)
        )

        # This loop only handles client-facing concerns:
        # incoming requests and health checks.
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        poller.register(self.health_socket, zmq.POLLIN)

        try:
            while not stop.is_set():
                events = dict(await poller.poll(timeout=100))

                # ── health checks ─────────────────────────────────
                if self.health_socket in events:
                    await self.health_socket.recv()
                    await self.health_socket.send(_HEALTH_RESPONSE)

                # ── client requests ───────────────────────────────
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

        except asyncio.CancelledError:
            pass
        finally:
            poller.unregister(self.frontend)
            poller.unregister(self.health_socket)
            router_task.cancel()
            await asyncio.gather(router_task, return_exceptions=True)

    async def close(self) -> None:
        self.frontend.close()
        self.health_socket.close()
        self.ctx.term()

        self.logger.info("ZMQEnvServer shut down")
