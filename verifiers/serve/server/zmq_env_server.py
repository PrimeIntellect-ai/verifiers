"""ZMQ-based environment server.

Adds a ZMQ ROUTER frontend socket on top of :class:`EnvServer`.  Client
requests are forwarded to the :class:`EnvRouter` worker pool; the router's
``run()`` loop handles responses, stats, and periodic checks in the background.

Health checks are handled inline on the ROUTER socket — clients send a
``b"ping"`` payload and receive a pre-serialized health response back on the
same connection.  No separate port is needed.
"""

import asyncio

import msgpack
import zmq
import zmq.asyncio

from verifiers.serve.server.env_server import EnvServer

# Pre-serialized health response — avoids repeated packing on every ping.
_HEALTH_RESPONSE = msgpack.packb({"success": True, "error": None}, use_bin_type=True)

# Sentinel payload used by health-check probes.
_HEALTH_PING = b"ping"


class ZMQEnvServer(EnvServer):
    """ZMQ ROUTER frontend + EnvRouter worker pool."""

    def __init__(self, *args, address: str = "tcp://127.0.0.1:5000", **kwargs):
        super().__init__(*args, **kwargs)
        self.address = address

        # Client-facing ROUTER socket (also serves health checks)
        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(self.address)

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

        stop = stop_event or asyncio.Event()

        # Start router background loop (drains responses, stats, periodic checks)
        router_task = asyncio.create_task(
            self.router.run(on_response=self.send_response, stop_event=stop)
        )

        # This loop handles client-facing concerns:
        # incoming requests, cancellations, and health pings.
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)

        try:
            while not stop.is_set():
                if router_task.done():
                    exc = router_task.exception()
                    if exc is not None:
                        self.logger.error(f"Router task died: {exc}")
                        raise RuntimeError("Router task failed unexpectedly") from exc
                    else:
                        self.logger.info("Router task exited normally")
                        break

                events = dict(await poller.poll(timeout=100))

                if self.frontend in events:
                    frames = await self.frontend.recv_multipart()
                    if len(frames) != 3:
                        self.logger.warning(
                            f"Invalid message: expected 3 frames, got {len(frames)}"
                        )
                    else:
                        client_id, request_id, payload = frames
                        if payload == _HEALTH_PING:
                            # Health check — respond immediately
                            try:
                                await self.frontend.send_multipart(
                                    [client_id, request_id, _HEALTH_RESPONSE]
                                )
                            except zmq.ZMQError:
                                pass  # peer disconnected between ping and pong
                        elif not payload:
                            await self.router.forward_cancel(request_id, client_id)
                        else:
                            try:
                                await self.router.dispatch_request(
                                    client_id, request_id, payload
                                )
                            except zmq.ZMQError as e:
                                self.logger.error(f"Failed to dispatch request: {e}")

        except asyncio.CancelledError:
            pass
        finally:
            poller.unregister(self.frontend)
            router_task.cancel()
            await asyncio.gather(router_task, return_exceptions=True)

    async def close(self) -> None:
        self.frontend.close()
        self.ctx.term()

        self.logger.info("ZMQEnvServer shut down")
