import asyncio
import hashlib
import json
from typing import cast

import msgpack
import zmq
import zmq.asyncio
from openai import AsyncOpenAI

from verifiers.types import ClientConfig, RolloutInput, SamplingArgs
from verifiers.utils.client_utils import setup_client
from verifiers.workers.server.env_server import EnvServer
from verifiers.workers.utils import msgpack_encoder


class ZMQEnvServer(EnvServer):
    """Server that exposes an environment via ZMQ."""

    def __init__(self, *args, address: str = "tcp://127.0.0.1:5000", **kwargs):
        super().__init__(*args, **kwargs)
        self.address = address

        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(address)

        self.clients: dict[str, AsyncOpenAI] = {}

    async def run(self, stop_event: asyncio.Event | None = None):
        self.logger.info(f"Environment server listening on {self.address}")

        # Create a task to wait for stop signal
        stop_task = asyncio.create_task(stop_event.wait()) if stop_event else None

        try:
            while True:
                # exit gracefully on stop signal
                if stop_event and stop_event.is_set():
                    self.logger.info("Stop event received, shutting down gracefully")
                    break

                try:
                    # receive with timeout to periodically check stop_event
                    frames = await asyncio.wait_for(
                        self.socket.recv_multipart(),
                        timeout=1.0 if stop_event else None,
                    )

                    if len(frames) != 3:
                        self.logger.warning(
                            f"Invalid message: expected 3 frames, got {len(frames)}"
                        )
                        continue

                    client_id, request_id, payload_bytes = frames

                    # Process in background with concurrency limit
                    asyncio.create_task(
                        self._process_request(client_id, request_id, payload_bytes)
                    )

                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in server loop: {e}", exc_info=True)
        finally:
            if stop_task and not stop_task.done():
                stop_task.cancel()

    async def close(self):
        # TODO: close clients

        # close zmq socket
        self.socket.close()
        self.ctx.term()
        self.logger.info("Environment server shut down")

    async def _process_request(
        self,
        client_id: bytes,
        request_id: bytes,
        payload_bytes: bytes,
    ):
        async with self.semaphore:
            try:
                # deserialize request
                request = msgpack.unpackb(payload_bytes, raw=False)
                action = request.get("action")

                # route to handler
                if action == "health":
                    response = await self._handle_health(request)
                elif action == "run_rollout":
                    response = await self._handle_run_rollout(request)
                else:
                    response = {"status": "error", "error": f"Unknown action: {action}"}

                # serialize response
                response_bytes = cast(
                    bytes,
                    msgpack.packb(response, default=msgpack_encoder, use_bin_type=True),
                )

                # send response: [client_id, request_id, response]
                await self.socket.send_multipart(
                    [client_id, request_id, response_bytes]
                )

                self.logger.info(
                    f"Sent {action} response ({len(response_bytes)} bytes)"
                )

            except Exception as e:
                self.logger.error(f"Error processing request: {e}", exc_info=True)

                # send error response
                error_response = {"status": "error", "error": str(e)}
                error_bytes = msgpack.packb(
                    error_response, default=msgpack_encoder, use_bin_type=True
                )
                await self.socket.send_multipart([client_id, request_id, error_bytes])

    async def _handle_health(self, request: dict) -> dict:
        return {"status": "ok"}

    async def _handle_run_rollout(self, request: dict) -> dict:
        client = self._get_client(
            ClientConfig.model_construct(**request.get("client_config", {}))
        )
        state = await self.env.run_rollout(
            input=RolloutInput(**request.get("input", {})),
            client=client,
            model=request.get("model", ""),
            gen_sampling_args=SamplingArgs(**request.get("sampling_args", {})),
            gen_sem=self.semaphore,
        )
        # Remove non-serializable fields before sending over the wire
        state.pop("client", None)
        return state

    def _get_client(self, client_config: ClientConfig) -> AsyncOpenAI:
        config_hash = hashlib.sha256(
            json.dumps(client_config.model_dump()).encode()
        ).hexdigest()
        client = self.clients.get(config_hash)
        if client is None:
            client = setup_client(client_config)
            self.clients[config_hash] = client
        return client


async def main():
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="ZMQ Environment Server")
    parser.add_argument("env_id", type=str, help="Environment ID to load")
    parser.add_argument(
        "--env-args", type=json.loads, default={}, help="Environment args as JSON"
    )
    parser.add_argument(
        "--address", default="tcp://127.0.0.1:5555", help="ZMQ bind address"
    )

    args = parser.parse_args()

    # initialize server
    server = ZMQEnvServer(
        env_id=args.env_id,
        env_args=args.env_args,
        address=args.address,
    )

    # setup graceful shutdown for SIGTERM (K8s, Docker, Slurm) and SIGINT (Ctrl+C)
    stop_event = asyncio.Event()

    def signal_handler(sig):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    try:
        await server.run(stop_event=stop_event)
    finally:
        await server.close()


if __name__ == "__main__":
    asyncio.run(main())
