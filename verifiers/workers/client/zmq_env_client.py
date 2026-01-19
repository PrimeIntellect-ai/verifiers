import asyncio
import logging
import uuid

import msgpack
import zmq
import zmq.asyncio

import verifiers as vf
from verifiers.workers.client.env_client import EnvClient
from verifiers.workers.utils import msgpack_encoder


class ZMQEnvClient(EnvClient):
    def __init__(self, address: str = "tcp://127.0.0.1:5555", timeout: float = 60.0):
        super().__init__(address=address)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.address = address
        self.timeout = timeout

        # DEALER socket for async request/response
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.socket.setsockopt(zmq.LINGER, 0)

        # TCP keepalive for faster dead server detection
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.socket.setsockopt(
            zmq.TCP_KEEPALIVE_IDLE, 10
        )  # Start probes after 10s idle
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 2)  # Probe every 2s
        self.socket.setsockopt(
            zmq.TCP_KEEPALIVE_CNT, 3
        )  # Give up after 3 failed probes

        # Connect to all endpoints
        self.logger.debug(f"Connecting ZMQ client to {address}")
        self.socket.connect(address)

        self.pending: dict[bytes, asyncio.Future] = {}
        self._receiver_task: asyncio.Task | None = None

    async def health(self) -> bool:
        response = await self._send_request("health")
        return response.get("status") == "ok"

    async def run_rollout(
        self,
        input: vf.RolloutInput,
        client_config: vf.ClientConfig,
        model: str,
        sampling_args: vf.SamplingArgs,
    ) -> vf.State:
        response = await self._send_request(
            "run_rollout",
            input=input,
            client_config=client_config,
            model=model,
            sampling_args=sampling_args,
        )
        return vf.State(**response)

    def _fail_all_pending(self, reason: str):
        """Fail all pending futures with the given reason."""
        for request_id, future in list(self.pending.items()):
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self.pending.clear()

    async def _receive_loop(self):
        """Continuously receive responses from environment servers."""
        while True:
            try:
                # Receive multipart: [request_id, payload]
                msg = await self.socket.recv_multipart()

                if len(msg) < 2:
                    self.logger.error(
                        f"Invalid message format: expected 2 frames, got {len(msg)}"
                    )
                    continue

                request_id, response_data = msg[0], msg[1]

                if request_id in self.pending:
                    future = self.pending.pop(request_id)
                    if not future.done():
                        try:
                            response = msgpack.unpackb(response_data, raw=False)
                            future.set_result(response)
                        except Exception as unpack_error:
                            # Unpacking failed - fail the specific future
                            self.logger.error(
                                f"Failed to unpack response for request {request_id.hex()}: {unpack_error}"
                            )
                            future.set_exception(
                                RuntimeError(
                                    f"Failed to deserialize response: {unpack_error}"
                                )
                            )
                else:
                    self.logger.warning(
                        f"Received response for unknown request_id: {request_id.hex()}"
                    )

            except asyncio.CancelledError:
                break
            except zmq.ZMQError as e:
                # Socket-level error - fail all pending futures and exit
                self.logger.error(f"ZMQ socket error in receive loop: {e}")
                self._fail_all_pending(f"ZMQ socket error: {e}")
                break
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in ZMQ receive loop: {e}", exc_info=True
                )
                # Don't break - log and continue for non-socket errors

    async def start(self):
        self._receiver_task = asyncio.create_task(self._receive_loop())
        self.logger.debug("ZMQ client started")

    async def _send_request(self, action: str, **kwargs) -> dict:
        """
        Send request to environment.

        Args:
            action: Action type (generate, get_dataset, etc.)
            **kwargs: Action-specific parameters

        Returns:
            Response dict
        """
        # Auto-start receiver if not already running
        if self._receiver_task is None:
            await self.start()

        request_id = uuid.uuid4().bytes

        # Let msgpack traverse dicts/lists in C - only call msgpack_encoder for unknown types
        payload_bytes = msgpack.packb(
            {"action": action, **kwargs},
            default=msgpack_encoder,
            use_bin_type=True,
        )

        future = asyncio.Future()
        self.pending[request_id] = future

        await self.socket.send_multipart([request_id, payload_bytes])

        try:
            response = await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            self.pending.pop(request_id, None)
            raise TimeoutError(f"Environment timeout for action: {action}")

        if response.get("status") == "error":
            raise RuntimeError(f"Environment error: {response.get('error')}")

        return response


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="ZMQ Environment Client")
    parser.add_argument(
        "--address", type=str, default="tcp://127.0.0.1:5555", help="ZMQ bind address"
    )

    args = parser.parse_args()

    # initialize client
    client = ZMQEnvClient(address=args.address)

    response = await client.health()
    print(response)

    state = await client.run_rollout(
        input=vf.RolloutInput(
            example_id=0,
            task="default",
            prompt=[{"role": "user", "content": "What is 1+1?"}],
        ),
        client_config=vf.ClientConfig(),
        model="openai/gpt-4.1-mini",
        sampling_args=vf.SamplingArgs(temperature=0.5),
    )
    print(state)


if __name__ == "__main__":
    asyncio.run(main())
