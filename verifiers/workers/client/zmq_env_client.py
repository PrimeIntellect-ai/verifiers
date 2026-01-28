import asyncio
import logging
import uuid
from typing import cast

import msgpack
import zmq
import zmq.asyncio

import verifiers as vf
from verifiers.utils.worker_utils import msgpack_encoder
from verifiers.workers.client.env_client import EnvClient
from verifiers.workers.types import (
    BaseRequest,
    BaseResponseT,
    HealthRequest,
    HealthResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)


class ZMQEnvClient(EnvClient):
    def __init__(self, address: str = "tcp://127.0.0.1:5000"):
        super().__init__(address=address)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.address = address

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

        self.pending: dict[str, asyncio.Future] = {}
        self._receiver_task: asyncio.Task | None = None

    async def health(self) -> bool:
        request = HealthRequest()
        response = await self._send_request(request, HealthResponse, timeout=1.0)
        return response.success

    async def run_rollout(
        self,
        input: vf.RolloutInput,
        client_config: vf.ClientConfig,
        model: str,
        sampling_args: vf.SamplingArgs,
    ) -> vf.RolloutOutput:
        request = RunRolloutRequest(
            input=input,
            client_config=client_config,
            model=model,
            sampling_args=sampling_args,
        )
        response = await self._send_request(
            request, RunRolloutResponse, timeout=36000.0
        )
        assert response.output is not None
        return response.output

    async def run_group(
        self,
        group_inputs: list[vf.RolloutInput],
        client_config: vf.ClientConfig,
        model: str,
        sampling_args: vf.SamplingArgs,
    ) -> list[vf.RolloutOutput]:
        request = RunGroupRequest(
            group_inputs=group_inputs,
            client_config=client_config,
            model=model,
            sampling_args=sampling_args,
        )
        response = await self._send_request(request, RunGroupResponse, timeout=36000.0)
        assert response.outputs is not None
        return response.outputs

    def _fail_all_pending(self, reason: str):
        """Fail all pending futures with the given reason."""
        for _, future in list(self.pending.items()):
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

                request_id_bytes, response_data = msg[0], msg[1]
                request_id = request_id_bytes.decode()

                if request_id in self.pending:
                    future = self.pending.pop(request_id)
                    if not future.done():
                        try:
                            response = msgpack.unpackb(response_data, raw=False)
                            future.set_result(response)
                        except Exception as unpack_error:
                            # Unpacking failed - fail the specific future
                            self.logger.error(
                                f"Failed to unpack response for request {request_id}: {unpack_error}"
                            )
                            future.set_exception(
                                RuntimeError(
                                    f"Failed to deserialize response: {unpack_error}"
                                )
                            )
                else:
                    self.logger.warning(
                        f"Received response for unknown request_id: {request_id}"
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

    async def _start(self):
        self._receiver_task = asyncio.create_task(self._receive_loop())
        self.socket.connect(self.address)
        self.logger.debug("ZMQ client started")

    async def _send_request(
        self,
        request: BaseRequest,
        response_type: type[BaseResponseT],
        timeout: float | None = None,
    ) -> BaseResponseT:
        """
        Send typed request to environment and parse typed response.

        Args:
            request: Pydantic request model (contains action and request_id)
            response_type: Expected Pydantic response type

        Returns:
            Validated response of type T
        """
        # Auto-start receiver if not already running
        if self._receiver_task is None:
            await self._start()

        # Use request_id from Pydantic model, encode to bytes for ZMQ frame
        request_id = uuid.uuid4().hex

        # Serialize using Pydantic
        payload_bytes = cast(
            bytes,
            msgpack.packb(
                request.model_dump(mode="python"),
                default=msgpack_encoder,
                use_bin_type=True,
            ),
        )

        future: asyncio.Future[dict] = asyncio.Future()
        self.pending[request_id] = future

        await self.socket.send_multipart([request_id.encode(), payload_bytes])

        try:
            raw_response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self.pending.pop(request_id, None)
            raise TimeoutError(
                f"Environment timeout for {request.request_type} request after {timeout}s"
            )

        # validate response with Pydantic
        response = response_type.model_validate(raw_response)

        if not response.success:
            raise RuntimeError(f"Server error: {response.error}")

        return response


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="ZMQ Environment Client")
    parser.add_argument(
        "--address", type=str, default="tcp://127.0.0.1:5000", help="ZMQ bind address"
    )
    args = parser.parse_args()

    # initialize client
    client = ZMQEnvClient(address=args.address)

    is_healthy = await client.health()
    assert is_healthy, "ZMQEnvServer is not healthy"
    print("Checked that ZMQEnvServer is running and healthy.")
    results = await client.run_rollout(
        input=vf.RolloutInput(
            example_id=0,
            prompt=[{"role": "user", "content": "What is 2+2?"}],
            answer="4",
            task="test",
        ),
        model="openai/gpt-4.1-mini",
        client_config=vf.ClientConfig(),
        sampling_args={},
    )
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
