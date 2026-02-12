import asyncio
from typing import cast

import msgpack
import zmq
import zmq.asyncio

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

    def __init__(self, *args, address: str = "tcp://127.0.0.1:5000", **kwargs):
        super().__init__(*args, **kwargs)
        self.address = address

        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(self.address)

    async def run(self, stop_event: asyncio.Event | None = None):
        self.logger.info(f"{self.__class__.__name__} started on {self.address}")

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

                    # Process in background, tracking the task for cleanup
                    task = asyncio.create_task(
                        self._process_request(client_id, request_id, payload_bytes)
                    )
                    self.pending_tasks.add(task)
                    task.add_done_callback(self.pending_tasks.discard)

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
        # cancel and await all pending tasks
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
                response = await self._handle_health(request)
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
                response.model_dump(mode="python"),
                default=msgpack_encoder,
                use_bin_type=True,
            ),
        )

        # send response: [client_id, request_id, response]
        await self.socket.send_multipart(
            [client_id, request_id.encode(), response_bytes]
        )
