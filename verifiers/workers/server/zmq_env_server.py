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
        self._recv_count = 0
        self._send_count = 0
        self._send_lock = asyncio.Lock()

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
                    frames = await self.socket.recv_multipart()

                    if len(frames) != 2:
                        self.logger.warning(
                            f"Invalid message: expected 2 frames, got {len(frames)}"
                        )
                        continue

                    client_id, payload_bytes = frames
                    self._recv_count += 1
                    self.logger.debug(
                        f"[server-recv] recv_total={self._recv_count} "
                        f"pending_tasks={len(self.pending_tasks)}"
                    )

                    task = asyncio.create_task(
                        self._process_request(client_id, payload_bytes)
                    )
                    self.pending_tasks.add(task)
                    task.add_done_callback(self.pending_tasks.discard)

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
        payload_bytes: bytes,
    ):
        response: BaseResponse
        request_id = "unknown"

        try:
            raw = msgpack.unpackb(payload_bytes, raw=False)
            request_id = raw.pop("_zmq_request_id", "unknown")
            request_type = raw.get("request_type")

            self.logger.debug(
                f"[server-recv] request_id={request_id[:8]} type={request_type} "
                f"(recv_total={self._recv_count}, pending_tasks={len(self.pending_tasks)})"
            )

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
            self.logger.warning(
                f"[server-cancelled] request_id={request_id[:8]} "
                f"(pending_tasks={len(self.pending_tasks)})"
            )
            return

        except Exception as e:
            self.logger.error(
                f"Error processing request {request_id}: {e}", exc_info=True
            )
            response = BaseResponse(
                success=False,
                error=repr(e),
            )

        # Serialize and send response — wrapped in try/except to guarantee
        # we always send a response back. Without this, serialization errors
        # (e.g. non-msgpack-serializable types in rollout outputs) silently
        # drop the response, causing the client to hang forever.
        try:
            response_dict = response.model_dump(mode="python", warnings=False)
            response_dict["_zmq_request_id"] = request_id
            response_bytes = cast(
                bytes,
                msgpack.packb(
                    response_dict, default=msgpack_encoder, use_bin_type=True
                ),
            )
        except Exception as e:
            self.logger.error(
                f"[server-send] Failed to serialize response for "
                f"request_id={request_id[:8]}: {e}",
                exc_info=True,
            )
            fallback = BaseResponse(success=False, error=repr(e))
            response_dict = fallback.model_dump(mode="python", warnings=False)
            response_dict["_zmq_request_id"] = request_id
            response_bytes = cast(
                bytes,
                msgpack.packb(
                    response_dict, default=msgpack_encoder, use_bin_type=True
                ),
            )

        # ROUTER requires [client_id, payload] — lock prevents interleaving
        # from concurrent _process_request tasks
        async with self._send_lock:
            await self.socket.send_multipart([client_id, response_bytes])
        self._send_count += 1
        self.logger.debug(
            f"[server-send] request_id={request_id[:8]} success={response.success} "
            f"(send_total={self._send_count}, pending_tasks={len(self.pending_tasks)})"
        )
