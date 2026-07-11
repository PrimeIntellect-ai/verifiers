import asyncio
import contextlib
import logging

import msgpack
import zmq
import zmq.asyncio

from verifiers.utils.process_utils import use_threading_tqdm_lock
from verifiers.utils.serve_utils import msgpack_encoder
from verifiers.v1.clients import ModelContext, resolve_client
from verifiers.v1.clients.client import Client
from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.env import EnvConfig
from verifiers.v1.serve.types import (
    BaseResponse,
    HealthResponse,
    InfoResponse,
    RunRequest,
    RunResponse,
)
from verifiers.v1.topology import resolve_topology_runner
from verifiers.v1.types import SamplingConfig

logger = logging.getLogger(__name__)


class EnvServer:
    def __init__(
        self, config: EnvConfig, address: str = "tcp://127.0.0.1:5000"
    ) -> None:
        self.address = address
        self.runner = resolve_topology_runner(config)
        self.tasks = self.runner.tasks
        self._clients: dict[
            tuple[str, str], Client
        ] = {}  # (client_config, model) -> Client

        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(self.address)
        # Resolve the concrete endpoint — when bound to an OS-assigned port
        # (address ending in `:0`), this is how we learn the actual port.
        self.address = self.frontend.getsockopt_string(zmq.LAST_ENDPOINT)

    @classmethod
    def run_server(cls, address_queue=None, **kwargs) -> None:
        """Run a spawned server and report its concrete address when requested."""
        # This worker loads the taskset (and any HF datasets it pulls in) and is killed at
        # teardown; pin tqdm to a threading lock first so it never leaks a multiprocessing
        # semaphore (resource_tracker warning at shutdown).
        use_threading_tqdm_lock()
        server = cls(**kwargs)
        if address_queue is not None:
            address_queue.put(server.address)
        try:
            asyncio.run(server.run())
        except KeyboardInterrupt:
            # SIGTERM arrives as KeyboardInterrupt (see serve.pool._arm_teardown) so the event
            # loop runs its cleanup finallys; swallow it for a clean spawned-worker exit instead
            # of a spurious multiprocessing traceback, matching serve_env's own handling.
            pass

    def _client(self, client_config: ClientConfig, model: str) -> Client:
        """Cache clients because renderer initialization builds a tokenizer pool."""
        key = (client_config.model_dump_json(), model)
        if key not in self._clients:
            self._clients[key] = resolve_client(client_config)
        return self._clients[key]

    def _context(
        self, client_config: ClientConfig, model: str, sampling: SamplingConfig
    ) -> ModelContext:
        return ModelContext(
            client=self._client(client_config, model), model=model, sampling=sampling
        )

    def serving(self):
        """Hold topology services for this worker's full request-serving lifetime."""
        return self.runner.serving()

    async def _run(self, req: RunRequest) -> RunResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        graph = await self.runner.run_instance(self.tasks[req.task_idx], ctx)
        return RunResponse.model_construct(graph=graph)

    async def _dispatch(self, route: str, raw: dict) -> BaseResponse:
        if route == "health":
            return HealthResponse()
        if route == "info":
            return InfoResponse(
                num_tasks=len(self.tasks),
                task_ids=[task.data.idx for task in self.tasks],
            )
        if route == "run":
            return await self._run(RunRequest.model_validate(raw))
        return BaseResponse(success=False, error=f"unknown method {route!r}")

    async def _handle(
        self, client_id: bytes, request_id: bytes, method: bytes, payload: bytes
    ) -> None:
        try:
            route = method.decode()
            raw = msgpack.unpackb(payload, raw=False)
            response = await self._dispatch(route, raw)
        except (
            Exception
        ) as e:  # a failed request is data, not a crash — report and keep serving
            logger.warning("request failed: %s", e, exc_info=True)
            response = BaseResponse(success=False, error=f"{type(e).__name__}: {e}")
        data = msgpack.packb(
            response.model_dump(mode="python"),
            default=msgpack_encoder,
            use_bin_type=True,
        )
        try:
            # Let ZMQ retain the packed response instead of copying large traces.
            await self.frontend.send_multipart(
                [client_id, request_id, data], copy=False
            )
        except zmq.ZMQError as e:
            logger.warning("failed to send response: %s", e)

    async def run(self) -> None:
        logger.info(
            "EnvServer up: address=%s tasks=%d",
            self.address,
            len(self.tasks),
        )
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        tasks: set[asyncio.Task] = set()
        # Shared servers and the interception pool live across requests in this worker.
        async with self.serving():
            try:
                while True:
                    events = dict(await poller.poll(timeout=100))
                    if self.frontend not in events:
                        continue
                    frames = await self.frontend.recv_multipart()
                    if len(frames) != 4:
                        logger.warning(
                            "invalid message: expected 4 frames, got %d", len(frames)
                        )
                        continue
                    client_id, request_id, method, payload = frames
                    task = asyncio.create_task(
                        self._handle(client_id, request_id, method, payload)
                    )
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
            finally:
                for task in tasks:
                    task.cancel()
                for client in self._clients.values():
                    with contextlib.suppress(Exception):
                        await client.close()
                self.frontend.close()
                self.ctx.term()
                logger.info("EnvServer down")
