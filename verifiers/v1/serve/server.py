import asyncio
import logging

import msgpack
import zmq
import zmq.asyncio

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import ClientConfig
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.serve.encoding import msgpack_encoder
from verifiers.v1.serve.types import (
    BaseResponse,
    HealthResponse,
    RunRequest,
    RunResponse,
)
from verifiers.v1.task import Task
from verifiers.v1.types import SamplingConfig
from verifiers.v1.utils.loaders import load_environment

logger = logging.getLogger(__name__)


class EnvServer:
    def __init__(
        self,
        config: EnvConfig,
        address: str = "tcp://127.0.0.1:5000",
        max_concurrent: int | None = None,
    ) -> None:
        self.address = address
        self.taskset_id = config.taskset.id
        self.env = load_environment(config)
        self.task_cls = type(self.env.taskset).task_type()
        self.data_cls = self.task_cls.data_type()
        # A dispatched task is its client-side model_dump(): a field excluded from
        # serialization would vanish on the wire and rebuild silently defaulted, so
        # refuse to serve such a taskset.
        excluded = [
            name for name, field in self.data_cls.model_fields.items() if field.exclude
        ]
        if excluded:
            raise ValueError(
                f"{self.data_cls.__name__} excludes {excluded} from serialization — "
                "a served task must survive the wire whole (drop exclude=True)"
            )
        # This worker's episode bound (`--max-concurrent`), spanning requests.
        self._gate = asyncio.Semaphore(max_concurrent) if max_concurrent else None
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

    def _build_task(self, task_data: dict) -> Task:
        """Rebuild a request's task from its wire data: validate into the taskset's
        declared `TaskData` type and wrap it in the declared `Task` with the config's
        task subtree — the same construction the taskset's own `load()` performs. The
        client owns the taskset; this server never `load()`s data, so pool workers
        don't each pull the dataset."""
        data = self.data_cls.model_validate(task_data)
        return self.task_cls(data, self.env.config.taskset.task)

    def _context(
        self, client_config: ClientConfig, model: str, sampling: SamplingConfig
    ) -> ModelContext:
        """The request's sampling context. No client is built or cached here — each
        rollout constructs its own from `client_config` and closes it, so a request's
        endpoint (and a training run's changing model) leaves nothing behind."""
        return ModelContext(client=client_config, model=model, sampling=sampling)

    async def _run(self, req: RunRequest) -> RunResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        (slot,) = self.env.slots(self._build_task(req.task_data))
        # The gate spans requests: `--max-concurrent` bounds this worker's episodes
        # in flight the same way the in-process eval's semaphore does.
        episode = await self.env.run_slot(slot, ctx, self._gate)
        # Trust the env-minted episode; serialize it once before client-side re-typing.
        return RunResponse.model_construct(episode=episode)

    async def _handle(
        self, client_id: bytes, request_id: bytes, method: bytes, payload: bytes
    ) -> None:
        try:
            route = method.decode()
            raw = msgpack.unpackb(payload, raw=False)
            if route == "health":
                response: BaseResponse = HealthResponse()
            elif route == "run":
                response = await self._run(RunRequest.model_validate(raw))
            else:
                response = BaseResponse(
                    success=False, error=f"unknown method {route!r}"
                )
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
            "EnvServer up: taskset=%s address=%s",
            self.taskset_id,
            self.address,
        )
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        tasks: set[asyncio.Task] = set()
        # Shared servers and the interception live across requests in this worker.
        async with self.env.serving():
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
                self.frontend.close()
                self.ctx.term()
                logger.info("EnvServer down: taskset=%s", self.taskset_id)
