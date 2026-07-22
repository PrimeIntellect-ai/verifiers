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
from verifiers.v1.loaders import load_environment
from verifiers.v1.serve.types import (
    BaseResponse,
    HealthResponse,
    InfoResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRequest,
    RunResponse,
)
from verifiers.v1.task import Task, task_data_cls
from verifiers.v1.types import SamplingConfig

logger = logging.getLogger(__name__)


class EnvServer:
    def __init__(
        self, config: EnvConfig, address: str = "tcp://127.0.0.1:5000"
    ) -> None:
        self.address = address
        self.taskset_id = config.taskset.id
        self.env = load_environment(config)
        self.task_cls = type(self.env.taskset).task_type()
        self.data_cls = task_data_cls(self.task_cls)
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
        self.num_tasks: int | None = None
        # v1 envs never group-score (siblings score inside the env's own rollout);
        # only the legacy (v0) bridge sets this.
        self.requires_group_scoring = False
        self._gate = (
            asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
        )
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
        # Pin tqdm to a threading lock first, so a dataset pull (legacy bridge) never
        # leaks a multiprocessing semaphore (resource_tracker warning at shutdown).
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

    def _build_task(self, task_data: dict | None) -> Task:
        """Rebuild a request's task from its wire data: validate into the taskset's
        declared `TaskData` type and wrap it in the declared `Task` with the config's
        task subtree — the same construction the taskset's own `load()` performs. The
        client owns the taskset; this server never `load()`s data, so pool workers
        don't each pull the dataset."""
        if task_data is None:
            raise ValueError(
                "v1 env server requests carry task_data (task_idx addresses the legacy bridge)"
            )
        data = self.data_cls.model_validate(task_data)
        return self.task_cls(data, self.env.config.taskset.task)

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
        """The env's serving resources, entered for the server's lifetime so they're
        reused across requests. The legacy v0 bridge overrides this (no v1 serving)."""
        return self.env.serving()

    async def _run(self, req: RunRequest) -> RunResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        (slot,) = self.env.slots(self._build_task(req.task_data))
        # The gate spans requests: `--env.max-concurrent` bounds this worker's
        # agent runs the same way the in-process eval's semaphore does.
        episode = await self.env.run_slot(slot, ctx, self._gate)
        # Trust the env-minted episode; serialize it once before client-side re-typing.
        return RunResponse.model_construct(episode=episode)

    async def _run_group(self, req: RunGroupRequest) -> RunGroupResponse:
        # The route survives for the legacy (v0) bridge (`LegacyEnvServer` overrides
        # this); a dispatcher calling it on a v1 env gets a loud error.
        raise RuntimeError(
            "run_group is a legacy (v0) route; v1 envs score sibling-dependent "
            "signals inside their own rollout — request run instead"
        )

    async def _handle(
        self, client_id: bytes, request_id: bytes, method: bytes, payload: bytes
    ) -> None:
        try:
            route = method.decode()
            raw = msgpack.unpackb(payload, raw=False)
            if route == "health":
                response: BaseResponse = HealthResponse()
            elif route == "info":
                response = InfoResponse(
                    num_tasks=self.num_tasks,
                    requires_group_scoring=self.requires_group_scoring,
                )
            elif route == "run":
                response = await self._run(RunRequest.model_validate(raw))
            elif route == "run_group":
                response = await self._run_group(RunGroupRequest.model_validate(raw))
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
            "EnvServer up: taskset=%s address=%s tasks=%s group_scoring=%s",
            self.taskset_id,
            self.address,
            self.num_tasks if self.num_tasks is not None else "client-side",
            self.requires_group_scoring,
        )
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        tasks: set[asyncio.Task] = set()
        # Shared servers and the interception live across requests in this worker.
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
                logger.info("EnvServer down: taskset=%s", self.taskset_id)
