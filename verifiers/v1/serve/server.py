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
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.env import EnvConfig
from verifiers.v1.loaders import load_environment
from verifiers.v1.serve.types import (
    PROTOCOL_VERSION,
    BaseResponse,
    HealthResponse,
    InfoResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)
from verifiers.v1.types import SamplingConfig

logger = logging.getLogger(__name__)

MAX_LAZY_TASKS = 1_000_000
"""Most tasks an infinite taskset's generator is willing to build (and cache) per worker."""


class EnvServer:
    def __init__(
        self, config: EnvConfig, address: str = "tcp://127.0.0.1:5000"
    ) -> None:
        self.address = address
        self.taskset_id = config.taskset.id
        self.env = load_environment(config)
        # A finite taskset is materialized up front (its count is served via `info`); an
        # infinite one is pulled off its generator on demand (see `_task`), so
        # `num_tasks=None` on the wire ⟺ the taskset is infinite.
        self._task_iter = iter(self.env.taskset.load())
        self._tasks: list = []
        self.num_tasks: int | None = None
        if not type(self.env.taskset).INFINITE:
            self._tasks = list(self._task_iter)
            self.num_tasks = len(self._tasks)
        # One task type per taskset (the authoring contract; its `load()` constructs it),
        # so group scoring is a run-wide property.
        first = self._task(0) if self.num_tasks != 0 else None
        self.requires_group_scoring = first is not None and bool(
            discover_decorated(first, "group_reward")
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

    def _task(self, idx: int):
        """The task at `idx`; an infinite taskset is generated (and cached) up to `idx`
        on demand. Generation must be deterministic — every pool worker runs its own
        `load()`, so idx-addressing relies on all of them producing the same sequence.
        Lazy generation is capped at `MAX_LAZY_TASKS`: an idx that far ahead is a
        runaway driver, and generating (and caching) toward it would hang the worker
        and exhaust memory instead of failing the one request."""
        while len(self._tasks) <= idx:
            if idx >= MAX_LAZY_TASKS:
                raise IndexError(
                    f"task_idx {idx} exceeds the lazy-generation cap ({MAX_LAZY_TASKS})"
                )
            try:
                self._tasks.append(next(self._task_iter))
            except StopIteration:
                raise IndexError(
                    f"task_idx {idx} out of range ({len(self._tasks)} tasks)"
                ) from None
        return self._tasks[idx]

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
        """Context for the server's eval-level serving resources (shared tool servers +
        interception), entered for the server's lifetime so they're reused across
        requests; episodes built inside it inherit them (see `Environment.serving`). The
        legacy v0 bridge overrides this (it runs its own rollouts, with no v1 serving)."""
        return self.env.serving()

    async def _run_rollout(self, req: RunRolloutRequest) -> RunRolloutResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        episode = self.env.episode(self._task(req.task_idx), ctx, n=1)
        records = await episode.run()
        # Trust the env-minted record; serialize it once before client-side re-typing.
        return RunRolloutResponse.model_construct(record=records[0])

    async def _run_group(self, req: RunGroupRequest) -> RunGroupResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        episode = self.env.episode(self._task(req.task_idx), ctx, n=req.n)
        records = await episode.run()
        # run_group still speaks flat traces (it exists only for @group_reward and dies
        # with it); avoid a dump-and-validate copy for every trusted trace in the group.
        return RunGroupResponse.model_construct(
            traces=[trace for record in records for trace in record.traces]
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
                    protocol=PROTOCOL_VERSION,
                )
            elif route == "run_rollout":
                response = await self._run_rollout(
                    RunRolloutRequest.model_validate(raw)
                )
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
            self.num_tasks if self.num_tasks is not None else "infinite",
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
