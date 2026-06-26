"""The env server: load a taskset once, run rollouts on request by pulling tasks.

A ZMQ ROUTER front end (msgpack frames) over a v1 `Environment`. The server
owns the environment — taskset, harness, runtime — and is the only process that ever
loads it, including task scheduling (shuffle + epoch). A caller (e.g. the orchestrator)
stays env-agnostic: it asks `info` for the task count (None if infinite) + whether group
scoring is needed, then pulls via `run_rollout` / `run_group` (no task index — the server
hands out the next task). Per request the server resolves a `Client` from the request's
`client` config (cached, so a renderer's tokenizer is built once), wraps it in a
`RolloutContext`, and runs `env.episode(task, ctx, n).run()`, returning each
`Trace` as a JSON dict (with its computed `branches`).

Minimal port of `verifiers.serve` (ROUTER + msgpack), single async process: each
request is its own `asyncio.Task`, so many rollouts run concurrently. No worker
pool / heartbeats yet — the rollout's own runtime already isolates execution, and
the structure leaves room to add a pool later. Health is just another request type
(no separate socket), which is enough at this scale.
"""

import asyncio
import contextlib
import logging
import random
from collections.abc import Iterator

import msgpack
import zmq
import zmq.asyncio

from verifiers.utils.process_utils import use_threading_tqdm_lock
from verifiers.utils.serve_utils import msgpack_encoder
from verifiers.v1.clients import RolloutContext, resolve_client
from verifiers.v1.clients.client import Client
from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.env import EnvConfig, Environment
from verifiers.v1.taskset import SHUFFLE_SEED
from verifiers.v1.serve.types import (
    BaseResponse,
    HealthResponse,
    InfoResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
    SampleRequest,
    SampleResponse,
)
from verifiers.v1.types import SamplingConfig

logger = logging.getLogger(__name__)


class EnvServer:
    """Serve one v1 environment over ZMQ. The only process that loads the env."""

    def __init__(
        self,
        config: EnvConfig,
        address: str = "tcp://127.0.0.1:5000",
        seed_offset: int = 0,
    ) -> None:
        self.address = address
        self.taskset_id = config.taskset.id
        self.env = Environment(config)
        # Task order: a `ShuffleConfig` shuffles a finite taskset (reshuffled per epoch) and seeds
        # an INFINITE one's generation. Each pool worker offsets the seed by its index (when
        # shuffling) so workers draw divergent streams; `taskset.seed` carries it to `load_tasks`.
        shuffle = config.taskset.shuffle
        self._shuffle = shuffle is not None
        self._seed = (shuffle.seed + seed_offset) if self._shuffle else SHUFFLE_SEED
        self.env.taskset.seed = self._seed
        # The server owns task scheduling: callers pull (`run_rollout`/`run_group` carry no task
        # index), and the server hands out the next task. A finite taskset is materialized + counted
        # and served as a (re)shuffled permutation that loops over epochs; an INFINITE one is
        # streamed straight off its generator (which owns its own order). Pull is sequential and
        # tasks are echoed back by value, so no random-access index/cache is needed.
        tasks = iter(self.env.taskset.load_tasks())  # a list is iterable too
        try:
            first = next(tasks)  # peek: non-empty + representative task
        except StopIteration:
            raise ValueError("taskset load_tasks() produced no tasks") from None
        self._first_task = first  # representative task for `serving()` + the task type
        self._task_type = type(first)
        if self.env.taskset.INFINITE:
            self._iter: Iterator | None = tasks  # forward-only stream
            self._tasks: list = []
            self._pending = first  # the peeked task, served on the opening pull
            self.num_tasks: int | None = None
        else:
            self._iter = None
            self._tasks = [first, *tasks]  # materialized for indexed access
            self._pending = None
            self.num_tasks = len(self._tasks)
        self._init_scheduler(self.num_tasks)
        self.requires_group_scoring = bool(
            discover_decorated(self.env.taskset, "group_reward")
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
        """Build and run a server (entry point for a spawned process). If
        `address_queue` is given, report the concrete bound address on it (so a
        spawner that passed a `:0` address learns the OS-assigned port) before
        serving."""
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
        """Resolve (and cache) a `Client` for this config+model. Cached because a
        renderer client builds the model's tokenizer pool on first use — doing that
        per request would be ruinous."""
        key = (client_config.model_dump_json(), model)
        if key not in self._clients:
            self._clients[key] = resolve_client(client_config)
        return self._clients[key]

    def _context(
        self, client_config: ClientConfig, model: str, sampling: SamplingConfig
    ) -> RolloutContext:
        return RolloutContext(
            client=self._client(client_config, model), model=model, sampling=sampling
        )

    def serving(self):
        """Context for the server's eval-level serving resources (shared tool servers +
        interception pool), entered for the server's lifetime so they're reused across
        requests; episodes built inside it inherit them (see `Environment.serving`). The
        legacy v0 bridge overrides this (it runs its own rollouts, with no v1 serving)."""
        # Shared tool servers are task-agnostic, but `Environment.serving` builds them from a
        # task, so hand it a representative one (the taskset is guaranteed non-empty).
        return self.env.serving([self._first_task])

    def _init_scheduler(self, count: int | None) -> None:
        """Set up the finite-taskset cursor: `count` tasks (None = infinite → streamed, no
        scheduler). Reads `self._shuffle`/`self._seed` (set in `__init__`). Shared by the native
        server and the v0 bridge (always a finite dataset)."""
        self._count = count
        self._cursor = 0
        self._order: list[int] | None = None  # current epoch's permutation
        self._epoch = 0

    def _next_index(self) -> int:
        """The next index of a *finite* taskset to serve: walk a (re)shuffled permutation that loops
        over epochs (reshuffled each epoch from `self._seed`, which already includes the per-worker
        offset; the `seed-epoch` string seed keeps (worker, epoch) pairs distinct). Synchronous and
        never awaits, so concurrent rollouts on one event loop can't interleave the cursor (no lock
        needed). Used by native-finite (`_next_task`) and the v0 bridge."""
        pos = self._cursor % self._count
        if pos == 0:  # (re)build the epoch's permutation at each boundary
            self._order = list(range(self._count))
            if self._shuffle:
                random.Random(f"{self._seed}-{self._epoch}").shuffle(self._order)
            self._epoch += 1
        self._cursor += 1
        return self._order[pos]

    def _next_task(self):
        """The next task to serve: a finite taskset is indexed via `_next_index`; an infinite one
        is streamed off its generator (the peeked first, then `next`)."""
        if self._count is None:  # infinite: stream (no index, no cache)
            if self._pending is not None:
                task, self._pending = self._pending, None
                return task
            return next(self._iter)
        return self._tasks[self._next_index()]

    def _sample(self, req: SampleRequest) -> SampleResponse:
        # Pull the next task (cursor + shuffle/epoch are server-owned); the caller echoes it back
        # to `run_rollout` to run rollouts of it. Lets the caller bound concurrency per rollout
        # (each `run_rollout` is one permit) while a group still shares one task.
        return SampleResponse.model_construct(task=self._next_task())

    async def _run_rollout(self, req: RunRolloutRequest) -> RunRolloutResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        task = self._task_type.model_validate(req.task.model_dump())
        episode = self.env.episode(task, ctx, n=1)
        traces = await episode.run()
        # Trust the concrete trace; serialize it once before client-side re-typing.
        return RunRolloutResponse.model_construct(trace=traces[0])

    async def _run_group(self, req: RunGroupRequest) -> RunGroupResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        task = self._task_type.model_validate(req.task.model_dump())
        episode = self.env.episode(task, ctx, n=req.n)
        traces = await episode.run()
        # Avoid a dump-and-validate copy for every trusted trace in the group.
        return RunGroupResponse.model_construct(traces=traces)

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
            elif route == "sample":
                response = self._sample(SampleRequest.model_validate(raw))
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
            self.num_tasks if self.num_tasks is not None else "lazy",
            self.requires_group_scoring,
        )
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        tasks: set[asyncio.Task] = set()
        # Enter the serving resources (shared tool servers + interception pool) for the
        # server's lifetime so they're reused across requests; episodes built per request
        # inherit them (the legacy bridge overrides this to a no-op).
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
