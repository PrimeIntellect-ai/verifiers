"""The env server: load a taskset once, run rollouts on request by task index.

A ZMQ ROUTER front end (msgpack frames) over a v1 `Environment`. The server
owns the environment — taskset, harness, runtime — and is the only process that ever
loads it. A caller (e.g. the orchestrator) stays env-agnostic: it asks `info` for
the task count + whether group scoring is needed, then `run_rollout` / `run_group`
by task index. Per request the server resolves a `Client` from the request's
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
from collections.abc import Iterable, Iterator, Sized

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
from verifiers.v1.task import Task
from verifiers.v1.serve.types import (
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


class EnvServer:
    """Serve one v1 environment over ZMQ. The only process that loads the env."""

    def __init__(
        self, config: EnvConfig, address: str = "tcp://127.0.0.1:5000"
    ) -> None:
        self.address = address
        self.taskset_id = config.taskset.id
        self.env = Environment(config)
        # The server is purely index-addressed (`run_rollout`/`run_group` by `task_idx`), so it
        # doesn't need the whole taskset up front. A `load_tasks` with a free length (a list) is
        # materialized and its count reported via `info`; a generator is consumed lazily by index
        # (see `_task`) with no count — so an unbounded taskset is served without materializing.
        self._load_tasks(self.env.taskset.load_tasks())
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

    def _load_tasks(self, tasks: Iterable[Task]) -> None:
        """Set up index-addressed task resolution from `load_tasks`'s result. A `Sized` (a
        list) has a free length, so materialize it and report `num_tasks`; any other iterable
        is a lazy stream — cached on demand by `_task`, with `num_tasks=None` (unknown / possibly
        unbounded)."""
        if isinstance(tasks, Sized):
            self._tasks: list[Task] = list(tasks)
            self._task_iter: Iterator[Task] | None = None
            self.num_tasks: int | None = len(self._tasks)
        else:
            self._tasks = []
            self._task_iter = iter(tasks)
            self.num_tasks = None

    def _task(self, idx: int) -> Task:
        """Resolve task `idx`, extending the lazy cache from `load_tasks` on demand (a
        materialized list is already fully cached). Synchronous and non-awaiting, so concurrent
        rollouts can't interleave a partial extension of the cache. The cache grows to the
        highest index served (bounded for a finite taskset; TODO: evict for very long unbounded
        runs, which needs a contract on index-access order)."""
        if self._task_iter is not None:
            while len(self._tasks) <= idx:
                try:
                    self._tasks.append(next(self._task_iter))
                except StopIteration:
                    raise IndexError(
                        f"task_idx {idx} out of range: taskset {self.taskset_id} yielded "
                        f"{len(self._tasks)} task(s)"
                    ) from None
        return self._tasks[idx]

    def serving(self):
        """Context for the server's eval-level serving resources (shared tool servers +
        interception pool), entered for the server's lifetime so they're reused across
        requests; episodes built inside it inherit them (see `Environment.serving`). The
        legacy v0 bridge overrides this (it runs its own rollouts, with no v1 serving)."""
        # Shared tool servers are task-agnostic; build them from a representative task (the
        # first, resolved lazily). An empty taskset has none.
        try:
            sample = [self._task(0)]
        except IndexError:
            sample = []
        return self.env.serving(sample)

    async def _run_rollout(self, req: RunRolloutRequest) -> RunRolloutResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        episode = self.env.episode(self._task(req.task_idx), ctx, n=1)
        traces = await episode.run()
        # Trust the concrete trace; serialize it once before client-side re-typing.
        return RunRolloutResponse.model_construct(trace=traces[0])

    async def _run_group(self, req: RunGroupRequest) -> RunGroupResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        episode = self.env.episode(self._task(req.task_idx), ctx, n=req.n)
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
