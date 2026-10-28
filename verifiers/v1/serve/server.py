import asyncio
import logging
import threading
import time
from typing import Protocol

import msgpack
import zmq
import zmq.asyncio

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.agent import agent_config_fields
from verifiers.v1.configs.client import ClientConfig
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.episode import Episode
from verifiers.v1.runtimes import backend_cls, make_backend
from verifiers.v1.serve.delta import DeltaStreamer, TraceSummary, dump, pack
from verifiers.v1.serve.encoding import msgpack_encoder
from verifiers.v1.serve.types import (
    BaseResponse,
    CancelRequest,
    CancelResponse,
    DiscardRequest,
    DiscardResponse,
    HealthResponse,
    RestoreRequest,
    RunRequest,
    RunResponse,
)
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import Checkpoint, RestorePoint, Trace
from verifiers.v1.utils.loaders import load_environment

logger = logging.getLogger(__name__)


class EpisodeHook(Protocol):
    """Takes finished episodes' per-token training arrays (routing rows, sampling masks,
    top-k heads) before the episodes leave the worker."""

    def keep(self, episode: Episode, sampling: SamplingConfig) -> None:
        """Keep the episode's arrays; may leave references to them in `trace.info`.
        Runs off the event loop, before the episode's final flush."""

    def discard(self, episode: Episode) -> None:
        """Free what `keep` kept for an episode that was not delivered."""

    def watch(self, trace: Trace) -> None:
        """See a trace as it is minted, or restored, before its first delta streams."""

    def restore(self, trace: Trace) -> None:
        """Give a restored trace's nodes back their arrays (a client's assembly never
        has them). Runs before the rollout rewinds the trace to its restore point."""


_episode_hook: EpisodeHook | None = None

LATE_CANCEL_SECONDS = 10.0
"""How long a delivered episode stays discardable: a client whose caller cancels the run
after the reply left drops the reply and sends a cancel right away."""


def set_episode_hook(hook: EpisodeHook | None) -> None:
    """Install a hook that keeps finished episodes' training arrays. Streamed nodes never
    carry them, so the hook is the only way they leave the worker. Every episode it
    keeps is either delivered in its run's reply or handed back to `discard`, and so is
    a delivered one whose client cancels the run within `LATE_CANCEL_SECONDS`."""
    global _episode_hook
    _episode_hook = hook


class _Keeping:
    """One episode's `keep`, run off the event loop. A failed or abandoned `keep` hands
    the episode to `discard` exactly once, and only after `keep` has returned: in the
    thread when `keep` finishes last, else in `abandon`."""

    def __init__(self, hook: EpisodeHook, episode: Episode) -> None:
        self.hook = hook
        self.episode = episode
        self._lock = threading.Lock()
        self._kept = False
        self._abandoned = False

    def run(self, sampling: SamplingConfig) -> None:
        try:
            self.hook.keep(self.episode, sampling)
        except BaseException:
            self.hook.discard(self.episode)
            raise
        with self._lock:
            self._kept = True
            discard = self._abandoned
        if discard:
            self.hook.discard(self.episode)

    def abandon(self) -> None:
        with self._lock:
            self._abandoned = True
            discard = self._kept
        if discard:
            self.hook.discard(self.episode)


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
        # In-flight run tasks by wire request_id, so a `cancel` can abort them
        self._running: dict[str, asyncio.Task] = {}
        # Episodes the hook kept, by run request, until their reply is sent.
        self._kept: dict[bytes, tuple[EpisodeHook, Episode]] = {}
        self._delivered: dict[str, tuple[float, EpisodeHook, Episode]] = {}
        """Recently delivered kept episodes, oldest first, for a cancel that arrives late."""
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

    def _restore(self, restore: RestoreRequest, task: Task) -> tuple[Trace, Checkpoint]:
        """The trace to continue and the checkpoint to continue it from: the trace as
        the client assembled it, typed for this taskset, and its restore point with
        the live-only fields back."""
        trace = Trace.model_validate(restore.trace)
        trace.task.data = task.data
        if _episode_hook is not None:
            _episode_hook.restore(trace)
        checkpoint = RestorePoint.model_validate(restore.point).live(
            state_cls(type(task))
        )
        # The trace's own record of the checkpoint lacks the live fields.
        trace.checkpoints = [
            checkpoint if c.id == checkpoint.id else c for c in trace.checkpoints
        ]
        if checkpoint not in trace.checkpoints:
            trace.checkpoints.append(checkpoint)
        return trace, checkpoint

    async def _discard(self, req: DiscardRequest) -> DiscardResponse:
        """Delete snapshots through the backend of the env's agents' runtime."""
        for agent in agent_config_fields(self.env.config).values():
            if backend_cls(agent.runtime).SUPPORTS_CHECKPOINTS:
                backend = make_backend(agent.runtime)
                return DiscardResponse(
                    deleted=await backend.discard_checkpoints(req.checkpoints)
                )
        return DiscardResponse()

    async def _run(
        self, req: RunRequest, client_id: bytes, request_id: bytes
    ) -> RunResponse:
        ctx = self._context(req.client, req.model, req.sampling)
        task = self._build_task(req.task_data)
        restore = self._restore(req.restore, task) if req.restore is not None else None
        (slot,) = self.env.slots(task)

        async def send_delta(delta: dict) -> None:
            await self.frontend.send_multipart(
                [client_id, request_id, b"delta", pack(delta)], copy=False
            )

        # The gate spans requests: `--max-concurrent` bounds this worker's episodes
        # in flight the same way the in-process eval's semaphore does. The streamer
        # ships each trace as it changes; the reply below carries only the rest.
        hook = _episode_hook
        async with DeltaStreamer(lambda: slot.traces, send_delta) as streamer:

            def watch(trace: Trace) -> None:
                if hook is not None:
                    hook.watch(trace)
                streamer.watch(trace)

            episode = await self.env.run_slot(
                slot, ctx, self._gate, on_trace=watch, restore=restore
            )
            if hook is not None:
                # Before the final flush, so what the hook writes to the traces streams.
                keeping = _Keeping(hook, episode)
                try:
                    # The thread runs to the end even when this await is cancelled.
                    await asyncio.shield(asyncio.to_thread(keeping.run, ctx.sampling))
                except asyncio.CancelledError:
                    keeping.abandon()
                    raise
                self._kept[request_id] = (hook, episode)
        return RunResponse(
            head=dump(episode, exclude={"traces"}),
            traces=[
                TraceSummary(
                    id=trace.id, nodes=len(trace.nodes), calls=len(trace.calls)
                )
                for trace in episode.traces
            ],
        )

    async def _handle(
        self, client_id: bytes, request_id: bytes, method: bytes, payload: bytes
    ) -> None:
        try:
            route = method.decode()
            raw = msgpack.unpackb(payload, raw=False)
            if route == "health":
                response: BaseResponse = HealthResponse()
            elif route == "run":
                # Registered in the dispatch loop, before this task first runs
                try:
                    response = await self._run(
                        RunRequest.model_validate(raw), client_id, request_id
                    )
                finally:
                    self._running.pop(request_id.decode(), None)
            elif route == "discard":
                response = await self._discard(DiscardRequest.model_validate(raw))
            elif route == "cancel":
                target = CancelRequest.model_validate(raw).request_id
                running = self._running.get(target)
                if running is not None:
                    running.cancel()
                    logger.info("cancelled run %s on client request", target)
                # The client dropped the reply, or will: nothing will take the arrays.
                self._expire_delivered()
                kept = self._kept.pop(target.encode(), None)
                if kept is None and (late := self._delivered.pop(target, None)):
                    kept = late[1:]
                if kept is not None:
                    hook, episode = kept
                    hook.discard(episode)
                response = CancelResponse(cancelled=running is not None)
            else:
                response = BaseResponse(
                    success=False, error=f"unknown method {route!r}"
                )
        except asyncio.CancelledError:
            # An aborted run still replies so broker/client accounting stays
            # exact; the requesting client has already given up on the result
            response = BaseResponse(success=False, error="Cancelled: rollout aborted")
        except (
            Exception
        ) as e:  # a failed request is data, not a crash — report and keep serving
            logger.warning("request failed: %s", e, exc_info=True)
            response = BaseResponse(success=False, error=f"{type(e).__name__}: {e}")
        try:
            data = msgpack.packb(
                response.model_dump(mode="python"),
                default=msgpack_encoder,
                use_bin_type=True,
            )
        except Exception as e:
            # Encoding failures must also reply so clients don't wait indefinitely.
            logger.warning("response encoding failed: %s", e, exc_info=True)
            response = BaseResponse(success=False, error=f"{type(e).__name__}: {e}")
            data = msgpack.packb(response.model_dump(), use_bin_type=True)
        delivered = False
        try:
            await self.frontend.send_multipart(
                [client_id, request_id, b"reply", data], copy=False
            )
            delivered = response.success
        except zmq.ZMQError as e:
            logger.warning("failed to send response: %s", e)
        finally:
            # Also on a cancel (shutdown) while the reply is sending.
            kept = self._kept.pop(request_id, None)
            if kept is not None:
                hook, episode = kept
                if delivered:
                    self._remember_delivered(request_id.decode(), hook, episode)
                else:
                    hook.discard(episode)

    def _remember_delivered(
        self, request_id: str, hook: EpisodeHook, episode: Episode
    ) -> None:
        self._delivered[request_id] = (time.monotonic(), hook, episode)
        asyncio.get_running_loop().call_later(
            LATE_CANCEL_SECONDS, self._expire_delivered
        )

    def _expire_delivered(self) -> None:
        now = time.monotonic()
        for request_id, (delivered_at, _, _) in list(self._delivered.items()):
            if now - delivered_at < LATE_CANCEL_SECONDS:
                break
            del self._delivered[request_id]

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
                    if method == b"run":
                        # Register at dispatch, not handler start: a cancel
                        # processed ahead of the run's handler task must still
                        # find its target (ZMQ preserves per-client frame
                        # order, so the run always arrives first)
                        self._running[request_id.decode()] = task
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
