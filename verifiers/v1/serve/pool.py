"""Env-server worker pool: a ROUTER broker over N worker processes.

A lone `EnvServer` runs every rollout as an `asyncio.Task` on one event loop, so
CPU-bound work (renderer tokenization, scoring) competes for that loop; the pool
spreads that work across processes.

A broker binds the client-facing ROUTER (the *same* wire protocol as a lone
`EnvServer`, so `EnvClient` is unchanged), starts one worker process and scales up to
`max_workers` on demand — each an ordinary `EnvServer` bound to its own ipc address —
load-balancing requests to the least-busy worker over a `DEALER` per worker. The
worker's `client_id` (its reply identity) is the broker's DEALER identity; the broker
holds the real client identity in `pending` and routes the reply back by `request_id`.
`health` is answered inline (no worker needed); everything else goes to a worker.

Scaling is elastic but upscale-only: a new worker is spawned when in-flight requests
reach 90% of current capacity (`workers * multiplex`). Workers are spawned `spawn`-style
(own env, own loop) and monitor a death pipe so an orphaned worker self-exits if the
broker dies. The broker watches each worker process in turn: when one dies, what it
held is answered (its requests as failed, its share of a steer as not queued) and nothing
is routed to it again; when the last one dies, the broker stops. TODO: downscale idle
workers, restart a dead worker, stats/lag monitors (v0 had them; omitted here — rollout
errors are returned as data, not crashes, so worker death is rare).
"""

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import signal
import tempfile
import threading
from collections.abc import Callable

import msgpack
import zmq
import zmq.asyncio

from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import (
    BaseResponse,
    CancelResponse,
    HealthResponse,
    SteerResponse,
)

logger = logging.getLogger(__name__)

_HEALTH = msgpack.packb(HealthResponse().model_dump(mode="json"), use_bin_type=True)
_CANCEL_MISS = msgpack.packb(
    CancelResponse(cancelled=False).model_dump(mode="json"), use_bin_type=True
)
_WORKER_DIED = "the env worker serving it died"
_LOST = msgpack.packb(
    BaseResponse(success=False, error=_WORKER_DIED).model_dump(mode="json"),
    use_bin_type=True,
)
_STEER_LOST = msgpack.packb(
    SteerResponse(queued=False, reason=_WORKER_DIED).model_dump(mode="json"),
    use_bin_type=True,
)


def _arm_teardown(death_pipe=None) -> None:
    """Arm a spawned process (serve_env broker/single server, or pool worker) for clean
    teardown: it inherits no signal handlers, so by default SIGTERM kills it abruptly, skipping
    asyncio.run()'s serving() cleanup and orphaning host tunnels (and sandboxes).

    - SIGTERM -> KeyboardInterrupt so the event loop runs its finallys (serve_env swallows it);
    - with `death_pipe`, self-SIGTERM when the parent dies (pipe EOF, even on its SIGKILL) so no
      child is orphaned (main -> serve_env and broker -> worker are both armed this way)."""

    def on_sigterm(*_) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    if death_pipe is None:
        return

    def _watch() -> None:
        with contextlib.suppress(Exception):
            death_pipe.recv()
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_watch, daemon=True).start()


class EnvServerPool:
    """ROUTER broker that elastically scales worker processes (least-busy dispatch).

    With `elastic=True` (default) it starts with a single worker and spawns another
    whenever in-flight requests reach 90% of current capacity (`workers * multiplex`), up
    to `max_workers`. Upscale-only for now — workers are never reclaimed. `elastic=False`
    pre-spawns all `max_workers` upfront (fixed-pool behavior)."""

    def __init__(
        self,
        server_kwargs: dict,
        max_workers: int | None,
        address: str,
        log_setup: Callable[[], None] | None = None,
        multiplex: int = 128,
        elastic: bool = True,
    ) -> None:
        self.server_kwargs = server_kwargs
        self.max_workers = max_workers
        self.multiplex = multiplex
        self.elastic = elastic
        self.log_setup = log_setup
        self._ipc_dir = tempfile.mkdtemp(prefix="vf-pool-", dir="/tmp")
        self.workers: list[dict] = []
        self._spawned = 0
        """Workers spawned so far: the next one's index (a dead one's is never reused)."""
        self._mpctx = mp.get_context("spawn")
        self._poller: zmq.asyncio.Poller | None = None

        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(address)
        self.address = self.frontend.getsockopt_string(zmq.LAST_ENDPOINT)

    def _worker_path(self, i: int) -> str:
        return f"{self._ipc_dir}/{i}"

    def _spawn_worker(self) -> None:
        i = self._spawned
        self._spawned += 1
        address = f"ipc://{self._worker_path(i)}"
        parent_conn, child_conn = self._mpctx.Pipe()
        proc = self._mpctx.Process(
            target=serve_env,
            kwargs=dict(
                max_workers=1,
                address=address,
                death_pipe=child_conn,
                log_setup=self.log_setup,
                **self.server_kwargs,
            ),
            daemon=False,
        )
        proc.start()
        child_conn.close()  # parent keeps the write end (its close signals death)
        dealer = self.ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.connect(address)  # connect before bind is fine — ZMQ queues
        self.workers.append(
            {
                "process": proc,
                "dealer": dealer,
                "pipe": parent_conn,
                "active": 0,
                "index": i,
            }
        )
        if self._poller is not None:
            self._poller.register(dealer, zmq.POLLIN)
            # Readable once the process exits.
            self._poller.register(proc.sentinel, zmq.POLLIN)

    def _maybe_scale_up(self, in_flight: int) -> None:
        """Spawn one more worker when in-flight rollout slots reach 90% of capacity.

        A new worker starts at `active=0`, so least-busy dispatch funnels the backlog to
        it as it comes online (a few seconds to load the env) — fine, since we only scale
        up once already saturated. `max_workers=None` scales without a cap."""
        if self.max_workers is not None and len(self.workers) >= self.max_workers:
            return
        if in_flight >= 0.9 * len(self.workers) * self.multiplex:
            self._spawn_worker()
            logger.info(
                "EnvServerPool scaled up to %d/%s workers (in_flight=%d)",
                len(self.workers),
                self._cap_str,
                in_flight,
            )

    @property
    def _cap_str(self) -> str:
        return "inf" if self.max_workers is None else str(self.max_workers)

    async def run(self) -> None:
        self._poller = zmq.asyncio.Poller()
        self._poller.register(self.frontend, zmq.POLLIN)
        # Elastic: start with one and scale up on demand. Otherwise pre-spawn the lot
        # (`max_workers` is a concrete count when elastic is off).
        for _ in range(1 if self.elastic else (self.max_workers or 1)):
            self._spawn_worker()
        # request_id -> {client_id, worker}
        pending: dict[bytes, dict] = {}
        # A steer asks every worker (each holds its own rollouts' registry): its
        # per-worker part ids (`_steer_part`) -> the client's request id, and per
        # request the client, the parts still out, and whether it was answered.
        steer_parts: dict[bytes, bytes] = {}
        steers: dict[bytes, dict] = {}
        logger.info(
            "EnvServerPool up: address=%s workers=%d/%s multiplex=%d elastic=%s",
            self.address,
            len(self.workers),
            self._cap_str,
            self.multiplex,
            self.elastic,
        )
        try:
            in_flight = 0
            while True:
                events = dict(await self._poller.poll())
                if self.frontend in events:
                    frames = await self.frontend.recv_multipart()
                    # Drop malformed requests without skipping ready worker replies.
                    if len(frames) != 4:
                        logger.warning(
                            "frontend: dropping malformed request (%d frames, expected 4)",
                            len(frames),
                        )
                    else:
                        client_id, request_id, method, payload = frames
                        if method == b"health":
                            await self.frontend.send_multipart(
                                [client_id, request_id, b"reply", _HEALTH]
                            )
                        elif method == b"steer":
                            steers[request_id] = {
                                "client_id": client_id,
                                "waiting": len(self.workers),
                                "answered": False,
                            }
                            for w in self.workers:
                                part = _steer_part(request_id, w)
                                steer_parts[part] = request_id
                                await w["dealer"].send_multipart(
                                    [part, method, payload]
                                )
                        elif method == b"cancel":
                            # Route the cancel to the worker holding the target run;
                            # an unknown/finished target (or an unparseable payload
                            # — one bad frame must not tear down the broker) is
                            # answered inline
                            try:
                                body = msgpack.unpackb(payload, raw=False)
                                target = str(body.get("request_id", "")).encode()
                            except Exception:  # noqa: BLE001 - one bad frame must not kill the broker
                                target = b""
                            target_entry = pending.get(target)
                            if target_entry is None:
                                await self.frontend.send_multipart(
                                    [client_id, request_id, b"reply", _CANCEL_MISS]
                                )
                            else:
                                worker = target_entry["worker"]
                                worker["active"] += 1
                                pending[request_id] = {
                                    "client_id": client_id,
                                    "worker": worker,
                                }
                                in_flight += 1
                                await worker["dealer"].send_multipart(
                                    [request_id, method, payload]
                                )
                        else:
                            worker = min(self.workers, key=lambda w: w["active"])
                            worker["active"] += 1
                            pending[request_id] = {
                                "client_id": client_id,
                                "worker": worker,
                            }
                            in_flight += 1
                            # forward without client_id — the DEALER identity is the worker's
                            # `client_id`; we hold the real one in `pending`.
                            await worker["dealer"].send_multipart(
                                [request_id, method, payload]
                            )
                            if self.elastic:
                                self._maybe_scale_up(in_flight)
                for w in self.workers:
                    if w["dealer"] in events:
                        frames = await w["dealer"].recv_multipart(copy=False)
                        if await self._relay(frames, pending, steer_parts, steers):
                            in_flight -= 1
                for w in list(self.workers):
                    if w["process"].sentinel in events:
                        in_flight -= await self._drop_worker(
                            w, pending, steer_parts, steers
                        )
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            self._shutdown()

    async def _relay(
        self,
        frames: list,
        pending: dict[bytes, dict],
        steer_parts: dict[bytes, bytes],
        steers: dict[bytes, dict],
    ) -> bool:
        """Route one worker message to its client. Returns whether it closed a request
        out (its `reply`; a `delta` is one of many for its request)."""
        request_id, kind, data = frames
        parent = steer_parts.pop(request_id.bytes, None)
        if parent is not None:
            await self._answer_steer(steers, parent, data.bytes)
            return False
        # Copy only the routing key; relay the Frames unchanged.
        closed = kind.bytes == b"reply"
        if closed:
            entry = pending.pop(request_id.bytes, None)
            if entry is not None:
                entry["worker"]["active"] -= 1
        else:
            entry = pending.get(request_id.bytes)
        if entry is None:
            return False
        with contextlib.suppress(zmq.ZMQError):
            await self.frontend.send_multipart(
                [entry["client_id"], request_id, kind, data], copy=False
            )
        return closed

    async def _drop_worker(
        self,
        w: dict,
        pending: dict[bytes, dict],
        steer_parts: dict[bytes, bytes],
        steers: dict[bytes, dict],
    ) -> int:
        """A worker's process died: relay what it sent before it did, answer what it
        still held (its requests as failed, its share of a steer as not queued), and
        route nothing to it again. Returns how many requests that closed out. Raises
        when no worker is left."""
        closed = 0
        while True:
            try:
                frames = await w["dealer"].recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                break
            closed += await self._relay(frames, pending, steer_parts, steers)
        w["process"].join()  # exited already: its sentinel is ready
        logger.error(
            "EnvServerPool: worker %d died (exit code %s)",
            w["index"],
            w["process"].exitcode,
        )
        for part, parent in list(steer_parts.items()):
            if part == _steer_part(parent, w):
                del steer_parts[part]
                await self._answer_steer(steers, parent, _STEER_LOST)
        for request_id, entry in list(pending.items()):
            if entry["worker"] is w:
                del pending[request_id]
                closed += 1
                with contextlib.suppress(zmq.ZMQError):
                    await self.frontend.send_multipart(
                        [entry["client_id"], request_id, b"reply", _LOST]
                    )
        self.workers.remove(w)
        if self._poller is not None:
            self._poller.unregister(w["dealer"])
            self._poller.unregister(w["process"].sentinel)
        w["dealer"].close()
        with contextlib.suppress(OSError):
            os.unlink(self._worker_path(w["index"]))
        if not self.workers:
            raise RuntimeError("EnvServerPool: every env worker died")
        return closed

    async def _answer_steer(
        self, steers: dict[bytes, dict], request_id: bytes, data: bytes
    ) -> None:
        """One worker's answer to a steer: the first that queued it answers the
        client, else the last one in."""
        group = steers[request_id]
        group["waiting"] -= 1
        if group["waiting"] == 0:
            del steers[request_id]
        if group["answered"]:
            return
        try:
            queued = bool(msgpack.unpackb(data, raw=False).get("queued"))
        except Exception:  # noqa: BLE001 - one bad frame must not kill the broker
            queued = False
        if queued or group["waiting"] == 0:
            group["answered"] = True
            with contextlib.suppress(zmq.ZMQError):
                await self.frontend.send_multipart(
                    [group["client_id"], request_id, b"reply", data]
                )

    def _shutdown(self) -> None:
        for w in self.workers:
            with contextlib.suppress(Exception):
                w["pipe"].close()
            with contextlib.suppress(Exception):
                w["process"].terminate()
        for w in self.workers:
            with contextlib.suppress(Exception):
                w["process"].join(timeout=10)
            if w["process"].is_alive():
                with contextlib.suppress(Exception):
                    w["process"].kill()
            with contextlib.suppress(Exception):
                w["dealer"].close()
            with contextlib.suppress(OSError):
                os.unlink(self._worker_path(w["index"]))
        with contextlib.suppress(OSError):
            os.rmdir(self._ipc_dir)
        self.frontend.close()
        self.ctx.term()
        logger.info("EnvServerPool down")


def env_config_data(env: EnvConfig) -> dict:
    """A picklable dict of a (possibly dynamically-narrowed, unpicklable) env config —
    ship this across a process boundary, then rebuild via `resolve_env_config`
    (re-narrowing to the env's concrete config class)."""
    return env.model_dump(mode="json")


def _steer_part(request_id: bytes, worker: dict) -> bytes:
    """The id of one worker's share of a steer (`EnvServerPool.run`)."""
    return request_id + b"/" + str(worker["index"]).encode()


def serve_env(
    *,
    max_workers: int | None,
    address: str | None = None,
    address_queue=None,
    death_pipe=None,
    log_setup: Callable[[], None] | None = None,
    multiplex: int = 128,
    elastic: bool = True,
    **server_kwargs,
) -> None:
    """Serve one env over ZMQ: a single in-process `EnvServer` when `max_workers <= 1`,
    else an `EnvServerPool` broker over up to `max_workers` worker processes (`None` =
    unbounded). The frontend speaks the same protocol either way, so the client is
    identical. Reports the bound address on `address_queue` (for a spawner that passed an
    OS-assigned `:0`). `address` None binds the default loopback `tcp://127.0.0.1:5000`.

    `elastic` (default True) starts the pool at one worker and scales up to `max_workers`
    as load grows; `multiplex` is the per-worker capacity for the scale-up trigger (spawn
    the next worker at 90% of `workers * multiplex` in-flight). `elastic=False` pre-spawns
    all `max_workers`.

    The env config may be passed as `config` (an object) or `config_data` (the
    picklable dict from `env_config_data`, for callers that spawn this function and so
    can't pickle a dynamically-narrowed config type).

    `log_setup` (a picklable callable) configures logging for this process and every
    spawned worker — without it a spawned server inherits no handlers and its INFO logs
    (rollout start/done, the pool line) are silently dropped.

    `death_pipe` (when spawned by a parent, e.g. the eval main process) makes this server
    self-terminate if that parent dies abruptly — see `_arm_teardown`."""
    # Graceful SIGTERM (run asyncio teardown) + self-terminate if the parent dies. The
    # re-raised KeyboardInterrupt is swallowed below for a clean exit (no spurious traceback).
    _arm_teardown(death_pipe)
    if log_setup is not None:
        log_setup()
    address = address or "tcp://127.0.0.1:5000"
    try:
        if max_workers is None or max_workers > 1:
            if (
                "config" in server_kwargs
            ):  # dict-ify for the workers (config_data is picklable)
                server_kwargs = {**server_kwargs}
                server_kwargs["config_data"] = env_config_data(
                    server_kwargs.pop("config")
                )
            pool = EnvServerPool(
                server_kwargs,
                max_workers,
                address,
                log_setup,
                multiplex,
                elastic,
            )
            if address_queue is not None:
                address_queue.put(pool.address)
            asyncio.run(pool.run())
        else:
            if (
                "config_data" in server_kwargs
            ):  # rebuild the env config for an in-process server
                from verifiers.v1.utils.loaders import resolve_env_config

                server_kwargs = {**server_kwargs}
                server_kwargs["config"] = resolve_env_config(
                    server_kwargs.pop("config_data")
                )
            EnvServer.run_server(
                address=address, address_queue=address_queue, **server_kwargs
            )
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Env server failed")
        raise
