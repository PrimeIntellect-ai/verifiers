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
broker dies. A worker that exits is noticed by the liveness sweep: every request still
pending on it is answered with an error (so no client waits on a silent peer) and, once
no worker is left, `health` reports unhealthy and the broker exits. TODO: downscale idle
workers, per-worker restart-on-death, stats/lag monitors (v0 had them; omitted here —
rollout errors are returned as data, not crashes, so worker death is rare).
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
from verifiers.v1.serve.types import BaseResponse, CancelResponse, HealthResponse

logger = logging.getLogger(__name__)

_HEALTH = msgpack.packb(HealthResponse().model_dump(mode="json"), use_bin_type=True)
_HEALTH_DOWN = msgpack.packb(
    HealthResponse(success=False, error="no live env worker").model_dump(mode="json"),
    use_bin_type=True,
)
_CANCEL_MISS = msgpack.packb(
    CancelResponse(cancelled=False).model_dump(mode="json"), use_bin_type=True
)

# A request never carries a deadline of its own (a healthy rollout has no default
# duration), so the poll loop wakes on this interval purely to notice a worker that
# died while requests were in flight on it. Without that sweep a run dispatched to a
# dead worker's DEALER is queued by ZMQ and pends forever.
_LIVENESS_INTERVAL_MS = 500


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
        i = len(self.workers)  # upscale-only, so the next index is the current count
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
                events = dict(await self._poller.poll(_LIVENESS_INTERVAL_MS))
                if self._reap_dead_workers():
                    in_flight = await self._fail_pending(pending, in_flight)
                if not any(w["process"].is_alive() for w in self.workers):
                    # Nothing can serve a request any more. In-flight requests were just
                    # failed by the sweep; let a health probe learn the same thing instead
                    # of being told everything is fine, then exit. Error propagation must
                    # precede shutdown — an exit alone leaves the untimed DEALER client
                    # waiting on a peer that will never reply.
                    if self.frontend in events:
                        frames = await self.frontend.recv_multipart()
                        if len(frames) == 4:
                            client_id, request_id, method, _ = frames
                            if method == b"health":
                                await self.frontend.send_multipart(
                                    [client_id, request_id, b"reply", _HEALTH_DOWN]
                                )
                    logger.error("EnvServerPool has no live workers left; exiting")
                    return
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
                            # Workers are alive at this point, so a run can be served.
                            await self.frontend.send_multipart(
                                [client_id, request_id, b"reply", _HEALTH]
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
                                    "method": method,
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
                                "method": method,
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
                        request_id, kind, data = await w["dealer"].recv_multipart(
                            copy=False
                        )
                        # Copy only the routing key; relay the Frames unchanged. A
                        # `delta` is one of many for its request — only the `reply`
                        # closes the request out.
                        if kind.bytes == b"reply":
                            entry = pending.pop(request_id.bytes, None)
                            if entry is None:
                                continue
                            entry["worker"]["active"] -= 1
                            in_flight -= 1
                        else:
                            entry = pending.get(request_id.bytes)
                            if entry is None:
                                continue
                        with contextlib.suppress(zmq.ZMQError):
                            await self.frontend.send_multipart(
                                [entry["client_id"], request_id, kind, data], copy=False
                            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            self._shutdown()

    def _reap_dead_workers(self) -> bool:
        """Drop workers whose process exited (a crash inside `serving()`, a task
        failure at startup) and report whether any died. Their DEALER is closed and
        unregistered so the poller stops watching it; the caller answers the requests
        still pending on them."""
        live, dead = [], []
        for worker in self.workers:
            (live if worker["process"].is_alive() else dead).append(worker)
        if not dead:
            return False
        for worker in dead:
            logger.error(
                "EnvServerPool worker %d exited with code %s; failing its in-flight work",
                worker["index"],
                worker["process"].exitcode,
            )
            if self._poller is not None:
                with contextlib.suppress(KeyError):
                    self._poller.unregister(worker["dealer"])
            with contextlib.suppress(Exception):
                worker["dealer"].close()
            with contextlib.suppress(Exception):
                worker["pipe"].close()
        self.workers = live
        return True

    async def _fail_pending(self, pending: dict[bytes, dict], in_flight: int) -> int:
        """Close out every request still pending on a dead worker. A `run` gets an error
        reply, so the client raises naming the worker instead of awaiting an answer that
        can never come; a `cancel` is a fire-and-forget notice, so it keeps its own
        `cancelled=False` meaning (`missed`) rather than a failure its caller cannot act
        on. `health` is answered by the caller; without this a request dispatched to a
        dead DEALER is queued by ZMQ and pends forever."""
        live = {id(worker) for worker in self.workers}
        for request_id, entry in list(pending.items()):
            if id(entry["worker"]) in live:
                continue
            pending.pop(request_id)
            in_flight -= 1
            worker = entry["worker"]
            if entry["method"] == b"cancel":
                answer = CancelResponse(cancelled=False)
            else:
                answer = BaseResponse(
                    success=False,
                    error=(
                        f"env worker {worker['index']} exited with code "
                        f"{worker['process'].exitcode}; the request was not served"
                    ),
                )
            data = msgpack.packb(answer.model_dump(mode="json"), use_bin_type=True)
            with contextlib.suppress(zmq.ZMQError):
                await self.frontend.send_multipart(
                    [entry["client_id"], request_id, b"reply", data]
                )
        return in_flight

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
            if os.path.exists(self._worker_path(w["index"])):
                # A dead worker is dropped from `self.workers`, so a later spawn can
                # reuse its index; unlink only while the path still exists.
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
