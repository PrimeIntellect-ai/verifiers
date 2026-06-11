"""Env-server worker pool: a ROUTER broker over N worker processes.

A lone `EnvServer` runs every rollout as an `asyncio.Task` on one event loop, so
CPU-bound work (renderer tokenization, scoring) competes for that loop. v0 relieved
this with a router + worker pool; this reinstates it for v1.

A broker binds the client-facing ROUTER (the *same* wire protocol as a lone
`EnvServer`, so `EnvClient` is unchanged), starts one worker process and scales up to
`max_workers` on demand — each an ordinary `EnvServer` / `LegacyEnvServer` bound to its
own ipc address — load-balancing requests to the least-busy worker over a `DEALER` per
worker. The worker's `client_id` (its reply identity) is the broker's DEALER identity;
the broker holds the real client identity in `pending` and routes the reply back by
`request_id`. `health` is answered inline (no worker needed); everything else goes to a
worker.

Scaling is elastic but upscale-only: a new worker is spawned when in-flight requests
reach 90% of current capacity (`workers * multiplex`). Workers are spawned `spawn`-style
(own env, own loop) and monitor a death pipe so an orphaned worker self-exits if the
broker dies. TODO: downscale idle workers, per-worker restart-on-death, stats/lag
monitors (v0 had them; omitted here — rollout errors are returned as data, not crashes,
so worker death is rare).
"""

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import signal
import threading
import uuid
from collections.abc import Callable

import msgpack
import zmq
import zmq.asyncio

from verifiers.v1.env import EnvConfig
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import HealthResponse

logger = logging.getLogger(__name__)

_HEALTH = msgpack.packb(HealthResponse().model_dump(mode="json"), use_bin_type=True)


def _monitor_parent(conn) -> None:
    """Self-SIGTERM when the parent (broker) dies: its write end of the pipe closes on
    exit (even on SIGKILL), so the blocking `recv` returns/raises and we shut down —
    no orphaned worker holding a sandbox."""

    def watch() -> None:
        with contextlib.suppress(Exception):
            conn.recv()
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=watch, daemon=True).start()


def _worker_entry(
    *, server_kwargs: dict, address: str, death_pipe, legacy: bool, log_setup=None
) -> None:
    """Spawned worker: an ordinary EnvServer/LegacyEnvServer bound to `address` (ipc).
    A native config arrives as a dict (`config_data`): the eval/serve CLI's narrowed
    config type is dynamic and unpicklable, so we rebuild it here via EnvConfig's
    id-resolving validator. `log_setup` (if given) configures this fresh process's
    logging so per-rollout logs surface — a spawned worker inherits no handlers."""
    from verifiers.v1.legacy import LegacyEnvServer

    if log_setup is not None:
        log_setup()
    _monitor_parent(death_pipe)
    if "config_data" in server_kwargs:
        server_kwargs = {
            "config": EnvConfig.model_validate(server_kwargs["config_data"])
        }
    cls = LegacyEnvServer if legacy else EnvServer
    cls.run_server(address=address, **server_kwargs)


class EnvServerPool:
    """ROUTER broker that elastically scales worker processes (least-busy dispatch).

    Starts with a single worker and spawns another whenever in-flight requests reach
    90% of current capacity (`workers * multiplex`), up to `max_workers`. Upscale-only
    for now — workers are never reclaimed. The broker forwards opaque request frames, so
    workers can be `EnvServer` (v1) or `LegacyEnvServer` (v0) without the broker caring."""

    def __init__(
        self,
        server_kwargs: dict,
        max_workers: int,
        address: str,
        legacy: bool,
        log_setup: Callable[[], None] | None = None,
        multiplex: int = 128,
    ) -> None:
        self.server_kwargs = server_kwargs
        self.max_workers = max_workers
        self.multiplex = multiplex
        self.legacy = legacy
        self.log_setup = log_setup
        self.session = uuid.uuid4().hex[:12]
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
        return f"/tmp/vf-pool-{self.session}-{i}"

    def _spawn_worker(self) -> None:
        i = len(self.workers)  # upscale-only, so the next index is the current count
        address = f"ipc://{self._worker_path(i)}"
        parent_conn, child_conn = self._mpctx.Pipe()
        proc = self._mpctx.Process(
            target=_worker_entry,
            kwargs=dict(
                server_kwargs=self.server_kwargs,
                address=address,
                death_pipe=child_conn,
                legacy=self.legacy,
                log_setup=self.log_setup,
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
        """Spawn one more worker when in-flight requests reach 90% of current capacity.

        A new worker starts at `active=0`, so least-busy dispatch funnels the backlog to
        it as it comes online (a few seconds to load the env) — fine, since we only scale
        up once already saturated."""
        if len(self.workers) >= self.max_workers:
            return
        if in_flight >= 0.9 * len(self.workers) * self.multiplex:
            self._spawn_worker()
            logger.info(
                "EnvServerPool scaled up to %d/%d workers (in_flight=%d)",
                len(self.workers),
                self.max_workers,
                in_flight,
            )

    async def run(self) -> None:
        self._poller = zmq.asyncio.Poller()
        self._poller.register(self.frontend, zmq.POLLIN)
        self._spawn_worker()  # start with one; scale up on demand
        pending: dict[bytes, dict] = {}  # request_id -> {client_id, worker}
        logger.info(
            "EnvServerPool up: address=%s multiplex=%d max_workers=%d",
            self.address,
            self.multiplex,
            self.max_workers,
        )
        try:
            while True:
                events = dict(await self._poller.poll())
                if self.frontend in events:
                    (
                        client_id,
                        request_id,
                        method,
                        payload,
                    ) = await self.frontend.recv_multipart()
                    if method == b"health":
                        await self.frontend.send_multipart(
                            [client_id, request_id, _HEALTH]
                        )
                    else:
                        worker = min(self.workers, key=lambda w: w["active"])
                        worker["active"] += 1
                        pending[request_id] = {"client_id": client_id, "worker": worker}
                        # forward without client_id — the DEALER identity is the worker's
                        # `client_id`; we hold the real one in `pending`.
                        await worker["dealer"].send_multipart(
                            [request_id, method, payload]
                        )
                        self._maybe_scale_up(len(pending))
                for w in self.workers:
                    if w["dealer"] in events:
                        request_id, data = await w["dealer"].recv_multipart()
                        entry = pending.pop(request_id, None)
                        if entry is None:
                            continue
                        entry["worker"]["active"] -= 1
                        with contextlib.suppress(zmq.ZMQError):
                            await self.frontend.send_multipart(
                                [entry["client_id"], request_id, data]
                            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            self._shutdown()

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
        self.frontend.close()
        self.ctx.term()
        logger.info("EnvServerPool down")


def env_config_data(config) -> dict:
    """The picklable `EnvConfig` fields of a (possibly dynamically-narrowed, unpicklable)
    config object — ship this across a process boundary, then rebuild via
    `EnvConfig.model_validate` (its validator re-resolves the concrete taskset/harness)."""
    data = config.model_dump(mode="json")
    return {k: v for k, v in data.items() if k in EnvConfig.model_fields}


def serve_env(
    *,
    num_workers: int,
    legacy: bool = False,
    address: str = "tcp://127.0.0.1:5000",
    address_queue=None,
    log_setup: Callable[[], None] | None = None,
    multiplex: int = 128,
    **server_kwargs,
) -> None:
    """Serve one env over ZMQ: a single in-process `EnvServer` when `num_workers <= 1`,
    else an `EnvServerPool` broker that scales from one worker up to `num_workers`. The
    frontend speaks the same protocol either way, so the client is identical. Reports the
    bound address on `address_queue` (for a spawner that passed an OS-assigned `:0`).

    `multiplex` is the per-worker capacity used by the pool's scale-up trigger (it spawns
    the next worker at 90% of `workers * multiplex` in-flight).

    A native env config may be passed as `config` (an object) or `config_data` (the
    picklable dict from `env_config_data`, for callers that spawn this function and so
    can't pickle a dynamically-narrowed config type); legacy passes `env_id`/`env_args`/
    `extra_env_kwargs`.

    `log_setup` (a picklable callable) configures logging for this process and every
    spawned worker — without it a spawned server inherits no handlers and its INFO logs
    (rollout start/done, the pool line) are silently dropped."""
    # SIGTERM -> KeyboardInterrupt so a killed server runs its teardown (pool: kill the
    # workers; single: close clients). It can fire inside a C call (e.g. zmq getsockopt),
    # raising in the loop machinery rather than at an `await`, so it escapes the server's
    # own run loop; `asyncio.run` still tears the server down via task cancellation, and we
    # swallow the re-raised KeyboardInterrupt here for a clean exit (no spurious traceback).
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if log_setup is not None:
        log_setup()
    try:
        if num_workers > 1:
            if (
                "config" in server_kwargs
            ):  # dict-ify for the workers (config_data is picklable)
                server_kwargs = {
                    "config_data": env_config_data(server_kwargs["config"])
                }
            pool = EnvServerPool(
                server_kwargs, num_workers, address, legacy, log_setup, multiplex
            )
            if address_queue is not None:
                address_queue.put(pool.address)
            asyncio.run(pool.run())
        else:
            from verifiers.v1.legacy import LegacyEnvServer

            if (
                "config_data" in server_kwargs
            ):  # rebuild the env config for an in-process server
                server_kwargs = {
                    "config": EnvConfig.model_validate(server_kwargs["config_data"])
                }
            cls = LegacyEnvServer if legacy else EnvServer
            cls.run_server(
                address=address, address_queue=address_queue, **server_kwargs
            )
    except KeyboardInterrupt:
        pass
