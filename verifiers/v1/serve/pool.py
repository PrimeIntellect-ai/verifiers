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
broker dies. TODO: downscale idle workers, stats/lag monitors (v0 had them).

A worker that dies is reaped: `EnvClient` awaits rollout replies untimed, so an
unanswered request is an unbounded hang, and a dead worker's `active` count never drains
— least-busy dispatch would elect the corpse for every subsequent request. The broker
therefore polls its workers' exit codes, answers their in-flight requests with an error,
and replaces them (up to `MAX_WORKER_RESTARTS`). Readiness is a worker handshake, not a
socket bind: the broker reports its address only once a worker answers `health`, so a
spawner is never told "ready" about a pool whose env failed to load.
"""

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import uuid
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from pathlib import Path

import msgpack
import zmq
import zmq.asyncio

from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import BaseResponse, HealthResponse, RunGroupRequest

logger = logging.getLogger(__name__)

_HEALTH = msgpack.packb(HealthResponse().model_dump(mode="json"), use_bin_type=True)

MAX_WORKER_RESTARTS = 3
"""Replacements the broker spawns over its lifetime: enough to ride out a one-off crash
(an OOM, a segfaulting sandbox), few enough that an env dying on every load fails loudly
instead of respawning forever."""

ENV_SERVER_SPAWN_TIMEOUT = 600.0
"""Default deadline for a spawned server to report its address. Generous: it reports only
once its env is loaded, which for a legacy bridge includes pulling a dataset."""

# Broker loop wake-up, so exited workers are reaped even while no socket is readable.
_REAP_INTERVAL_S = 0.25
_REAP_INTERVAL_MS = int(_REAP_INTERVAL_S * 1000)
# Startup handshake: a `health` request whose reply proves a worker loaded its env.
_STARTUP_PROBE = b"startup"
_EMPTY_PAYLOAD = msgpack.packb({}, use_bin_type=True)


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
    to `max_workers`. Upscale-only for now — a live worker is never reclaimed, though a dead
    one is (see the module docstring). `elastic=False` pre-spawns all `max_workers` upfront
    (the old fixed-pool behavior). The broker forwards opaque request frames, so workers can
    be `EnvServer` (v1) or `LegacyEnvServer` (v0) without the broker caring."""

    def __init__(
        self,
        server_kwargs: dict,
        max_workers: int | None,
        address: str,
        legacy: bool,
        log_setup: Callable[[], None] | None = None,
        multiplex: int = 128,
        elastic: bool = True,
        address_queue=None,
    ) -> None:
        self.server_kwargs = server_kwargs
        self.max_workers = max_workers
        self.multiplex = multiplex
        self.elastic = elastic
        self.legacy = legacy
        self.log_setup = log_setup
        self.address_queue = address_queue
        self.session = uuid.uuid4().hex[:12]
        self.workers: list[dict] = []
        # request_id -> {client_id, worker, rollout_slots}
        self.pending: dict[bytes, dict] = {}
        self.in_flight = 0
        self._mpctx = mp.get_context("spawn")
        self._poller: zmq.asyncio.Poller | None = None
        self._next_index = 0
        self._restarts = 0
        self._last_reap = 0.0

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
        i = self._next_index  # monotonic: a retired worker's path is never reused
        self._next_index += 1
        address = f"ipc://{self._worker_path(i)}"
        parent_conn, child_conn = self._mpctx.Pipe()
        proc = self._mpctx.Process(
            target=serve_env,
            kwargs=dict(
                max_workers=1,
                address=address,
                death_pipe=child_conn,
                legacy=self.legacy,
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

    def _try_spawn_worker(self) -> bool:
        """Spawn a worker mid-serve, where failing to spawn must not be fatal: this process
        holds every client's in-flight requests, so dying here strands all of them. A pool
        one worker short still serves. (At startup the opposite holds — a pool with no
        workers has nothing to serve — so `run` lets that failure propagate.)"""
        try:
            self._spawn_worker()
            return True
        except Exception:
            logger.exception("EnvServerPool failed to spawn a worker")
            return False

    def _maybe_scale_up(self) -> None:
        """Spawn one more worker when in-flight rollout slots reach 90% of capacity.

        A new worker starts at `active=0`, so least-busy dispatch funnels the backlog to
        it as it comes online (a few seconds to load the env) — fine, since we only scale
        up once already saturated. `max_workers=None` scales without a cap."""
        if self.max_workers is not None and len(self.workers) >= self.max_workers:
            return
        if (
            self.in_flight >= 0.9 * len(self.workers) * self.multiplex
            and self._try_spawn_worker()
        ):
            logger.info(
                "EnvServerPool scaled up to %d/%s workers (in_flight=%d)",
                len(self.workers),
                self._cap_str,
                self.in_flight,
            )

    @property
    def _cap_str(self) -> str:
        return "inf" if self.max_workers is None else str(self.max_workers)

    async def _wait_for_worker(self, w: dict) -> None:
        """Block until a worker answers `health`.

        A worker binds its socket before loading the env, so its first reply — not its
        bind — is what proves the env loaded. Untimed on purpose: a slow load (a legacy
        dataset pull) is the spawner's deadline to enforce, not ours. We only insist that
        the process is still alive."""
        dealer = w["dealer"]
        await dealer.send_multipart([_STARTUP_PROBE, b"health", _EMPTY_PAYLOAD])
        while not await dealer.poll(timeout=_REAP_INTERVAL_MS):
            if w["process"].exitcode is not None:
                raise RuntimeError(
                    f"env server worker {w['index']} died while loading the env "
                    f"(exit code {w['process'].exitcode})"
                )
        await dealer.recv_multipart()

    async def _reply_error(
        self, client_id: bytes, request_id: bytes, error: str
    ) -> None:
        """Answer a request with a failure the client raises on. `EnvClient` awaits
        rollout replies untimed, so leaving one unanswered is an unbounded hang."""
        data = msgpack.packb(
            BaseResponse(success=False, error=error).model_dump(mode="json"),
            use_bin_type=True,
        )
        with contextlib.suppress(zmq.ZMQError):
            await self.frontend.send_multipart([client_id, request_id, data])

    def _retire(self, w: dict) -> None:
        """Drop a worker from dispatch and release everything it held."""
        self.workers.remove(w)
        if self._poller is not None:
            self._poller.unregister(w["dealer"])
        with contextlib.suppress(Exception):
            w["dealer"].close()
        with contextlib.suppress(Exception):
            w["pipe"].close()
        with contextlib.suppress(OSError):
            os.unlink(self._worker_path(w["index"]))

    def _replace_worker(self) -> None:
        """Spawn a replacement for a dead worker, within the restart budget."""
        if self._restarts >= MAX_WORKER_RESTARTS:
            logger.error(
                "EnvServerPool exhausted its %d worker restart(s) — not replacing",
                MAX_WORKER_RESTARTS,
            )
            return
        self._restarts += 1  # counts attempts, so a failing spawn can't loop hot
        if self._try_spawn_worker():
            logger.warning(
                "EnvServerPool replaced a dead worker (restart %d/%d)",
                self._restarts,
                MAX_WORKER_RESTARTS,
            )

    async def _reap_dead_workers(self) -> None:
        """Retire exited workers, fail their in-flight requests, and replace them.

        Rate-limited: reading `exitcode` is a waitpid per worker, and the broker loop turns
        once per request."""
        now = time.monotonic()
        if now - self._last_reap < _REAP_INTERVAL_S:
            return
        self._last_reap = now
        for w in [w for w in self.workers if w["process"].exitcode is not None]:
            exitcode = w["process"].exitcode
            # Relay what it already answered before closing its socket: a worker can queue
            # several replies and then exit, and retiring it first would discard them —
            # reporting finished rollouts as deaths.
            await self._drain_replies(w)
            self._retire(w)
            orphaned = [
                rid for rid, entry in self.pending.items() if entry["worker"] is w
            ]
            logger.error(
                "EnvServerPool worker %d died (exit code %s) — failing %d in-flight request(s)",
                w["index"],
                exitcode,
                len(orphaned),
            )
            for request_id in orphaned:
                entry = self.pending.pop(request_id)
                self.in_flight -= entry["rollout_slots"]
                await self._reply_error(
                    entry["client_id"],
                    request_id,
                    f"env server worker died (exit code {exitcode})",
                )
            self._replace_worker()

    async def _on_request(
        self, client_id: bytes, request_id: bytes, method: bytes, payload: bytes
    ) -> None:
        """Answer `health` inline; dispatch everything else to the least-busy worker."""
        if method == b"health":
            await self.frontend.send_multipart([client_id, request_id, _HEALTH])
            return
        if not self.workers:
            await self._reply_error(
                client_id, request_id, "env server has no live workers"
            )
            return
        # Pool capacity is measured in rollouts; one group request carries n.
        rollout_slots = 1
        if method == b"run_group":
            with contextlib.suppress(Exception):
                request = RunGroupRequest.model_validate(
                    msgpack.unpackb(payload, raw=False)
                )
                rollout_slots = max(1, request.n)
        worker = min(self.workers, key=lambda w: w["active"])
        worker["active"] += rollout_slots
        self.pending[request_id] = {
            "client_id": client_id,
            "worker": worker,
            "rollout_slots": rollout_slots,
        }
        self.in_flight += rollout_slots
        # forward without client_id — the DEALER identity is the worker's `client_id`;
        # we hold the real one in `pending`.
        await worker["dealer"].send_multipart([request_id, method, payload])
        if self.elastic:
            self._maybe_scale_up()

    async def _drain_replies(self, w: dict) -> None:
        """Relay every reply already queued from this worker, then return. The serve loop
        takes one reply per pass (so a chatty worker can't starve the frontend); a worker
        about to be retired instead has to be emptied, since its socket is closing."""
        while await w["dealer"].poll(timeout=0):
            await self._on_reply(w)

    async def _on_reply(self, w: dict) -> None:
        """Relay one worker reply back to the client that asked for it."""
        request_id, data = await w["dealer"].recv_multipart(copy=False)
        # Copy only the routing key; relay the response Frames unchanged.
        entry = self.pending.pop(request_id.bytes, None)
        if entry is None:
            return
        entry["worker"]["active"] -= entry["rollout_slots"]
        self.in_flight -= entry["rollout_slots"]
        with contextlib.suppress(zmq.ZMQError):
            await self.frontend.send_multipart(
                [entry["client_id"], request_id, data], copy=False
            )

    async def run(self) -> None:
        self._poller = zmq.asyncio.Poller()
        self._poller.register(self.frontend, zmq.POLLIN)
        # Elastic: start with one and scale up on demand. Otherwise pre-spawn the lot
        # (`max_workers` is a concrete count when elastic is off).
        for _ in range(1 if self.elastic else (self.max_workers or 1)):
            self._spawn_worker()
        try:
            # Report the address only once a worker can actually serve. The frontend binds
            # before any env is loaded, so a spawner told "ready" at bind time would connect
            # happily to a pool whose workers are still loading — or already dead.
            for w in list(self.workers):
                await self._wait_for_worker(w)
            if self.address_queue is not None:
                self.address_queue.put(self.address)
            logger.info(
                "EnvServerPool up: address=%s workers=%d/%s multiplex=%d elastic=%s",
                self.address,
                len(self.workers),
                self._cap_str,
                self.multiplex,
                self.elastic,
            )
            while True:
                events = dict(await self._poller.poll(timeout=_REAP_INTERVAL_MS))
                # Drain replies before reaping, so a worker that answered and then died
                # still has its reply relayed instead of failed.
                for w in list(self.workers):
                    if w["dealer"] in events:
                        await self._on_reply(w)
                await self._reap_dead_workers()
                if self.frontend in events:
                    await self._on_request(*await self.frontend.recv_multipart())
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


def env_config_data(env: EnvConfig) -> dict:
    """A picklable dict of a (possibly dynamically-narrowed, unpicklable) env config —
    ship this across a process boundary, then rebuild via `resolve_env_config`
    (re-narrowing to the env's concrete config class)."""
    return env.model_dump(mode="json")


def serve_env(
    *,
    max_workers: int | None,
    legacy: bool = False,
    address: str = "tcp://127.0.0.1:5000",
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
    identical. Reports the served address on `address_queue` (for a spawner that passed an
    OS-assigned `:0`) once the env is loaded and serving — read it with `wait_for_address`,
    which fails fast if this process dies first.

    `elastic` (default True) starts the pool at one worker and scales up to `max_workers`
    as load grows; `multiplex` is the per-worker capacity for the scale-up trigger (spawn
    the next worker at 90% of `workers * multiplex` in-flight). `elastic=False` pre-spawns
    all `max_workers`.

    A native env config may be passed as `config` (an object) or `config_data` (the
    picklable dict from `env_config_data`, for callers that spawn this function and so
    can't pickle a dynamically-narrowed config type); legacy passes `env_id`/`env_args`/
    `extra_env_kwargs`.

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
    try:
        if max_workers is None or max_workers > 1:
            if (
                "config" in server_kwargs
            ):  # dict-ify for the workers (config_data is picklable)
                server_kwargs = {
                    "config_data": env_config_data(server_kwargs["config"])
                }
            pool = EnvServerPool(
                server_kwargs,
                max_workers,
                address,
                legacy,
                log_setup,
                multiplex,
                elastic,
                # The pool reports its own address, once a worker answers `health` — see
                # EnvServerPool.run.
                address_queue=address_queue,
            )
            asyncio.run(pool.run())
        else:
            from verifiers.v1.legacy import LegacyEnvServer

            if (
                "config_data" in server_kwargs
            ):  # rebuild the env config for an in-process server
                from verifiers.v1.loaders import resolve_env_config

                server_kwargs = {
                    "config": resolve_env_config(server_kwargs["config_data"])
                }
            cls = LegacyEnvServer if legacy else EnvServer
            cls.run_server(
                address=address, address_queue=address_queue, **server_kwargs
            )
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Env server failed")
        raise


def _log_tail(log_file: str | Path | None, lines: int = 20) -> str:
    """The tail of a spawned server's log, to quote in a failure message. A spawner that
    redirects the child's output has the real traceback there, so this is what turns "the
    server died" into the actual cause."""
    if log_file is None:
        return ""
    with contextlib.suppress(OSError):
        tail = Path(log_file).read_text(errors="replace").splitlines()[-lines:]
        if tail:
            return "\n".join(["", f"--- tail of {log_file} ---", *tail])
    return ""


async def wait_for_address(
    process: BaseProcess,
    address_queue,
    *,
    name: str = "env server",
    timeout: float = ENV_SERVER_SPAWN_TIMEOUT,
    log_file: str | Path | None = None,
) -> str:
    """Wait for a spawned `serve_env` process to report its address, failing fast if it
    dies first.

    The address arrives only once the env is loaded and serving, so a server that dies on
    startup (a bad import, missing credentials, an OOM) never reports one. Waiting on the
    queue alone would then block for the full `timeout` and blame a timeout for what was a
    crash — so poll the exit code alongside it and raise as soon as the process is gone."""
    deadline = time.monotonic() + timeout
    while True:
        with contextlib.suppress(queue.Empty):
            return await asyncio.to_thread(address_queue.get, timeout=1.0)
        if process.exitcode is not None:
            raise RuntimeError(
                f"{name} died on startup with exit code {process.exitcode}"
                f"{_log_tail(log_file)}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{name} did not report its address within {timeout}s{_log_tail(log_file)}"
            )
