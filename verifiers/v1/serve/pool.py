"""Env-server worker pool: a ROUTER broker over N worker processes.

A lone `EnvServer` runs every rollout on one event loop, so CPU-bound work
(renderer tokenization, scoring) competes for it; the pool spreads that across
worker processes. The broker binds the client-facing ROUTER with the *same* wire
protocol as a lone `EnvServer` (so `EnvClient` is unchanged) and dispatches each
request to the least-busy worker over a per-worker `DEALER`; the real client
identity is held in `pending` and the reply routed back by `request_id`.

Scaling is elastic but upscale-only: a new worker spawns when in-flight requests
reach 90% of `workers * multiplex`. Workers monitor a death pipe so an orphan
self-exits if the broker dies. A worker that dies fails its in-flight requests
back to their clients and leaves dispatch; when every worker is dead the pool
shuts down (clients error instead of hanging). TODO: downscale idle workers,
per-worker restart-on-death.
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
from verifiers.v1.serve.types import BaseResponse, HealthResponse, RunGroupRequest

logger = logging.getLogger(__name__)

_HEALTH = msgpack.packb(HealthResponse().model_dump(mode="json"), use_bin_type=True)


def _error(message: str) -> bytes:
    return msgpack.packb(
        BaseResponse(success=False, error=message).model_dump(mode="json"),
        use_bin_type=True,
    )


def _arm_teardown(death_pipe=None) -> None:
    """Arm a spawned process for clean teardown — by default SIGTERM would kill it
    abruptly, skipping asyncio's serving() cleanup and orphaning tunnels/sandboxes.
    SIGTERM -> KeyboardInterrupt so the event loop runs its finallys; with
    `death_pipe`, self-SIGTERM when the parent dies (pipe EOF, even on SIGKILL) so
    no child is orphaned."""

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
    The broker forwards opaque request frames, so workers can be `EnvServer` (v1)
    or `LegacyEnvServer` (v0) without the broker caring."""

    def __init__(
        self,
        server_kwargs: dict,
        max_workers: int | None,
        address: str,
        legacy: bool,
        log_setup: Callable[[], None] | None = None,
        multiplex: int = 128,
        elastic: bool = True,
    ) -> None:
        self.server_kwargs = server_kwargs
        self.max_workers = max_workers
        self.multiplex = multiplex
        self.elastic = elastic
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
                "dead": False,
            }
        )
        if self._poller is not None:
            self._poller.register(dealer, zmq.POLLIN)

    def _maybe_scale_up(self, in_flight: int) -> None:
        """Spawn one more worker when in-flight rollout slots reach 90% of capacity.
        A new worker starts at `active=0`, so least-busy dispatch funnels the
        backlog to it as it comes online. `max_workers=None` scales without a cap."""
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
        # Elastic starts with one worker; otherwise pre-spawn the lot.
        for _ in range(1 if self.elastic else (self.max_workers or 1)):
            self._spawn_worker()
        # request_id -> {client_id, worker, rollout_slots}
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
                # Bounded poll so worker liveness is checked even when idle.
                events = dict(await self._poller.poll(timeout=1000))
                in_flight -= await self._reap_dead_workers(pending)
                if all(w["dead"] for w in self.workers):
                    logger.error(
                        "EnvServerPool: all %d worker(s) died; shutting down",
                        len(self.workers),
                    )
                    # Give requests racing toward the dead pool a beat to land,
                    # then refuse them, so their clients error instead of hanging.
                    await asyncio.sleep(0.5)
                    await self._refuse_queued_requests()
                    raise RuntimeError(
                        "all env workers died — check the worker logs for the cause"
                    )
                if self.frontend in events:
                    frames = await self.frontend.recv_multipart()
                    if len(frames) != 4:
                        logger.warning(
                            "invalid message: expected 4 frames, got %d", len(frames)
                        )
                        continue
                    client_id, request_id, method, payload = frames
                    if method == b"health":
                        with contextlib.suppress(zmq.ZMQError):
                            await self.frontend.send_multipart(
                                [client_id, request_id, _HEALTH]
                            )
                    else:
                        # Pool capacity is measured in rollouts; one group request carries n.
                        rollout_slots = 1
                        if method == b"run_group":
                            with contextlib.suppress(Exception):
                                request = RunGroupRequest.model_validate(
                                    msgpack.unpackb(payload, raw=False)
                                )
                                rollout_slots = max(1, request.n)
                        worker = min(
                            (w for w in self.workers if not w["dead"]),
                            key=lambda w: w["active"],
                        )
                        worker["active"] += rollout_slots
                        pending[request_id] = {
                            "client_id": client_id,
                            "worker": worker,
                            "rollout_slots": rollout_slots,
                        }
                        in_flight += rollout_slots
                        # forward without client_id — the DEALER identity is the worker's
                        # `client_id`; we hold the real one in `pending`.
                        await worker["dealer"].send_multipart(
                            [request_id, method, payload]
                        )
                        if self.elastic:
                            self._maybe_scale_up(in_flight)
                for w in self.workers:
                    if not w["dead"] and w["dealer"] in events:
                        request_id, data = await w["dealer"].recv_multipart(copy=False)
                        # Copy only the routing key; relay the response Frames unchanged.
                        entry = pending.pop(request_id.bytes, None)
                        if entry is None:
                            continue
                        entry["worker"]["active"] -= entry["rollout_slots"]
                        in_flight -= entry["rollout_slots"]
                        with contextlib.suppress(zmq.ZMQError):
                            await self.frontend.send_multipart(
                                [entry["client_id"], request_id, data], copy=False
                            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            self._shutdown()

    async def _reap_dead_workers(self, pending: dict[bytes, dict]) -> int:
        """Fail a dead worker's in-flight requests back to their clients and drop it
        from dispatch (restart-on-death is deliberately deferred). Replies the worker
        managed to send before dying are relayed, not failed. Returns the rollout
        slots released."""
        released = 0
        for w in self.workers:
            if w["dead"] or w["process"].is_alive():
                continue
            w["dead"] = True
            while await w["dealer"].poll(timeout=0):
                request_id, data = await w["dealer"].recv_multipart(copy=False)
                entry = pending.pop(request_id.bytes, None)
                if entry is None:
                    continue
                released += entry["rollout_slots"]
                with contextlib.suppress(zmq.ZMQError):
                    await self.frontend.send_multipart(
                        [entry["client_id"], request_id, data], copy=False
                    )
            lost = [rid for rid, e in pending.items() if e["worker"] is w]
            error = _error(
                f"env worker {w['index']} died (exit code {w['process'].exitcode}) "
                "with the request in flight — check the worker logs for the crash"
            )
            for request_id in lost:
                entry = pending.pop(request_id)
                released += entry["rollout_slots"]
                with contextlib.suppress(zmq.ZMQError):
                    await self.frontend.send_multipart(
                        [entry["client_id"], request_id, error]
                    )
            logger.error(
                "EnvServerPool: worker %d died (exit code %s); "
                "failed %d in-flight request(s)",
                w["index"],
                w["process"].exitcode,
                len(lost),
            )
            if self._poller is not None:
                with contextlib.suppress(KeyError):
                    self._poller.unregister(w["dealer"])
            w["dealer"].close()
            with contextlib.suppress(Exception):
                w["pipe"].close()
        return released

    async def _refuse_queued_requests(self) -> None:
        """Error-reply everything already queued on the frontend so those clients
        fail fast; anything sent after the pool exits is unanswerable."""
        error = _error("env server pool shut down: all workers died")
        while await self.frontend.poll(timeout=0):
            frames = await self.frontend.recv_multipart()
            if len(frames) != 4:
                continue
            client_id, request_id, _, _ = frames
            with contextlib.suppress(zmq.ZMQError):
                await self.frontend.send_multipart([client_id, request_id, error])

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
    else an `EnvServerPool` broker over up to `max_workers` workers (`None` =
    unbounded). The frontend speaks the same protocol either way. Reports the bound
    address on `address_queue` (for a spawner that passed an OS-assigned `:0`).

    A native env config may come as `config` (an object) or `config_data` (the
    picklable dict from `env_config_data` — a spawning caller can't pickle a
    dynamically-narrowed config type); legacy passes `env_id`/`env_args`/
    `extra_env_kwargs`. `log_setup` (a picklable callable) configures logging for
    this process and every worker — without it a spawned server's logs are silently
    dropped. `death_pipe` makes this server self-terminate if its parent dies
    abruptly (see `_arm_teardown`)."""
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
            )
            if address_queue is not None:
                address_queue.put(pool.address)
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
