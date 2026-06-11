"""Env-server worker pool: a ROUTER broker over N worker processes.

A lone `EnvServer` runs every rollout as an `asyncio.Task` on one event loop, so
CPU-bound work (renderer tokenization, scoring) competes for that loop. v0 relieved
this with a router + worker pool; this reinstates it for v1.

A broker binds the client-facing ROUTER (the *same* wire protocol as a lone
`EnvServer`, so `EnvClient` is unchanged), spawns `num_workers` worker processes —
each an ordinary `EnvServer` / `LegacyEnvServer` bound to its own ipc address — and
load-balances requests to the least-busy worker over a `DEALER` per worker. The
worker's `client_id` (its reply identity) is the broker's DEALER identity; the broker
holds the real client identity in `pending` and routes the reply back by `request_id`.
`health` is answered inline (no worker needed); everything else goes to a worker.

Workers are spawned `spawn`-style (own env, own loop) and monitor a death pipe so an
orphaned worker self-exits if the broker dies. TODO: per-worker restart-on-death and
stats/lag monitors (v0 had them; omitted here — rollout errors are returned as data,
not crashes, so worker death is rare).
"""

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import signal
import threading
import uuid

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
    *, server_kwargs: dict, address: str, death_pipe, legacy: bool
) -> None:
    """Spawned worker: an ordinary EnvServer/LegacyEnvServer bound to `address` (ipc).
    A native config arrives as a dict (`config_data`): the eval/serve CLI's narrowed
    config type is dynamic and unpicklable, so we rebuild it here via EnvConfig's
    id-resolving validator."""
    from verifiers.v1.legacy import LegacyEnvServer

    _monitor_parent(death_pipe)
    if "config_data" in server_kwargs:
        server_kwargs = {
            "config": EnvConfig.model_validate(server_kwargs["config_data"])
        }
    cls = LegacyEnvServer if legacy else EnvServer
    cls.run_server(address=address, **server_kwargs)


class EnvServerPool:
    """ROUTER broker over `num_workers` worker processes (least-busy dispatch)."""

    def __init__(
        self,
        server_kwargs: dict,
        num_workers: int,
        address: str,
        legacy: bool,
    ) -> None:
        self.server_kwargs = server_kwargs
        self.num_workers = num_workers
        self.legacy = legacy
        self.session = uuid.uuid4().hex[:12]
        self.workers: list[dict] = []

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

    def _start_workers(self) -> None:
        mpctx = mp.get_context("spawn")
        for i in range(self.num_workers):
            address = f"ipc://{self._worker_path(i)}"
            parent_conn, child_conn = mpctx.Pipe()
            proc = mpctx.Process(
                target=_worker_entry,
                kwargs=dict(
                    server_kwargs=self.server_kwargs,
                    address=address,
                    death_pipe=child_conn,
                    legacy=self.legacy,
                ),
                daemon=False,
            )
            proc.start()
            child_conn.close()  # parent keeps the write end (its close signals death)
            dealer = self.ctx.socket(zmq.DEALER)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.connect(address)  # connect before bind is fine — ZMQ queues
            self.workers.append(
                {"process": proc, "dealer": dealer, "pipe": parent_conn, "active": 0}
            )

    async def run(self) -> None:
        self._start_workers()
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        for w in self.workers:
            poller.register(w["dealer"], zmq.POLLIN)
        pending: dict[bytes, dict] = {}  # request_id -> {client_id, worker}
        logger.info(
            "EnvServerPool up: address=%s workers=%d", self.address, self.num_workers
        )
        try:
            while True:
                events = dict(await poller.poll())
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
        for i, w in enumerate(self.workers):
            with contextlib.suppress(Exception):
                w["process"].join(timeout=10)
            if w["process"].is_alive():
                with contextlib.suppress(Exception):
                    w["process"].kill()
            with contextlib.suppress(Exception):
                w["dealer"].close()
            with contextlib.suppress(OSError):
                os.unlink(self._worker_path(i))
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
    **server_kwargs,
) -> None:
    """Serve one env over ZMQ: a single in-process `EnvServer` when `num_workers <= 1`,
    else an `EnvServerPool` broker over `num_workers` worker processes. The frontend
    speaks the same protocol either way, so the client is identical. Reports the bound
    address on `address_queue` (for a spawner that passed an OS-assigned `:0`).

    A native env config may be passed as `config` (an object) or `config_data` (the
    picklable dict from `env_config_data`, for callers that spawn this function and so
    can't pickle a dynamically-narrowed config type); legacy passes `env_id`/`env_args`/
    `extra_env_kwargs`."""
    # SIGTERM -> KeyboardInterrupt so a killed server runs its teardown (pool: kill the
    # workers; single: close clients).
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if num_workers > 1:
        if (
            "config" in server_kwargs
        ):  # dict-ify for the workers (config_data is picklable)
            server_kwargs = {"config_data": env_config_data(server_kwargs["config"])}
        pool = EnvServerPool(server_kwargs, num_workers, address, legacy)
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
        cls.run_server(address=address, address_queue=address_queue, **server_kwargs)
