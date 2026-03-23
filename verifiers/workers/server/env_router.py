"""Worker pool manager with IPC dispatch.

Owns per-worker PUSH sockets and shared response/stats PULL sockets.
Handles worker spawning, least-pending dispatch, heartbeat monitoring,
dead-worker restart with transparent re-dispatch, and stats aggregation.

This is an internal component owned by :class:`EnvServer` — it has no
client-facing socket knowledge.
"""

import logging
import multiprocessing as mp
import os
import time
import uuid
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from typing import Any

import msgpack
import zmq
import zmq.asyncio

from verifiers.utils.logging_utils import print_time
from verifiers.utils.process_utils import terminate_process
from verifiers.utils.worker_utils import make_ipc_address
from verifiers.workers.types import WorkerStats


@dataclass
class ActiveRequestInfo:
    client_id: bytes
    request_id: bytes
    worker_id: int
    payload: bytes


@dataclass
class WorkerHandle:
    worker_id: int
    process: BaseProcess
    address: str
    socket: zmq.asyncio.Socket
    active_requests: dict[bytes, ActiveRequestInfo] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)
    stats: WorkerStats | None = None

    @property
    def active_count(self) -> int:
        return len(self.active_requests)


class EnvRouter:
    """Manages a pool of ZMQEnvWorker processes via IPC PUSH/PULL."""

    def __init__(
        self,
        env_id: str,
        env_args: dict[str, Any] | None = None,
        extra_env_kwargs: dict[str, Any] | None = None,
        log_level: str | None = None,
        log_file: str | None = None,
        log_file_level: str | None = None,
        *,
        num_workers: int = 1,
        worker_heartbeat_timeout: float = 30.0,
        stats_log_interval: float = 10.0,
    ):
        self.logger = logging.getLogger(f"{__name__}.EnvRouter")

        # Config forwarded to workers
        self.env_id = env_id
        self.env_args = env_args or {}
        self.extra_env_kwargs = extra_env_kwargs or {}
        self.log_level = log_level
        self.log_file = log_file
        self.log_file_level = log_file_level

        self.num_workers = num_workers
        self.worker_heartbeat_timeout = worker_heartbeat_timeout
        self.stats_log_interval = stats_log_interval

        self.session_id = uuid.uuid4().hex[:12]

        # setup sockets
        self.ctx = zmq.asyncio.Context()

        self.response_address = make_ipc_address(self.session_id, "responses")
        self.response_pull = self.ctx.socket(zmq.PULL)
        self.response_pull.setsockopt(zmq.RCVHWM, 0)
        self.response_pull.setsockopt(zmq.LINGER, 0)
        self.response_pull.bind(self.response_address)

        self.stats_address = make_ipc_address(self.session_id, "stats")
        self.stats_pull = self.ctx.socket(zmq.PULL)
        self.stats_pull.setsockopt(zmq.RCVHWM, 0)
        self.stats_pull.setsockopt(zmq.LINGER, 0)
        self.stats_pull.bind(self.stats_address)

        # setup state
        self.workers: dict[int, WorkerHandle] = {}
        self._request_worker: dict[bytes, int] = {}  # request_id → worker_id

        self.ipc_paths: list[str] = [
            self.response_address.replace("ipc://", ""),
            self.stats_address.replace("ipc://", ""),
        ]

    @property
    def active_requests(self) -> dict[bytes, ActiveRequestInfo]:
        """All active requests across all workers."""
        return {
            rid: info
            for handle in self.workers.values()
            for rid, info in handle.active_requests.items()
        }

    # ── worker lifecycle ─────────────────────────────────────────

    def get_worker_name(self, worker_id: int) -> str:
        """Get the name of an env worker."""
        return f"{self.env_id}-{worker_id}"

    def get_worker_address(self, worker_id: int) -> str:
        """Get the address of an env worker."""
        worker_name = self.get_worker_name(worker_id)
        return make_ipc_address(self.session_id, worker_name)

    def start_worker(self, worker_id: int) -> WorkerHandle:
        """Start an EnvWorker process."""
        from verifiers.workers.server.env_worker import EnvWorker

        worker_name = self.get_worker_name(worker_id)
        worker_addr = self.get_worker_address(worker_id)
        self.ipc_paths.append(worker_addr.replace("ipc://", ""))

        ctx = mp.get_context("spawn")
        process = ctx.Process(
            target=EnvWorker.run_worker,
            args=(
                self.env_id,
                self.env_args,
                self.extra_env_kwargs,
                self.log_level,
                self.log_file,
                self.log_file_level,
            ),
            kwargs=dict(
                worker_id=worker_id,
                worker_name=worker_name,
                request_address=worker_addr,
                response_address=self.response_address,
                stats_address=self.stats_address,
            ),
            name=worker_name,
            daemon=False,
        )
        process.start()

        socket = self.ctx.socket(zmq.PUSH)
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.LINGER, 5000)
        socket.connect(worker_addr)

        self.logger.info(
            f"Started {worker_name} (id={worker_id}, name={worker_name}, address={worker_addr}, pid={process.pid})"
        )

        return WorkerHandle(
            worker_id=worker_id,
            process=process,
            socket=socket,
            address=worker_addr,
        )

    def start_workers(self) -> None:
        """Spawn all worker processes."""
        for wid in range(self.num_workers):
            self.workers[wid] = self.start_worker(wid)

    async def restart_worker(self, worker_id: int) -> None:
        old = self.workers.get(worker_id)
        to_redispatch: list[ActiveRequestInfo] = []
        if old is not None:
            to_redispatch = list(old.active_requests.values())
            old.active_requests.clear()
            terminate_process(old.process)
            old.socket.close()

        self.workers[worker_id] = self.start_worker(worker_id)

        for info in to_redispatch:
            new_wid = self.select_worker()
            handle = self.workers[new_wid]
            try:
                await handle.socket.send_multipart(
                    [info.client_id, info.request_id, info.payload]
                )
                info.worker_id = new_wid
                handle.active_requests[info.request_id] = info
                self._request_worker[info.request_id] = new_wid
                self.logger.info(
                    f"Re-dispatched request {info.request_id[:7]} "
                    f"from dead worker {worker_id} to worker {new_wid}"
                )
            except zmq.ZMQError as e:
                self._request_worker.pop(info.request_id, None)
                self.logger.error(f"Failed to re-dispatch request: {e}")

    # ── dispatch ─────────────────────────────────────────────────

    def select_worker(self) -> int:
        """Select the least-busy worker."""
        return min(self.workers, key=lambda wid: self.workers[wid].active_count)

    async def dispatch(
        self, client_id: bytes, request_id: bytes, payload: bytes
    ) -> None:
        """Send a request to the least-busy worker."""
        wid = self.select_worker()
        handle = self.workers[wid]
        await handle.socket.send_multipart([client_id, request_id, payload])
        info = ActiveRequestInfo(
            client_id=client_id,
            request_id=request_id,
            payload=payload,
            worker_id=wid,
        )
        handle.active_requests[request_id] = info
        self._request_worker[request_id] = wid

    async def forward_cancel(self, request_id: bytes, client_id: bytes) -> None:
        """Forward a cancel signal to the worker owning this request."""
        wid = self._request_worker.get(request_id)
        if wid is not None:
            handle = self.workers.get(wid)
            if handle is not None:
                try:
                    await handle.socket.send_multipart([client_id, request_id, b""])
                except zmq.ZMQError:
                    pass

    def complete_request(self, request_id: bytes) -> ActiveRequestInfo | None:
        """Update bookkeeping after a response is received. Returns the info or None."""
        wid = self._request_worker.pop(request_id, None)
        if wid is None:
            return None
        handle = self.workers.get(wid)
        if handle is None:
            return None
        return handle.active_requests.pop(request_id, None)

    def handle_stats_message(self, data: bytes) -> None:
        """Parse a stats message and update the worker handle."""
        try:
            raw = msgpack.unpackb(data, raw=False)
            stats = WorkerStats.model_validate(raw)
            handle = self.workers.get(stats.worker_id)
            if handle is not None:
                handle.stats = stats
                handle.last_heartbeat = stats.timestamp
        except Exception:
            pass

    # ── periodic checks ──────────────────────────────────────────

    async def check_workers(self) -> None:
        """Restart dead or unresponsive workers."""
        now = time.time()
        for wid, handle in list(self.workers.items()):
            if not handle.process.is_alive():
                self.logger.warning(
                    f"Worker {wid} (pid={handle.process.pid}) died, restarting"
                )
                await self.restart_worker(wid)
            elif (
                now - handle.last_heartbeat > self.worker_heartbeat_timeout
                and handle.last_heartbeat > 0
            ):
                self.logger.warning(
                    f"Worker {wid} heartbeat timeout "
                    f"({now - handle.last_heartbeat:.1f}s), restarting"
                )
                await self.restart_worker(wid)

    def log_aggregate_stats(self) -> None:
        """Log aggregate stats for all workers."""
        total_active = 0
        per_worker = []
        lag_means: list[float] = []
        lag_p99s: list[float] = []
        lag_maxes: list[float] = []

        for wid in sorted(self.workers):
            handle = self.workers[wid]
            total_active += handle.active_count
            per_worker.append(f"W{wid}:{handle.active_count}")
            if handle.stats and handle.stats.lag.n > 0:
                lag_means.append(handle.stats.lag.mean)
                lag_p99s.append(handle.stats.lag.p99)
                lag_maxes.append(handle.stats.lag.max)

        parts = [
            f"Workers: {len(self.workers)}",
            f"Active: {total_active} ({', '.join(per_worker)})",
        ]
        if lag_means:
            parts.append(
                f"Lag: mean={print_time(sum(lag_means) / len(lag_means))} "
                f"p99={print_time(max(lag_p99s))} max={print_time(max(lag_maxes))}"
            )
        self.logger.info(" | ".join(parts))

    # ── shutdown ─────────────────────────────────────────────────

    async def close(self) -> None:
        """Close all router resources."""
        for handle in self.workers.values():
            terminate_process(handle.process)
            handle.socket.close()

        self.workers.clear()
        self._request_worker.clear()

        self.response_pull.close()
        self.stats_pull.close()
        self.ctx.term()

        for path in self.ipc_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        self.logger.info("Router shut down")
