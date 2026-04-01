"""Worker pool manager with IPC dispatch.

Owns per-worker PUSH sockets and shared response/stats PULL sockets.
Handles worker spawning, least-pending dispatch, heartbeat monitoring,
dead-worker restart with transparent re-dispatch, and stats aggregation.

This is an internal component owned by :class:`EnvServer` — it has no
client-facing socket knowledge.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

import msgpack
import zmq
import zmq.asyncio
from pydantic import BaseModel

from verifiers.serve.server.env_worker import EnvWorkerStats
from verifiers.utils.async_utils import EventLoopLagMonitor, EventLoopLagStats
from verifiers.utils.process_utils import terminate_process, terminate_processes
from verifiers.utils.serve_utils import make_ipc_address

# Callback type: (client_id, request_id, response_bytes) -> awaitable
OnResponseCallback = Callable[[bytes, bytes, bytes], Awaitable[None]]


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
    last_heartbeat: float = 0.0
    stats: EnvWorkerStats | None = None

    @property
    def active_count(self) -> int:
        return len(self.active_requests)


class EnvRouterStats(BaseModel):
    lag: EventLoopLagStats = EventLoopLagStats()
    workers: dict[int, EnvWorkerStats | None] = {}

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    @property
    def active_tasks(self) -> int:
        return sum(w.active_tasks for w in self.workers.values() if w is not None)

    def __str__(self) -> str:
        worker_counts = ", ".join(
            f"W{wid}: {w.active_tasks if w is not None else '?'}"
            for wid, w in sorted(self.workers.items())
        )
        header = f"Active tasks: {self.active_tasks} ({worker_counts})"

        # compute label width for alignment
        labels = ["Server"] + [f"W{wid}" for wid in sorted(self.workers)]
        pad = max(len(label) for label in labels)

        lines = [header, f"  {'Server':<{pad}} | Lag: {self.lag}"]
        for wid in sorted(self.workers):
            lines.append(
                f"  {f'W{wid}':<{pad}} | {self.workers[wid] or 'no stats yet'}"
            )
        return "\n".join(lines)


class EnvRouter:
    """Manages a pool of EnvWorker processes via IPC PUSH/PULL."""

    def __init__(
        self,
        env_id: str,
        env_args: dict[str, Any] | None = None,
        extra_env_kwargs: dict[str, Any] | None = None,
        log_level: str | None = None,
        log_dir: str | None = None,
        console_logging: bool = True,
        *,
        num_workers: int = 1,
        worker_heartbeat_timeout: float = 30.0,
        stats_log_interval: float = 10.0,
        death_pipe: Connection | None = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Config forwarded to workers
        self.env_id = env_id
        self.env_args = env_args or {}
        self.extra_env_kwargs = extra_env_kwargs or {}
        self.log_level = log_level
        self.log_dir = log_dir
        self.console_logging = console_logging

        self.num_workers = num_workers
        self.worker_heartbeat_timeout = worker_heartbeat_timeout
        self.stats_log_interval = stats_log_interval
        self.death_pipe = death_pipe

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
        self.request_to_worker: dict[bytes, int] = {}  # request_id → worker_id
        self.lag_monitor = EventLoopLagMonitor()

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

    def get_worker_name(self, worker_id: int) -> str:
        """Get the name of an env worker."""
        return f"{self.env_id}-{worker_id}"

    def get_worker_address(self, worker_id: int) -> str:
        """Get the address of an env worker."""
        worker_name = self.get_worker_name(worker_id)
        return make_ipc_address(self.session_id, worker_name)

    def start_worker(self, worker_id: int) -> WorkerHandle:
        """Start an EnvWorker process."""
        from verifiers.serve.server.env_worker import EnvWorker

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
                self.log_dir,
                self.console_logging,
            ),
            kwargs=dict(
                worker_id=worker_id,
                worker_name=worker_name,
                request_address=worker_addr,
                response_address=self.response_address,
                stats_address=self.stats_address,
                death_pipe=self.death_pipe,
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
            f"Started worker (id={worker_id}, name={worker_name}, address={worker_addr}, pid={process.pid})"
        )

        return WorkerHandle(
            worker_id=worker_id,
            process=process,
            socket=socket,
            address=worker_addr,
        )

    def start_workers(self) -> None:
        """Spawn all worker processes."""
        self.lag_task = asyncio.create_task(self.lag_monitor.run())
        for worker_id in range(self.num_workers):
            self.workers[worker_id] = self.start_worker(worker_id)

    async def run(
        self,
        on_response: OnResponseCallback,
        stop_event: asyncio.Event,
    ) -> None:
        """Background loop: drain worker responses/stats and run periodic checks.

        Args:
            on_response: Called for each completed response with
                ``(client_id, request_id, response_bytes)``.
            stop_event: Set to signal shutdown.
        """
        self.start_workers()

        poller = zmq.asyncio.Poller()
        poller.register(self.response_pull, zmq.POLLIN)
        poller.register(self.stats_pull, zmq.POLLIN)

        last_stats_log = time.time()
        last_heartbeat_check = time.time()

        try:
            while not stop_event.is_set():
                events = dict(await poller.poll(timeout=100))

                # ── worker responses ───────────────────────────────
                if self.response_pull in events:
                    while True:
                        try:
                            frames = await self.response_pull.recv_multipart(
                                zmq.NOBLOCK
                            )
                        except zmq.Again:
                            break
                        if len(frames) != 3:
                            continue
                        client_id, request_id, response_bytes = frames
                        self.complete_request(request_id)
                        await on_response(client_id, request_id, response_bytes)

                # ── worker stats ───────────────────────────────────
                if self.stats_pull in events:
                    while True:
                        try:
                            data = await self.stats_pull.recv(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        self.handle_stats_message(data)

                # ── periodic checks ────────────────────────────────
                now = time.time()
                if now - last_stats_log >= self.stats_log_interval:
                    self.log_stats()
                    last_stats_log = now
                if now - last_heartbeat_check >= 5.0:
                    await self.check_workers()
                    last_heartbeat_check = now

        except asyncio.CancelledError:
            pass
        finally:
            poller.unregister(self.response_pull)
            poller.unregister(self.stats_pull)

    def select_worker(self) -> int:
        """Select the least-busy worker."""
        return min(self.workers, key=lambda wid: self.workers[wid].active_count)

    async def restart_worker(self, worker_id: int) -> None:
        """Restart a dead or unresponsive worker. Re-dispatches all pending requests."""
        old_worker = self.workers.get(worker_id)
        to_redispatch: list[ActiveRequestInfo] = []
        if old_worker is not None:
            to_redispatch = list(old_worker.active_requests.values())
            old_worker.active_requests.clear()
            for info in to_redispatch:
                self.request_to_worker.pop(info.request_id, None)

        # Start the replacement *before* terminating the old process so that
        # self.workers is never empty
        self.workers[worker_id] = self.start_worker(worker_id)

        if old_worker is not None:
            await asyncio.to_thread(terminate_process, old_worker.process)
            old_worker.socket.close()

        for info in to_redispatch:
            new_worker_id = self.select_worker()
            new_worker = self.workers[new_worker_id]
            try:
                await new_worker.socket.send_multipart(
                    [info.client_id, info.request_id, info.payload]
                )
                info.worker_id = new_worker_id
                new_worker.active_requests[info.request_id] = info
                self.request_to_worker[info.request_id] = new_worker_id
                self.logger.debug(
                    f"Re-dispatched request {info.request_id[:7]} "
                    f"from dead worker {worker_id} to worker {new_worker_id}"
                )
            except zmq.ZMQError as e:
                self.request_to_worker.pop(info.request_id, None)
                self.logger.error(f"Failed to re-dispatch request: {e}")

    async def dispatch_request(
        self, client_id: bytes, request_id: bytes, payload: bytes
    ) -> None:
        """Send a request to the least-busy worker."""
        worker_id = self.select_worker()
        worker = self.workers[worker_id]
        await worker.socket.send_multipart([client_id, request_id, payload])
        info = ActiveRequestInfo(
            client_id=client_id,
            request_id=request_id,
            payload=payload,
            worker_id=worker_id,
        )
        worker.active_requests[request_id] = info
        self.request_to_worker[request_id] = worker_id

    async def forward_cancel(self, request_id: bytes, client_id: bytes) -> None:
        """Forward a cancel signal to the worker owning this request."""
        worker_id = self.request_to_worker.get(request_id)
        if worker_id is not None:
            worker = self.workers.get(worker_id)
            if worker is not None:
                try:
                    await worker.socket.send_multipart([client_id, request_id, b""])
                except zmq.ZMQError:
                    pass

    def complete_request(self, request_id: bytes) -> ActiveRequestInfo | None:
        """Update bookkeeping after a response is received. Returns the info or None."""
        worker_id = self.request_to_worker.pop(request_id, None)
        if worker_id is None:
            return None
        worker = self.workers.get(worker_id)
        if worker is not None:
            return worker.active_requests.pop(request_id, None)
        return None

    def handle_stats_message(self, data: bytes) -> None:
        """Parse a stats message and update the worker handle."""
        try:
            raw = msgpack.unpackb(data, raw=False)
            stats = EnvWorkerStats.model_validate(raw)
            worker = self.workers.get(stats.worker_id)
            if worker is not None:
                worker.stats = stats
                worker.last_heartbeat = stats.timestamp
        except Exception:
            pass

    async def check_workers(self) -> None:
        """Restart dead or unresponsive workers."""
        now = time.time()
        for worker_id, worker in list(self.workers.items()):
            if not worker.process.is_alive():
                self.logger.warning(
                    f"Worker {worker_id} (pid={worker.process.pid}) died, restarting"
                )
                await self.restart_worker(worker_id)
            elif (
                now - worker.last_heartbeat > self.worker_heartbeat_timeout
                and worker.last_heartbeat > 0
            ):
                self.logger.warning(
                    f"Worker {worker_id} heartbeat timeout "
                    f"({now - worker.last_heartbeat:.1f}s), restarting"
                )
                await self.restart_worker(worker_id)

    def log_stats(self) -> None:
        """Log server lag + per-worker stats."""
        stats = EnvRouterStats(
            lag=EventLoopLagStats.from_monitor(self.lag_monitor),
            workers={wid: w.stats for wid, w in sorted(self.workers.items())},
        )
        self.logger.info(stats)

    async def close(self) -> None:
        """Close all router resources."""
        if hasattr(self, "lag_task"):
            self.lag_task.cancel()
            await asyncio.gather(self.lag_task, return_exceptions=True)

        terminate_processes([w.process for w in self.workers.values()])
        for worker in self.workers.values():
            worker.socket.close()

        self.workers.clear()
        self.request_to_worker.clear()

        self.response_pull.close()
        self.stats_pull.close()
        self.ctx.term()

        for path in self.ipc_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        self.logger.info("Router shut down")
