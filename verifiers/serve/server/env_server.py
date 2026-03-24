"""Base environment server.

Owns an :class:`EnvRouter` (worker pool), sets up logging, and provides
the ``run()`` / ``run_server()`` lifecycle.  Subclasses implement the
client-facing transport in ``serve()`` and ``close()``.
"""

import asyncio
import logging
import signal
from abc import ABC, abstractmethod
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import verifiers as vf
from verifiers.serve.server.env_router import EnvRouter
from verifiers.utils.process_utils import monitor_death_pipe


class EnvServer(ABC):
    """Base class for environment server.

    Manages a pool of worker processes via an :class:`EnvRouter`.
    Subclasses add the client-facing protocol (e.g. ZMQ ROUTER socket).
    """

    def __init__(
        self,
        env_id: str,
        env_args: dict[str, Any] | None = None,
        extra_env_kwargs: dict[str, Any] | None = None,
        log_level: str | None = None,
        log_dir: str | None = None,
        console_logging: bool = True,
        file_logging: bool = True,
        json_logging: bool = False,
        *,
        num_workers: int = 1,
        worker_heartbeat_timeout: float = 30.0,
        stats_log_interval: float = 10.0,
        death_pipe: Connection | None = None,
    ):
        self.death_pipe = death_pipe

        logger_kwargs: dict[str, Any] = {
            "console_logging": console_logging,
            "file_logging": file_logging and log_dir is not None,
            "json_logging": json_logging,
        }
        if log_level is not None:
            logger_kwargs["level"] = log_level
        if log_dir is not None:
            server_log = EnvServer.get_log_file(log_dir)
            server_log.parent.mkdir(parents=True, exist_ok=True)
            logger_kwargs["log_file"] = str(server_log)
        vf.setup_logging(**logger_kwargs)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(
            f"Initializing {self.__class__.__name__} to serve {env_id} "
            f"({env_args=}, {extra_env_kwargs=}, {num_workers=})"
        )

        self.router = EnvRouter(
            env_id=env_id,
            env_args=env_args,
            extra_env_kwargs=extra_env_kwargs,
            log_level=log_level,
            log_dir=log_dir,
            console_logging=console_logging,
            num_workers=num_workers,
            worker_heartbeat_timeout=worker_heartbeat_timeout,
            stats_log_interval=stats_log_interval,
            death_pipe=death_pipe,
        )

    @abstractmethod
    async def serve(self, stop_event: asyncio.Event | None = None) -> None:
        """Client-facing serve loop. Subclasses implement this."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up client-facing resources (sockets, health process, etc.)."""

    async def run(self) -> None:
        """Run the server with signal-based graceful shutdown."""
        if self.death_pipe is not None:
            monitor_death_pipe(self.death_pipe)

        stop_event = asyncio.Event()

        def signal_handler(sig, frame):
            stop_event.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            await self.serve(stop_event=stop_event)
        finally:
            # Ignore signals during cleanup to avoid interrupting teardown.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            await self.router.close()
            await self.close()

    @staticmethod
    def get_log_file(log_dir: str) -> Path:
        """Return the server log file path for a given log directory."""
        return Path(log_dir) / "env_server.log"

    @staticmethod
    def get_all_log_files(log_dir: str, num_workers: int) -> list[Path]:
        """Return all log file paths: the server log followed by each worker log."""
        from verifiers.serve.server.env_worker import EnvWorker

        server_log = EnvServer.get_log_file(log_dir)
        worker_logs = [
            EnvWorker.get_log_file(log_dir, wid) for wid in range(num_workers)
        ]
        return [server_log, *worker_logs]

    @classmethod
    def run_server(cls, *args, **kwargs):
        server = cls(*args, **kwargs)
        return asyncio.run(server.run())
