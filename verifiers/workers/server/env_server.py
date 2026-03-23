"""Base environment server.

Owns an :class:`EnvRouter` (worker pool), sets up logging, and provides
the ``run()`` / ``run_server()`` lifecycle.  Subclasses implement the
client-facing transport in ``serve()`` and ``close()``.
"""

import asyncio
import logging
import signal
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import verifiers as vf
from verifiers.workers.server.env_router import EnvRouter


def request_parent_death_signal() -> None:
    """Ask the Linux kernel to send SIGTERM when the parent process dies."""
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


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
        log_file: str | None = None,
        log_file_level: str | None = None,
        json_logging: bool = False,
        *,
        num_workers: int = 1,
        worker_heartbeat_timeout: float = 30.0,
        stats_log_interval: float = 10.0,
    ):
        logger_kwargs: dict[str, Any] = {"json_logging": json_logging}
        if log_level is not None:
            logger_kwargs["level"] = log_level
        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            logger_kwargs["log_file"] = log_file
            logger_kwargs["log_file_level"] = log_file_level
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
            log_file=log_file,
            log_file_level=log_file_level,
            num_workers=num_workers,
            worker_heartbeat_timeout=worker_heartbeat_timeout,
            stats_log_interval=stats_log_interval,
        )

    @abstractmethod
    async def serve(self, stop_event: asyncio.Event | None = None) -> None:
        """Client-facing serve loop. Subclasses implement this."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up client-facing resources (sockets, health process, etc.)."""

    async def run(self) -> None:
        """Run the server with signal-based graceful shutdown."""
        request_parent_death_signal()

        stop_event = asyncio.Event()

        def signal_handler(sig, frame):
            stop_event.set()
            if sig == signal.SIGTERM:
                raise SystemExit(143)
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            await self.serve(stop_event=stop_event)
        finally:
            await self.close()
            await self.router.close()

    @classmethod
    def run_server(cls, *args, **kwargs):
        server = cls(*args, **kwargs)
        return asyncio.run(server.run())
