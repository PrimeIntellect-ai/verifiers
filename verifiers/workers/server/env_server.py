import asyncio
import logging
import multiprocessing
from abc import ABC, abstractmethod
from typing import Any

import verifiers as vf
from verifiers.utils.async_utils import maybe_semaphore


class EnvServer(ABC):
    """Server that exposes an environment as a service."""

    def __init__(
        self,
        # environment
        env_id: str,
        env_args: dict[str, Any] = {},
        extra_env_kwargs: dict[str, Any] = {},
        # server
        max_concurrent: int = -1,
        num_workers: int = 1,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(
            f"Initializing {self.__class__.__name__} with env_id={env_id}, env_args={env_args}, extra_env_kwargs={extra_env_kwargs}, max_concurrent={max_concurrent}, num_workers={num_workers}"
        )

        self.env_id = env_id
        self.env_args = env_args
        self.extra_env_kwargs = extra_env_kwargs

        self.max_concurrent = max_concurrent
        self.num_workers = num_workers

        self.workers = multiprocessing.Pool(num_workers)
        self.semaphore = maybe_semaphore(max_concurrent)

        # load environment
        self.env = vf.load_environment(env_id, **self.env_args)
        if self.extra_env_kwargs:
            self.env.set_kwargs(**self.extra_env_kwargs)

        self.logger.info(
            f"Initialized {self.num_workers} worker process(es) to serve {self.env_id}"
        )

    @abstractmethod
    async def run(self, stop_event: asyncio.Event | None = None):
        pass

    @classmethod
    def run_server(cls, *args, **kwargs):
        server = cls(*args, **kwargs)
        return asyncio.run(server.run())
