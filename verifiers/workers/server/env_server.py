import asyncio
import logging
import signal
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import ClientConfig
from verifiers.utils.async_utils import maybe_semaphore
from verifiers.utils.client_utils import setup_client
from verifiers.workers.types import (
    HealthRequest,
    HealthResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)


class EnvServer(ABC):
    """Server that exposes an environment as a service."""

    def __init__(
        self,
        # environment
        env_id: str,
        env_args: dict[str, Any] = {},
        extra_env_kwargs: dict[str, Any] = {},
        log_level: str | None = None,
        log_file: str | None = None,
        log_file_level: str | None = None,
    ):
        # setup logging
        log_file = log_file or f"logs/{env_id}.log"
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        if log_level is None:
            vf.setup_logging(log_file=log_file, log_file_level=log_file_level)
        else:
            vf.setup_logging(
                level=log_level, log_file=log_file, log_file_level=log_file_level
            )

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(
            f"Initializing {self.__class__.__name__} to serve {env_id} ({env_args=}, {extra_env_kwargs=})"
        )

        self.env_id = env_id
        self.env_args = env_args
        self.extra_env_kwargs = extra_env_kwargs
        self.clients: dict[str, AsyncOpenAI] = {}

        # load environment
        self.env = vf.load_environment(env_id, **self.env_args)
        if self.extra_env_kwargs:
            self.logger.info(
                f"Setting extra environment kwargs: {self.extra_env_kwargs}"
            )
            self.env.set_kwargs(**self.extra_env_kwargs)

        self.no_limit = maybe_semaphore(None)

    @abstractmethod
    async def run(self, stop_event: asyncio.Event | None = None):
        pass

    @abstractmethod
    async def close(self):
        pass

    @classmethod
    def run_server(cls, *args, **kwargs):
        server = cls(*args, **kwargs)

        async def run_with_graceful_shutdown():
            # setup graceful shutdown for SIGTERM (K8s, Docker, Slurm) and SIGINT (Ctrl+C)
            stop_event = asyncio.Event()

            def signal_handler(sig):
                server.logger.debug(
                    f"Received signal {sig.name}, initiating graceful shutdown"
                )
                stop_event.set()

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

            try:
                await server.run(stop_event=stop_event)
            finally:
                await server.close()

        return asyncio.run(run_with_graceful_shutdown())

    async def _handle_health(self, _request: HealthRequest) -> HealthResponse:
        return HealthResponse()

    async def _handle_run_rollout(
        self, request: RunRolloutRequest
    ) -> RunRolloutResponse:
        client = await self._resolve_client(request.client_config)
        # Server always runs locally with AsyncOpenAI, so we always get State back
        output = await self.env.run_rollout(
            request.input,
            client,
            request.model,
            request.sampling_args,
        )
        return RunRolloutResponse(output=output)

    async def _handle_run_group(self, request: RunGroupRequest) -> RunGroupResponse:
        client = await self._resolve_client(request.client_config)
        # Server always runs locally with AsyncOpenAI, so we always get list[State] back
        outputs = await self.env.run_group(
            request.group_inputs,
            client,
            request.model,
            request.sampling_args,
        )
        return RunGroupResponse(outputs=outputs)

    async def _resolve_client(self, client_config: ClientConfig) -> AsyncOpenAI:
        client_key = client_config.model_dump_json()
        if client_key in self.clients:
            return self.clients[client_key]
        client = setup_client(client_config)
        self.clients[client_key] = client
        return client
