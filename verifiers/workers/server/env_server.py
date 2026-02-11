import asyncio
import logging
import signal
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import ClientConfig
from verifiers.utils.client_utils import resolve_client_config, setup_client
from verifiers.workers.types import (
    HealthRequest,
    HealthResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)


class EnvServer(ABC):
    """Base class for environment server."""

    def __init__(
        self,
        # environment
        env_id: str,
        env_args: dict[str, Any] | None = None,
        extra_env_kwargs: dict[str, Any] | None = None,
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
        self.env_args = env_args or {}
        self.extra_env_kwargs = extra_env_kwargs or {}

        self.clients: dict[str, AsyncOpenAI] = {}
        self.pending_tasks: set[asyncio.Task] = set()

        # load environment
        self.logger.info(f"Loading environment {env_id} with {env_args=}")
        self.env = vf.load_environment(self.env_id, **self.env_args)
        if self.extra_env_kwargs:
            self.logger.info(
                f"Setting extra environment kwargs: {self.extra_env_kwargs}"
            )
            self.env.set_kwargs(**self.extra_env_kwargs)

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
                server.logger.info(
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
        output = await self.env.run_rollout(
            input=request.input,
            client=client,
            model=request.model,
            sampling_args=request.sampling_args,
            max_retries=request.max_retries,
            state_columns=request.state_columns,
        )
        return RunRolloutResponse(output=output)

    async def _handle_run_group(self, request: RunGroupRequest) -> RunGroupResponse:
        client = await self._resolve_client(request.client_config)
        outputs = await self.env.run_group(
            group_inputs=request.group_inputs,
            client=client,
            model=request.model,
            sampling_args=request.sampling_args,
            max_retries=request.max_retries,
            state_columns=request.state_columns,
        )
        return RunGroupResponse(outputs=outputs)

    async def _resolve_client(self, client_config: ClientConfig) -> AsyncOpenAI:
        resolved_client_config = resolve_client_config(client_config)
        client_key = resolved_client_config.model_dump_json()
        if client_key in self.clients:
            return self.clients[client_key]
        client = setup_client(resolved_client_config)
        self.clients[client_key] = client
        return client

    async def _close_cached_clients(self) -> None:
        for client in self.clients.values():
            await client.close()
        self.clients.clear()
