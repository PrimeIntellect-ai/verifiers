import asyncio
import logging
import signal
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import verifiers as vf
from verifiers.workers.types import (
    EvaluateRequest,
    EvaluateResponse,
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

        # load environment
        self.env = vf.load_environment(env_id, **self.env_args)
        if self.extra_env_kwargs:
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
        state = await self.env.run_rollout(
            request.input,
            request.client_config,
            request.model,
            request.sampling_args,
            request.score,
        )
        return RunRolloutResponse(state=state)

    async def _handle_run_group(self, request: RunGroupRequest) -> RunGroupResponse:
        states = await self.env.run_group(
            request.group_inputs,
            request.client_config,
            request.model,
            request.sampling_args,
            request.score,
        )
        return RunGroupResponse(states=states)

    async def _handle_evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        from pathlib import Path

        results_path = Path(request.results_path) if request.results_path else None
        results = await self.env.evaluate(
            request.client_config,
            request.model,
            request.sampling_args,
            request.num_examples,
            request.rollouts_per_example,
            request.max_concurrent,
            results_path,
            request.state_columns,
            request.save_results,
            request.save_every,
            request.independent_scoring,
        )
        return EvaluateResponse(results=results)
