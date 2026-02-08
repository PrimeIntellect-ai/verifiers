import logging
import sys
from typing import Any, cast

import tenacity as tc
from prime_sandboxes import CommandTimeoutError


from verifiers.envs.sandbox_env import (
    CreateSandboxRequest,
    SandboxCreationError,
    SandboxNotReadyError,
    ThreadedAsyncSandboxClient,
)


class SandboxExecutorMixin:
    """Small helper mixin for sandbox lifecycle + retries."""

    def _init_sandbox_executor(
        self,
        *,
        sandbox_client_max_workers: int = 50,
        sandbox_client_max_connections: int = 100,
        sandbox_client_max_keepalive_connections: int = 50,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
    ) -> None:
        self._sandbox_logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self._sandbox_client = ThreadedAsyncSandboxClient(
            max_workers=sandbox_client_max_workers,
            max_connections=sandbox_client_max_connections,
            max_keepalive_connections=sandbox_client_max_keepalive_connections,
        )
        self._sandbox_with_retry = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(max_retries),
            wait=tc.wait_exponential_jitter(
                initial=base_delay,
                exp_base=backoff_factor,
                max=max_backoff_seconds,
                jitter=jitter,
            ),
            before_sleep=tc.before_sleep_log(
                cast(Any, self._sandbox_logger),
                logging.WARNING,
            ),
            reraise=True,
        ).wraps

    async def _create_sandbox(self, request: CreateSandboxRequest) -> Any:
        try:
            return await self._sandbox_with_retry(self._sandbox_client.create)(request)
        except Exception as e:
            raise SandboxCreationError(e)

    async def _wait_for_sandbox_ready(self, sandbox_id: str) -> None:
        try:
            await self._sandbox_client.wait_for_creation(sandbox_id)
        except Exception as e:
            raise SandboxNotReadyError(e)

    async def _execute_sandbox_command(
        self,
        sandbox_id: str,
        command: str,
        *,
        working_dir: str | None = None,
        timeout: int = 30,
    ) -> Any:
        try:
            return await self._sandbox_client.execute_command(
                sandbox_id,
                command,
                working_dir=working_dir,
                timeout=timeout,
            )
        except CommandTimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(e)

    async def _delete_sandbox(self, sandbox_id: str, *, use_retry: bool = True) -> None:
        if getattr(sys, "is_finalizing", lambda: False)():
            return
        executor = getattr(self._sandbox_client, "executor", None)
        if executor is not None and getattr(executor, "_shutdown", False):
            return
        try:
            if use_retry:
                await self._sandbox_with_retry(self._sandbox_client.delete)(sandbox_id)
            else:
                await self._sandbox_client.delete(sandbox_id)
        except RuntimeError as exc:
            message = str(exc)
            if "cannot schedule new futures after interpreter shutdown" in message:
                return
            if "Event loop is closed" in message:
                return
            raise

    def _teardown_sandbox_client(self) -> None:
        self._sandbox_client.teardown()
