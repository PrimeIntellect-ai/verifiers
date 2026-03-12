import asyncio
import io
import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

import httpx
import tenacity as tc
from prime_sandboxes import (
    APIError,
    CommandTimeoutError,
    CreateSandboxRequest,
    SandboxClient,
    SandboxOOMError,
    SandboxTimeoutError,
)
from prime_sandboxes.core import APIClient

import verifiers as vf
from verifiers.utils.threaded_sandbox_client import ThreadedAsyncSandboxClient

# Enable httpx debug logging if HTTPX_LOG_LEVEL is set
_httpx_log_level = os.environ.get("HTTPX_LOG_LEVEL", "").upper()
if _httpx_log_level:
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(getattr(logging, _httpx_log_level, logging.DEBUG))
    httpcore_logger = logging.getLogger("httpcore")
    httpcore_logger.setLevel(getattr(logging, _httpx_log_level, logging.DEBUG))


@dataclass
class SandboxScorer:
    """A scorer that runs a script inside the sandbox after rollout.

    The scorer uploads ``script`` and any ``files`` to ``dest_dir`` via bundle,
    then executes the script.  The last line of stdout is parsed as a float and
    stored in ``state[state_key]``.

    Attributes:
        name: Human-readable name for logging.
        script: Python source code to execute in the sandbox.
        state_key: Key under which the score is stored in state.
        dest_dir: Directory inside the sandbox to upload files to.
        files_fn: Callable that receives ``state`` and returns a dict of
            ``{filename: content}`` to upload alongside the script.
        timeout: Timeout in seconds for the scorer script execution.
    """

    name: str
    script: str
    state_key: str
    dest_dir: str = "/app"
    files_fn: Callable[[dict], dict[str, str]] = field(default_factory=lambda: lambda state: {})
    timeout: int = 30


class SandboxCreationError(vf.SandboxError): ...


class SandboxNotReadyError(vf.SandboxError): ...


class SandboxSetupError(vf.SandboxError): ...


class SandboxMonitorRubric(vf.Rubric):
    """Monitor rubric that tracks sandbox execution failures."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.sandbox_oom)
        self.add_metric(self.sandbox_timeout)

    async def sandbox_oom(self, state: vf.State) -> float:
        """Whether the sandbox was OOM-killed."""
        return float(bool(state.get("sandbox_oom")))

    async def sandbox_timeout(self, state: vf.State) -> float:
        """Whether the sandbox timed out."""
        return float(bool(state.get("sandbox_timeout")))


class SandboxMixin:
    """Mixin providing sandbox lifecycle management with retry, tracking, and cleanup."""

    active_sandboxes: set[str]
    sandbox_client: ThreadedAsyncSandboxClient
    sandbox_wait_for_creation_max_attempts: int
    sandbox_scorers: list[SandboxScorer]
    with_retry: Callable

    def init_sandbox_client(
        self,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        sandbox_client_max_workers: int = 10,
        sandbox_client_max_connections: int = 100,
        sandbox_client_max_keepalive_connections: int = 50,
        sandbox_wait_for_creation_max_attempts: int = 120,
    ):
        """Initialize sandbox client and retry wrapper. Call from subclass __init__."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.active_sandboxes = set()
        if not hasattr(self, "sandbox_scorers"):
            self.sandbox_scorers = []
        self.sandbox_wait_for_creation_max_attempts = (
            sandbox_wait_for_creation_max_attempts
        )
        self.sandbox_client = ThreadedAsyncSandboxClient(
            max_workers=sandbox_client_max_workers,
            max_connections=sandbox_client_max_connections,
            max_keepalive_connections=sandbox_client_max_keepalive_connections,
        )
        self.with_retry = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(max_retries),
            wait=tc.wait_exponential_jitter(
                initial=base_delay,
                exp_base=backoff_factor,
                max=max_backoff_seconds,
                jitter=jitter,
            ),
            before_sleep=tc.before_sleep_log(
                cast(Any, self.logger),
                logging.WARNING,
            ),
            reraise=True,
        ).wraps

    async def create_sandbox(self, state, request: CreateSandboxRequest) -> str:
        """Create sandbox with retry, tracking, wait_for_creation, and post-setup hook.

        Raises:
            SandboxCreationError: If sandbox creation fails after retries.
            SandboxNotReadyError: If sandbox fails to become ready.
            SandboxSetupError: If post_sandbox_setup hook fails.
        """
        try:
            sandbox = await self.with_retry(self.sandbox_client.create)(request)
        except Exception as e:
            raise SandboxCreationError(f"Failed to create sandbox: {e}") from e

        self.active_sandboxes.add(sandbox.id)
        state["sandbox_id"] = sandbox.id
        self.logger.debug(f"Created sandbox {sandbox.id}")

        try:
            await self.sandbox_client.wait_for_creation(
                sandbox.id,
                max_attempts=self.sandbox_wait_for_creation_max_attempts,
            )
        except Exception as e:
            raise SandboxNotReadyError(
                f"Sandbox {sandbox.id} failed to become ready: {e}"
            ) from e

        try:
            await self.post_sandbox_setup(state)
        except vf.SandboxError:
            raise
        except Exception as e:
            raise SandboxSetupError(f"Sandbox {sandbox.id} setup failed: {e}") from e

        return sandbox.id

    async def post_sandbox_setup(self, state):
        """Hook for subclasses to run setup after sandbox is ready."""
        pass

    async def delete_sandbox(self, sandbox_id: str):
        """Delete sandbox with retry and tracking."""

        async def _delete(sandbox_id: str):
            await self.sandbox_client.delete(sandbox_id)
            self.active_sandboxes.discard(sandbox_id)
            self.logger.debug(f"Deleted sandbox {sandbox_id}")

        try:
            await self.with_retry(_delete)(sandbox_id)
        except Exception as e:
            self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def bulk_delete_sandboxes(self, sandbox_ids: list[str]) -> None:
        """Delete multiple sandboxes by their IDs."""
        try:
            await self.with_retry(self.sandbox_client.bulk_delete)(sandbox_ids)
            self.logger.debug(f"Bulk deleted sandboxes: {sandbox_ids}")
            self.active_sandboxes.difference_update(sandbox_ids)
        except Exception as e:
            self.logger.error(f"Failed to bulk delete sandboxes {sandbox_ids}: {e}")

    async def run_background_job(
        self,
        state: dict[str, Any],
        command: str,
        timeout: int,
        working_dir: str | None = None,
        poll_interval: int = 3,
    ):
        """Run a command as a background job and poll until completion or timeout."""
        sandbox_id = state["sandbox_id"]
        start_job = self.with_retry(self.sandbox_client.start_background_job)
        get_job = self.with_retry(self.sandbox_client.get_background_job)

        try:
            job = await start_job(
                sandbox_id=sandbox_id, command=command, working_dir=working_dir
            )
        except (CommandTimeoutError, httpx.ReadTimeout) as e:
            self.logger.error(f"Failed to start background job: {repr(e)}")
            raise vf.SandboxError() from e
        except SandboxOOMError as e:
            state["sandbox_oom"] = True
            self.logger.error(f"Sandbox OOM during background job: {repr(e)}")
            raise vf.SandboxError() from e
        except SandboxTimeoutError as e:
            state["sandbox_timeout"] = True
            self.logger.error(f"Sandbox timeout during background job: {repr(e)}")
            raise vf.SandboxError() from e

        try:
            for elapsed in range(0, timeout + poll_interval, poll_interval):
                results = await get_job(sandbox_id, job)
                if results.completed:
                    return results
                self.logger.debug(
                    f"{sandbox_id=}: Polling job... {elapsed} / {timeout} seconds elapsed"
                )
                await asyncio.sleep(poll_interval)
        except SandboxOOMError as e:
            state["sandbox_oom"] = True
            self.logger.error(f"Sandbox OOM during polling: {repr(e)}")
            raise vf.SandboxError() from e
        except SandboxTimeoutError as e:
            state["sandbox_timeout"] = True
            self.logger.error(f"Sandbox timeout during polling: {repr(e)}")
            raise vf.SandboxError() from e

        raise CommandTimeoutError(
            sandbox_id=sandbox_id, command=command, timeout=timeout
        )

    async def upload_file(
        self,
        sandbox_id: str,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Upload a local file to the sandbox."""
        try:
            await self.sandbox_client.upload_file(sandbox_id, remote_path, local_path)
        except SandboxOOMError as e:
            raise vf.SandboxError(
                f"Sandbox {sandbox_id} OOM during upload to {remote_path}"
            ) from e
        except SandboxTimeoutError as e:
            raise vf.SandboxError(
                f"Sandbox {sandbox_id} timeout during upload to {remote_path}"
            ) from e
        except APIError as e:
            raise vf.SandboxError(
                f"API error uploading to {remote_path} in {sandbox_id}: {e}"
            ) from e

    async def upload_content(
        self,
        sandbox_id: str,
        content: str,
        remote_path: str,
    ) -> None:
        """Upload a string as a file to the sandbox."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            local_path = f.name
        try:
            await self.upload_file(sandbox_id, remote_path, local_path)
        finally:
            Path(local_path).unlink(missing_ok=True)

    async def upload_bundle(
        self,
        sandbox_id: str,
        file_map: dict[str, str],
        dest_dir: str,
    ) -> None:
        """Upload a bundle of files to the sandbox.

        Builds a tar.gz archive from ``file_map`` (relative path → UTF-8
        content), uploads it, and extracts into ``dest_dir``.
        """
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for rel_path, content in file_map.items():
                data = content.encode("utf-8")
                info = tarfile.TarInfo(name=rel_path)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        bundle_bytes = buf.getvalue()

        archive_remote = f"{dest_dir}/_bundle.tar.gz"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as f:
            f.write(bundle_bytes)
            tmp_path = f.name
        try:
            await self.upload_file(sandbox_id, archive_remote, tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        extract_cmd = (
            f"mkdir -p {dest_dir} && "
            f'python3 -c "import tarfile; '
            f"tarfile.open('{archive_remote}', 'r:gz').extractall('{dest_dir}')\" && "
            f"rm -f {archive_remote}"
        )
        result = await self.sandbox_client.execute_command(
            sandbox_id,
            extract_cmd,
            timeout=60,
        )
        if result.exit_code != 0:
            raise vf.SandboxError(
                f"Bundle extract failed in {sandbox_id} (exit={result.exit_code}): "
                f"{(result.stderr or '')[:200]}"
            )

    def add_sandbox_scorer(self, scorer: SandboxScorer) -> None:
        """Register a sandbox scorer to run during post_rollout."""
        self.sandbox_scorers.append(scorer)

    async def run_sandbox_scorers(self, state: dict[str, Any]) -> None:
        """Run all registered sandbox scorers sequentially.

        Each scorer uploads its files + script to the sandbox via bundle, runs
        the script, and stores the result in ``state[scorer.state_key]``.
        Skipped when there is no sandbox or an error occurred.
        """
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return
        if state.get("error") or state.get("sandbox_error"):
            return

        for scorer in self.sandbox_scorers:
            try:
                files = scorer.files_fn(state)
                files["score.py"] = scorer.script
                await self.upload_bundle(sandbox_id, file_map=files, dest_dir=scorer.dest_dir)
                result = await self.sandbox_client.execute_command(
                    sandbox_id,
                    f"python3 {scorer.dest_dir}/score.py",
                    working_dir=scorer.dest_dir,
                    timeout=scorer.timeout,
                )
                if result.exit_code == 0 and result.stdout.strip():
                    state[scorer.state_key] = float(result.stdout.strip().splitlines()[-1])
                else:
                    stderr = (result.stderr or "")[:200]
                    self.logger.warning(
                        f"Sandbox scorer '{scorer.name}' failed (exit={result.exit_code}): {stderr}"
                    )
                    state[scorer.state_key] = 0.0
            except Exception as e:
                self.logger.warning(
                    f"Sandbox scorer '{scorer.name}' error: {type(e).__name__}: {e}"
                )
                state[scorer.state_key] = 0.0

    def teardown_sandboxes(self):
        """Delete all active sandboxes using sync client.

        Uses the synchronous SandboxClient for teardown to avoid event loop issues
        during signal handling and interpreter shutdown.
        """
        if not self.active_sandboxes:
            return
        self.logger.info(f"Deleting {len(self.active_sandboxes)} remaining sandboxes")
        sync_client = SandboxClient(APIClient())
        sandbox_ids = list(self.active_sandboxes)
        batch_size = 100
        for i in range(0, len(sandbox_ids), batch_size):
            batch = sandbox_ids[i : i + batch_size]
            try:
                sync_client.bulk_delete(sandbox_ids=batch)
                for sandbox_id in batch:
                    self.active_sandboxes.discard(sandbox_id)
                self.logger.debug(f"Bulk deleted batch of {len(batch)} sandboxes")
            except Exception as e:
                self.logger.warning(f"Bulk delete failed for batch: {e}")

    def teardown_sandbox_client(self):
        """Teardown the threaded sandbox client."""
        self.sandbox_client.teardown()

    @vf.teardown(priority=-10)
    async def teardown_mixin_sandboxes(self) -> None:
        """Default teardown handler for deleting tracked sandboxes.

        Override ``teardown_sandboxes`` in subclasses to customize behavior while
        keeping this auto-registered handler.
        """
        self.teardown_sandboxes()

    @vf.teardown(priority=-20)
    async def teardown_mixin_sandbox_client(self) -> None:
        """Default teardown handler for threaded sandbox client shutdown.

        Override ``teardown_sandbox_client`` in subclasses to customize behavior
        while keeping this auto-registered handler.
        """
        self.teardown_sandbox_client()
