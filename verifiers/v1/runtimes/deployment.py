"""Attempt-scoped target ownership, independent of placement."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable, Collection, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field

from verifiers.v1.configs.deployment import DeploymentConfig
from verifiers.v1.errors import HarnessError, SandboxError
from verifiers.v1.runtimes import (
    DockerBackend,
    PrimeConfig,
    Runtime,
    RuntimeBackend,
    RuntimeConfig,
    provision_backend,
)
from verifiers.v1.runtimes.base import (
    RuntimeProcess,
    register,
)
from verifiers.v1.runtimes.compose import ComposeSpec
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import clip, get_logger

logger = get_logger(__name__)

_MAX_RECEIPTS = 100_000
_MAX_RECEIPT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class ServicesSpec:
    """Named Prime VMs beside the task's `main` workspace, on one private LAN.

    Each config selects a service's image and resources. Task setup starts its
    processes through `deployment.target(name)` and waits for them to be ready.
    `deployment.hosts` maps every target, including `main`, to its LAN hostname.
    """

    services: dict[str, PrimeConfig]


_MARK = "VF_EXEC_ID"
"""Environment variable that marks a worker and every process its tools start, which
inherit it through detaching and reparenting."""
_EXITED = "vf-exec: worker exited"
_WRAPPER = f'"$@"; echo "{_EXITED} ($?)"'
"""The script a worker runs under, which reports the worker's exit."""
_STOP = """
for proc in /proc/[0-9]*; do
    while IFS= read -r -d '' entry; do
        if [ "$entry" = "$1" ]; then kill -9 "${proc#/proc/}" 2>/dev/null; break; fi
    done 2>/dev/null < "$proc/environ"
done
exit 0
"""
"""Kill every process whose environment holds the marker `$1`, as the worker's user,
so it reaches nothing that user could not signal itself."""
_STOP_GROUP = """
marker=$1 wrapper=$2 group=
marked() {
    while IFS= read -r -d '' entry; do
        [ "$entry" = "$marker" ] && return 0
    done 2>/dev/null < "$1/environ"
    return 1
}
group_of() {
    local stat
    read -r stat 2>/dev/null < "$1/stat" || return
    stat=${stat##*) }
    set -- $stat
    echo "$3"
}
for proc in /proc/[0-9]*; do
    marked "$proc" || continue
    while IFS= read -r -d '' arg; do
        if [ "$arg" = "$wrapper" ]; then group=$(group_of "$proc"); break; fi
    done 2>/dev/null < "$proc/cmdline"
    [ -n "$group" ] && break
done
[ -n "$group" ] || exit 0
for proc in /proc/[0-9]*; do
    if marked "$proc" && [ "$(group_of "$proc")" = "$group" ]; then
        kill -9 "${proc#/proc/}" 2>/dev/null
    fi
done
exit 0
"""
"""Kill the worker whose wrapper runs the script `$2`, with what runs in its process
group: the processes there whose environment holds the marker `$1`, as the worker's
user."""


class WorkerGroup:
    """Reap marked workers and their descendants, including detached children."""

    def __init__(self):
        self._listeners: list[tuple[Runtime, str, bool]] = []

    def mark(self, runtime: Runtime, *, keep_detached: bool = False) -> dict[str, str]:
        worker = uuid.uuid4().hex
        self._listeners.append((runtime, f"{_MARK}={worker}", keep_detached))
        return {_MARK: worker}

    async def close(self):
        workers, self._listeners = self._listeners, []

        async def reap_one(runtime: Runtime, marker: str, keep_detached: bool) -> None:
            # Cleanup is trusted control code. Do not source task BASH_ENV or use
            # task PATH while reaping the worker under its original Unix user.
            runtime = runtime.with_env({})
            argv = [
                "/usr/bin/env",
                "-i",
                "PATH=/usr/bin:/bin",
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                _STOP_GROUP if keep_detached else _STOP,
                "worker-cleanup",
                marker,
                *([_WRAPPER] if keep_detached else []),
            ]
            try:
                result = await runtime.run(argv, {})
                if result.exit_code:
                    logger.warning(
                        "Failed to stop execution worker - %s",
                        clip(result.stderr or result.stdout),
                    )
            except Exception as error:
                logger.warning(
                    "Failed to stop execution worker - %s: %s",
                    type(error).__name__,
                    clip(str(error)),
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )

        if workers:
            await asyncio.gather(
                *(
                    reap_one(runtime, marker, keep_detached)
                    for runtime, marker, keep_detached in workers
                )
            )


@dataclass
class _Worker:
    name: str
    process: RuntimeProcess
    stdout_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stderr_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stdin_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    eof: set[str] = field(default_factory=set)


class ExecutionConnection:
    """Host-owned registration, stdio relay, and lifecycle for worker-sandbox processes.

    Harnesses can open only registered workers; neither model nor MCP credentials
    enter the worker sandbox. Receipts preserve calls across transport retries.
    """

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.commands: dict[str, tuple[list[str], dict[str, str]]] = {}
        self.workers: dict[str, _Worker] = {}
        self.receipts: dict[str, tuple[bytes, asyncio.Task]] = {}
        self._open_lock = asyncio.Lock()
        self._receipt_bytes = 0
        self.closed = False
        self._workers = WorkerGroup()
        self._stdio_marker: dict[str, str] | None = None
        self._abandon_on_close = False

    def _retain(self, size: int) -> None:
        if self._receipt_bytes + size > _MAX_RECEIPT_BYTES:
            raise HarnessError("split execution exceeded its replay buffer budget")
        self._receipt_bytes += size

    def register(self, name: str, argv: list[str], env: dict[str, str]) -> None:
        if self.closed or name in self.commands or not argv:
            raise HarnessError(f"invalid split worker registration {name!r}")
        self.commands[name] = (list(argv), dict(env))

    async def _exec(self, argv: list[str], env: dict[str, str]) -> RuntimeProcess:
        if self._stdio_marker is None:
            self._stdio_marker = self._workers.mark(self.runtime)
        return await self.runtime.exec(argv, {**env, **self._stdio_marker})

    async def handle(self, body: dict) -> dict:
        if self.closed:
            raise HarnessError("split execution bridge is closed")
        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise HarnessError("split execution requires a request_id")
        if body.get("operation") in {"kill", "wait"}:
            return await self._handle(body)
        encoded = json.dumps(body, sort_keys=True).encode()
        fingerprint = hashlib.sha256(encoded).digest()
        receipt = self.receipts.get(request_id)
        if receipt is None:
            if len(self.receipts) >= _MAX_RECEIPTS:
                raise HarnessError(
                    "split execution exceeded its request receipt budget"
                )
            size = len(encoded)
            self._retain(size + len(fingerprint))

            async def execute():
                try:
                    return await self._handle(body)
                finally:
                    self._receipt_bytes -= size

            task = asyncio.create_task(execute())
            self.receipts[request_id] = (fingerprint, task)
        else:
            previous, task = receipt
            if previous != fingerprint:
                raise HarnessError("split execution request_id was reused")
        del encoded
        return await asyncio.shield(task)

    async def _handle(self, body: dict) -> dict:
        operation = body.get("operation")
        process_id = body.get("process_id")
        if not isinstance(process_id, str) or not process_id:
            raise HarnessError("split execution requires a process_id")
        if operation == "open":
            name = body.get("worker")
            if not isinstance(name, str) or name not in self.commands:
                raise HarnessError("split execution requested an unregistered worker")
            argv, env = self.commands[name]
            async with self._open_lock:
                existing = self.workers.get(process_id)
                if existing is not None and existing.name == name:
                    return {"process_id": process_id}
                if process_id in self.workers:
                    raise HarnessError("split process_id was reused")

                async def start() -> None:
                    process = await self._exec(argv, env)
                    self.workers[process_id] = _Worker(name, process)

                await run_shielded(start())
            return {"process_id": process_id}
        worker = self.workers.get(process_id)
        if worker is None:
            raise HarnessError("split execution requested an unknown process")
        if operation == "write":
            try:
                data = base64.b64decode(body["data"], validate=True)
            except (KeyError, ValueError, TypeError) as error:
                raise HarnessError("invalid split stdin payload") from error
            async with worker.stdin_lock:
                await worker.process.write(data)
            return {}
        if operation == "read":
            stream = body.get("stream")
            if stream not in ("stdout", "stderr"):
                raise HarnessError("invalid split output stream")
            async with getattr(worker, f"{stream}_lock"):
                chunk = (
                    None
                    if stream in worker.eof
                    else await anext(getattr(worker.process, stream), None)
                )
                if chunk is None:
                    worker.eof.add(stream)
            self._retain(((len(chunk) + 2) // 3) * 4 if chunk is not None else 0)
            return {
                "data": base64.b64encode(chunk).decode() if chunk is not None else None,
                "eof": chunk is None,
            }
        if operation in {"wait", "kill"}:
            if operation == "kill":
                await worker.process.kill()
            return {"exit_code": await worker.process.wait()}
        raise HarnessError(f"unknown split execution operation {operation!r}")

    async def serve(
        self,
        runtime: Runtime,
        argv: list[str],
        env: dict[str, str],
        announce: re.Pattern[str],
        *,
        keep_detached: bool = False,
    ) -> int:
        """Start `argv` in the worker sandbox and return the port it announces."""
        marker = self._workers.mark(runtime, keep_detached=keep_detached)
        log = runtime.temp_path(f"vf-exec-{marker[_MARK]}.log")
        await runtime.run_background(
            ["sh", "-c", _WRAPPER, "sh", *argv],
            {**env, **marker},
            log,
        )
        output = ""
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(60):
                while True:
                    with contextlib.suppress(SandboxError):
                        output = (await runtime.read(log)).decode(errors="replace")
                    if _EXITED in output:
                        break
                    if found := announce.search(output):
                        return int(found.group(1))
                    await asyncio.sleep(0.2)
        raise SandboxError(f"execution worker did not start: {output[-2000:]}")

    async def abandon(self) -> None:
        if self.closed:
            return
        self.closed = True
        tasks = [task for _, task in self.receipts.values()]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(
            *(worker.process.abandon() for worker in self.workers.values()),
            return_exceptions=True,
        )
        self.receipts.clear()
        self.workers.clear()
        self._receipt_bytes = 0

    async def close(self) -> None:
        if self.closed:
            return
        if self._abandon_on_close:
            await self.abandon()
            self._workers._listeners.clear()
            return
        self.closed = True
        tasks = [task for _, task in self.receipts.values()]
        for task in tasks:
            if not task.done():
                task.cancel()
        try:
            async with asyncio.timeout(10):
                await asyncio.gather(*tasks, return_exceptions=True)
                processes = [worker.process for worker in self.workers.values()]
                await asyncio.gather(
                    *(proc.kill() for proc in processes),
                    return_exceptions=True,
                )
                await asyncio.gather(
                    *(proc.wait() for proc in processes),
                    return_exceptions=True,
                )
        except TimeoutError:
            logger.warning(
                "Timed out waiting for split workers to exit",
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
        self.receipts.clear()
        self.workers.clear()
        self._receipt_bytes = 0
        await self._workers.close()


class _DeploymentStrategy:
    """Placement, connectivity and lifecycle hooks for one deployment attempt."""

    async def start(self, deployment: Deployment) -> None:
        raise NotImplementedError

    async def prepare_execution(
        self, deployment: Deployment, routes: list[str], state_route: str | None = None
    ) -> None:
        if deployment.harness_backend is not None:
            await deployment.harness_backend.prepare_execution(routes)
        else:
            await deployment.execution_backend.prepare_execution(routes)

    async def stop_execution(self, deployment: Deployment) -> None:
        return

    async def close(self, deployment: Deployment) -> None:
        return

    async def prepare_verifier(self, deployment: Deployment) -> None:
        return

    @asynccontextmanager
    async def verifier(
        self, deployment: Deployment, config: RuntimeConfig, env: dict[str, str]
    ) -> AsyncIterator[RuntimeBackend]:
        async with provision_backend(config, env=env, workspace=True) as backend:
            yield backend


class Deployment:
    """Coordinate the execution resources for one rollout attempt.

    Provisions and connects the harness, task workspace, and supporting services,
    and cleans up resources it acquires, including fresh verification environments.
    Runtimes implement resource operations; the rollout controls agent execution.
    """

    def __init__(
        self,
        backend: RuntimeBackend,
        *,
        spec: ComposeSpec | ServicesSpec | None = None,
        tool_vms: Mapping[str, PrimeConfig] | None = None,
        tool_services: Collection[str] = (),
        env: dict[str, str] | None = None,
        config: DeploymentConfig | None = None,
        borrowed_runtime: Runtime | None = None,
        for_agent: bool = False,
        execution_network: RuntimeConfig | None = None,
        harness_image: Callable[[str], str] | None = None,
    ):
        self.backend = backend
        self.spec = spec
        self.tool_vms = dict(tool_vms or {})
        """Tool servers on Prime runtimes of their own, by server name, which join the
        workspace over the rollout's private network."""
        self.tool_services = frozenset(tool_services)
        """Task services that host tool servers, which the harness connects to."""
        self.env = dict(env or {})
        self.config = config or DeploymentConfig()
        self.borrowed_runtime = borrowed_runtime
        self.execution_network = execution_network or backend.config
        self.harness_image = harness_image
        self.for_agent = for_agent
        self.control_routes: list[str] = []
        from verifiers.v1.runtimes.strategies import _deployment_strategy

        self._strategy = _deployment_strategy(
            self.config,
            backend,
            spec,
            self.tool_vms,
            has_harness=for_agent,
        )
        self.target_backends: dict[str, RuntimeBackend] = {}
        self.hosts: dict[str, str] = {}
        """LAN hostnames of native services or tool servers on their own VMs."""
        self.harness_backend: RuntimeBackend | None = None
        self.execution_backend: RuntimeBackend | None = None
        self._stack = AsyncExitStack()
        self._owned: set[int] = set()
        self._close_task: asyncio.Task[None] | None = None

    async def own(self, backend: RuntimeBackend) -> RuntimeBackend:
        """Acquire a target; register cleanup before startup can partially fail."""
        if id(backend) in self._owned:
            raise ValueError("runtime already owned by this deployment")
        self._owned.add(id(backend))
        register(backend)
        self._stack.push_async_callback(backend.stop)
        await backend.start()
        return backend

    async def enter(self, resource):
        """Own a provider resource such as a network or composed service group."""
        return await self._stack.enter_async_context(resource)

    async def start(self) -> None:
        try:
            # Only an owned Docker workspace needs filtering support at creation.
            # Prime applies policies dynamically; local Compose owns its networks.
            if (
                self.for_agent
                and self.borrowed_runtime is None
                and self.spec is None
                and isinstance(self.backend, DockerBackend)
            ):
                self.backend.filter_workspace = True
            await self._strategy.start(self)
            if self.execution_backend is None or (
                self.for_agent and self.harness_backend is None
            ):
                raise ValueError(
                    "deployment strategy must bind execution and harness roles"
                )
        except BaseException:
            try:
                await self.close()
            except Exception:
                logger.warning(
                    "Failed to tear down the deployment after a startup error",
                    exc_info=True,
                )
            raise

    def connection(self, workspace: Runtime) -> ExecutionConnection:
        """Bind workspace execution to one connection, independent of the runtime backend."""
        if (
            self.harness_backend is None
            or self.harness_backend is self.execution_backend
        ):
            raise ValueError(
                "harness execution requires a dedicated harness runtime apart from the workspace"
            )
        return ExecutionConnection(workspace)

    @property
    def execution_name(self) -> str:
        return next(
            (
                name
                for name, target_backend in self.target_backends.items()
                if target_backend is self.execution_backend
            ),
            "main",
        )

    @property
    def workspace(self) -> Runtime:
        """Runtime view where task commands execute and task files live.

        This is the single runtime for ordinary tasks, or the selected Compose
        service; it names an execution role rather than a separate resource type.
        """
        if self.borrowed_runtime is not None:
            return self.borrowed_runtime.with_env(self.env)
        return self.execution_backend.bind(env=self.env)

    @property
    def _harness_runtime(self) -> Runtime:
        """The trusted harness view, separate from the workspace in split mode."""
        return self.harness_backend.bind(env={})

    def target(self, name: str) -> Runtime:
        """Access a named service or workspace through an execution view."""
        return (
            self.workspace
            if self.target_backends[name] is self.execution_backend
            else self.target_backends[name].bind()
        )

    @asynccontextmanager
    async def verifier(
        self, config: RuntimeConfig, env: dict[str, str]
    ) -> AsyncIterator[Runtime]:
        # Parent ownership is registered before acquisition, including cancellation.
        stack = AsyncExitStack()
        self._stack.push_async_callback(stack.aclose)
        async with stack:
            backend = await stack.enter_async_context(
                self._strategy.verifier(self, config, env)
            )
            yield backend.bind()

    async def prepare_verifier(self) -> None:
        """Release solver resources after collection, outside verifier budgets.

        Finish cleanup even if the caller is cancelled, then propagate cancellation.
        """
        await run_shielded(self._strategy.prepare_verifier(self))

    async def stop_execution(self) -> None:
        await self._strategy.stop_execution(self)

    async def prepare_execution(
        self, routes: list[str], *, state_route: str | None = None
    ) -> None:
        """Apply each target's execution policy: the harness's keeps `routes`, and a tool
        server's own VM keeps only `state_route`, its state channel."""
        await self._strategy.prepare_execution(self, routes, state_route)

    async def close(self) -> None:
        async def release():
            try:
                await self._strategy.close(self)
            finally:
                await self._stack.aclose()

        if self._close_task is None:
            self._close_task = asyncio.create_task(release())
        await run_shielded(self._close_task)
