"""Run a packaged evaluator in a trusted Prime VM, beside its split deployments.

The launcher owns every VM and private network. Only provisioning, release and
the job result use its pipe. The evaluator keeps its Python callbacks, trace,
model client and interception server in one process for the entire job.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.v1.interception.server import InterceptionServer

from verifiers.v1.configs.runtime import PrimeConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime, provision_runtime
from verifiers.v1.runtimes.base import register
from verifiers.v1.runtimes.rpc_wire import serve_stream
from verifiers.v1.runtimes.stdio_rpc import StreamRPC
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)


class ControllerJob:
    def __init__(self, launcher: StreamRPC, server: InterceptionServer):
        self.launcher = launcher
        self.server = server


current_job: ContextVar[ControllerJob | None] = ContextVar(
    "prime_controller", default=None
)


@asynccontextmanager
async def prime_controller(
    program: Callable, config: PrimeConfig | None = None
) -> AsyncIterator[PrimeController]:
    """Install a trusted evaluator once and lend a controller for repeated jobs.

    The packaged async program receives its JSON payload and a live Interception
    to pass to its agents. Each job has an independent process and private LANs.
    """
    from verifiers.v1.mcp.launch import install_in_runtime

    if program.__module__ == "__main__" or "<locals>" in program.__qualname__:
        raise ValueError("the evaluator must be importable from its installed package")
    async with provision_runtime(
        config or PrimeConfig(), name="vf-controller"
    ) as runtime:
        python = await install_in_runtime(program, runtime, label="trusted controller")
        controller = PrimeController(runtime, python, program)
        try:
            yield controller
        finally:
            await run_shielded(controller.close())


class PrimeController:
    def __init__(self, runtime: Runtime, python: str, program: Callable):
        self.runtime = runtime
        self._python = python
        self._program = program
        self._jobs: set[asyncio.Task] = set()
        self._closed = False

    async def close(self):
        self._closed = True
        jobs = list(self._jobs)
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)

    async def run(self, payload: dict, *, credentials: dict[str, str]):
        """Run the evaluator on C and return its JSON result.

        Deliver only the provider and model credentials the trusted program needs.
        They travel on the private launcher pipe to this process alone.
        """
        if self._closed:
            raise RuntimeError("the controller is closed")
        job = asyncio.create_task(self._run(payload, credentials))
        self._jobs.add(job)
        try:
            return await job
        finally:
            self._jobs.discard(job)

    async def _run(self, payload: dict, credentials: dict[str, str]):
        from verifiers.v1.runtimes.backends.prime import PrimeBackend, leased_client
        from verifiers.v1.runtimes.deployment import WorkerGroup

        result = None
        finished = False

        async with AsyncExitStack() as stack:
            workers = WorkerGroup()
            stack.push_async_callback(workers.close)
            process = await self.runtime.exec(
                [
                    self._python,
                    "-I",
                    "-u",
                    "-c",
                    "import asyncio; from verifiers.v1.runtimes.controller import _main; asyncio.run(_main())",
                ],
                workers.mark(self.runtime),
            )
            client = await stack.enter_async_context(leased_client())
            vms = {}
            networks = set()

            async def release_network(identifier):
                if identifier in networks:
                    await client.delete_lan_network(identifier)
                    networks.remove(identifier)

            async def handle(message):
                nonlocal finished, result
                operation = message["operation"]
                if operation == "job":
                    return {
                        "module": self._program.__module__,
                        "name": self._program.__qualname__,
                        "payload": payload,
                        "credentials": credentials,
                        "sandbox_id": self.runtime.info.id,
                    }
                if operation == "create":
                    backend = PrimeBackend(
                        PrimeConfig.model_validate(message["config"]), message["name"]
                    )
                    register(backend)
                    backend.env = message["env"]
                    backend.lan = message["lan"]
                    stack.push_async_callback(backend.stop)
                    try:
                        await backend.start()
                        await backend.wait_lan_ready()
                    except BaseException:
                        await backend.stop()
                        raise
                    vms[backend.info.id] = backend
                    auth = None
                    with contextlib.suppress(Exception):
                        auth = await backend._client._auth_cache.get_or_refresh(
                            backend.info.id
                        )
                    return {
                        "id": backend.info.id,
                        "image_cached": backend.info.image_cached,
                        "gateway_auth": auth,
                        "arch": backend.cache.arch,
                        "system_python": backend.cache.system_python,
                        "prepared_json": getattr(backend, "_prepared_json", ""),
                    }
                if operation == "stop":
                    await vms[message["sandbox_id"]].stop()
                elif operation == "expose":
                    return await vms[message["sandbox_id"]].expose(message["port"])
                elif operation == "network":
                    from prime_sandboxes import LanRule

                    async def create_network():
                        network = await client.create_lan_network(
                            message["name"],
                            rules=[
                                LanRule.model_validate(rule)
                                for rule in message["rules"]
                            ],
                        )
                        networks.add(network.id)
                        stack.push_async_callback(release_network, network.id)
                        return network

                    network = await run_shielded(create_network())
                    try:
                        if not message["join_controller"]:
                            return network.id
                        await client.add_lan_member(
                            network.id, self.runtime.info.id, ["vf-controller"]
                        )
                        # The provider can acknowledge enrollment before its VM layer
                        # has installed the directed ACL. Admit solvers only after that.
                        async with asyncio.timeout(60):
                            while True:
                                state = await client.get_lan_network(network.id)
                                if state.synced and any(
                                    member.sandbox_id == self.runtime.info.id
                                    and member.synced
                                    for member in state.members or ()
                                ):
                                    break
                                await asyncio.sleep(0.2)
                        return network.id
                    except BaseException:
                        await run_shielded(release_network(network.id))
                        raise
                elif operation == "join_network":
                    from verifiers.v1.runtimes.backends.prime import _join_lan_fast

                    await _join_lan_fast(
                        client,
                        message["network_id"],
                        message["sandbox_id"],
                        message["tags"],
                    )
                elif operation == "leave_network":
                    await client.remove_lan_member(
                        message["network_id"], message["sandbox_id"]
                    )
                elif operation == "delete_network":
                    await release_network(message["network_id"])
                elif operation == "result":
                    result, finished = message["value"], True
                else:
                    raise ValueError("unknown controller lifecycle operation")
                return None

            tail = bytearray()

            async def drain():
                async for chunk in process.stderr:
                    logger.debug(
                        "Controller output - %s",
                        chunk[-4000:].decode(errors="replace").rstrip(),
                    )
                    tail.extend(chunk)
                    del tail[:-4000]

            stderr = asyncio.create_task(drain())
            try:
                await serve_stream(handle, process.stdout, process.write)
                code = await process.wait()
                if code or not finished:
                    raise SandboxError(
                        f"trusted evaluator exited ({code}) - {tail.decode(errors='replace')}"
                    )
                return result
            finally:
                try:
                    await run_shielded(process.kill())
                finally:
                    stderr.cancel()
                    await asyncio.gather(stderr, return_exceptions=True)


async def _main():
    from verifiers.v1.interception.server import InterceptionServer

    # Keep ordinary evaluator stdout, including subprocess output, off the framed pipe.
    writer = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    reader = asyncio.StreamReader()
    loop = asyncio.get_running_loop()
    transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer
    )
    output, protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, writer
    )
    stream = asyncio.StreamWriter(output, protocol, None, loop)

    async def chunks():
        while chunk := await reader.read(65536):
            yield chunk

    async def write(packet):
        stream.write(packet)
        await stream.drain()

    rpc = StreamRPC(chunks(), write)
    try:
        args = await rpc.call("job")
        os.environ.update(args.pop("credentials"))
        os.environ["PATH"] = (
            os.path.expanduser("~/.local/bin") + os.pathsep + os.environ["PATH"]
        )
        server = InterceptionServer(lan_host=f"{args['sandbox_id']}.sandbox.internal")
        job = ControllerJob(rpc, server)
        token = current_job.set(job)
        try:
            program = importlib.import_module(args["module"])
            for part in args["name"].split("."):
                program = getattr(program, part)
            async with server:
                value = await program(args["payload"], server)
                await rpc.call("result", value=value)
        finally:
            current_job.reset(token)
    finally:
        await rpc.close()
        transport.close()
        stream.close()
