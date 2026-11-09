"""Evaluator-owned Prime sandboxes for RewardKit Worker calls."""

import asyncio
import json
from pathlib import Path

from pydantic import Field

from verifiers.v1.errors import TaskError
from verifiers.v1.runtimes import PrimeConfig, Runtime, provision_runtime
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.artifacts import MAX_ARTIFACT_BYTES


class HarborWorkerConfig(PrimeConfig):
    image: str = "python:3.12-slim"
    allow: list[str] = Field(default_factory=list)


async def run_programmatic(
    runtime: Runtime, argv: list[str], env: dict[str, str], config: PrimeConfig, trace
) -> ProgramResult:
    from rewardkit import sandbox

    if config.checkpoint or config.mounts:
        raise TaskError("RewardKit Workers require fresh images without shared mounts")
    source = Path(sandbox.__file__).parent
    sources = {
        name: (source / name).read_bytes()
        for name in ("sandbox.py", "_worker_bootstrap.py", "_worker_cleanup.py")
    }
    sources["worker.py"] = Path(__file__).with_name("worker.py").read_bytes()
    process = await runtime.exec(argv, env)
    stderr = bytearray()

    async def drain_errors():
        async for chunk in process.stderr:
            stderr.extend(chunk)
            del stderr[:-16384]

    async def serve():
        pending = bytearray()
        async for chunk in process.stdout:
            pending.extend(chunk)
            if len(pending) > 65536:
                raise TaskError("Worker request exceeds the size limit")
            while b"\n" in pending:
                line, _, tail = pending.partition(b"\n")
                pending[:] = tail
                path = json.loads(line)["path"]
                archive = await runtime.read(
                    path + "/request.tar", max_bytes=MAX_ARTIFACT_BYTES
                )
                # Never inherit the verifier's image, credentials, mounts,
                # checkpoint or LAN. Only the authored Worker inputs cross over.
                async with provision_runtime(config) as child:
                    trace.info.setdefault("scoring_diagnostics", {}).setdefault(
                        "harbor_workers", []
                    ).append(child.info.model_dump(mode="json"))
                    root = "/tmp/vf-worker"
                    for name, data in sources.items():
                        await child.write(f"{root}/{name}", data)
                    await child.write(f"{root}/request.tar", archive)
                    await child.backend.prepare_execution([])
                    result = await child.run(
                        ["python3", "-I", f"{root}/worker.py", root], {}
                    )
                    if result.exit_code:
                        raise TaskError(
                            f"Worker executor failed (exit {result.exit_code}): "
                            f"{result.stderr[-1000:]}"
                        )
                    response = await child.read(
                        f"{root}/result.json", max_bytes=MAX_ARTIFACT_BYTES
                    )
                # Run native teardown before responding; its atexit backstop also
                # retains the sandbox ID if the provider's deletion fails.
                await runtime.write(path + "/result.json", response)
                await process.write(b"{}\n")
        if pending:
            raise TaskError("Incomplete Worker request")

    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(drain_errors())
            group.create_task(serve())
            finished = group.create_task(process.wait())
        return ProgramResult(finished.result(), "", stderr.decode(errors="replace"))
    except ExceptionGroup as exc:
        cause = exc
        while isinstance(cause, ExceptionGroup):
            cause = cause.exceptions[0]
        raise cause from exc
    finally:
        if await process.poll() is None:
            await run_shielded(process.kill())
