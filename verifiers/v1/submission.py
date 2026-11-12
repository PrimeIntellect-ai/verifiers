"""Execute a submission in a disposable runtime containing no grader state."""

import asyncio
import contextlib
from collections.abc import Mapping
from pathlib import PurePosixPath

from verifiers.v1.configs.runtime import NetworkPolicyConfig, RuntimeConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import ProgramResult, backend_cls, provision_runtime
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.artifacts import MAX_ARTIFACT_BYTES


async def run_submission(
    config: RuntimeConfig,
    argv: list[str],
    files: Mapping[str, bytes],
    *,
    user: str = "nobody",
    timeout: float = 60,
    max_output_bytes: int = 1024 * 1024,
) -> ProgramResult:
    """Run untrusted files as a non-root user in a fresh, offline box.

    `config.image` must be a clean execution image, without tests, reference
    solutions, or secrets. Input names are relative to `/submission`. The caller's
    runtime, mounts, environment, and scoring files never enter this box. Both
    output streams share a byte ceiling. The entire box is destroyed before the
    result returns, including children that detached from the submitted command.
    """
    backend = backend_cls(config)
    if (
        not backend.SUPPORTS_TASK_IMAGES
        or not backend.SUPPORTS_USER_SWITCHING
        or not backend.SUPPORTS_STOP_CONFIRMATION
    ):
        raise ValueError(
            "submission execution requires isolation, user switching, and confirmed teardown"
        )
    if not isinstance(config, NetworkPolicyConfig):
        raise TypeError("submission execution requires enforced network restrictions")
    if getattr(config, "mounts", None) or getattr(config, "checkpoint", None):
        raise ValueError(
            "submission execution requires a clean image without mounts or checkpoints"
        )
    if not argv or not user or timeout <= 0 or max_output_bytes <= 0:
        raise ValueError(
            "submission execution requires a command, user, and positive limits"
        )
    if sum(len(data) for data in files.values()) > MAX_ARTIFACT_BYTES:
        raise ValueError("submission inputs exceed the artifact byte limit")
    for name in files:
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\x00" in name
            or path == PurePosixPath(".")
        ):
            raise ValueError(f"submission file must be a relative path: {name!r}")
    config = type(config).model_validate(
        {**config.model_dump(), "workdir": "/submission", "allow": [], "block": ["*"]}
    )
    async with provision_runtime(config, env={}) as runtime:
        async with asyncio.timeout(timeout):
            # The execution image is trusted; submitted bytes are inert until all
            # provisioning and privilege checks have completed.
            prepared = await runtime.run(
                ["sh", "-c", "mkdir -p /submission && chmod 777 /submission"], {}
            )
            if prepared.exit_code:
                raise SandboxError("could not prepare the submission directory")
            child = runtime.with_user(user)
            identity = await child.run(["id", "-u"], {})
            if (
                identity.exit_code
                or not identity.stdout.strip().isdigit()
                or int(identity.stdout.strip()) == 0
            ):
                raise SandboxError("submission user must resolve to a non-root account")
            for name, data in files.items():
                await runtime.write(str(PurePosixPath("/submission") / name), data)
            await runtime.backend.prepare_execution([])
            process = await child.exec(
                [
                    "env",
                    "-i",
                    "PATH=/usr/local/bin:/usr/bin:/bin",
                    "HOME=/submission",
                    "TMPDIR=/submission",
                    "sh",
                    "-c",
                    'exec "$@" < /dev/null',
                    "sh",
                    *argv,
                ],
                {},
            )
            used = 0

            async def drain(stream) -> bytes:
                nonlocal used
                chunks = []
                async for chunk in stream:
                    used += len(chunk)
                    if used > max_output_bytes:
                        raise SandboxError("submission exceeded its output byte limit")
                    chunks.append(chunk)
                return b"".join(chunks)

            try:
                async with asyncio.TaskGroup() as group:
                    stdout = group.create_task(drain(process.stdout))
                    stderr = group.create_task(drain(process.stderr))
                    status = group.create_task(process.wait())
            except BaseException:
                with contextlib.suppress(Exception):
                    await run_shielded(process.kill())
                raise
            result = ProgramResult(
                status.result(),
                stdout.result().decode(errors="replace"),
                stderr.result().decode(errors="replace"),
            )
    await runtime.backend.verify_stopped()
    return result
