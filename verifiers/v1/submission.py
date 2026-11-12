"""Execute a submission in a disposable runtime containing no grader state."""

import asyncio
import math
from collections.abc import Mapping
from pathlib import PurePosixPath

from verifiers.v1.configs.runtime import NetworkPolicyConfig, RuntimeConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import ProgramResult, backend_cls, make_backend
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
    solutions, or secrets. The backend must enforce finite positive `memory` and
    `disk` limits; an advisory disk request is insufficient. Input names are
    relative to `/submission`. Files belong to `user` and have mode 0644; invoke
    scripts through their interpreter or explicitly chmod them in the command.
    The caller's runtime, mounts, environment, and scoring files never enter this
    box. Both output streams share a byte ceiling. Teardown is confirmed on every
    exit, including failures and cancellation, before returning or raising.
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
    if not backend.SUPPORTS_MEMORY_LIMIT or not backend.SUPPORTS_DISK_LIMIT:
        raise ValueError(
            "submission execution requires enforced memory and disk limits; "
            f"the {config.type} runtime does not provide both"
        )
    for name in ("memory", "disk"):
        value = getattr(config, name, None)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"submission execution requires a finite positive {name} limit"
            )
    if getattr(config, "mounts", None) or getattr(config, "checkpoint", None):
        raise ValueError(
            "submission execution requires a clean image without mounts or checkpoints"
        )
    if (
        not argv
        or not user
        or not math.isfinite(timeout)
        or timeout <= 0
        or not isinstance(max_output_bytes, int)
        or max_output_bytes <= 0
    ):
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
    # Own the backend before start: provisioning may allocate a resource and then
    # fail before a provision_runtime context can yield its Runtime handle.
    resource = make_backend(config)
    primary_error: BaseException | None = None

    async def stop_and_confirm() -> None:
        try:
            await resource.stop()
        finally:
            await resource.verify_stopped()

    try:
        await resource.start()
        runtime = resource.bind(env={})
        async with asyncio.timeout(timeout):
            # The execution image is trusted; submitted bytes are inert until all
            # provisioning and privilege checks have completed.
            child = runtime.with_user(user)
            identity = await child.run(["sh", "-c", "id -u; id -g"], {})
            ids = identity.stdout.split()
            if (
                identity.exit_code
                or len(ids) != 2
                or not all(value.isdigit() for value in ids)
                or int(ids[0]) == 0
            ):
                raise SandboxError("submission user must resolve to a non-root account")
            prepared = await runtime.run(
                [
                    "sh",
                    "-c",
                    'mkdir -p /submission && chown "$1:$2" /submission && chmod 700 /submission',
                    "submission",
                    *ids,
                ],
                {},
            )
            if prepared.exit_code:
                raise SandboxError("could not prepare the submission directory")
            for name, data in files.items():
                path = str(PurePosixPath("/submission") / name)
                await child.write(path, data)
                permissions = await child.run(["chmod", "0644", "--", path], {})
                if permissions.exit_code:
                    raise SandboxError(
                        f"could not set submission input permissions: {name!r}"
                    )
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
            except ExceptionGroup as failed:
                # Keep framework errors catchable just as in other native task
                # groups; the group remains the context for concurrent failures.
                raise failed.exceptions[0] from failed
            return ProgramResult(
                status.result(),
                stdout.result().decode(errors="replace"),
                stderr.result().decode(errors="replace"),
            )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        # Stop and confirmation form one shielded operation. Cancellation during
        # either must not interrupt the other or leave it running as an orphan.
        try:
            await run_shielded(stop_and_confirm())
        except BaseException as cleanup_error:
            if isinstance(primary_error, asyncio.CancelledError):
                raise primary_error from cleanup_error
            if primary_error is not None:
                raise cleanup_error from primary_error
            raise
