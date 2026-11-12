"""A bubblewrap filesystem boundary for one agent's runtime view."""

from __future__ import annotations

import asyncio
import contextlib
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from verifiers.v1.configs.runtime import FilesystemPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.utils.aio import run_shielded

if TYPE_CHECKING:
    from verifiers.v1.runtimes.runtime import Runtime


# These variables can run or load agent-controlled code before bubblewrap enters
# its namespace. Restore their requested values only inside the namespace.
_BOOTSTRAP_EMPTY = (
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "LD_DEBUG",
    "LD_DEBUG_OUTPUT",
    "LD_PROFILE",
    "LD_ORIGIN_PATH",
    "GCONV_PATH",
    "LOCPATH",
    "GLIBC_TUNABLES",
    "BASH_ENV",
    "ENV",
)
_BOOTSTRAP_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
_PROBE = r"""
set -eu
binary=$(command -v bwrap) || { echo 'bubblewrap is not installed' >&2; exit 1; }
readlink -f -- "$binary"
pwd -P
for path do
    canonical=$(readlink -f -- "$path")
    [ "$canonical" = "$path" ] || {
        printf 'filesystem path must not contain symlinks: %s\n' "$path" >&2
        exit 1
    }
    if [ -d "$path" ]; then
        printf 'directory\n'
    elif [ -f "$path" ]; then
        printf 'file\n'
    else
        printf 'filesystem path must be an existing directory or regular file: %s\n' "$path" >&2
        exit 1
    fi
done
"""


def bootstrap_env(env: dict[str, str]) -> dict[str, str]:
    return {
        **{
            key: value
            for key, value in env.items()
            if not key.startswith(("LD_", "BASH_FUNC_"))
        },
        **dict.fromkeys(_BOOTSTRAP_EMPTY, ""),
        "PATH": _BOOTSTRAP_PATH,
        "SHELL": "/bin/sh",
    }


@dataclass(frozen=True)
class FilesystemSandbox:
    prefix: tuple[str, ...]

    def command(self, argv: list[str], env: dict[str, str]) -> list[str]:
        return [
            # Docker exec can be parented by a short-lived runc thread. Keep a
            # shell waiting for bwrap so --die-with-parent tracks this process,
            # rather than killing the sandbox when that thread exits.
            "/bin/sh",
            "-c",
            '"$@"; status=$?; exit "$status"',
            "filesystem-policy",
            *self.prefix,
            *(arg for key, value in env.items() for arg in ("--setenv", key, value)),
            "--",
            *argv,
        ]


async def prepare_filesystem(
    runtime: Runtime, policy: FilesystemPolicyConfig
) -> FilesystemSandbox:
    if not runtime.backend.SUPPORTS_FILESYSTEM_POLICY:
        raise SandboxError(
            f"the {runtime.type} runtime does not enforce filesystem policies"
        )
    paths = list(dict.fromkeys((*policy.readable, *policy.writable, *policy.hidden)))
    probe = await runtime.run(
        ["/bin/sh", "-c", _PROBE, "filesystem-policy", *paths],
        bootstrap_env({}),
    )
    fields = probe.stdout.splitlines()
    if probe.exit_code or len(fields) != len(paths) + 2:
        raise SandboxError(
            f"cannot prepare filesystem policy (exit code {probe.exit_code}) - "
            + probe.stderr.strip()[-1000:]
        )
    binary, workdir, *kinds = fields
    if not binary.startswith(("/usr/", "/bin/", "/sbin/")):
        raise SandboxError(
            "bubblewrap must be installed in a protected system directory"
        )
    types = dict(zip(paths, kinds, strict=True))
    # Every process gets its own mount and PID namespaces. Disabling additional
    # user namespaces and dropping capabilities prevents remounting masked trees.
    command = [
        binary,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--disable-userns",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--cap-drop",
        "ALL",
    ]
    mounts = {path: "--ro-bind" for path in policy.readable}
    mounts.update(dict.fromkeys(policy.writable, "--bind"))
    for path, flag in sorted(
        mounts.items(), key=lambda item: (len(PurePosixPath(item[0]).parts), item[0])
    ):
        command.extend((flag, path, path))
    command.extend(("--proc", "/proc", "--dev", "/dev"))
    for path in sorted(policy.hidden, key=lambda path: len(PurePosixPath(path).parts)):
        if types[path] == "directory":
            command.extend(("--tmpfs", path, "--remount-ro", path))
        else:
            command.extend(("--ro-bind", "/dev/null", path))
    # The empty sandbox root otherwise remains writable even with no writable
    # paths. This does not remount explicitly writable child mounts.
    command.extend(("--remount-ro", "/", "--chdir", workdir))
    sandbox = FilesystemSandbox(tuple(command))
    result = await runtime.run(
        sandbox.command(["/bin/sh", "-c", ":"], {}), bootstrap_env({})
    )
    if result.exit_code:
        raise SandboxError(
            "filesystem isolation requires working bubblewrap user namespaces, "
            "--disable-userns and an accessible shell/workdir "
            f"(exit code {result.exit_code}) - " + result.stderr.strip()[-1000:]
        )
    return sandbox


async def read_file(runtime: Runtime, path: str, max_bytes: int | None) -> bytes:
    argv = ["/bin/cat"] if max_bytes is None else ["/bin/head", "-c", str(max_bytes)]
    process = await runtime.exec([*argv, "--", path], {})

    async def drain(stream, limit: int | None = None) -> bytes:
        data = bytearray()
        async for chunk in stream:
            data.extend(chunk)
            if limit is not None and len(data) > limit:
                raise SandboxError("filesystem read exceeded its byte limit")
        return bytes(data)

    try:
        data, error, code = await asyncio.gather(
            drain(process.stdout, max_bytes), drain(process.stderr), process.wait()
        )
    except BaseException:
        with contextlib.suppress(Exception):
            await run_shielded(process.kill())
        raise
    if code:
        raise SandboxError(f"read {path!r}: {error.decode(errors='replace')[-500:]}")
    return data


async def write_file(runtime: Runtime, path: str, data: bytes) -> None:
    # Read exactly the payload length, so no backend-specific stdin-close API is
    # needed. Both directory creation and opening the destination are confined.
    process = await runtime.exec(
        [
            "/bin/sh",
            "-c",
            '/bin/mkdir -p -- "$(/usr/bin/dirname -- "$1")" && /bin/head -c "$2" > "$1"',
            "filesystem-write",
            path,
            str(len(data)),
        ],
        {},
    )

    async def drain(stream) -> bytes:
        data = bytearray()
        async for chunk in stream:
            data.extend(chunk)
            del data[:-1000]
        return bytes(data)

    output = asyncio.gather(
        drain(process.stdout), drain(process.stderr), process.wait()
    )
    try:
        if data:
            await process.write(data)
        _, error, code = await output
    except BaseException as error:
        with contextlib.suppress(Exception):
            await run_shielded(process.kill())
        output.cancel()
        with contextlib.suppress(BaseException):
            await output
        if isinstance(error, Exception):
            raise SandboxError(f"write {path!r}: {str(error)[-500:]}") from error
        raise
    if code:
        raise SandboxError(f"write {path!r}: {error.decode(errors='replace')[-500:]}")


def background_command(argv: list[str], log: str) -> list[str]:
    return [
        "/bin/sh",
        "-c",
        f'exec "$@" > {shlex.quote(log)} 2>&1 < /dev/null',
        "filesystem-background",
        *argv,
    ]
