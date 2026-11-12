"""A bubblewrap filesystem boundary for one agent's runtime view."""

from __future__ import annotations

import asyncio
import contextlib
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from verifiers.v1.configs.runtime import (
    FILESYSTEM_DEFAULT_HIDDEN,
    FRAMEWORK_PRIVATE_DIR,
    FilesystemPolicyConfig,
)
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
_INSTALL_BUBBLEWRAP = r"""
set -eu
if ! command -v bwrap >/dev/null; then
    if command -v apt-get >/dev/null; then
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq bubblewrap
    elif command -v apk >/dev/null; then
        apk add --no-cache bubblewrap
    else
        echo 'install bubblewrap in the task image to enable filesystem confinement' >&2
        exit 1
    fi
fi
bwrap --help | grep -q -- --disable-userns || {
    echo 'filesystem confinement requires bubblewrap with --disable-userns support' >&2
    exit 1
}
"""
_DEFAULT_WRITABLE = r"""
set -eu
for path in "$(pwd -P)" "${HOME:-}" "${TMPDIR:-/tmp}" /tmp /var/tmp /logs/artifacts; do
    [ -n "$path" ] || continue
    case "$path" in /*) ;; *) echo 'default writable paths must be absolute' >&2; exit 1;; esac
    canonical=$(readlink -f -- "$path")
    # A container can start at /; that must never make its entire filesystem writable.
    [ "$canonical" != / ] || continue
    printf '%s\0' "$canonical"
done
"""
_PREPARE_DIRECTORIES = r"""
set -eu
mode=$1
shift
for path do
    [ "$(readlink -m -- "$path")" = "$path" ] || {
        printf 'default hidden path must not contain symlinks: %s\n' "$path" >&2
        exit 1
    }
    mkdir -p -m "$mode" -- "$path"
done
"""
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
_HARDLINK_PROBE = r"""
set -eu
label=$1
shift
command -v find >/dev/null || {
    echo 'filesystem alias validation requires find' >&2
    exit 1
}
for path do
    alias=$(find "$path" -type f -links +1 -print -quit)
    [ -z "$alias" ] || {
        printf 'filesystem %s tree contains a hard-linked file: %s\n' "$label" "$alias" >&2
        exit 1
    }
done
"""
_MOUNT_ALIAS_PROBE = r"""
set -eu
mode=$1
protected=$2
shift 2

contains() {
    [ "$1" = / ] && return 0
    [ "$1" = "$2" ] && return 0
    case "$2" in "$1"/*) return 0;; esac
    return 1
}

resolve_identity() {
    target=$1
    resolved_device=
    resolved_root=
    resolved_length=-1
    resolved_ambiguous=false
    while IFS=' ' read -r _ _ device root mountpoint _; do
        root=$(printf '%b' "$root")
        mountpoint=$(printf '%b' "$mountpoint")
        contains "$mountpoint" "$target" || continue
        length=${#mountpoint}
        [ "$length" -lt "$resolved_length" ] && continue
        if [ "$length" -eq "$resolved_length" ]; then
            resolved_ambiguous=true
            continue
        fi
        resolved_ambiguous=false
        resolved_device=$device
        if [ "$mountpoint" = / ]; then
            suffix=$target
        else
            suffix=${target#"$mountpoint"}
        fi
        if [ "$root" = / ]; then
            resolved_root=${suffix:-/}
        else
            resolved_root=${root%/}$suffix
        fi
        resolved_length=$length
    done < /proc/self/mountinfo
    if [ "$resolved_ambiguous" = true ]; then
        printf 'ambiguous filesystem identity for %s: stacked mounts\n' "$target" >&2
        exit 1
    fi
    [ -n "$resolved_device" ] || {
        printf 'cannot resolve filesystem identity for %s\n' "$target" >&2
        exit 1
    }
}

check_identity() {
    protected_path=$1
    protected_device=$2
    protected_root=$3
    shift 3
    while IFS=' ' read -r _ _ device root mountpoint _; do
        [ "$device" = "$protected_device" ] || continue
        root=$(printf '%b' "$root")
        mountpoint=$(printf '%b' "$mountpoint")
        for candidate do
            if contains "$candidate" "$mountpoint"; then
                visible_path=$mountpoint
                visible_root=$root
            elif contains "$mountpoint" "$candidate"; then
                visible_path=$candidate
                if [ "$mountpoint" = / ]; then
                    suffix=$candidate
                else
                    suffix=${candidate#"$mountpoint"}
                fi
                visible_root=${root%/}$suffix
            else
                continue
            fi
            # A candidate inside a mount exposes only that source subtree, not
            # the whole mount (e.g. /alias/work does not expose /alias/etc).
            if contains "$visible_root" "$protected_root"; then
                if [ "$visible_root" = / ]; then
                    suffix=$protected_root
                else
                    suffix=${protected_root#"$visible_root"}
                fi
                alias_path=${visible_path%/}$suffix
                alias_path=${alias_path:-/}
            elif contains "$protected_root" "$visible_root"; then
                alias_path=$visible_path
            else
                continue
            fi
            # Compare the exposed path, not its containing mount: hiding a bind
            # target must also reject the original source still visible at /.
            contains "$protected_path" "$alias_path" && continue
            printf 'filesystem %s path aliases protected tree: %s and %s\n' \
                "$mode" "$alias_path" "$protected_path" >&2
            exit 1
        done
    done < /proc/self/mountinfo
}

check_protected() {
    protected_path=$1
    shift
    resolve_identity "$protected_path"
    check_identity "$protected_path" "$resolved_device" "$resolved_root" "$@"
    # A protected tree can contain mounts backed by unrelated devices or roots.
    # Masking the parent hides these too, so reject their exposed aliases as well.
    while IFS=' ' read -r _ _ child_device child_root child_path _; do
        child_path=$(printf '%b' "$child_path")
        [ "$child_path" = "$protected_path" ] && continue
        contains "$protected_path" "$child_path" || continue
        child_root=$(printf '%b' "$child_root")
        check_identity "$protected_path" "$child_device" "$child_root" "$@"
    done < /proc/self/mountinfo
}

if [ "$protected" = bootstrap ]; then
    for protected_path in /usr /bin /sbin /lib /lib64 /etc; do
        check_protected "$protected_path" "$@"
    done
else
    check_protected "$protected" "$@"
fi
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


async def prepare_filesystem_tools(runtime: Runtime) -> None:
    """Install the namespace tool during trusted setup, before restricting egress."""
    trusted = runtime.backend.bind(env=bootstrap_env({})).with_user("root")
    result = await trusted.run(["/bin/sh", "-c", _INSTALL_BUBBLEWRAP], {})
    if result.exit_code:
        raise SandboxError(
            "cannot prepare filesystem confinement - "
            + (result.stderr or result.stdout).strip()[-1000:]
        )


async def prepare_private_directory(runtime: Runtime) -> str:
    """Keep trusted setup scratch on disk, outside every agent-writable tree."""
    trusted = runtime.backend.bind(env=bootstrap_env({})).with_user("root")
    result = await trusted.run(
        [
            "/bin/sh",
            "-c",
            _PREPARE_DIRECTORIES,
            "filesystem-policy",
            "1777",
            FRAMEWORK_PRIVATE_DIR,
        ],
        {},
    )
    if result.exit_code:
        raise SandboxError(
            "cannot prepare framework private directory - "
            + result.stderr.strip()[-1000:]
        )
    return FRAMEWORK_PRIVATE_DIR


async def prepare_filesystem(
    runtime: Runtime, policy: FilesystemPolicyConfig
) -> FilesystemSandbox:
    if not runtime.backend.SUPPORTS_FILESYSTEM_POLICY:
        raise SandboxError(
            f"the {runtime.type} runtime does not enforce filesystem policies"
        )
    await prepare_private_directory(runtime)
    trusted = runtime.backend.bind(env=bootstrap_env({})).with_user("root")
    defaults = await trusted.run(
        [
            "/bin/sh",
            "-c",
            _PREPARE_DIRECTORIES,
            "filesystem-policy",
            "755",
            *FILESYSTEM_DEFAULT_HIDDEN[1:],
        ],
        {},
    )
    if defaults.exit_code:
        raise SandboxError(
            "cannot prepare default hidden directories - "
            + defaults.stderr.strip()[-1000:]
        )
    if policy.writable is None:
        scratch = await trusted.run(
            [
                "/bin/sh",
                "-c",
                _PREPARE_DIRECTORIES,
                "filesystem-policy",
                "1777",
                "/tmp",
                "/var/tmp",
                "/logs/artifacts",
            ],
            {},
        )
        if scratch.exit_code:
            raise SandboxError(
                "cannot prepare default writable directories - "
                + scratch.stderr.strip()[-1000:]
            )
        discovered = await runtime.run(
            ["/bin/sh", "-c", _DEFAULT_WRITABLE], bootstrap_env({})
        )
        if discovered.exit_code:
            raise SandboxError(
                "cannot resolve default writable directories - "
                + discovered.stderr.strip()[-1000:]
            )
        roots = list(dict.fromkeys(discovered.stdout.rstrip("\0").split("\0")))
        # HOME, workdir and TMPDIR can nest; bind each tree only once.
        roots = [
            path
            for path in roots
            if not any(
                path != other and PurePosixPath(path).is_relative_to(other)
                for other in roots
            )
        ]
        policy = FilesystemPolicyConfig.model_validate(
            {**policy.model_dump(), "writable": roots}
        )
    assert policy.writable is not None
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
    for label, roots in (("writable", policy.writable), ("hidden", policy.hidden)):
        if not roots:
            continue
        links = await trusted.run(
            ["/bin/sh", "-c", _HARDLINK_PROBE, "filesystem-policy", label, *roots],
            bootstrap_env({}),
        )
        if links.exit_code:
            raise SandboxError(
                "cannot prepare filesystem policy - " + links.stderr.strip()[-1000:]
            )
    if policy.writable:
        aliases = await trusted.run(
            [
                "/bin/sh",
                "-c",
                _MOUNT_ALIAS_PROBE,
                "filesystem-policy",
                "writable",
                "bootstrap",
                *policy.writable,
            ],
            bootstrap_env({}),
        )
        if aliases.exit_code:
            raise SandboxError(
                "cannot prepare filesystem policy - " + aliases.stderr.strip()[-1000:]
            )
    visible = list(dict.fromkeys((*policy.readable, *policy.writable)))
    for hidden in policy.hidden:
        aliases = await trusted.run(
            [
                "/bin/sh",
                "-c",
                _MOUNT_ALIAS_PROBE,
                "filesystem-policy",
                "readable",
                hidden,
                *visible,
            ],
            bootstrap_env({}),
        )
        if aliases.exit_code:
            raise SandboxError(
                "cannot prepare filesystem policy - " + aliases.stderr.strip()[-1000:]
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
        if any(
            path != other and PurePosixPath(path).is_relative_to(other)
            for other in policy.hidden
        ):
            continue
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
