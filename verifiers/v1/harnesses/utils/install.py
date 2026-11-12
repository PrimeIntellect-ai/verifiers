"""Host-side helpers for installing a harness's program into a runtime."""

import shlex

from verifiers.v1.runtimes import Runtime

_COPY_LINKED_FILES = r"""
find "$1" -type f -links +1 -exec sh -c '
    set -eu
    for file do
        temporary=$(mktemp "${file}.vf-XXXXXX")
        if cp -p -- "$file" "$temporary" && mv -f -- "$temporary" "$file"; then
            :
        else
            rm -f -- "$temporary"
            exit 1
        fi
    done
' sh {} +
"""


async def ensure_installed(
    runtime: Runtime,
    *,
    directory: str,
    install: str,
    env: dict[str, str],
    label: str,
    ready: str | None = None,
    lock: str | None = None,
    shell: tuple[str, ...] = ("sh", "-c"),
) -> None:
    """Run `install` in `runtime` under a lock, so concurrent rollouts sharing the runtime
    install once and the rest wait. `ready` is a shell test that skips the install when it
    already holds. The lock is `directory/install.lock` unless the install replaces
    `directory` itself, in which case pass a `lock` path beside it. `shell` runs the script
    (e.g. `bash -o pipefail -c` for pipelines).

    The lock is `flock` (Linux, busybox) or `lockf` (macOS/BSD), both released when the holder
    dies; an image with neither falls back to a symlink spinlock whose owner is recorded as
    `pid:starttime` so a dead holder is reaped even if its pid was reused."""
    # Writable trees are checked for hard-link aliases before agent confinement.
    env = {"UV_LINK_MODE": "copy", **env}
    lock = shlex.quote(lock or f"{directory}/install.lock")
    script = f"{ready} || ({install})" if ready else install
    # npm post-install scripts can hard-link their binaries. Detach only this
    # framework-owned installation, under its lock, before the agent can use it.
    script = (
        f"({script}) && sh -c {shlex.quote(_COPY_LINKED_FILES)} "
        f"harness-install {shlex.quote(directory)}"
    )
    run = f"{shlex.join(shell)} {shlex.quote(script)}"
    # The owner token is pid:starttime (starttime from /proc where readable), so a reused pid
    # does not pass as the live holder. A lock that is not a symlink (left behind by
    # flock/lockf) or whose owner is gone is stale and reaped; `kill -0 ""` succeeds under
    # busybox ash, hence the explicit -z.
    ident = (
        'ident() { if [ -r "/proc/$1/stat" ] && s=$(sed "s/^.*) //" "/proc/$1/stat" '
        '| cut -d" " -f20) && [ -n "$s" ]; then echo "$1:$s"; else echo "$1"; fi; }; '
        "me=$(ident $$); "
    )
    spinlock = (
        f"{ident}"
        f'until ln -s "$me" {lock} 2>/dev/null; do owner=$(readlink {lock} 2>/dev/null); '
        f'pid=${{owner%%:*}}; if [ -z "$owner" ] || ! kill -0 "$pid" 2>/dev/null '
        f'|| [ "$(ident "$pid")" != "$owner" ]; then '
        f'[ "$(readlink {lock} 2>/dev/null)" != "$owner" ] || rm -f {lock}; fi; '
        f"sleep 0.1; done; "
        f'trap \'[ "$(readlink {lock} 2>/dev/null)" != "$me" ] || rm -f {lock}\' EXIT; {run}'
    )
    guarded = (
        f'mkdir -p "$(dirname {lock})" && '
        f'if l=$(command -v flock || command -v lockf); then "$l" {lock} {run}; else {spinlock}; fi'
    )
    if runtime.user is None:
        result = await runtime.run(["sh", "-c", guarded], env)
    else:
        # Install as root (system packages), keeping uv's Python in `directory`, then
        # open it to the agent user: some installers leave it private.
        python_dir = {
            "UV_PYTHON_INSTALL_DIR": f"{directory}/python",
            "UV_CACHE_DIR": f"{directory}/uv-cache",
        }
        shareable = f"{guarded} && chmod -R a+rX {shlex.quote(directory)}"
        result = await runtime.with_user(None).run(
            ["sh", "-c", shareable], {**python_dir, **env}
        )
    if result.exit_code != 0:
        detail = (result.stderr.strip() or result.stdout.strip())[-500:]
        raise RuntimeError(f"{label} install failed: {detail}")


async def remove_dir(runtime: Runtime, path: str, label: str) -> None:
    """Delete `path` in `runtime`; a failure names `label` (what the path held)."""
    result = await runtime.run(["rm", "-rf", path], {})
    if result.exit_code != 0:
        raise RuntimeError(
            f"failed to clean up {label}: {result.stderr.strip()[-500:]}"
        )
