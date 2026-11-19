"""Host-side helpers for installing a harness's program into a runtime."""

import hashlib
import shlex

from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.prepared import load_prepared, record_arguments


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
    await load_prepared(runtime.with_user(None))
    cache = runtime.backend.cache
    install_key = (
        f"install:{directory}:"
        + hashlib.sha256(f"{ready or ''}\0{install}".encode()).hexdigest()[:16]
    )
    if ready and install_key in cache.installed:
        return
    record = shlex.join(
        record_arguments(
            runtime.with_user(None),
            install_key,
            {"install": directory, **({"ready": ready} if ready else {})},
        )
    )
    save_record = f'if py=$(command -v python3 || command -v python 2>/dev/null); then "$py" {record} || true; fi'
    lock = shlex.quote(lock or f"{directory}/install.lock")
    script = f"({ready}) || ({install})" if ready else install
    script = f"({script}) && {save_record}"
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
    unlock = f'[ "$(readlink {lock} 2>/dev/null)" != "$me" ] || rm -f {lock}'
    spinlock = (
        f"{ident}"
        f'until ln -s "$me" {lock} 2>/dev/null; do owner=$(readlink {lock} 2>/dev/null); '
        f'pid=${{owner%%:*}}; if [ -z "$owner" ] || ! kill -0 "$pid" 2>/dev/null '
        f'|| [ "$(ident "$pid")" != "$owner" ]; then '
        f'[ "$(readlink {lock} 2>/dev/null)" != "$owner" ] || rm -f {lock}; fi; '
        f"sleep 0.1; done; "
        f"trap {shlex.quote(unlock)} EXIT; {run}"
    )
    guarded = (
        f'mkdir -p "$(dirname {lock})" && '
        f'if l=$(command -v flock || command -v lockf); then "$l" {lock} {run}; else {spinlock}; fi'
    )
    # Different harness/script/package installers share uv and the OS package
    # manager, even when their target directories and filesystem locks differ.
    async with cache.setup_lock:
        if ready and install_key in cache.installed:
            return
        if runtime.user is None:
            result = await runtime.run(["sh", "-c", guarded], env)
        else:
            # Install as root, keeping uv's Python open to the agent user.
            python_dir = {
                "UV_PYTHON_INSTALL_DIR": f"{directory}/python",
                "UV_CACHE_DIR": f"{directory}/uv-cache",
            }
            shareable = f"{guarded} && chmod -R a+rX {shlex.quote(directory)}"
            result = await runtime.with_user(None).run(
                ["sh", "-c", shareable], {**python_dir, **env}
            )
        if result.exit_code == 0 and ready:
            cache.installed.add(install_key)
    if result.exit_code != 0:
        detail = (result.stderr.strip() or result.stdout.strip())[-500:]
        raise RuntimeError(f"{label} install failed: {detail}")


async def remove_dir(runtime: Runtime, path: str, label: str) -> None:
    """Delete `path` in `runtime`; a failure names `label` (what the path held)."""
    if (
        runtime.user is None
        and (lease := getattr(runtime, "_lease", None)) is not None
        and (daemon := getattr(lease.entry, "daemon", None)) is not None
        and daemon.alive
    ):
        base_dir = (
            runtime.workdir
            or getattr(runtime.backend.config, "workdir", "/app")
            or "/app"
        )
        target = path if path.startswith("/") else f"{base_dir.rstrip('/')}/{path}"
        await daemon.remove_dir(target)
        return
    result = await runtime.run(["rm", "-rf", path], {})
    if result.exit_code != 0:
        raise RuntimeError(
            f"failed to clean up {label}: {result.stderr.strip()[-500:]}"
        )
